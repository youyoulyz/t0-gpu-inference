//! QK-norm op — per-head RMSNorm for queries and keys (inference only).
//!
//! Qwen3 applies RMSNorm independently to each attention head's Q and K vectors.
//! Input: [seq_len, n_heads * head_dim] f32
//! Output: [seq_len, n_heads * head_dim] f32
//!
//! This is equivalent to reshaping to [seq_len * n_heads, head_dim], applying
//! RMSNorm with gamma=[head_dim], and reshaping back.

#[cfg(feature = "rocm")]
use std::sync::Arc;
#[cfg(feature = "rocm")]
use super::super::tensor::Tensor;
#[cfg(feature = "rocm")]
use super::super::gpu_context::GpuRuntime;

/// Apply per-head RMSNorm.
///
/// # Arguments
/// - `x`: [seq_len, n_heads * head_dim] f32
/// - `gamma`: [head_dim] f32 (per-head scale, shared across all heads)
/// - `n_heads`: number of attention heads
/// - `head_dim`: dimension per head (must be <= 256)
/// - `runtime`: GPU runtime
///
/// # Returns
/// - output: [seq_len, n_heads * head_dim] f32
#[cfg(feature = "rocm")]
pub fn qk_norm(
    x: &Tensor,
    gamma: &Tensor,
    n_heads: usize,
    head_dim: usize,
    runtime: &Arc<GpuRuntime>,
) -> Result<Tensor, String> {
    crate::profile_scope!("qk_norm");
    crate::profiler::set_shapes(
        vec![crate::profiler::ShapeInfo::new(x.shape())],
        vec![crate::profiler::ShapeInfo::new(x.shape())],
    );
    let shape = x.shape();
    assert!(shape.len() == 2, "qk_norm: expected 2D, got {:?}", shape);
    assert_eq!(shape[1], n_heads * head_dim, "qk_norm: dim mismatch");
    assert_eq!(gamma.numel(), head_dim, "qk_norm: gamma size mismatch");
    assert!(head_dim <= 256, "qk_norm: head_dim {} exceeds kernel limit", head_dim);

    let seq_len = shape[0];
    let total_rows = seq_len * n_heads; // treat each head as a separate row

    // Reinterpret the buffer as [seq_len * n_heads, head_dim] — no copy needed.
    // The data is already laid out as [seq_len, n_heads * head_dim], which is
    // contiguous in memory as [seq_len * n_heads, head_dim] (row-major).
    let out_buf = runtime.alloc_f32(seq_len * n_heads * head_dim)?;

    // Ensure gamma is f32 — the RMSNorm kernel reads f32 from gamma buffer.
    // Safetensors loads weights as bf16, so we need to convert.
    let gamma_f32 = if gamma.dtype() == super::super::tensor::DType::BF16 {
        let data = gamma.to_f32_vec();
        let buf = runtime.upload_f32(&data)?;
        super::super::tensor::Tensor::from_buffer(
            std::sync::Arc::new(buf), &runtime, gamma.shape(),
            super::super::tensor::DType::F32, "gamma_f32"
        )
    } else {
        gamma.clone()
    };

    let kernel = runtime.ensure_kernel_blockdsl(&format!("rmsnorm_d{}", head_dim), || {
        crate::t0::rmsnorm_kernels::build_rmsnorm_forward()
    })?;

    let ka = crate::kernargs![
        x.gpu_addr() => u64,
        gamma_f32.gpu_addr() => u64,
        out_buf.gpu_addr() => u64,
        head_dim as u32 => u32,
        1e-6f32 => f32  // eps
    ];
    let (grid_x, _) = crate::t0::rmsnorm_kernels::rmsnorm_grid(total_rows as u32);
    runtime.dispatch(&kernel, [grid_x, 1, 1], &ka)?;

    let out_arc = Arc::new(out_buf);
    Ok(Tensor::from_buffer(out_arc, runtime, shape,
        super::super::tensor::DType::F32, "qk_norm_out"))
}

/// CPU reference: per-head RMSNorm.
///
/// Treats x as [seq_len * n_heads, head_dim] and applies RMSNorm with gamma to each row.
pub fn cpu_qk_norm(x: &[f32], gamma: &[f32], n_heads: usize, head_dim: usize, seq_len: usize) -> Vec<f32> {
    let eps = 1e-6f32;
    let total_rows = seq_len * n_heads;
    let mut out = vec![0.0f32; total_rows * head_dim];

    for row in 0..total_rows {
        let base = row * head_dim;
        // RMS = sqrt(mean(x^2) + eps)
        let sum_sq: f32 = x[base..base + head_dim].iter().map(|v| v * v).sum();
        let rms = (sum_sq / head_dim as f32 + eps).sqrt();
        let inv_rms = 1.0 / rms;

        for d in 0..head_dim {
            out[base + d] = x[base + d] * inv_rms * gamma[d];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_qk_norm_ones_gamma() {
        // With gamma=1, output should be x / rms(x)
        let head_dim = 8;
        let n_heads = 2;
        let seq_len = 3;
        let gamma = vec![1.0f32; head_dim];

        let x: Vec<f32> = (0..seq_len * n_heads * head_dim)
            .map(|i| (i as f32 + 1.0) * 0.5)
            .collect();

        let out = cpu_qk_norm(&x, &gamma, n_heads, head_dim, seq_len);

        // Verify each head's output has rms ≈ 1
        for row in 0..seq_len * n_heads {
            let base = row * head_dim;
            let sum_sq: f32 = out[base..base + head_dim].iter().map(|v| v * v).sum();
            let rms = (sum_sq / head_dim as f32).sqrt();
            assert!((rms - 1.0).abs() < 1e-4,
                "row {} rms={} expected ~1.0", row, rms);
        }
    }

    #[test]
    fn test_cpu_qk_norm_with_gamma() {
        // gamma scales each element after normalization
        let head_dim = 4;
        let n_heads = 1;
        let seq_len = 1;
        let gamma = vec![2.0, 0.5, 1.0, 3.0];

        let x = vec![4.0, 0.0, 0.0, 0.0]; // only first element nonzero
        let out = cpu_qk_norm(&x, &gamma, n_heads, head_dim, seq_len);

        // rms = sqrt(16/4 + 1e-6) = sqrt(4) = 2
        // x[0]/rms = 4/2 = 2, * gamma[0]=2 → 4
        // x[1]/rms = 0, * gamma[1]=0.5 → 0
        assert!((out[0] - 4.0).abs() < 1e-4, "out[0]={}", out[0]);
        assert!((out[1]).abs() < 1e-4, "out[1]={}", out[1]);
        assert!((out[2]).abs() < 1e-4, "out[2]={}", out[2]);
        assert!((out[3]).abs() < 1e-4, "out[3]={}", out[3]);
    }

    #[test]
    fn test_cpu_qk_norm_heads_independent() {
        // Each head should be normalized independently
        let head_dim = 4;
        let n_heads = 2;
        let seq_len = 1;

        // Head 0: large values, Head 1: small values
        let x = vec![
            100.0, 200.0, 300.0, 400.0,  // head 0
            0.01, 0.02, 0.03, 0.04,       // head 1
        ];
        let gamma = vec![1.0; head_dim];
        let out = cpu_qk_norm(&x, &gamma, n_heads, head_dim, seq_len);

        // Head 0 rms = sqrt((100^2+200^2+300^2+400^2)/4) = sqrt(75000) ≈ 273.86
        let head0_rms_sq: f32 = out[0..4].iter().map(|v| v * v).sum::<f32>() / 4.0;
        assert!((head0_rms_sq - 1.0).abs() < 1e-2, "head0 rms_sq={}", head0_rms_sq);

        // Head 1 rms should also be ~1
        let head1_rms_sq: f32 = out[4..8].iter().map(|v| v * v).sum::<f32>() / 4.0;
        assert!((head1_rms_sq - 1.0).abs() < 1e-2, "head1 rms_sq={}", head1_rms_sq);
    }

    #[test]
    fn test_cpu_qk_norm_multiple_tokens() {
        // Multiple tokens should be normalized independently
        let head_dim = 8;
        let n_heads = 2;
        let seq_len = 4;
        let gamma = vec![1.0; head_dim];

        let x: Vec<f32> = (0..seq_len * n_heads * head_dim)
            .map(|i| ((i as f32) * 0.7 + 1.0).sin())
            .collect();

        let out = cpu_qk_norm(&x, &gamma, n_heads, head_dim, seq_len);

        // Verify each row is normalized
        for row in 0..seq_len * n_heads {
            let base = row * head_dim;
            let sum_sq: f32 = out[base..base + head_dim].iter().map(|v| v * v).sum();
            let rms = (sum_sq / head_dim as f32).sqrt();
            assert!((rms - 1.0).abs() < 1e-4, "row {} rms={}", row, rms);
        }
    }

    #[test]
    fn test_qk_norm_rmsnorm_kernel_compiles() {
        let kb = crate::t0::rmsnorm_kernels::build_rmsnorm_forward();
        let ck = kb.compile_via_ssa(crate::t0::ir::Target::GFX1100).expect("rmsnorm compile");
        assert!(!ck.elf.is_empty());
    }
}
