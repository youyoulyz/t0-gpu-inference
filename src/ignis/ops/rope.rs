//! RoPE (Rotary Position Embedding) op — inference only, no tape recording.
//!
//! Wraps the T0 BlockDSL RoPE kernel for use in Ignis.
//!
//! Input: [n_tokens, dim] f32
//! Output: [n_tokens, dim] f32 (rotated)

#[cfg(feature = "rocm")]
use std::sync::Arc;
#[cfg(feature = "rocm")]
use super::super::tensor::Tensor;
#[cfg(feature = "rocm")]
use super::super::gpu_context::GpuRuntime;

/// Apply RoPE to a tensor.
///
/// # Arguments
/// - `x`: [n_rows, dim] f32 tensor (may be multi-head: n_rows = seq_len * n_heads)
/// - `pos_offset`: starting position index (0 for prefill, current_pos for decode)
/// - `rope_theta`: base frequency (Qwen3 uses 1_000_000.0)
/// - `n_heads`: number of heads (1 if single-head, >1 if multi-head reshaped)
/// - `runtime`: GPU runtime
///
/// # Returns
/// - output: [n_rows, dim] f32 tensor with RoPE applied
#[cfg(feature = "rocm")]
pub fn rope_forward(
    x: &Tensor,
    pos_offset: usize,
    rope_theta: f32,
    n_heads: usize,
    runtime: &Arc<GpuRuntime>,
) -> Result<Tensor, String> {
    crate::profile_scope!("rope");
    crate::profiler::set_shapes(
        vec![crate::profiler::ShapeInfo::new(x.shape())],
        vec![crate::profiler::ShapeInfo::new(x.shape())],
    );
    let shape = x.shape();
    assert!(shape.len() == 2, "rope: expected 2D [n_rows, dim], got {:?}", shape);
    let n_rows = shape[0];
    let dim = shape[1];
    assert!(dim % 2 == 0, "rope: dim must be even, got {}", dim);
    assert!(dim <= 256, "rope: dim {} exceeds kernel limit 256", dim);

    let out_buf = runtime.alloc_f32(n_rows * dim)?;

    let n_heads_shift = n_heads.trailing_zeros();
    let kernel_name = format!("rope_fwd_s{}_theta{}", n_heads_shift, rope_theta as u32);
    let kernel = runtime.ensure_kernel_blockdsl(&kernel_name, || {
        crate::t0::rope_kernels::build_rope_forward(n_heads_shift)
    })?;

    let ka = crate::kernargs![
        x.gpu_addr() => u64,
        out_buf.gpu_addr() => u64,
        dim as u32 => u32,
        0u32 => u32,  // reserved (was n_tokens)
        pos_offset as u32 => u32,
        rope_theta => f32
    ];
    let (grid_x, _) = crate::t0::rope_kernels::rope_grid(n_rows as u32);
    runtime.dispatch(&kernel, [grid_x, 1, 1], &ka)?;

    let out_arc = Arc::new(out_buf);
    Ok(Tensor::from_buffer(out_arc, runtime, &[n_rows, dim],
        super::super::tensor::DType::F32, "rope_out"))
}

/// CPU reference implementation of RoPE forward (rotate_half style, for testing).
pub fn cpu_rope_forward(x: &[f32], n_tokens: usize, dim: usize, base: f32, pos_offset: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n_tokens * dim];
    let half_d = dim / 2;
    for t in 0..n_tokens {
        let pos = (pos_offset + t) as f32;
        for i in 0..half_d {
            let freq = 1.0 / base.powf(2.0 * i as f32 / dim as f32);
            let theta = pos * freq;
            let cos_t = theta.cos();
            let sin_t = theta.sin();
            let first = t * dim + i;
            let second = first + half_d;
            out[first]  = x[first] * cos_t - x[second] * sin_t;
            out[second] = x[first] * sin_t + x[second] * cos_t;
        }
    }
    out
}

/// CPU reference: inverse RoPE (transpose rotation, rotate_half style).
pub fn cpu_rope_inverse(x: &[f32], n_tokens: usize, dim: usize, base: f32, pos_offset: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n_tokens * dim];
    let half_d = dim / 2;
    for t in 0..n_tokens {
        let pos = (pos_offset + t) as f32;
        for i in 0..half_d {
            let freq = 1.0 / base.powf(2.0 * i as f32 / dim as f32);
            let theta = pos * freq;
            let cos_t = theta.cos();
            let sin_t = theta.sin();
            let first = t * dim + i;
            let second = first + half_d;
            out[first]  =  x[first] * cos_t + x[second] * sin_t;
            out[second] = -x[first] * sin_t + x[second] * cos_t;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rope_pos_zero_is_identity() {
        // At position 0, theta=0, cos=1, sin=0 → identity
        let x: Vec<f32> = (0..16).map(|i| (i as f32) * 0.1).collect();
        let out = cpu_rope_forward(&x, 1, 16, 10000.0, 0);
        for i in 0..16 {
            assert!((out[i] - x[i]).abs() < 1e-6,
                "pos=0 should be identity: out[{}]={} != x[{}]={}", i, out[i], i, x[i]);
        }
    }

    #[test]
    fn test_rope_pos_offset_matches_shifted() {
        // RoPE with pos_offset=N on 1 token should equal RoPE with pos_offset=0 on token N
        // (using the same input data for both)
        let dim = 32;
        let x_multi: Vec<f32> = (0..6 * dim).map(|i| (i as f32) * 0.13).collect();

        // Single token at position 5: use the same data as token 5 in the multi-token case
        let x_single = &x_multi[5 * dim..6 * dim];
        let out_offset = cpu_rope_forward(x_single, 1, dim, 10000.0, 5);

        // Multi-token prefill, take token 5
        let out_multi = cpu_rope_forward(&x_multi, 6, dim, 10000.0, 0);

        for i in 0..dim {
            assert!((out_offset[i] - out_multi[5 * dim + i]).abs() < 1e-5,
                "pos_offset mismatch at [{}]: offset={} multi={}",
                i, out_offset[i], out_multi[5 * dim + i]);
        }
    }

    #[test]
    fn test_rope_inverse_roundtrip() {
        // Forward then inverse should recover original
        let dim = 64;
        let n_tokens = 4;
        let x: Vec<f32> = (0..n_tokens * dim).map(|i| ((i as f32 * 0.17).sin() * 2.0)).collect();

        let rotated = cpu_rope_forward(&x, n_tokens, dim, 10000.0, 0);
        let recovered = cpu_rope_inverse(&rotated, n_tokens, dim, 10000.0, 0);

        for i in 0..n_tokens * dim {
            assert!((recovered[i] - x[i]).abs() < 1e-4,
                "roundtrip[{}]: recovered={} expected={} err={}",
                i, recovered[i], x[i], (recovered[i] - x[i]).abs());
        }
    }

    #[test]
    fn test_rope_preserves_norm() {
        // RoPE is a rotation, so it preserves vector norm per pair
        let dim = 32;
        let x: Vec<f32> = (0..dim).map(|i| ((i as f32 * 0.31).cos() * 1.5)).collect();
        let out = cpu_rope_forward(&x, 1, dim, 10000.0, 7);

        // Check each rotate_half pair preserves x[i]^2 + x[i+d/2]^2
        let half_d = dim / 2;
        for i in 0..half_d {
            let in_norm = x[i].powi(2) + x[i + half_d].powi(2);
            let out_norm = out[i].powi(2) + out[i + half_d].powi(2);
            assert!((in_norm - out_norm).abs() < 1e-5,
                "pair {} norm: in={} out={}", i, in_norm, out_norm);
        }
    }

    #[test]
    fn test_rope_different_positions_different_output() {
        let dim = 16;
        let x: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.1).collect();
        let out0 = cpu_rope_forward(&x, 1, dim, 10000.0, 0);
        let out1 = cpu_rope_forward(&x, 1, dim, 10000.0, 1);
        // At least one element should differ
        let differs = out0.iter().zip(out1.iter()).any(|(a, b)| (a - b).abs() > 1e-6);
        assert!(differs, "different positions should produce different output");
    }

    #[test]
    fn test_rope_multi_token_independent() {
        // Each token in a multi-token input should be rotated independently
        let dim = 16;
        let n = 3;
        let x: Vec<f32> = (0..n * dim).map(|i| (i as f32) * 0.1).collect();
        let out_all = cpu_rope_forward(&x, n, dim, 10000.0, 0);

        for t in 0..n {
            let single_in = &x[t * dim..(t + 1) * dim];
            let single_out = cpu_rope_forward(single_in, 1, dim, 10000.0, t);
            for i in 0..dim {
                assert!((out_all[t * dim + i] - single_out[i]).abs() < 1e-6,
                    "token {} [{}]: batch={} single={}", t, i, out_all[t * dim + i], single_out[i]);
            }
        }
    }

    #[test]
    fn test_rope_compile_fwd() {
        let kb = crate::t0::rope_kernels::build_rope_forward(0);
        let ck = kb.compile_via_ssa(crate::t0::ir::Target::GFX1100).expect("RoPE fwd compile");
        assert!(!ck.elf.is_empty());
    }

    #[test]
    fn test_rope_compile_bwd() {
        let kb = crate::t0::rope_kernels::build_rope_backward();
        let ck = kb.compile_via_ssa(crate::t0::ir::Target::GFX1100).expect("RoPE bwd compile");
        assert!(!ck.elf.is_empty());
    }
}
