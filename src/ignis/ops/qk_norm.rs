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

    let kernel = runtime.ensure_kernel_blockdsl(&format!("rmsnorm_d{}", head_dim), || {
        crate::t0::rmsnorm_kernels::build_rmsnorm_forward()
    })?;

    let ka = crate::kernargs![
        x.gpu_addr() => u64,
        gamma.gpu_addr() => u64,
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
