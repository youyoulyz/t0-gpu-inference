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
/// - `x`: [n_tokens, dim] f32 tensor
/// - `pos_offset`: starting position index (0 for prefill, current_pos for decode)
/// - `rope_theta`: base frequency (Qwen3 uses 1_000_000.0). Note: the kernel uses
///   a fixed ln(10000) base; this parameter is kept for API compatibility.
/// - `runtime`: GPU runtime
///
/// # Returns
/// - output: [n_tokens, dim] f32 tensor with RoPE applied
#[cfg(feature = "rocm")]
pub fn rope_forward(
    x: &Tensor,
    pos_offset: usize,
    _rope_theta: f32,
    runtime: &Arc<GpuRuntime>,
) -> Result<Tensor, String> {
    let shape = x.shape();
    assert!(shape.len() == 2, "rope: expected 2D [n_tokens, dim], got {:?}", shape);
    let n_tokens = shape[0];
    let dim = shape[1];
    assert!(dim % 2 == 0, "rope: dim must be even, got {}", dim);
    assert!(dim <= 256, "rope: dim {} exceeds kernel limit 256", dim);

    let out_buf = runtime.alloc_f32(n_tokens * dim)?;

    let kernel = runtime.ensure_kernel_blockdsl("rope_fwd", || {
        crate::t0::rope_kernels::build_rope_forward()
    })?;

    let ka = crate::kernargs![
        x.gpu_addr() => u64,
        out_buf.gpu_addr() => u64,
        dim as u32 => u32,
        n_tokens as u32 => u32,
        pos_offset as u32 => u32  // pos_base
    ];
    let (grid_x, _) = crate::t0::rope_kernels::rope_grid(n_tokens as u32);
    runtime.dispatch(&kernel, [grid_x, 1, 1], &ka)?;

    let out_arc = Arc::new(out_buf);
    Ok(Tensor::from_buffer(out_arc, runtime, &[n_tokens, dim],
        super::super::tensor::DType::F32, "rope_out"))
}

/// CPU reference implementation of RoPE forward (for testing).
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
            let even = t * dim + 2 * i;
            let odd = even + 1;
            out[even] = x[even] * cos_t - x[odd] * sin_t;
            out[odd]  = x[even] * sin_t + x[odd] * cos_t;
        }
    }
    out
}
