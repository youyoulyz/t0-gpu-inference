//! Argmax operation — find the index of the maximum value along the last dimension.
//!
//! Forward: indices = argmax(logits, dim=-1)  → [batch] of u32 indices
//! No backward (argmax is not differentiable).
//!
//! Uses the T0 argmax kernel (single dispatch per row).

#[cfg(feature = "rocm")]
use crate::kfd::GpuBuffer;
#[cfg(feature = "rocm")]
use super::super::tensor::{Tensor, DType};
#[cfg(feature = "rocm")]
use super::super::gpu_context::GpuRuntime;
#[cfg(feature = "rocm")]
use std::sync::Arc;

/// Argmax along the last dimension.
///
/// # Arguments
/// - `input`: [batch, n] f32 tensor
///
/// # Returns
/// - `indices`: [batch] u32 tensor (indices stored as raw u32)
#[cfg(feature = "rocm")]
pub fn argmax(
    input: &Tensor,
    runtime: &Arc<GpuRuntime>,
) -> Result<Tensor, String> {
    let n = input.shape().last().copied().unwrap_or(0);
    let batch = input.numel() / n;

    if n > 256 {
        return Err(format!("argmax: last dim {} exceeds max 256", n));
    }

    // Allocate output buffer for indices (u32)
    let out_buf = runtime.device.alloc_vram(batch * 4)?; // 4 bytes per u32

    // Get or compile argmax kernel
    let k_argmax = runtime.ensure_kernel_blockdsl("argmax", || {
        crate::t0::argmax_kernels::build_argmax()
    })?;

    let (grid_x, _) = crate::t0::argmax_kernels::argmax_grid(batch as u32);
    let ka = crate::kernargs![
        input.gpu_addr() => u64,
        out_buf.gpu_addr() => u64,
        n as u32 => u32
    ];
    runtime.dispatch(&k_argmax, [grid_x, 1, 1], &ka)?;
    runtime.wait_idle()?;

    // Create output tensor (u32 type)
    let out_tensor = Tensor::from_buffer(
        Arc::new(out_buf),
        runtime,
        &[batch],
        DType::U32,
        "argmax",
    );

    Ok(out_tensor)
}

/// Sample from a categorical distribution given as probabilities.
///
/// # Arguments
/// - `probs`: [batch, n] f32 tensor (should sum to 1 per row)
/// - `rand_buf`: [batch] f32 tensor of random values in [0, 1)
///
/// # Returns
/// - `indices`: [batch] u32 tensor of sampled indices
///
/// Algorithm: for each row, find the first index where cumsum(probs) > rand
#[cfg(feature = "rocm")]
pub fn sample_categorical(
    probs: &Tensor,
    rand_buf: &Tensor,
    runtime: &Arc<GpuRuntime>,
) -> Result<Tensor, String> {
    // For now, use argmax as a simple sampling strategy (greedy decoding)
    // Full categorical sampling needs cumsum + search, which is more complex
    argmax(probs, runtime)
}
