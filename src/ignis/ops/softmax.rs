//! Large-row softmax operation — softmax along the last dimension for vocab projection.
//!
//! Forward: probs = softmax(logits, dim=-1)  → same shape as input
//! No backward (inference only).
//!
//! Uses the T0 chunked softmax kernel (three-pass: max → sum → normalize).
//! Supports cols > 256 (up to 65536+).

#[cfg(feature = "rocm")]
use super::super::tensor::{Tensor, DType};
#[cfg(feature = "rocm")]
use super::super::gpu_context::GpuRuntime;
#[cfg(feature = "rocm")]
use std::sync::Arc;

/// Softmax along the last dimension.
///
/// # Arguments
/// - `input`: [batch, n] f32 tensor (n can be > 256)
///
/// # Returns
/// - `output`: [batch, n] f32 tensor (probabilities, sum to 1 per row)
#[cfg(feature = "rocm")]
pub fn softmax(
    input: &Tensor,
    runtime: &Arc<GpuRuntime>,
) -> Result<Tensor, String> {
    let n = input.shape().last().copied().unwrap_or(0);
    let batch = input.numel() / n;

    // Use small softmax if cols ≤ 256
    if n <= 256 {
        return softmax_small(input, runtime);
    }

    // Use large chunked softmax for cols > 256
    softmax_large(input, runtime, n, batch)
}

/// Small softmax (cols ≤ 256) using the original single-pass kernel.
#[cfg(feature = "rocm")]
fn softmax_small(
    input: &Tensor,
    runtime: &Arc<GpuRuntime>,
) -> Result<Tensor, String> {
    let n = input.shape().last().copied().unwrap_or(0);
    let batch = input.numel() / n;

    let output_buf = runtime.alloc_f32(batch * n)?;

    let k_softmax = runtime.ensure_kernel_blockdsl("softmax_fwd", || {
        crate::t0::softmax_kernels::build_softmax_forward()
    })?;

    let (grid_x, _) = crate::t0::softmax_kernels::softmax_grid(batch as u32);
    let ka = crate::kernargs![
        input.gpu_addr() => u64,
        output_buf.gpu_addr() => u64,
        n as u32 => u32
    ];
    runtime.dispatch(&k_softmax, [grid_x, 1, 1], &ka)?;
    runtime.wait_idle()?;

    let out_tensor = Tensor::from_buffer(
        Arc::new(output_buf),
        runtime,
        input.shape(),
        DType::F32,
        "softmax",
    );

    Ok(out_tensor)
}

/// Large chunked softmax (cols > 256).
#[cfg(feature = "rocm")]
fn softmax_large(
    input: &Tensor,
    runtime: &Arc<GpuRuntime>,
    n: usize,
    batch: usize,
) -> Result<Tensor, String> {
    let n_chunks = crate::t0::softmax_large::softmax_n_chunks(n as u32);

    let output_buf = runtime.alloc_f32(batch * n)?;

    // Use TileSSA-based kernel with vector accumulators
    let k_softmax = runtime.ensure_kernel_precompiled("softmax_large", || {
        let ck = crate::t0::softmax_large::compile_softmax_large()?;
        Ok((ck.elf, ck.workgroup_size, ck.lds_size))
    })?;

    let (grid_x, _) = crate::t0::softmax_large::softmax_large_grid(batch as u32);
    let ka = crate::kernargs![
        input.gpu_addr() => u64,
        output_buf.gpu_addr() => u64,
        n as u32 => u32,
        n_chunks => u32
    ];
    runtime.dispatch(&k_softmax, [grid_x, 1, 1], &ka)?;
    runtime.wait_idle()?;

    let out_tensor = Tensor::from_buffer(
        Arc::new(output_buf),
        runtime,
        input.shape(),
        DType::F32,
        "softmax_large",
    );

    Ok(out_tensor)
}
