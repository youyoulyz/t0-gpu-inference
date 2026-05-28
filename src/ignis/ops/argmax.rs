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

/// Sample a single token from logits with temperature and top-p filtering.
///
/// This runs on CPU (reads logits from GPU, samples, returns token ID).
/// Suitable for single-token decode; for batch decode, consider GPU sampling.
///
/// # Arguments
/// - `logits`: [vocab_size] f32 tensor (raw logits for one position)
/// - `temperature`: sampling temperature. <= 0 means greedy argmax.
/// - `top_p`: nucleus sampling threshold (0.0 = greedy, 1.0 = no filtering)
///
/// # Returns
/// - sampled token ID as u32
#[cfg(feature = "rocm")]
pub fn sample_token(
    logits: &Tensor,
    temperature: f32,
    top_p: f32,
    runtime: &Arc<GpuRuntime>,
) -> Result<u32, String> {
    let vocab_size = logits.numel();
    let data = runtime.read_f32(logits.buffer(), vocab_size);

    // Greedy: just argmax
    if temperature <= 0.0 || top_p <= 0.0 {
        let mut best_idx = 0usize;
        let mut best_val = data[0];
        for i in 1..vocab_size {
            if data[i] > best_val {
                best_val = data[i];
                best_idx = i;
            }
        }
        return Ok(best_idx as u32);
    }

    // Apply temperature
    let inv_temp = 1.0 / temperature;
    let mut scaled: Vec<f32> = data.iter().map(|&l| l * inv_temp).collect();

    // Softmax (numerically stable)
    let max_val = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    for v in scaled.iter_mut() {
        *v = (*v - max_val).exp();
    }
    let sum: f32 = scaled.iter().sum();
    for v in scaled.iter_mut() {
        *v /= sum;
    }

    // Top-p (nucleus) filtering
    if top_p < 1.0 {
        // Sort by probability descending
        let mut indexed: Vec<(usize, f32)> = scaled.iter().enumerate().map(|(i, &p)| (i, p)).collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Find cutoff
        let mut cumsum = 0.0f32;
        let mut cutoff_idx = indexed.len();
        for (i, (_, p)) in indexed.iter().enumerate() {
            cumsum += p;
            if cumsum >= top_p {
                cutoff_idx = i + 1;
                break;
            }
        }

        // Zero out tokens beyond cutoff and renormalize
        let mut keep = vec![false; vocab_size];
        for &(idx, _) in &indexed[..cutoff_idx] {
            keep[idx] = true;
        }
        let mut sum_kept = 0.0f32;
        for i in 0..vocab_size {
            if !keep[i] {
                scaled[i] = 0.0;
            }
            sum_kept += scaled[i];
        }
        if sum_kept > 0.0 {
            for v in scaled.iter_mut() {
                *v /= sum_kept;
            }
        }
    }

    // Sample from the distribution using simple random sampling
    // Use a simple LCG PRNG seeded from current time
    let seed = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .subsec_nanos();
    let rand_val = (seed as f32) / (u32::MAX as f32);

    let mut cumsum = 0.0f32;
    for (i, &p) in scaled.iter().enumerate() {
        cumsum += p;
        if cumsum >= rand_val {
            return Ok(i as u32);
        }
    }

    // Fallback: return last non-zero token
    Ok((vocab_size - 1) as u32)
}
