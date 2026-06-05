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
/// For greedy mode (temperature ≤ 0 or top_p ≤ 0), uses GPU argmax kernel
/// to avoid downloading the full logits vector (595KB for Qwen3).
/// For sampling modes, falls back to CPU with full logits download.
#[cfg(feature = "rocm")]
pub fn sample_token(
    logits: &Tensor,
    temperature: f32,
    top_p: f32,
    runtime: &Arc<GpuRuntime>,
) -> Result<u32, String> {
    let vocab_size = logits.numel();

    // Greedy: GPU argmax (single dispatch, ~5KB readback vs 607KB full download)
    if temperature <= 0.0 || top_p <= 0.0 {
        return gpu_argmax(logits, runtime);
    }

    // Sampling: CPU fallback (requires full logits download)
    let data = runtime.read_f32(logits.buffer(), vocab_size);

    let inv_temp = 1.0 / temperature;
    let mut scaled: Vec<f32> = data.iter().map(|&l| l * inv_temp).collect();

    let max_val = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    for v in scaled.iter_mut() {
        *v = (*v - max_val).exp();
    }
    let sum: f32 = scaled.iter().sum();
    for v in scaled.iter_mut() {
        *v /= sum;
    }

    if top_p < 1.0 {
        let mut indexed: Vec<(usize, f32)> = scaled.iter().enumerate().map(|(i, &p)| (i, p)).collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut cumsum = 0.0f32;
        let mut cutoff_idx = indexed.len();
        for (i, (_, p)) in indexed.iter().enumerate() {
            cumsum += p;
            if cumsum >= top_p {
                cutoff_idx = i + 1;
                break;
            }
        }

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

    Ok((vocab_size - 1) as u32)
}

/// GPU argmax: find the index of the maximum value in a 1D tensor.
///
/// Uses a single-dispatch chunked reduce kernel. Each workgroup processes
/// 256 elements and writes (max_val, global_idx) to partial buffers.
/// CPU then reduces the partial results (a few KB vs 607KB full download).
#[cfg(feature = "rocm")]
fn gpu_argmax(logits: &Tensor, runtime: &Arc<GpuRuntime>) -> Result<u32, String> {
    let vocab_size = logits.numel();
    let chunk_size = 256usize;
    let n_chunks = (vocab_size + chunk_size - 1) / chunk_size;

    let vals_buf = runtime.device.alloc_vram(n_chunks * 4)?;
    let idxs_buf = runtime.device.alloc_vram(n_chunks * 4)?;

    let k = runtime.ensure_kernel_blockdsl("argmax_reduce", || {
        crate::t0::argmax_kernels::build_argmax_reduce()
    })?;

    let (grid_x, _) = crate::t0::argmax_kernels::argmax_reduce_grid(n_chunks as u32);
    let ka = crate::kernargs![
        logits.gpu_addr() => u64,
        vals_buf.gpu_addr() => u64,
        idxs_buf.gpu_addr() => u64,
        vocab_size as u32 => u32
    ];
    runtime.dispatch(&k, [grid_x, 1, 1], &ka)?;

    let vals = runtime.read_f32(&vals_buf, n_chunks);
    let idxs = runtime.read_f32(&idxs_buf, n_chunks);

    let mut best_val = f32::NEG_INFINITY;
    let mut best_idx = 0u32;
    for c in 0..n_chunks {
        if vals[c] > best_val {
            best_val = vals[c];
            best_idx = idxs[c] as u32;
        }
    }

    Ok(best_idx)
}

/// CPU reference: greedy argmax over logits (for testing).
pub fn cpu_argmax(logits: &[f32]) -> usize {
    logits.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// CPU reference: sample a token from logits with temperature and top-p (for testing).
///
/// Returns (token_id, probs_vector) for verification.
pub fn cpu_sample_token(logits: &[f32], temperature: f32, top_p: f32, rand_val: f32) -> (usize, Vec<f32>) {
    let vocab_size = logits.len();

    // Greedy
    if temperature <= 0.0 || top_p <= 0.0 {
        return (cpu_argmax(logits), vec![]);
    }

    // Temperature scaling + softmax
    let inv_temp = 1.0 / temperature;
    let mut scaled: Vec<f32> = logits.iter().map(|&l| l * inv_temp).collect();
    let max_val = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    for v in scaled.iter_mut() {
        *v = (*v - max_val).exp();
    }
    let sum: f32 = scaled.iter().sum();
    for v in scaled.iter_mut() {
        *v /= sum;
    }

    // Top-p filtering
    if top_p < 1.0 {
        let mut indexed: Vec<(usize, f32)> = scaled.iter().enumerate().map(|(i, &p)| (i, p)).collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut cumsum = 0.0f32;
        let mut cutoff_idx = indexed.len();
        for (i, (_, p)) in indexed.iter().enumerate() {
            cumsum += p;
            if cumsum >= top_p {
                cutoff_idx = i + 1;
                break;
            }
        }

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

    // Sample using provided rand_val
    let mut cumsum = 0.0f32;
    for (i, &p) in scaled.iter().enumerate() {
        cumsum += p;
        if cumsum >= rand_val {
            return (i, scaled);
        }
    }
    (vocab_size - 1, scaled)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_argmax_basic() {
        assert_eq!(cpu_argmax(&[1.0, 3.0, 2.0]), 1);
        assert_eq!(cpu_argmax(&[5.0, 1.0, 2.0]), 0);
        assert_eq!(cpu_argmax(&[1.0, 2.0, 5.0]), 2);
    }

    #[test]
    fn test_cpu_argmax_tie() {
        // With equal values, max_by returns the last one
        assert_eq!(cpu_argmax(&[3.0, 3.0, 1.0]), 1);
    }

    #[test]
    fn test_cpu_argmax_single() {
        assert_eq!(cpu_argmax(&[42.0]), 0);
    }

    #[test]
    fn test_cpu_sample_greedy() {
        let logits = vec![0.1, 0.5, 0.3, 0.2];
        // temperature <= 0 → greedy
        let (tok, _) = cpu_sample_token(&logits, 0.0, 1.0, 0.5);
        assert_eq!(tok, 1); // argmax
    }

    #[test]
    fn test_cpu_sample_greedy_top_p_zero() {
        let logits = vec![0.1, 0.5, 0.3, 0.2];
        let (tok, _) = cpu_sample_token(&logits, 1.0, 0.0, 0.5);
        assert_eq!(tok, 1); // greedy when top_p <= 0
    }

    #[test]
    fn test_cpu_sample_temperature_scales() {
        // With high temperature, distribution should be more uniform
        let logits = vec![10.0, 0.0, 0.0, 0.0];

        // Low temperature → peaked at index 0
        let (_, probs_low) = cpu_sample_token(&logits, 0.1, 1.0, 0.5);
        assert!(probs_low[0] > 0.99, "low temp: probs[0]={} should be >0.99", probs_low[0]);

        // High temperature → more spread
        let (_, probs_high) = cpu_sample_token(&logits, 100.0, 1.0, 0.5);
        assert!(probs_high[0] < 0.5, "high temp: probs[0]={} should be <0.5", probs_high[0]);
    }

    #[test]
    fn test_cpu_sample_top_p_filters() {
        // With top_p=0.1, only the highest-prob token should survive
        let logits = vec![5.0, 1.0, 1.0, 1.0];
        let (_, probs) = cpu_sample_token(&logits, 1.0, 0.1, 0.5);

        // Only index 0 should have nonzero probability
        assert!(probs[0] > 0.99, "top_p filter: probs[0]={}", probs[0]);
        assert!(probs[1] < 1e-6, "top_p filter: probs[1]={}", probs[1]);
    }

    #[test]
    fn test_cpu_sample_rand_val_selects() {
        // With rand_val very small, should select first token in sorted order
        let logits = vec![0.0, 10.0, 0.0, 0.0];
        let (tok, _) = cpu_sample_token(&logits, 1.0, 1.0, 0.001);
        assert_eq!(tok, 1); // highest prob token
    }

    #[test]
    fn test_cpu_sample_rand_val_large() {
        // With rand_val very large (but < 1.0), should select last token
        let logits = vec![0.0, 0.0, 0.0, 10.0];
        let (tok, _) = cpu_sample_token(&logits, 1.0, 1.0, 0.999);
        assert_eq!(tok, 3);
    }

    #[test]
    fn test_cpu_sample_probs_sum_to_one() {
        let logits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let (_, probs) = cpu_sample_token(&logits, 1.0, 1.0, 0.5);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "probs sum={}", sum);
    }

    #[test]
    fn test_cpu_sample_top_p_probs_sum_to_one() {
        let logits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let (_, probs) = cpu_sample_token(&logits, 1.0, 0.5, 0.5);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "top_p probs sum={}", sum);
    }

    #[test]
    fn test_cpu_sample_uniform_logits() {
        // All logits equal → uniform distribution
        let logits = vec![1.0; 10];
        let (_, probs) = cpu_sample_token(&logits, 1.0, 1.0, 0.5);
        for (i, &p) in probs.iter().enumerate() {
            assert!((p - 0.1).abs() < 1e-5, "uniform probs[{}]={}", i, p);
        }
    }
}
