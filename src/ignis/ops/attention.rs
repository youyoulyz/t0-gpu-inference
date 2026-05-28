//! Standard scaled dot-product attention with GQA support (inference only).
//!
//! Computes: output = softmax(Q @ K^T / sqrt(d_k)) @ V
//!
//! Supports Grouped Query Attention (GQA) where n_kv_heads < n_heads.
//!
//! # Arguments
//! - Q: [n_heads, seq_len, head_dim]
//! - K: [n_kv_heads, kv_len, head_dim]  (from KV cache)
//! - V: [n_kv_heads, kv_len, head_dim]  (from KV cache)
//!
//! # Returns
//! - output: [seq_len, n_heads * head_dim]

#[cfg(feature = "rocm")]
use std::sync::Arc;
#[cfg(feature = "rocm")]
use super::super::tensor::Tensor;
#[cfg(feature = "rocm")]
use super::super::gpu_context::GpuRuntime;

/// Standard scaled dot-product attention with GQA.
///
/// For inference: seq_len is typically 1 (decode) or prompt_len (prefill).
/// kv_len is the total number of tokens in the KV cache (including current).
///
/// # Arguments
/// - `q`: [seq_len, n_heads * head_dim] f32
/// - `k`: [kv_len, n_kv_heads * head_dim] f32 (from KV cache)
/// - `v`: [kv_len, n_kv_heads * head_dim] f32 (from KV cache)
/// - `n_heads`: number of query heads
/// - `n_kv_heads`: number of key/value heads (GQA)
/// - `head_dim`: dimension per head
/// - `runtime`: GPU runtime
///
/// # Returns
/// - output: [seq_len, n_heads * head_dim] f32
#[cfg(feature = "rocm")]
pub fn standard_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    runtime: &Arc<GpuRuntime>,
) -> Result<Tensor, String> {
    let q_shape = q.shape();
    let k_shape = k.shape();
    assert!(q_shape.len() == 2, "q: expected [seq, n_heads*head_dim]");
    assert!(k_shape.len() == 2, "k: expected [kv_len, n_kv_heads*head_dim]");

    let seq_len = q_shape[0];
    let kv_len = k_shape[0];
    let gqa_ratio = n_heads / n_kv_heads;

    assert_eq!(q_shape[1], n_heads * head_dim);
    assert_eq!(k_shape[1], n_kv_heads * head_dim);

    let scale = 1.0 / (head_dim as f32).sqrt();

    // Allocate output: [seq_len, n_heads * head_dim]
    let out_buf = runtime.alloc_f32(seq_len * n_heads * head_dim)?;

    // For each head, compute attention independently.
    // This is a simple loop-based implementation — each head's attention is:
    //   scores = q_head @ k_head^T * scale    [seq_len, kv_len]
    //   if causal and seq_len > 1: mask upper triangle
    //   weights = softmax(scores)              [seq_len, kv_len]
    //   out_head = weights @ v_head            [seq_len, head_dim]
    //
    // We use bf16_matmul for the GEMMs and softmax for the softmax.

    let q_data = q.to_f32_vec();
    let k_data = k.to_f32_vec();
    let v_data = v.to_f32_vec();

    let mut out_data = vec![0.0f32; seq_len * n_heads * head_dim];

    for h in 0..n_heads {
        let kv_h = h / gqa_ratio; // which KV head this query head maps to

        // Extract Q head: [seq_len, head_dim]
        let q_head: Vec<f32> = (0..seq_len)
            .flat_map(|s| {
                let base = s * n_heads * head_dim + h * head_dim;
                q_data[base..base + head_dim].to_vec()
            })
            .collect();

        // Extract K head: [kv_len, head_dim]
        let k_head: Vec<f32> = (0..kv_len)
            .flat_map(|s| {
                let base = s * n_kv_heads * head_dim + kv_h * head_dim;
                k_data[base..base + head_dim].to_vec()
            })
            .collect();

        // Extract V head: [kv_len, head_dim]
        let v_head: Vec<f32> = (0..kv_len)
            .flat_map(|s| {
                let base = s * n_kv_heads * head_dim + kv_h * head_dim;
                v_data[base..base + head_dim].to_vec()
            })
            .collect();

        // scores = Q @ K^T * scale  → [seq_len, kv_len]
        let mut scores = vec![0.0f32; seq_len * kv_len];
        for s in 0..seq_len {
            for k_idx in 0..kv_len {
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q_head[s * head_dim + d] * k_head[k_idx * head_dim + d];
                }
                scores[s * kv_len + k_idx] = dot * scale;
            }
        }

        // Causal mask: for prefill (seq_len > 1), mask future positions.
        // The causal mask applies when Q and K have the same sequence.
        // During prefill, kv_len == seq_len and positions are 0..seq_len-1.
        // During decode, seq_len=1 and no mask is needed (all past is visible).
        if seq_len > 1 && kv_len == seq_len {
            for s in 0..seq_len {
                for k_idx in (s + 1)..kv_len {
                    scores[s * kv_len + k_idx] = f32::NEG_INFINITY;
                }
            }
        }

        // softmax along last dim
        for s in 0..seq_len {
            let row = &mut scores[s * kv_len..(s + 1) * kv_len];
            // Online softmax
            let mut max_val = f32::NEG_INFINITY;
            for &v in row.iter() {
                if v > max_val { max_val = v; }
            }
            let mut sum = 0.0f32;
            for v in row.iter_mut() {
                *v = (*v - max_val).exp();
                sum += *v;
            }
            for v in row.iter_mut() {
                *v /= sum;
            }
        }

        // out = weights @ V  → [seq_len, head_dim]
        for s in 0..seq_len {
            for d in 0..head_dim {
                let mut val = 0.0f32;
                for k_idx in 0..kv_len {
                    val += scores[s * kv_len + k_idx] * v_head[k_idx * head_dim + d];
                }
                out_data[s * n_heads * head_dim + h * head_dim + d] = val;
            }
        }
    }

    // Upload result to GPU
    let out_gpu = runtime.upload_f32(&out_data)?;
    Ok(Tensor::from_buffer(Arc::new(out_gpu), runtime,
        &[seq_len, n_heads * head_dim],
        super::super::tensor::DType::F32, "attn_out"))
}
