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

/// CPU reference: scaled dot-product attention (for testing).
pub fn cpu_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    seq_len: usize,
    kv_len: usize,
) -> Vec<f32> {
    let scale = 1.0 / (head_dim as f32).sqrt();
    let gqa_ratio = n_heads / n_kv_heads;
    let mut out = vec![0.0f32; seq_len * n_heads * head_dim];

    for h in 0..n_heads {
        let kv_h = h / gqa_ratio;

        // scores = Q @ K^T * scale → [seq_len, kv_len]
        let mut scores = vec![0.0f32; seq_len * kv_len];
        for s in 0..seq_len {
            for ki in 0..kv_len {
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q[s * n_heads * head_dim + h * head_dim + d]
                         * k[ki * n_kv_heads * head_dim + kv_h * head_dim + d];
                }
                scores[s * kv_len + ki] = dot * scale;
            }
        }

        // Causal mask (prefill only)
        if seq_len > 1 && kv_len == seq_len {
            for s in 0..seq_len {
                for ki in (s + 1)..kv_len {
                    scores[s * kv_len + ki] = f32::NEG_INFINITY;
                }
            }
        }

        // softmax
        for s in 0..seq_len {
            let row = &mut scores[s * kv_len..(s + 1) * kv_len];
            let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for v in row.iter_mut() {
                *v = (*v - max_val).exp();
                sum += *v;
            }
            for v in row.iter_mut() {
                *v /= sum;
            }
        }

        // out = weights @ V
        for s in 0..seq_len {
            for d in 0..head_dim {
                let mut val = 0.0f32;
                for ki in 0..kv_len {
                    val += scores[s * kv_len + ki] * v[ki * n_kv_heads * head_dim + kv_h * head_dim + d];
                }
                out[s * n_heads * head_dim + h * head_dim + d] = val;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: naive softmax over a slice (for test verification).
    fn softmax_row(row: &[f32]) -> Vec<f32> {
        let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = row.iter().map(|v| (v - max_val).exp()).collect();
        let sum: f32 = exps.iter().sum();
        exps.iter().map(|v| v / sum).collect()
    }

    #[test]
    fn test_cpu_attention_identity_k() {
        // When K = e_i (one-hot per head), attention should select the corresponding V row.
        let head_dim = 4;
        let n_heads = 1;
        let n_kv_heads = 1;
        let seq_len = 1;
        let kv_len = 1;

        // Q = [1, 2, 3, 4], K = [1, 0, 0, 0] (unit vector), V = [10, 20, 30, 40]
        let q = vec![1.0, 2.0, 3.0, 4.0];
        let k = vec![1.0, 0.0, 0.0, 0.0];
        let v = vec![10.0, 20.0, 30.0, 40.0];

        let out = cpu_attention(&q, &k, &v, n_heads, n_kv_heads, head_dim, seq_len, kv_len);

        // scores = (1*1 + 2*0 + 3*0 + 4*0) / sqrt(4) = 0.5
        // softmax([0.5]) = [1.0]
        // out = 1.0 * [10, 20, 30, 40] = [10, 20, 30, 40]
        for d in 0..head_dim {
            assert!((out[d] - v[d]).abs() < 1e-5, "out[{}]={} expected {}", d, out[d], v[d]);
        }
    }

    #[test]
    fn test_cpu_attention_causal_mask() {
        // Prefill with 3 tokens: position 0 should not attend to positions 1 or 2
        let head_dim = 4;
        let n_heads = 1;
        let n_kv_heads = 1;
        let seq_len = 3;
        let kv_len = 3;

        // K and V: identity-like (each token's key is a different unit vector)
        let k: Vec<f32> = vec![
            1.0, 0.0, 0.0, 0.0, // token 0
            0.0, 1.0, 0.0, 0.0, // token 1
            0.0, 0.0, 1.0, 0.0, // token 2
        ];
        let v: Vec<f32> = vec![
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
        ];
        // Q: all ones → will dot with all K rows equally
        let q: Vec<f32> = vec![
            1.0, 1.0, 1.0, 1.0, // token 0
            1.0, 1.0, 1.0, 1.0, // token 1
            1.0, 1.0, 1.0, 1.0, // token 2
        ];

        let out = cpu_attention(&q, &k, &v, n_heads, n_kv_heads, head_dim, seq_len, kv_len);

        // Token 0: can only attend to token 0 (causal). score=1/sqrt(4)=0.5 for all, but mask=inf for 1,2.
        // After mask, only token 0 survives → out = [1,0,0,0]
        assert!((out[0] - 1.0).abs() < 1e-5, "token0 dim0={}", out[0]);
        assert!((out[1]).abs() < 1e-5, "token0 dim1={}", out[1]);

        // Token 2: can attend to all 3 tokens. Each has same dot=0.5, so uniform 1/3.
        let t2_base = 2 * n_heads * head_dim;
        let expected_t2_d0 = (1.0 + 0.0 + 0.0) / 3.0; // weighted avg of v[0][0], v[1][0], v[2][0]
        assert!((out[t2_base] - expected_t2_d0).abs() < 1e-4,
            "token2 dim0={} expected {}", out[t2_base], expected_t2_d0);
    }

    #[test]
    fn test_cpu_attention_gqa() {
        // 2 query heads, 1 KV head → GQA ratio = 2
        let head_dim = 4;
        let n_heads = 2;
        let n_kv_heads = 1;
        let seq_len = 1;
        let kv_len = 1;

        // Both query heads see the same K/V
        let q: Vec<f32> = vec![
            1.0, 2.0, 3.0, 4.0, // head 0
            5.0, 6.0, 7.0, 8.0, // head 1
        ];
        let k = vec![1.0, 1.0, 1.0, 1.0];
        let v = vec![0.1, 0.2, 0.3, 0.4];

        let out = cpu_attention(&q, &k, &v, n_heads, n_kv_heads, head_dim, seq_len, kv_len);

        // Head 0: dot = (1+2+3+4)/2 = 5.0, softmax = [1.0], out = v
        for d in 0..head_dim {
            assert!((out[d] - v[d]).abs() < 1e-5, "head0 dim{}: out={} v={}", d, out[d], v[d]);
        }
        // Head 1: dot = (5+6+7+8)/2 = 13.0, softmax = [1.0], out = v
        for d in 0..head_dim {
            let out_val = out[head_dim + d];
            assert!((out_val - v[d]).abs() < 1e-5, "head1 dim{}: out={} v={}", d, out_val, v[d]);
        }
    }

    #[test]
    fn test_cpu_attention_decode_single_token() {
        // Decode: seq_len=1, kv_len=5 (cached)
        let head_dim = 8;
        let n_heads = 2;
        let n_kv_heads = 2;
        let seq_len = 1;
        let kv_len = 5;

        let q: Vec<f32> = (0..n_heads * head_dim).map(|i| (i as f32) * 0.1).collect();
        let k: Vec<f32> = (0..kv_len * n_kv_heads * head_dim).map(|i| ((i as f32) * 0.13).sin()).collect();
        let v: Vec<f32> = (0..kv_len * n_kv_heads * head_dim).map(|i| ((i as f32) * 0.17).cos()).collect();

        let out = cpu_attention(&q, &k, &v, n_heads, n_kv_heads, head_dim, seq_len, kv_len);

        // Output should be a weighted combination of V rows
        // Verify output has correct shape and is finite
        assert_eq!(out.len(), seq_len * n_heads * head_dim);
        assert!(out.iter().all(|v| v.is_finite()), "output contains non-finite values");
    }

    #[test]
    fn test_cpu_attention_softmax_correctness() {
        // Verify the softmax in attention matches standalone softmax
        let head_dim = 4;
        let n_heads = 1;
        let n_kv_heads = 1;
        let seq_len = 1;
        let kv_len = 3;

        let q = vec![1.0, 0.0, 0.0, 0.0];
        let k = vec![
            1.0, 0.0, 0.0, 0.0, // dot=1
            0.0, 1.0, 0.0, 0.0, // dot=0
            0.0, 0.0, 1.0, 0.0, // dot=0
        ];
        let v = vec![
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
        ];

        let out = cpu_attention(&q, &k, &v, n_heads, n_kv_heads, head_dim, seq_len, kv_len);

        // scores = [1/2, 0, 0] (scale = 1/sqrt(4) = 0.5)
        let expected_weights = softmax_row(&[0.5, 0.0, 0.0]);
        // out = expected_weights[0]*v[0] + expected_weights[1]*v[1] + expected_weights[2]*v[2]
        let expected_out: Vec<f32> = (0..head_dim).map(|d| {
            expected_weights[0] * v[0 * head_dim + d] +
            expected_weights[1] * v[1 * head_dim + d] +
            expected_weights[2] * v[2 * head_dim + d]
        }).collect();

        for d in 0..head_dim {
            assert!((out[d] - expected_out[d]).abs() < 1e-5,
                "dim {}: out={} expected {}", d, out[d], expected_out[d]);
        }
    }

    #[test]
    fn test_cpu_attention_two_heads_different_values() {
        // Verify heads compute independently
        let head_dim = 4;
        let n_heads = 2;
        let n_kv_heads = 2;
        let seq_len = 1;
        let kv_len = 1;

        // Head 0 Q=[1,0,0,0], Head 1 Q=[0,1,0,0]
        let q = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let k = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let v = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0];

        let out = cpu_attention(&q, &k, &v, n_heads, n_kv_heads, head_dim, seq_len, kv_len);

        // Head 0: dot=1/sqrt(4)=0.5, out = v_head0 = [10,20,30,40]
        for d in 0..head_dim {
            assert!((out[d] - v[d]).abs() < 1e-5, "head0 dim{}: {}", d, out[d]);
        }
        // Head 1: dot=1/sqrt(4)=0.5, out = v_head1 = [50,60,70,80]
        for d in 0..head_dim {
            assert!((out[head_dim + d] - v[head_dim + d]).abs() < 1e-5,
                "head1 dim{}: {}", d, out[head_dim + d]);
        }
    }
}
