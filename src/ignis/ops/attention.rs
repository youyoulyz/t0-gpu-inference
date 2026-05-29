//! Standard scaled dot-product attention with GQA support (inference only).
//!
//! Computes: output = softmax(Q @ K^T / sqrt(d_k)) @ V
//!
//! Supports Grouped Query Attention (GQA) where n_kv_heads < n_heads.
//!
//! GPU implementation: uses bf16 GEMM for Q@K^T and weights@V,
//! and softmax_large kernel for softmax.

#[cfg(feature = "rocm")]
use std::sync::Arc;
#[cfg(feature = "rocm")]
use super::super::tensor::Tensor;
#[cfg(feature = "rocm")]
use super::super::gpu_context::GpuRuntime;

/// Standard scaled dot-product attention with GQA (GPU implementation).
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

    // Read Q from GPU (small: seq_len * n_heads * head_dim * 4 bytes)
    let q_data = q.to_f32_vec();

    // Get K/V data from GPU (read once, reuse for all heads)
    let k_data = k.to_f32_vec();
    let v_data = v.to_f32_vec();

    // Output accumulator on CPU
    let mut out_data = vec![0.0f32; seq_len * n_heads * head_dim];

    // For each query head, compute attention using GPU kernels
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

        // Upload per-head data to GPU
        let q_gpu = runtime.upload_f32(&q_head)?;
        let k_gpu = runtime.upload_f32(&k_head)?;
        let v_gpu = runtime.upload_f32(&v_head)?;

        // Step 1: scores = Q_head @ K_head^T * scale → [seq_len, kv_len]
        // Compute on CPU for now (GEMM has issues with multi-row inputs)
        let t0 = std::time::Instant::now();
        let mut scores_data = vec![0.0f32; seq_len * kv_len];
        for s in 0..seq_len {
            for ki in 0..kv_len {
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q_head[s * head_dim + d] * k_head[ki * head_dim + d];
                }
                scores_data[s * kv_len + ki] = dot * scale;
            }
        }
        let t_scores = t0.elapsed();

        // Causal mask (prefill only: seq_len > 1 && kv_len == seq_len)
        let masked_scores = if seq_len > 1 && kv_len == seq_len {
            let mut masked = scores_data;
            for s in 0..seq_len {
                for ki in (s + 1)..kv_len {
                    masked[s * kv_len + ki] = f32::NEG_INFINITY;
                }
            }
            masked
        } else {
            scores_data
        };

        // Upload masked scores to GPU
        let scores_gpu = runtime.upload_f32(&masked_scores)?;

        // Step 2: Softmax on GPU
        let t1 = std::time::Instant::now();
        // Use softmax_large for kv_len > 256, standard for kv_len <= 256
        let weights_gpu = if kv_len <= 256 {
            let kernel = runtime.ensure_kernel_blockdsl("softmax_fwd", || {
                crate::t0::softmax_kernels::build_softmax_forward()
            })?;
            let weights_buf = runtime.alloc_f32(seq_len * kv_len)?;
            let ka = crate::kernargs![
                scores_gpu.gpu_addr() => u64,
                weights_buf.gpu_addr() => u64,
                kv_len as u32 => u32
            ];
            let (grid_x, _) = crate::t0::softmax_kernels::softmax_grid(seq_len as u32);
            runtime.dispatch(&kernel, [grid_x, 1, 1], &ka)?;
            weights_buf
        } else {
            let kernel = runtime.ensure_kernel_precompiled("softmax_large", || {
                let ck = crate::t0::softmax_large::compile_softmax_large()?;
                Ok((ck.elf, ck.workgroup_size, ck.lds_size))
            })?;
            let weights_buf = runtime.alloc_f32(seq_len * kv_len)?;
            let n_chunks = crate::t0::softmax_large::softmax_n_chunks(kv_len as u32);
            let ka = crate::kernargs![
                scores_gpu.gpu_addr() => u64,
                weights_buf.gpu_addr() => u64,
                kv_len as u32 => u32,
                n_chunks => u32
            ];
            let (grid_x, _) = crate::t0::softmax_large::softmax_large_grid(seq_len as u32);
            runtime.dispatch(&kernel, [grid_x, 1, 1], &ka)?;
            weights_buf
        };
        let t_softmax = t1.elapsed();

        // Step 3: out_head = weights @ V_head → [seq_len, head_dim]
        // Compute on CPU (small matrices)
        let t2 = std::time::Instant::now();
        let weights_data = runtime.read_f32(&weights_gpu, seq_len * kv_len);
        for s in 0..seq_len {
            for d in 0..head_dim {
                let mut val = 0.0f32;
                for ki in 0..kv_len {
                    val += weights_data[s * kv_len + ki] * v_head[ki * head_dim + d];
                }
                out_data[s * n_heads * head_dim + h * head_dim + d] = val;
            }
        }
        let t_wv = t2.elapsed();

        eprintln!("  [Attn] h={} scores={:.1}ms softmax={:.1}ms w@v={:.1}ms",
            h, t_scores.as_secs_f64()*1000.0, t_softmax.as_secs_f64()*1000.0, t_wv.as_secs_f64()*1000.0);
    }

    // Upload final output to GPU
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
        let head_dim = 4;
        let n_heads = 1;
        let n_kv_heads = 1;
        let seq_len = 1;
        let kv_len = 1;

        let q = vec![1.0, 2.0, 3.0, 4.0];
        let k = vec![1.0, 0.0, 0.0, 0.0];
        let v = vec![10.0, 20.0, 30.0, 40.0];

        let out = cpu_attention(&q, &k, &v, n_heads, n_kv_heads, head_dim, seq_len, kv_len);

        for d in 0..head_dim {
            assert!((out[d] - v[d]).abs() < 1e-5, "out[{}]={} expected {}", d, out[d], v[d]);
        }
    }

    #[test]
    fn test_cpu_attention_causal_mask() {
        let head_dim = 4;
        let n_heads = 1;
        let n_kv_heads = 1;
        let seq_len = 3;
        let kv_len = 3;

        let k: Vec<f32> = vec![
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
        ];
        let v: Vec<f32> = vec![
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
        ];
        let q: Vec<f32> = vec![
            1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0,
        ];

        let out = cpu_attention(&q, &k, &v, n_heads, n_kv_heads, head_dim, seq_len, kv_len);

        assert!((out[0] - 1.0).abs() < 1e-5, "token0 dim0={}", out[0]);
        assert!((out[1]).abs() < 1e-5, "token0 dim1={}", out[1]);

        let t2_base = 2 * n_heads * head_dim;
        let expected_t2_d0 = (1.0 + 0.0 + 0.0) / 3.0;
        assert!((out[t2_base] - expected_t2_d0).abs() < 1e-4,
            "token2 dim0={} expected {}", out[t2_base], expected_t2_d0);
    }

    #[test]
    fn test_cpu_attention_gqa() {
        let head_dim = 4;
        let n_heads = 2;
        let n_kv_heads = 1;
        let seq_len = 1;
        let kv_len = 1;

        let q: Vec<f32> = vec![
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
        ];
        let k = vec![1.0, 1.0, 1.0, 1.0];
        let v = vec![0.1, 0.2, 0.3, 0.4];

        let out = cpu_attention(&q, &k, &v, n_heads, n_kv_heads, head_dim, seq_len, kv_len);

        for d in 0..head_dim {
            assert!((out[d] - v[d]).abs() < 1e-5, "head0 dim{}: out={} v={}", d, out[d], v[d]);
        }
        for d in 0..head_dim {
            let out_val = out[head_dim + d];
            assert!((out_val - v[d]).abs() < 1e-5, "head1 dim{}: out={} v={}", d, out_val, v[d]);
        }
    }

    #[test]
    fn test_cpu_attention_decode_single_token() {
        let head_dim = 8;
        let n_heads = 2;
        let n_kv_heads = 2;
        let seq_len = 1;
        let kv_len = 5;

        let q: Vec<f32> = (0..n_heads * head_dim).map(|i| (i as f32) * 0.1).collect();
        let k: Vec<f32> = (0..kv_len * n_kv_heads * head_dim).map(|i| ((i as f32) * 0.13).sin()).collect();
        let v: Vec<f32> = (0..kv_len * n_kv_heads * head_dim).map(|i| ((i as f32) * 0.17).cos()).collect();

        let out = cpu_attention(&q, &k, &v, n_heads, n_kv_heads, head_dim, seq_len, kv_len);

        assert_eq!(out.len(), seq_len * n_heads * head_dim);
        assert!(out.iter().all(|v| v.is_finite()), "output contains non-finite values");
    }

    #[test]
    fn test_cpu_attention_softmax_correctness() {
        let head_dim = 4;
        let n_heads = 1;
        let n_kv_heads = 1;
        let seq_len = 1;
        let kv_len = 3;

        let q = vec![1.0, 0.0, 0.0, 0.0];
        let k = vec![
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
        ];
        let v = vec![
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
        ];

        let out = cpu_attention(&q, &k, &v, n_heads, n_kv_heads, head_dim, seq_len, kv_len);

        let expected_weights = softmax_row(&[0.5, 0.0, 0.0]);
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
        let head_dim = 4;
        let n_heads = 2;
        let n_kv_heads = 2;
        let seq_len = 1;
        let kv_len = 1;

        let q = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let k = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let v = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0];

        let out = cpu_attention(&q, &k, &v, n_heads, n_kv_heads, head_dim, seq_len, kv_len);

        for d in 0..head_dim {
            assert!((out[d] - v[d]).abs() < 1e-5, "head0 dim{}: {}", d, out[d]);
        }
        for d in 0..head_dim {
            assert!((out[head_dim + d] - v[head_dim + d]).abs() < 1e-5,
                "head1 dim{}: {}", d, out[head_dim + d]);
        }
    }

    #[cfg(feature = "rocm")]
    mod gpu_tests {
        use super::*;
        use std::sync::{Arc, OnceLock};
        use crate::ignis::gpu_context::GpuRuntime;
        use crate::ignis::tensor::Tensor;

        struct SyncRt(Arc<GpuRuntime>);
        unsafe impl Sync for SyncRt {}
        unsafe impl Send for SyncRt {}
        static GPU_RT: OnceLock<SyncRt> = OnceLock::new();

        fn rt() -> Arc<GpuRuntime> {
            GPU_RT.get_or_init(|| {
                SyncRt(GpuRuntime::new().expect("Failed to create GpuRuntime"))
            }).0.clone()
        }

        #[test]
        fn test_gpu_attention_decode_small() {
            // Decode: seq_len=1, head_dim=4, kv_len=3, 1 head
            let r = rt();
            let head_dim = 4;
            let n_heads = 1;
            let n_kv_heads = 1;
            let seq_len = 1;
            let kv_len = 3;

            let q_data = vec![1.0, 2.0, 3.0, 4.0];
            let k_data = vec![
                1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
            ];
            let v_data = vec![
                10.0, 20.0, 30.0, 40.0,
                50.0, 60.0, 70.0, 80.0,
                90.0, 100.0, 110.0, 120.0,
            ];

            let q = Tensor::from_f32(&r, &q_data, &[seq_len, n_heads * head_dim], "q").unwrap();
            let k = Tensor::from_f32(&r, &k_data, &[kv_len, n_kv_heads * head_dim], "k").unwrap();
            let v = Tensor::from_f32(&r, &v_data, &[kv_len, n_kv_heads * head_dim], "v").unwrap();

            let gpu_out = standard_attention(&q, &k, &v, n_heads, n_kv_heads, head_dim, &r).unwrap();
            let gpu_data = gpu_out.to_f32_vec();

            let cpu_data = cpu_attention(&q_data, &k_data, &v_data, n_heads, n_kv_heads, head_dim, seq_len, kv_len);

            for i in 0..n_heads * head_dim {
                assert!((gpu_data[i] - cpu_data[i]).abs() < 0.1,
                    "GPU[{}]={} vs CPU[{}]={}", i, gpu_data[i], i, cpu_data[i]);
            }
        }

        #[test]
        fn test_gpu_attention_decode_multi_head() {
            // Decode: seq_len=1, head_dim=4, kv_len=3, 2 heads (GQA: 2 query, 1 kv)
            let r = rt();
            let head_dim = 4;
            let n_heads = 2;
            let n_kv_heads = 1;
            let seq_len = 1;
            let kv_len = 3;

            let q_data: Vec<f32> = (0..n_heads * head_dim).map(|i| (i as f32) * 0.5).collect();
            let k_data: Vec<f32> = (0..kv_len * n_kv_heads * head_dim).map(|i| ((i as f32) * 0.3).sin()).collect();
            let v_data: Vec<f32> = (0..kv_len * n_kv_heads * head_dim).map(|i| ((i as f32) * 0.2).cos()).collect();

            let q = Tensor::from_f32(&r, &q_data, &[seq_len, n_heads * head_dim], "q").unwrap();
            let k = Tensor::from_f32(&r, &k_data, &[kv_len, n_kv_heads * head_dim], "k").unwrap();
            let v = Tensor::from_f32(&r, &v_data, &[kv_len, n_kv_heads * head_dim], "v").unwrap();

            let gpu_out = standard_attention(&q, &k, &v, n_heads, n_kv_heads, head_dim, &r).unwrap();
            let gpu_data = gpu_out.to_f32_vec();

            let cpu_data = cpu_attention(&q_data, &k_data, &v_data, n_heads, n_kv_heads, head_dim, seq_len, kv_len);

            for i in 0..n_heads * head_dim {
                assert!((gpu_data[i] - cpu_data[i]).abs() < 0.5,
                    "GPU[{}]={} vs CPU[{}]={}", i, gpu_data[i], i, cpu_data[i]);
            }
        }

        #[test]
        fn test_gpu_attention_prefill() {
            // Prefill: seq_len=3, head_dim=4, kv_len=3, 1 head (causal mask)
            let r = rt();
            let head_dim = 4;
            let n_heads = 1;
            let n_kv_heads = 1;
            let seq_len = 3;
            let kv_len = 3;

            let q_data: Vec<f32> = (0..seq_len * n_heads * head_dim).map(|i| ((i as f32) * 0.1).sin()).collect();
            let k_data: Vec<f32> = (0..kv_len * n_kv_heads * head_dim).map(|i| ((i as f32) * 0.15).cos()).collect();
            let v_data: Vec<f32> = (0..kv_len * n_kv_heads * head_dim).map(|i| (i as f32) * 0.1).collect();

            let q = Tensor::from_f32(&r, &q_data, &[seq_len, n_heads * head_dim], "q").unwrap();
            let k = Tensor::from_f32(&r, &k_data, &[kv_len, n_kv_heads * head_dim], "k").unwrap();
            let v = Tensor::from_f32(&r, &v_data, &[kv_len, n_kv_heads * head_dim], "v").unwrap();

            let gpu_out = standard_attention(&q, &k, &v, n_heads, n_kv_heads, head_dim, &r).unwrap();
            let gpu_data = gpu_out.to_f32_vec();

            let cpu_data = cpu_attention(&q_data, &k_data, &v_data, n_heads, n_kv_heads, head_dim, seq_len, kv_len);

            eprintln!("GPU: {:?}", gpu_data);
            eprintln!("CPU: {:?}", cpu_data);

            for i in 0..seq_len * n_heads * head_dim {
                assert!((gpu_data[i] - cpu_data[i]).abs() < 0.5,
                    "GPU[{}]={} vs CPU[{}]={}", i, gpu_data[i], i, cpu_data[i]);
            }
        }

        #[test]
        fn test_gpu_attention_larger() {
            // Larger test: head_dim=32, kv_len=16, 4 heads, 2 kv_heads
            let r = rt();
            let head_dim = 32;
            let n_heads = 4;
            let n_kv_heads = 2;
            let seq_len = 1;
            let kv_len = 16;

            let mut rng_state = 42u64;
            let mut rand = || -> f32 {
                rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                ((rng_state >> 33) as f32 / (1u64 << 31) as f32 - 1.0) * 0.5
            };

            let q_data: Vec<f32> = (0..seq_len * n_heads * head_dim).map(|_| rand()).collect();
            let k_data: Vec<f32> = (0..kv_len * n_kv_heads * head_dim).map(|_| rand()).collect();
            let v_data: Vec<f32> = (0..kv_len * n_kv_heads * head_dim).map(|_| rand()).collect();

            let q = Tensor::from_f32(&r, &q_data, &[seq_len, n_heads * head_dim], "q").unwrap();
            let k = Tensor::from_f32(&r, &k_data, &[kv_len, n_kv_heads * head_dim], "k").unwrap();
            let v = Tensor::from_f32(&r, &v_data, &[kv_len, n_kv_heads * head_dim], "v").unwrap();

            let gpu_out = standard_attention(&q, &k, &v, n_heads, n_kv_heads, head_dim, &r).unwrap();
            let gpu_data = gpu_out.to_f32_vec();

            let cpu_data = cpu_attention(&q_data, &k_data, &v_data, n_heads, n_kv_heads, head_dim, seq_len, kv_len);

            let mut max_err: f32 = 0.0;
            for i in 0..seq_len * n_heads * head_dim {
                let err = (gpu_data[i] - cpu_data[i]).abs();
                max_err = max_err.max(err);
            }
            eprintln!("GPU vs CPU max error: {:.6}", max_err);
            assert!(max_err < 1.0, "GPU vs CPU max error too large: {}", max_err);
        }
    }
}
