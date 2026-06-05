//! Standard scaled dot-product attention with GQA support (inference only).
//!
//! Computes: output = softmax(Q @ K^T / sqrt(d_k)) @ V
//!
//! Supports Grouped Query Attention (GQA) where n_kv_heads < n_heads.
//!
//! Fully GPU: gather, transpose, GEMM, scale+mask, softmax, scatter
//! all run on GPU. Zero PCIe round-trips in the per-head loop.

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
    crate::profile_scope!("standard_attention");
    use crate::t0::attention_kernels as ak;

    let q_shape = q.shape();
    let k_shape = k.shape();
    crate::profiler::set_shapes(
        vec![
            crate::profiler::ShapeInfo::new(q_shape),
            crate::profiler::ShapeInfo::new(k_shape),
            crate::profiler::ShapeInfo::new(v.shape()),
        ],
        vec![crate::profiler::ShapeInfo::new(&[q_shape[0], n_heads * head_dim])],
    );
    assert!(q_shape.len() == 2, "q: expected [seq, n_heads*head_dim]");
    assert!(k_shape.len() == 2, "k: expected [kv_len, n_kv_heads*head_dim]");

    let seq_len = q_shape[0];
    let kv_len = k_shape[0];
    let gqa_ratio = n_heads / n_kv_heads;

    assert_eq!(q_shape[1], n_heads * head_dim);
    assert_eq!(k_shape[1], n_kv_heads * head_dim);

    let scale = 1.0 / (head_dim as f32).sqrt();
    let is_prefill = seq_len > 1 && kv_len == seq_len;

    // Get GPU buffer addresses (no CPU download)
    let q_buf = q.buffer_arc();
    let k_buf = k.buffer_arc();
    let v_buf = v.buffer_arc();

    // Compile GPU kernels (cached after first call)
    let gather_kernel = runtime.ensure_kernel_blockdsl("attn_gather", || ak::build_attn_gather())?;
    let transpose_kernel = runtime.ensure_kernel_blockdsl("attn_transpose", || ak::build_attn_transpose())?;
    let scale_kernel = if is_prefill {
        runtime.ensure_kernel_blockdsl("attn_scale_causal", || ak::build_attn_scale_causal())?
    } else {
        runtime.ensure_kernel_blockdsl("attn_scale", || ak::build_attn_scale())?
    };
    let scatter_kernel = runtime.ensure_kernel_blockdsl("attn_scatter", || ak::build_attn_scatter())?;

    // Allocate output buffer on GPU
    let out_buf = runtime.alloc_f32(seq_len * n_heads * head_dim)?;
    out_buf.zero();

    // Per-head scratch buffers (reused across heads)
    let q_head_buf = runtime.alloc_f32(seq_len * head_dim)?;
    let k_head_buf = runtime.alloc_f32(kv_len * head_dim)?;
    let v_head_buf = runtime.alloc_f32(kv_len * head_dim)?;
    let k_t_buf = runtime.alloc_f32(head_dim * kv_len)?;
    let scores_buf = runtime.alloc_f32(seq_len * kv_len)?;

    let q_stride = (n_heads * head_dim) as u32;
    let kv_stride = (n_kv_heads * head_dim) as u32;

    for h in 0..n_heads {
        let kv_h = h / gqa_ratio;
        let t0 = std::time::Instant::now();

        // Step 1: GPU gather Q_head [seq_len, head_dim] from Q[seq_len, n_heads*head_dim]
        let ka = crate::kernargs![
            q_buf.gpu_addr() => u64,
            q_head_buf.gpu_addr() => u64,
            head_dim as u32 => u32,
            q_stride => u32,
            h as u32 => u32,
            seq_len as u32 => u32
        ];
        let (gx, _) = ak::attn_gather_grid(seq_len as u32);
        runtime.dispatch(&gather_kernel, [gx, 1, 1], &ka)?;

        // Step 2: GPU gather K_head [kv_len, head_dim] from K
        let ka = crate::kernargs![
            k_buf.gpu_addr() => u64,
            k_head_buf.gpu_addr() => u64,
            head_dim as u32 => u32,
            kv_stride => u32,
            kv_h as u32 => u32,
            kv_len as u32 => u32
        ];
        let (gx, _) = ak::attn_gather_grid(kv_len as u32);
        runtime.dispatch(&gather_kernel, [gx, 1, 1], &ka)?;

        // Step 3: GPU gather V_head [kv_len, head_dim] from V
        let ka = crate::kernargs![
            v_buf.gpu_addr() => u64,
            v_head_buf.gpu_addr() => u64,
            head_dim as u32 => u32,
            kv_stride => u32,
            kv_h as u32 => u32,
            kv_len as u32 => u32
        ];
        let (gx, _) = ak::attn_gather_grid(kv_len as u32);
        runtime.dispatch(&gather_kernel, [gx, 1, 1], &ka)?;

        // Debug: check K and V values for head 0 during decode
        if crate::t0_debug() && h == 0 && seq_len == 1 {
            let mut k_data = vec![0f32; kv_len * head_dim];
            k_head_buf.read(unsafe { std::slice::from_raw_parts_mut(k_data.as_mut_ptr() as *mut u8, kv_len * head_dim * 4) });
            let k_norm: f32 = k_data.iter().map(|x| x*x).sum::<f32>().sqrt();
            let mut v_data = vec![0f32; kv_len * head_dim];
            v_head_buf.read(unsafe { std::slice::from_raw_parts_mut(v_data.as_mut_ptr() as *mut u8, kv_len * head_dim * 4) });
            let v_norm: f32 = v_data.iter().map(|x| x*x).sum::<f32>().sqrt();
            let last_k = (kv_len - 1) * head_dim;
            let prev_k = if kv_len > 1 { (kv_len - 2) * head_dim } else { 0 };
            eprintln!("    [attn decode h=0] kv_len={} K_norm={:.4} V_norm={:.4} K_last[0..3]={:.4} {:.4} {:.4} {:.4} K_prev[0..3]={:.4} {:.4} {:.4} {:.4}",
                kv_len, k_norm, v_norm,
                k_data[last_k], k_data[last_k+1], k_data[last_k+2], k_data[last_k+3],
                k_data[prev_k], k_data[prev_k+1], k_data[prev_k+2], k_data[prev_k+3]);
        }

        // Step 4: GPU transpose K_head [kv_len, head_dim] → K_T [head_dim, kv_len]
        let ka = crate::kernargs![
            k_head_buf.gpu_addr() => u64,
            k_t_buf.gpu_addr() => u64,
            head_dim as u32 => u32,
            kv_len as u32 => u32
        ];
        let (gx, _) = ak::attn_transpose_grid(kv_len as u32);
        runtime.dispatch(&transpose_kernel, [gx, 1, 1], &ka)?;

        let t_gather = t0.elapsed();

        // Step 5: GPU GEMM scores = Q_head @ K_T → [seq_len, kv_len]
        let t1 = std::time::Instant::now();
        let gemm_out = if std::env::var("T0_F32_GEMM").ok().as_deref() == Some("1") {
            let q_data = runtime.read_f32(&q_head_buf, seq_len * head_dim);
            let k_data = runtime.read_f32(&k_t_buf, head_dim * kv_len);
            let y = super::super::ops::bf16_matmul::gemm_f32_cpu(&q_data, &k_data, seq_len, head_dim, kv_len);
            let buf = runtime.alloc_f32(seq_len * kv_len)?;
            runtime.write_f32(&buf, &y);
            buf
        } else {
            super::super::ops::bf16_matmul::gemm_f32_raw(
                runtime, &q_head_buf, &k_t_buf, seq_len, head_dim, kv_len,
            )?
        };
        let t_gemm = t1.elapsed();

        // Step 6: GPU scale + causal mask
        let t2 = std::time::Instant::now();

        // CPU reference: scale + causal mask (debug only)
        let is_causal = seq_len > 1;
        if crate::t0_debug() && h == 0 {
            let scores_raw = runtime.read_f32(&gemm_out, seq_len * kv_len);
            let mut cpu_scaled = vec![0.0f32; seq_len * kv_len];
            for row in 0..seq_len {
                for col in 0..kv_len {
                    let s = scores_raw[row * kv_len + col] * scale;
                    cpu_scaled[row * kv_len + col] = if is_causal && col > row { f32::NEG_INFINITY } else { s };
                }
            }
            // GPU scale
            let ka = crate::kernargs![
                gemm_out.gpu_addr() => u64,
                scores_buf.gpu_addr() => u64,
                kv_len as u32 => u32,
                scale => f32
            ];
            let (gx, _) = ak::attn_scale_grid(seq_len as u32);
            runtime.dispatch(&scale_kernel, [gx, 1, 1], &ka)?;
            let gpu_scaled = runtime.read_f32(&scores_buf, seq_len * kv_len);
            let max_diff: f32 = cpu_scaled.iter().zip(gpu_scaled.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
            eprintln!("    [attn h={}] scale: max_diff={:.6}", h, max_diff);
        } else {
            let ka = crate::kernargs![
                gemm_out.gpu_addr() => u64,
                scores_buf.gpu_addr() => u64,
                kv_len as u32 => u32,
                scale => f32
            ];
            let (gx, _) = ak::attn_scale_grid(seq_len as u32);
            runtime.dispatch(&scale_kernel, [gx, 1, 1], &ka)?;
        }
        let t_scale = t2.elapsed();

        // Step 7: GPU softmax
        let t3 = std::time::Instant::now();
        let weights_buf = if kv_len <= 256 {
            let s_kernel = runtime.ensure_kernel_blockdsl("softmax_fwd", || {
                crate::t0::softmax_kernels::build_softmax_forward()
            })?;
            let wb = runtime.alloc_f32(seq_len * kv_len)?;
            let ka = crate::kernargs![
                scores_buf.gpu_addr() => u64,
                wb.gpu_addr() => u64,
                kv_len as u32 => u32
            ];
            let (gx, _) = crate::t0::softmax_kernels::softmax_grid(seq_len as u32);
            runtime.dispatch(&s_kernel, [gx, 1, 1], &ka)?;
            wb
        } else {
            let s_kernel = runtime.ensure_kernel_precompiled("softmax_large", || {
                let ck = crate::t0::softmax_large::compile_softmax_large()?;
                Ok((ck.elf, ck.workgroup_size, ck.lds_size))
            })?;
            let wb = runtime.alloc_f32(seq_len * kv_len)?;
            let n_chunks = crate::t0::softmax_large::softmax_n_chunks(kv_len as u32);
            let ka = crate::kernargs![
                scores_buf.gpu_addr() => u64,
                wb.gpu_addr() => u64,
                kv_len as u32 => u32,
                n_chunks => u32
            ];
            let (gx, _) = crate::t0::softmax_large::softmax_large_grid(seq_len as u32);
            runtime.dispatch(&s_kernel, [gx, 1, 1], &ka)?;
            wb
        };
        let t_softmax = t3.elapsed();

        // CPU softmax reference for head 0 (debug only)
        if crate::t0_debug() && h == 0 {
            let scores_data = runtime.read_f32(&scores_buf, seq_len * kv_len);
            let gpu_weights = runtime.read_f32(&weights_buf, seq_len * kv_len);
            let mut cpu_weights = vec![0.0f32; seq_len * kv_len];
            for row in 0..seq_len {
                let mut max_val = f32::NEG_INFINITY;
                for col in 0..kv_len {
                    let s = scores_data[row * kv_len + col];
                    if s.is_finite() && s > max_val { max_val = s; }
                }
                let mut sum = 0.0f32;
                for col in 0..kv_len {
                    let s = scores_data[row * kv_len + col];
                    let e = if s.is_finite() { (s - max_val).exp() } else { 0.0 };
                    cpu_weights[row * kv_len + col] = e;
                    sum += e;
                }
                for col in 0..kv_len { cpu_weights[row * kv_len + col] /= sum; }
            }
            let max_diff: f32 = cpu_weights.iter().zip(gpu_weights.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
            eprintln!("    [attn h={}] softmax: max_diff={:.6}", h, max_diff);
        }

        // Step 8: GPU GEMM out_head = weights @ V_head → [seq_len, head_dim]
        let t4 = std::time::Instant::now();
        let out_head_buf = if std::env::var("T0_F32_GEMM").ok().as_deref() == Some("1") {
            let w_data = runtime.read_f32(&weights_buf, seq_len * kv_len);
            let v_data = runtime.read_f32(&v_head_buf, kv_len * head_dim);
            let y = super::super::ops::bf16_matmul::gemm_f32_cpu(&w_data, &v_data, seq_len, kv_len, head_dim);
            let buf = runtime.alloc_f32(seq_len * head_dim)?;
            runtime.write_f32(&buf, &y);
            buf
        } else {
            super::super::ops::bf16_matmul::gemm_f32_raw(
                runtime, &weights_buf, &v_head_buf, seq_len, kv_len, head_dim,
            )?
        };
        let t_gemm2 = t4.elapsed();

        // CPU reference for weights@V (debug only)
        if crate::t0_debug() && h == 0 && std::env::var("T0_F32_GEMM").ok().as_deref() == Some("1") {
            let w_data = runtime.read_f32(&weights_buf, seq_len * kv_len);
            let v_data = runtime.read_f32(&v_head_buf, kv_len * head_dim);
            let mut cpu_out = vec![0.0f32; seq_len * head_dim];
            for row in 0..seq_len {
                for col in 0..head_dim {
                    let mut sum = 0.0f32;
                    for kk in 0..kv_len {
                        sum += w_data[row * kv_len + kk] * v_data[kk * head_dim + col];
                    }
                    cpu_out[row * head_dim + col] = sum;
                }
            }
            let gpu_out = runtime.read_f32(&out_head_buf, seq_len * head_dim);
            let max_diff: f32 = cpu_out.iter().zip(gpu_out.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
            let cpu_norm: f32 = cpu_out.iter().map(|x| x*x).sum::<f32>().sqrt();
            let gpu_norm: f32 = gpu_out.iter().map(|x| x*x).sum::<f32>().sqrt();
            eprintln!("    [attn h=0] weights@V: CPU={:.4} GPU={:.4} max_diff={:.6}", cpu_norm, gpu_norm, max_diff);
        }

        // Step 9: GPU scatter out_head → out_buf at position h
        let t5 = std::time::Instant::now();
        let ka = crate::kernargs![
            out_head_buf.gpu_addr() => u64,
            out_buf.gpu_addr() => u64,
            head_dim as u32 => u32,
            (n_heads * head_dim) as u32 => u32,
            h as u32 => u32,
            seq_len as u32 => u32
        ];
        let (gx, _) = ak::attn_scatter_grid(seq_len as u32);
        runtime.dispatch(&scatter_kernel, [gx, 1, 1], &ka)?;
        let t_scatter = t5.elapsed();

        if crate::t0_debug() {
            eprintln!("  [Attn] h={} gather={:.1}ms gemm_qk={:.1}ms scale={:.1}ms softmax={:.1}ms gemm_wv={:.1}ms scatter={:.1}ms",
                h, t_gather.as_secs_f64()*1000.0, t_gemm.as_secs_f64()*1000.0,
                t_scale.as_secs_f64()*1000.0, t_softmax.as_secs_f64()*1000.0,
                t_gemm2.as_secs_f64()*1000.0, t_scatter.as_secs_f64()*1000.0);
        }
    }

    Ok(Tensor::from_buffer(Arc::new(out_buf), runtime,
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

            // bf16 GEMM introduces precision loss; GPU bf16 rounding differs from CPU
            for i in 0..n_heads * head_dim {
                assert!((gpu_data[i] - cpu_data[i]).abs() < 0.5,
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

        #[test]
        fn test_gpu_attention_zerocopy_kv_cache() {
            // Verify that using from_gpu_addr (KV cache zero-copy) produces
            // identical results to using from_f32 (CPU upload path).
            use crate::ignis::kv_cache::{KvCache, KvCacheConfig};

            let r = rt();
            let head_dim = 8;
            let n_heads = 2;
            let n_kv_heads = 2;
            let seq_len = 1; // decode
            let kv_len = 5;  // 5 tokens already cached

            let cfg = KvCacheConfig {
                num_layers: 1,
                num_kv_heads: n_kv_heads,
                head_dim,
                max_seq_len: 32,
            };
            let cache = KvCache::new(&r, cfg).unwrap();

            // Fill cache with 5 tokens
            let kv_dim = n_kv_heads * head_dim;
            for pos in 0..kv_len {
                let k_data: Vec<f32> = (0..kv_dim).map(|i| (pos * 100 + i) as f32 * 0.01).collect();
                let v_data: Vec<f32> = (0..kv_dim).map(|i| (pos * 100 + i) as f32 * 0.02).collect();
                let key = Tensor::from_f32(&r, &k_data, &[n_kv_heads, head_dim], "k").unwrap();
                let val = Tensor::from_f32(&r, &v_data, &[n_kv_heads, head_dim], "v").unwrap();
                cache.append(&r, 0, &key, &val).unwrap();
                cache.advance();
            }

            // Q: random
            let q_data: Vec<f32> = (0..n_heads * head_dim).map(|i| ((i as f32) * 0.1).sin()).collect();
            let q = Tensor::from_f32(&r, &q_data, &[seq_len, n_heads * head_dim], "q").unwrap();

            // Path A: old way — CPU read + re-upload
            let k_cpu_data = cache.read_k_layer(&r, 0);
            let v_cpu_data = cache.read_v_layer(&r, 0);
            let k_old = Tensor::from_f32(&r, &k_cpu_data, &[kv_len, kv_dim], "k_old").unwrap();
            let v_old = Tensor::from_f32(&r, &v_cpu_data, &[kv_len, kv_dim], "v_old").unwrap();
            let out_old = standard_attention(&q, &k_old, &v_old, n_heads, n_kv_heads, head_dim, &r).unwrap();

            // Path B: zero-copy — from_gpu_addr
            let k_slice = cache.get_k(0);
            let v_slice = cache.get_v(0);
            let k_new = Tensor::from_gpu_addr(k_slice.gpu_addr, &r, &[kv_len, kv_dim], "k_new");
            let v_new = Tensor::from_gpu_addr(v_slice.gpu_addr, &r, &[kv_len, kv_dim], "v_new");
            let out_new = standard_attention(&q, &k_new, &v_new, n_heads, n_kv_heads, head_dim, &r).unwrap();

            let old_data = out_old.to_f32_vec();
            let new_data = out_new.to_f32_vec();

            let mut max_err: f32 = 0.0;
            for i in 0..n_heads * head_dim {
                let err = (old_data[i] - new_data[i]).abs();
                max_err = max_err.max(err);
            }
            eprintln!("Zero-copy vs CPU-upload max error: {:.8}", max_err);
            assert!(max_err < 1e-5, "Zero-copy mismatch: max_err={}", max_err);
        }
    }
}
