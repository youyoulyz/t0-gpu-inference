//! TransformerLayer — Attention + FFN with RMSNorm residual connections.
//!
//! Supports GQA (Grouped Query Attention): num_heads >= num_kv_heads.
//!
//! Architecture:
//!   x → RMSNorm → Q/K/V projections → attention → output projection → residual
//!   x → RMSNorm → gate/up projections → SiLU gate → down projection → residual

#[cfg(feature = "rocm")]
use std::sync::Arc;
#[cfg(feature = "rocm")]
use super::Module;
#[cfg(feature = "rocm")]
use super::linear::Linear;
#[cfg(feature = "rocm")]
use super::super::tensor::Tensor;
#[cfg(feature = "rocm")]
use super::super::gpu_context::GpuRuntime;
#[cfg(feature = "rocm")]
use super::super::ops;
#[cfg(feature = "rocm")]
use super::config::Qwen3Config;

/// Single Transformer layer with GQA attention.
///
/// Weight layout:
///   wq: [hidden_size, q_dim]                 — Q projection (q_dim = n_heads * head_dim)
///   wk: [hidden_size, kv_dim]                — K projection (kv_heads only)
///   wv: [hidden_size, kv_dim]                — V projection (kv_heads only)
///   wo: [q_dim, hidden_size]                 — output projection
///   w_gate: [hidden_size, intermediate_size] — FFN gate
///   w_up: [hidden_size, intermediate_size]   — FFN up
///   w_down: [intermediate_size, hidden_size] — FFN down
///   attn_norm_gamma: [hidden_size]
///   ffn_norm_gamma: [hidden_size]
///   q_norm_gamma: [head_dim]                 — QK-norm for queries
///   k_norm_gamma: [head_dim]                 — QK-norm for keys
#[cfg(feature = "rocm")]
pub struct TransformerLayer {
    pub wq: Linear,
    pub wk: Linear,
    pub wv: Linear,
    pub wo: Linear,
    pub w_gate: Linear,
    pub w_up: Linear,
    pub w_down: Linear,
    pub attn_norm_gamma: Tensor,
    pub ffn_norm_gamma: Tensor,
    pub q_norm_gamma: Tensor,
    pub k_norm_gamma: Tensor,
    pub dim: usize,
    pub d_head: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub q_dim: usize,
    pub kv_dim: usize,
    pub ffn_dim: usize,
    pub rope_theta: f32,
    runtime: Arc<GpuRuntime>,
}

#[cfg(feature = "rocm")]
impl TransformerLayer {
    /// Create from Qwen3Config.
    pub fn from_config(
        runtime: &Arc<GpuRuntime>,
        config: &Qwen3Config,
        layer_idx: usize,
    ) -> Result<Self, String> {
        let dim = config.hidden_size;
        let n_heads = config.num_attention_heads;
        let n_kv_heads = config.num_key_value_heads;
        let d_head = config.head_dim();
        let q_dim = config.q_dim();
        let kv_dim = config.kv_dim();
        let ffn_dim = config.intermediate_size;
        let prefix = format!("L{}", layer_idx);

        Ok(Self {
            // Q projection: [hidden_size, q_dim]  (q_dim = n_heads * head_dim, may != hidden_size)
            wq: Linear::new(runtime, dim, q_dim, &format!("{}_wq", prefix))?,
            // K projection: [hidden_size, kv_dim]  (GQA: kv_dim < q_dim when kv_heads < heads)
            wk: Linear::new(runtime, dim, kv_dim, &format!("{}_wk", prefix))?,
            // V projection: [hidden_size, kv_dim]
            wv: Linear::new(runtime, dim, kv_dim, &format!("{}_wv", prefix))?,
            // Output projection: [q_dim, hidden_size]
            wo: Linear::new(runtime, q_dim, dim, &format!("{}_wo", prefix))?,
            w_gate: Linear::new(runtime, dim, ffn_dim, &format!("{}_gate", prefix))?,
            w_up: Linear::new(runtime, dim, ffn_dim, &format!("{}_up", prefix))?,
            w_down: Linear::new(runtime, ffn_dim, dim, &format!("{}_down", prefix))?,
            attn_norm_gamma: {
                let mut g = Tensor::from_f32(runtime, &vec![1.0f32; dim], &[dim],
                    &format!("{}_attn_norm", prefix))?;
                g.set_requires_grad(true);
                g
            },
            ffn_norm_gamma: {
                let mut g = Tensor::from_f32(runtime, &vec![1.0f32; dim], &[dim],
                    &format!("{}_ffn_norm", prefix))?;
                g.set_requires_grad(true);
                g
            },
            q_norm_gamma: {
                let mut g = Tensor::from_f32(runtime, &vec![1.0f32; d_head], &[d_head],
                    &format!("{}_q_norm", prefix))?;
                g.set_requires_grad(true);
                g
            },
            k_norm_gamma: {
                let mut g = Tensor::from_f32(runtime, &vec![1.0f32; d_head], &[d_head],
                    &format!("{}_k_norm", prefix))?;
                g.set_requires_grad(true);
                g
            },
            dim,
            d_head,
            n_heads,
            n_kv_heads,
            q_dim,
            kv_dim,
            ffn_dim,
            rope_theta: config.rope_theta,
            runtime: runtime.clone(),
        })
    }

    /// Legacy constructor (MHA only, no GQA).
    pub fn new(
        runtime: &Arc<GpuRuntime>,
        dim: usize,
        n_heads: usize,
        ffn_mult: usize,
        layer_idx: usize,
    ) -> Result<Self, String> {
        let d_head = dim / n_heads;
        let ffn_dim = dim * ffn_mult;
        let prefix = format!("L{}", layer_idx);

        Ok(Self {
            wq: Linear::new(runtime, dim, dim, &format!("{}_wq", prefix))?,
            wk: Linear::new(runtime, dim, dim, &format!("{}_wk", prefix))?,
            wv: Linear::new(runtime, dim, dim, &format!("{}_wv", prefix))?,
            wo: Linear::new(runtime, dim, dim, &format!("{}_wo", prefix))?,
            w_gate: Linear::new(runtime, dim, ffn_dim, &format!("{}_gate", prefix))?,
            w_up: Linear::new(runtime, dim, ffn_dim, &format!("{}_up", prefix))?,
            w_down: Linear::new(runtime, ffn_dim, dim, &format!("{}_down", prefix))?,
            attn_norm_gamma: {
                let mut g = Tensor::from_f32(runtime, &vec![1.0f32; dim], &[dim],
                    &format!("{}_attn_norm", prefix))?;
                g.set_requires_grad(true);
                g
            },
            ffn_norm_gamma: {
                let mut g = Tensor::from_f32(runtime, &vec![1.0f32; dim], &[dim],
                    &format!("{}_ffn_norm", prefix))?;
                g.set_requires_grad(true);
                g
            },
            q_norm_gamma: {
                let mut g = Tensor::from_f32(runtime, &vec![1.0f32; d_head], &[d_head],
                    &format!("{}_q_norm", prefix))?;
                g.set_requires_grad(true);
                g
            },
            k_norm_gamma: {
                let mut g = Tensor::from_f32(runtime, &vec![1.0f32; d_head], &[d_head],
                    &format!("{}_k_norm", prefix))?;
                g.set_requires_grad(true);
                g
            },
            dim,
            d_head,
            n_heads,
            n_kv_heads: n_heads, // no GQA: kv_heads == heads
            q_dim: dim,
            kv_dim: dim,
            ffn_dim,
            rope_theta: 10000.0,
            runtime: runtime.clone(),
        })
    }

    /// Simple forward (no real attention — placeholder for testing).
    pub fn forward_simple(&self, x: &Tensor) -> Result<Tensor, String> {
        let device = &self.runtime.device;

        // Attention sub-layer
        let h = ops::rmsnorm::rmsnorm(x, &self.attn_norm_gamma, device)?;
        let q = self.wq.forward(&h)?;
        let _k = self.wk.forward(&h)?;
        let _v = self.wv.forward(&h)?;

        // Simplified: just Q @ Wo → residual
        let attn_out = ops::bf16_matmul::matmul(&q, &self.wo.weight, device)?;
        let x2 = ops::add::add(x, &attn_out, device)?;

        // FFN sub-layer
        let h2 = ops::rmsnorm::rmsnorm(&x2, &self.ffn_norm_gamma, device)?;
        let gate = self.w_gate.forward(&h2)?;
        let up = self.w_up.forward(&h2)?;
        let silu_out = ops::silu::silu_gate(&gate, &up, device)?;
        let ffn_out = self.w_down.forward(&silu_out)?;

        ops::add::add(&x2, &ffn_out, device)
    }

    /// Full forward with OCPA attention.
    pub fn forward_ocpa(&self, x: &Tensor, config: &ops::ocpa_attention::OcpaConfig) -> Result<Tensor, String> {
        let device = &self.runtime.device;

        let h = ops::rmsnorm::rmsnorm(x, &self.attn_norm_gamma, device)?;
        let q = self.wq.forward(&h)?;
        let k = self.wk.forward(&h)?;
        let v = self.wv.forward(&h)?;

        let attn_out = ops::ocpa_attention::ocpa_forward(&q, &k, &v, config, &self.runtime)?;
        let proj_out = self.wo.forward(&attn_out)?;
        let x2 = ops::add::add(x, &proj_out, device)?;

        let h2 = ops::rmsnorm::rmsnorm(&x2, &self.ffn_norm_gamma, device)?;
        let gate = self.w_gate.forward(&h2)?;
        let up = self.w_up.forward(&h2)?;
        let silu_out = ops::silu::silu_gate(&gate, &up, device)?;
        let ffn_out = self.w_down.forward(&silu_out)?;

        ops::add::add(&x2, &ffn_out, device)
    }

    /// Inference forward pass with RoPE, QK-norm, standard attention, and KV cache.
    ///
    /// # Arguments
    /// - `x`: [seq_len, hidden_size] f32 input
    /// - `pos`: starting position in the sequence (0 for prefill start, current_pos for decode)
    /// - `layer_idx`: this layer's index (for KV cache)
    /// - `kv_cache`: mutable KV cache for storing/retrieving K and V
    ///
    /// # Returns
    /// - output: [seq_len, hidden_size] f32
    pub fn forward_inference(
        &self,
        x: &Tensor,
        pos: usize,
        layer_idx: usize,
        kv_cache: &mut super::super::kv_cache::KvCache,
    ) -> Result<Tensor, String> {
        let device = &self.runtime.device;
        let seq_len = x.shape()[0];
        let dbg = layer_idx == 0 || layer_idx == 27;
        let norm = |t: &Tensor| -> f32 { t.to_f32_vec().iter().map(|v| v*v).sum::<f32>().sqrt() };

        // === Attention sub-layer ===
        let h = ops::rmsnorm::rmsnorm(x, &self.attn_norm_gamma, device)?;
        if dbg {
            let d = h.to_f32_vec();
            eprintln!("  [L{}] h_norm: {:.4} first5: {:.4} {:.4} {:.4} {:.4} {:.4}", layer_idx, norm(&h), d[0], d[1], d[2], d[3], d[4]);
        }

        // Q/K/V projections
        let q = self.wq.forward(&h)?;  // [seq, q_dim]
        let k = self.wk.forward(&h)?;  // [seq, kv_dim]
        let v = self.wv.forward(&h)?;  // [seq, kv_dim]
        if dbg {
            eprintln!("  [L{}] Q={:.4} K={:.4} V={:.4} wq_norm={:.4} wk_norm={:.4} wv_norm={:.4}",
                layer_idx, norm(&q), norm(&k), norm(&v),
                {
                    let w = self.wq.weight.to_f32_vec();
                    w.iter().map(|x| x*x).sum::<f32>().sqrt()
                },
                {
                    let w = self.wk.weight.to_f32_vec();
                    w.iter().map(|x| x*x).sum::<f32>().sqrt()
                },
                {
                    let w = self.wv.weight.to_f32_vec();
                    w.iter().map(|x| x*x).sum::<f32>().sqrt()
                });
            // CPU reference for Q projection (all tokens)
            let hd = h.to_f32_vec();
            let wq_w = self.wq.weight.to_f32_vec();
            let q_dim = self.q_dim;
            let dim = self.dim;
            let mut cpu_q = vec![0f32; seq_len * q_dim];
            for t in 0..seq_len {
                for j in 0..q_dim {
                    let mut sum = 0.0f32;
                    for i in 0..dim {
                        sum += hd[t * dim + i] * wq_w[i * q_dim + j];
                    }
                    cpu_q[t * q_dim + j] = sum;
                }
            }
            let cpu_q_norm: f32 = cpu_q.iter().map(|x| x*x).sum::<f32>().sqrt();
            let gpu_q = q.to_f32_vec();
            // Compare last token only
            let last = (seq_len - 1) * q_dim;
            let cpu_last_norm: f32 = cpu_q[last..last+q_dim].iter().map(|x| x*x).sum::<f32>().sqrt();
            let gpu_last_norm: f32 = gpu_q[last..last+q_dim].iter().map(|x| x*x).sum::<f32>().sqrt();
            eprintln!("  [L{}] CPU Q: total={:.4} last_tok={:.4}  GPU Q: total={:.4} last_tok={:.4}",
                layer_idx, cpu_q_norm, cpu_last_norm, norm(&q), gpu_last_norm);
            eprintln!("  [L{}] Q last_tok first3: CPU={:.6} {:.6} {:.6}  GPU={:.6} {:.6} {:.6}",
                layer_idx, cpu_q[last], cpu_q[last+1], cpu_q[last+2],
                gpu_q[last], gpu_q[last+1], gpu_q[last+2]);
        }

        // QK-norm: per-head RMSNorm on Q and K
        let q = ops::qk_norm::qk_norm(&q, &self.q_norm_gamma, self.n_heads, self.d_head, &self.runtime)?;
        let k = ops::qk_norm::qk_norm(&k, &self.k_norm_gamma, self.n_kv_heads, self.d_head, &self.runtime)?;
        if dbg { eprintln!("  [L{}] Q_qknorm={:.4} K_qknorm={:.4}", layer_idx, norm(&q), norm(&k)); }

        // RoPE: reshape to [seq*n_heads, head_dim], apply per-head, reshape back
        let q_2d = q.reshape(&[seq_len * self.n_heads, self.d_head]);
        let k_2d = k.reshape(&[seq_len * self.n_kv_heads, self.d_head]);
        let q_2d = ops::rope::rope_forward(&q_2d, pos, self.rope_theta, &self.runtime)?;
        let k_2d = ops::rope::rope_forward(&k_2d, pos, self.rope_theta, &self.runtime)?;
        let q = q_2d.reshape(&[seq_len, self.q_dim]);
        let k = k_2d.reshape(&[seq_len, self.kv_dim]);
        if dbg { eprintln!("  [L{}] Q_rope={:.4} K_rope={:.4}", layer_idx, norm(&q), norm(&k)); }

        // Store K/V in cache
        let kv_heads = self.n_kv_heads;
        let hd = self.d_head;
        let k_3d = k.reshape(&[seq_len, kv_heads, hd]);
        let v_3d = v.reshape(&[seq_len, kv_heads, hd]);

        // Write K/V at the current position (same for all layers, advance once after all layers)
        let write_pos = kv_cache.position();
        if layer_idx == 0 {
            eprintln!("  [L{}] kv_pos={} write_pos={} seq_len={} kv_len={}",
                layer_idx, kv_cache.position(), write_pos, seq_len, write_pos + seq_len);
        }
        if seq_len == 1 {
            kv_cache.append_at_pos(&self.runtime, layer_idx, write_pos, &k_3d, &v_3d)?;
        } else {
            kv_cache.append_many(&self.runtime, layer_idx, &k_3d, &v_3d)?;
        }

        // KV length includes the newly appended tokens
        let kv_len = write_pos + seq_len;
        // Get GPU addresses directly (bypass get_k/get_v which assert pos > 0)
        let k_addr = kv_cache.buf_gpu_addr() + kv_cache.k_offset(layer_idx, 0) as u64;
        let v_addr = kv_cache.buf_gpu_addr() + kv_cache.v_offset(layer_idx, 0) as u64;

        let k_cache = Tensor::from_gpu_addr(k_addr, &self.runtime, &[kv_len, self.kv_dim], "k_cache");
        let v_cache = Tensor::from_gpu_addr(v_addr, &self.runtime, &[kv_len, self.kv_dim], "v_cache");

        // Standard scaled dot-product attention with GQA
        let attn_out = ops::attention::standard_attention(
            &q, &k_cache, &v_cache,
            self.n_heads, self.n_kv_heads, self.d_head,
            &self.runtime,
        )?;
        if dbg {
            let ao = attn_out.to_f32_vec();
            let last = (seq_len - 1) * self.q_dim;
            let ao_last_norm: f32 = ao[last..last+self.q_dim].iter().map(|x| x*x).sum::<f32>().sqrt();
            eprintln!("  [L{}] attn_out: total={:.4} last_tok={:.4}", layer_idx, norm(&attn_out), ao_last_norm);
        }

        // Output projection
        let proj_out = self.wo.forward(&attn_out)?;
        if dbg {
            // CPU reference for proj_out (last token only)
            let ao = attn_out.to_f32_vec();
            let wo_w = self.wo.weight.to_f32_vec();
            let dim = self.dim;
            let qd = self.q_dim;
            let last_ao = (seq_len - 1) * qd;
            let mut cpu_proj = vec![0f32; dim];
            for j in 0..dim {
                let mut sum = 0.0f32;
                for i in 0..qd {
                    sum += ao[last_ao + i] * wo_w[i * dim + j];
                }
                cpu_proj[j] = sum;
            }
            let cpu_proj_norm: f32 = cpu_proj.iter().map(|x| x*x).sum::<f32>().sqrt();
            let po = proj_out.to_f32_vec();
            let last_po = (seq_len - 1) * dim;
            let gpu_proj_norm: f32 = po[last_po..last_po+dim].iter().map(|x| x*x).sum::<f32>().sqrt();
            eprintln!("  [L{}] proj_out: CPU_last={:.4} GPU_last={:.4} total={:.4}",
                layer_idx, cpu_proj_norm, gpu_proj_norm, norm(&proj_out));
        }
        let x2 = ops::add::add(x, &proj_out, device)?;
        if dbg {
            let d = x2.to_f32_vec();
            let xd = x.to_f32_vec();
            let pd = proj_out.to_f32_vec();
            // Verify add: x2[last][0] should equal x[last][0] + proj_out[last][0]
            let last_s = (seq_len - 1) * self.dim;
            eprintln!("  [L{}] x2={:.4} x[last][0]={:.4} proj[last][0]={:.4} sum={:.4} actual={:.4}",
                layer_idx, norm(&x2), xd[last_s], pd[last_s], xd[last_s]+pd[last_s], d[last_s]);
        }

        // === FFN sub-layer ===
        let h2 = ops::rmsnorm::rmsnorm(&x2, &self.ffn_norm_gamma, device)?;
        let gate = self.w_gate.forward(&h2)?;
        let up = self.w_up.forward(&h2)?;
        if dbg {
            // CPU RMSNorm reference for h2
            let x2d = x2.to_f32_vec();
            let gamma = self.ffn_norm_gamma.to_f32_vec();
            let dim = self.dim;
            let mut cpu_h2 = vec![0f32; seq_len * dim];
            for t in 0..seq_len {
                let mut sum_sq = 0.0f32;
                for i in 0..dim {
                    sum_sq += x2d[t * dim + i] * x2d[t * dim + i];
                }
                let rms = (sum_sq / dim as f32 + 1e-6f32).sqrt();
                let inv_rms = 1.0 / rms;
                for i in 0..dim {
                    cpu_h2[t * dim + i] = x2d[t * dim + i] * inv_rms * gamma[i];
                }
            }
            let cpu_h2_norm: f32 = cpu_h2.iter().map(|x| x*x).sum::<f32>().sqrt();
            let gpu_h2_norm = norm(&h2);
            let h2d = h2.to_f32_vec();
            let max_h2_diff: f32 = cpu_h2.iter().zip(h2d.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
            eprintln!("  [L{}] h2: CPU={:.4} GPU={:.4} max_diff={:.6}", layer_idx, cpu_h2_norm, gpu_h2_norm, max_h2_diff);
            // CPU reference for gate (all tokens)
            let h2d = h2.to_f32_vec();
            let gw = self.w_gate.weight.to_f32_vec();
            let ffn_dim = self.ffn_dim;
            let dim = self.dim;
            let mut cpu_gate_all = vec![0f32; seq_len * ffn_dim];
            for t in 0..seq_len {
                for j in 0..ffn_dim {
                    let mut sum = 0.0f32;
                    for i in 0..dim {
                        sum += h2d[t * dim + i] * gw[i * ffn_dim + j];
                    }
                    cpu_gate_all[t * ffn_dim + j] = sum;
                }
            }
            let cpu_gate_total: f32 = cpu_gate_all.iter().map(|x| x*x).sum::<f32>().sqrt();
            let cpu_gate_last: f32 = {
                let s = (seq_len-1)*ffn_dim;
                cpu_gate_all[s..s+ffn_dim].iter().map(|x| x*x).sum::<f32>().sqrt()
            };
            let gd = gate.to_f32_vec();
            let gpu_gate_last: f32 = {
                let s = (seq_len-1)*ffn_dim;
                gd[s..s+ffn_dim].iter().map(|x| x*x).sum::<f32>().sqrt()
            };
            // CPU reference for up
            let uw = self.w_up.weight.to_f32_vec();
            let mut cpu_up_all = vec![0f32; seq_len * ffn_dim];
            for t in 0..seq_len {
                for j in 0..ffn_dim {
                    let mut sum = 0.0f32;
                    for i in 0..dim {
                        sum += h2d[t * dim + i] * uw[i * ffn_dim + j];
                    }
                    cpu_up_all[t * ffn_dim + j] = sum;
                }
            }
            let cpu_up_total: f32 = cpu_up_all.iter().map(|x| x*x).sum::<f32>().sqrt();
            eprintln!("  [L{}] gate: CPU={:.4} GPU={:.4}  up: CPU={:.4} GPU={:.4}",
                layer_idx, cpu_gate_total, norm(&gate), cpu_up_total, norm(&up));
            // Weight norms
            let gw_norm: f32 = gw.iter().map(|x| x*x).sum::<f32>().sqrt();
            let uw_norm: f32 = uw.iter().map(|x| x*x).sum::<f32>().sqrt();
            let wd = self.w_down.weight.to_f32_vec();
            let wd_norm: f32 = wd.iter().map(|x| x*x).sum::<f32>().sqrt();
            eprintln!("  [L{}] w_gate={:.4} w_up={:.4} w_down={:.4} h2={:.4}", layer_idx, gw_norm, uw_norm, wd_norm, norm(&h2));
        }
        let silu_out = ops::silu::silu_gate(&gate, &up, device)?;

        if dbg {
            // CPU SiLU reference
            let gd = gate.to_f32_vec();
            let ud = up.to_f32_vec();
            let n_elems = gd.len();
            let mut cpu_silu = vec![0f32; n_elems];
            let mut cpu_silu_gate = vec![0f32; n_elems];
            for i in 0..n_elems {
                let sig = 1.0 / (1.0 + (-gd[i]).exp());
                cpu_silu_gate[i] = gd[i] * sig;
                cpu_silu[i] = gd[i] * sig * ud[i];
            }
            let cpu_silu_gate_norm: f32 = cpu_silu_gate.iter().map(|x| x*x).sum::<f32>().sqrt();
            let cpu_silu_norm: f32 = cpu_silu.iter().map(|x| x*x).sum::<f32>().sqrt();
            let sd = silu_out.to_f32_vec();
            let last = (seq_len - 1) * self.ffn_dim;
            let cpu_last_norm: f32 = cpu_silu[last..last+self.ffn_dim].iter().map(|x| x*x).sum::<f32>().sqrt();
            let gpu_last_norm: f32 = sd[last..last+self.ffn_dim].iter().map(|x| x*x).sum::<f32>().sqrt();
            // Compare CPU silu with GPU silu
            let sd = silu_out.to_f32_vec();
            let max_diff: f32 = cpu_silu.iter().zip(sd.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
            let mean_diff: f32 = cpu_silu.iter().zip(sd.iter()).map(|(a, b)| (a - b).abs()).sum::<f32>() / n_elems as f32;
            eprintln!("  [L{}] silu_gate={:.4} silu_out: CPU={:.4} GPU={:.4} max_diff={:.6} mean_diff={:.6}",
                layer_idx, cpu_silu_gate_norm, cpu_silu_norm, norm(&silu_out), max_diff, mean_diff);
            eprintln!("  [L{}] silu last first3: CPU={:.6} {:.6} {:.6}  GPU={:.6} {:.6} {:.6}",
                layer_idx, cpu_silu[last], cpu_silu[last+1], cpu_silu[last+2],
                sd[last], sd[last+1], sd[last+2]);
        }
        let ffn_out = self.w_down.forward(&silu_out)?;

        if dbg {
            // CPU reference for ffn_out (last token only)
            let sd = silu_out.to_f32_vec();
            let wd = self.w_down.weight.to_f32_vec();
            let in_f = self.w_down.in_features;
            let out_f = self.w_down.out_features;
            let last_sd = (seq_len - 1) * in_f;
            let mut cpu_ffn = vec![0f32; out_f];
            for j in 0..out_f {
                let mut sum = 0.0f32;
                for i in 0..in_f {
                    sum += sd[last_sd + i] * wd[i * out_f + j];
                }
                cpu_ffn[j] = sum;
            }
            let cpu_ffn_norm: f32 = cpu_ffn.iter().map(|x| x*x).sum::<f32>().sqrt();
            let fd = ffn_out.to_f32_vec();
            let last_fd = (seq_len - 1) * out_f;
            let gpu_ffn_norm: f32 = fd[last_fd..last_fd+out_f].iter().map(|x| x*x).sum::<f32>().sqrt();
            eprintln!("  [L{}] ffn_out: CPU_last={:.4} GPU_last={:.4} total={:.4}",
                layer_idx, cpu_ffn_norm, gpu_ffn_norm, norm(&ffn_out));
        }

        let result = ops::add::add(&x2, &ffn_out, device)?;
        if dbg {
            // CPU reference for x2 + ffn_out
            let x2d = x2.to_f32_vec();
            let fd = ffn_out.to_f32_vec();
            let rd = result.to_f32_vec();
            let cpu_sum_norm: f32 = x2d.iter().zip(fd.iter()).map(|(a, b)| (a + b).powi(2)).sum::<f32>().sqrt();
            let gpu_norm: f32 = rd.iter().map(|x| x*x).sum::<f32>().sqrt();
            let last_s = (seq_len - 1) * self.dim;
            eprintln!("  [L{}] final: CPU_sum={:.4} GPU={:.4} x2[last][0]={:.4} ffn[last][0]={:.4} result[last][0]={:.4}",
                layer_idx, cpu_sum_norm, gpu_norm, x2d[last_s], fd[last_s], rd[last_s]);
        }
        Ok(result)
    }
}

#[cfg(feature = "rocm")]
impl Module for TransformerLayer {
    fn forward(&self, input: &Tensor) -> Result<Tensor, String> {
        self.forward_simple(input)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        vec![
            &self.wq.weight, &self.wk.weight, &self.wv.weight, &self.wo.weight,
            &self.w_gate.weight, &self.w_up.weight, &self.w_down.weight,
            &self.attn_norm_gamma, &self.ffn_norm_gamma,
            &self.q_norm_gamma, &self.k_norm_gamma,
        ]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        vec![
            &mut self.wq.weight, &mut self.wk.weight, &mut self.wv.weight, &mut self.wo.weight,
            &mut self.w_gate.weight, &mut self.w_up.weight, &mut self.w_down.weight,
            &mut self.attn_norm_gamma, &mut self.ffn_norm_gamma,
            &mut self.q_norm_gamma, &mut self.k_norm_gamma,
        ]
    }
}
