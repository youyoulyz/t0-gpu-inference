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
    wqkv_bf16: std::sync::OnceLock<Option<crate::kfd::GpuBuffer>>,
    wgu_bf16: std::sync::OnceLock<Option<crate::kfd::GpuBuffer>>,
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
            wqkv_bf16: std::sync::OnceLock::new(),
            wgu_bf16: std::sync::OnceLock::new(),
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
            wqkv_bf16: std::sync::OnceLock::new(),
            wgu_bf16: std::sync::OnceLock::new(),
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

    pub fn set_wqkv_bf16(&self, buf: crate::kfd::GpuBuffer) {
        let _ = self.wqkv_bf16.set(Some(buf));
    }

    pub fn set_wgu_bf16(&self, buf: crate::kfd::GpuBuffer) {
        let _ = self.wgu_bf16.set(Some(buf));
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
        let dbg = crate::t0_debug();

        let use_defer = false && seq_len == 1;
        if use_defer {
            self.runtime.begin_defer_sync();
            eprintln!("[L{}] defer_sync BEGIN", layer_idx);
        }

        // === Attention sub-layer ===
        // For decode, fuse rmsnorm + f32→bf16 conversion to save 1 dispatch
        let attn_pad_k = (self.dim as u32 + 15) / 16 * 16;

        // h_attn holds the rmsnorm output in f32 (for prefill/debug) or None (fused decode)
        let (q, k, v, _qkv_holder, h_attn) = if seq_len == 1 {
            if let Some(Some(wt)) = self.wqkv_bf16.get() {
                let n_fused = self.q_dim + 2 * self.kv_dim;
                // Fused: rmsnorm → bf16 (single dispatch, skips f32 output)
                let h_bf16 = ops::rmsnorm::rmsnorm_to_bf16(
                    x, &self.attn_norm_gamma, 1, self.dim,
                    attn_pad_k as usize, 16, &self.runtime,
                )?;
                let qkv = ops::bf16_matmul::matmul_with_bf16_x(
                    &h_bf16, wt, 1, self.dim, n_fused, &self.runtime,
                )?;
                let base = qkv.gpu_addr();
                let q = Tensor::from_gpu_addr(base, &self.runtime, &[1, self.q_dim], "q_fused");
                let k = Tensor::from_gpu_addr(
                    base + (self.q_dim * 4) as u64, &self.runtime, &[1, self.kv_dim], "k_fused",
                );
                let v = Tensor::from_gpu_addr(
                    base + ((self.q_dim + self.kv_dim) * 4) as u64,
                    &self.runtime, &[1, self.kv_dim], "v_fused",
                );
                (q, k, v, Some(qkv), None)
            } else {
                let h = ops::rmsnorm::rmsnorm(x, &self.attn_norm_gamma, device)?;
                (self.wq.forward(&h)?, self.wk.forward(&h)?, self.wv.forward(&h)?, None, Some(h))
            }
        } else {
            let h = ops::rmsnorm::rmsnorm(x, &self.attn_norm_gamma, device)?;
            (self.wq.forward(&h)?, self.wk.forward(&h)?, self.wv.forward(&h)?, None, Some(h))
        };

        if dbg && layer_idx == 0 {
            let qd = q.to_f32_vec();
            let kd = k.to_f32_vec();
            let vd = v.to_f32_vec();
            let last_q = (seq_len - 1) * self.q_dim;
            let last_k = (seq_len - 1) * self.kv_dim;
            let last_v = (seq_len - 1) * self.kv_dim;
            if let Some(ref h) = h_attn {
                let hd = h.to_f32_vec();
                let last_h = (seq_len - 1) * self.dim;
                eprintln!("  [L0] h_last[:8]: {:.6} {:.6} {:.6} {:.6} {:.6} {:.6} {:.6} {:.6}",
                    hd[last_h], hd[last_h+1], hd[last_h+2], hd[last_h+3],
                    hd[last_h+4], hd[last_h+5], hd[last_h+6], hd[last_h+7]);
            }
            eprintln!("  [L0] Q_last[:8]: {:.6} {:.6} {:.6} {:.6} {:.6} {:.6} {:.6} {:.6}",
                qd[last_q], qd[last_q+1], qd[last_q+2], qd[last_q+3],
                qd[last_q+4], qd[last_q+5], qd[last_q+6], qd[last_q+7]);
            eprintln!("  [L0] K_last[:8]: {:.6} {:.6} {:.6} {:.6} {:.6} {:.6} {:.6} {:.6}",
                kd[last_k], kd[last_k+1], kd[last_k+2], kd[last_k+3],
                kd[last_k+4], kd[last_k+5], kd[last_k+6], kd[last_k+7]);
            eprintln!("  [L0] V_last[:8]: {:.6} {:.6} {:.6} {:.6} {:.6} {:.6} {:.6} {:.6}",
                vd[last_v], vd[last_v+1], vd[last_v+2], vd[last_v+3],
                vd[last_v+4], vd[last_v+5], vd[last_v+6], vd[last_v+7]);
        }

        // QK-norm: per-head RMSNorm on Q and K
        let q = ops::qk_norm::qk_norm(&q, &self.q_norm_gamma, self.n_heads, self.d_head, &self.runtime)?;
        let k = ops::qk_norm::qk_norm(&k, &self.k_norm_gamma, self.n_kv_heads, self.d_head, &self.runtime)?;
        if dbg && layer_idx == 0 {
            let qn = q.to_f32_vec();
            let kn = k.to_f32_vec();
            let last_q = (seq_len - 1) * self.q_dim;
            let last_k = (seq_len - 1) * self.kv_dim;
            eprintln!("  [L0] Q_after_qknorm[:8]: {:.6} {:.6} {:.6} {:.6} {:.6} {:.6} {:.6} {:.6}",
                qn[last_q], qn[last_q+1], qn[last_q+2], qn[last_q+3],
                qn[last_q+4], qn[last_q+5], qn[last_q+6], qn[last_q+7]);
            eprintln!("  [L0] K_after_qknorm[:8]: {:.6} {:.6} {:.6} {:.6} {:.6} {:.6} {:.6} {:.6}",
                kn[last_k], kn[last_k+1], kn[last_k+2], kn[last_k+3],
                kn[last_k+4], kn[last_k+5], kn[last_k+6], kn[last_k+7]);
        }

        // RoPE: reshape to [seq*n_heads, head_dim], apply per-head, reshape back
        let q_2d = q.reshape(&[seq_len * self.n_heads, self.d_head]);
        let k_2d = k.reshape(&[seq_len * self.n_kv_heads, self.d_head]);
        let q_2d = ops::rope::rope_forward(&q_2d, pos, self.rope_theta, self.n_heads, &self.runtime)?;
        let k_2d = ops::rope::rope_forward(&k_2d, pos, self.rope_theta, self.n_kv_heads, &self.runtime)?;
        let q = q_2d.reshape(&[seq_len, self.q_dim]);
        let k = k_2d.reshape(&[seq_len, self.kv_dim]);
        if dbg && layer_idx == 0 {
            let qd = q.to_f32_vec();
            let kd = k.to_f32_vec();
            let last_q = (seq_len - 1) * self.q_dim;
            let last_k = (seq_len - 1) * self.kv_dim;
            eprintln!("  [L0] Q_after_rope[:8]: {:.6} {:.6} {:.6} {:.6} {:.6} {:.6} {:.6} {:.6}",
                qd[last_q], qd[last_q+1], qd[last_q+2], qd[last_q+3],
                qd[last_q+4], qd[last_q+5], qd[last_q+6], qd[last_q+7]);
            eprintln!("  [L0] K_after_rope[:8]: {:.6} {:.6} {:.6} {:.6} {:.6} {:.6} {:.6} {:.6}",
                kd[last_k], kd[last_k+1], kd[last_k+2], kd[last_k+3],
                kd[last_k+4], kd[last_k+5], kd[last_k+6], kd[last_k+7]);
        }

        // Store K/V in cache
        let kv_heads = self.n_kv_heads;
        let hd = self.d_head;
        let k_3d = k.reshape(&[seq_len, kv_heads, hd]);
        let v_3d = v.reshape(&[seq_len, kv_heads, hd]);

        let write_pos = kv_cache.position();
        if dbg && layer_idx == 0 {
            eprintln!("  [L{}] kv_pos={} write_pos={} seq_len={} kv_len={}",
                layer_idx, kv_cache.position(), write_pos, seq_len, write_pos + seq_len);
        }
        if seq_len == 1 {
            kv_cache.append_at_pos(&self.runtime, layer_idx, write_pos, &k_3d, &v_3d)?;
        } else {
            kv_cache.append_many(&self.runtime, layer_idx, &k_3d, &v_3d)?;
        }

        let kv_len = write_pos + seq_len;
        let k_addr = kv_cache.buf_gpu_addr() + kv_cache.k_offset(layer_idx, 0) as u64;
        let v_addr = kv_cache.buf_gpu_addr() + kv_cache.v_offset(layer_idx, 0) as u64;

        let k_cache = Tensor::from_gpu_addr(k_addr, &self.runtime, &[kv_len, self.kv_dim], "k_cache");
        let v_cache = Tensor::from_gpu_addr(v_addr, &self.runtime, &[kv_len, self.kv_dim], "v_cache");

        // Attention — flash for decode, flash prefill for prefill
        let attn_out = if seq_len == 1 {
            ops::attention::flash_attention_decode(
                &q, &k_cache, &v_cache,
                self.n_heads, self.n_kv_heads, self.d_head,
                &self.runtime,
            )?
        } else {
            ops::attention::flash_attention_prefill(
                &q, &k_cache, &v_cache,
                self.n_heads, self.n_kv_heads, self.d_head,
                write_pos,
                &self.runtime,
            )?
        };

        // Output projection
        let proj_out = self.wo.forward(&attn_out)?;
        let x2 = ops::add::add(x, &proj_out, device)?;

        // === FFN sub-layer ===
        let ffn_pad_k = (self.dim as u32 + 15) / 16 * 16;

        let (gate, up, _gu_holder) = if seq_len == 1 {
            if let Some(Some(wt)) = self.wgu_bf16.get() {
                let n_fused = 2 * self.ffn_dim;
                // Fused: rmsnorm → bf16 (single dispatch, skips f32 output)
                let h2_bf16 = ops::rmsnorm::rmsnorm_to_bf16(
                    &x2, &self.ffn_norm_gamma, 1, self.dim,
                    ffn_pad_k as usize, 16, &self.runtime,
                )?;
                let gu = ops::bf16_matmul::matmul_with_bf16_x(
                    &h2_bf16, wt, 1, self.dim, n_fused, &self.runtime,
                )?;
                let base = gu.gpu_addr();
                let gate = Tensor::from_gpu_addr(base, &self.runtime, &[1, self.ffn_dim], "gate_fused");
                let up = Tensor::from_gpu_addr(
                    base + (self.ffn_dim * 4) as u64, &self.runtime, &[1, self.ffn_dim], "up_fused",
                );
                (gate, up, Some(gu))
            } else {
                let h2 = ops::rmsnorm::rmsnorm(&x2, &self.ffn_norm_gamma, device)?;
                (self.w_gate.forward(&h2)?, self.w_up.forward(&h2)?, None)
            }
        } else {
            let h2 = ops::rmsnorm::rmsnorm(&x2, &self.ffn_norm_gamma, device)?;
            (self.w_gate.forward(&h2)?, self.w_up.forward(&h2)?, None)
        };

        let silu_out = ops::silu::silu_gate(&gate, &up, device)?;
        let ffn_out = self.w_down.forward(&silu_out)?;

        let result = ops::add::add(&x2, &ffn_out, device)?;
        if use_defer {
            eprintln!("[L{}] defer_sync END (waiting...)", layer_idx);
            self.runtime.end_defer_sync()?;
            eprintln!("[L{}] defer_sync DONE", layer_idx);
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
