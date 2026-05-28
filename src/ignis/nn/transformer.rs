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
///   wq: [hidden_size, hidden_size]           — Q projection (all heads)
///   wk: [hidden_size, kv_dim]                — K projection (kv_heads only)
///   wv: [hidden_size, kv_dim]                — V projection (kv_heads only)
///   wo: [hidden_size, hidden_size]           — output projection
///   w_gate: [hidden_size, intermediate_size] — FFN gate
///   w_up: [hidden_size, intermediate_size]   — FFN up
///   w_down: [intermediate_size, hidden_size] — FFN down
///   attn_norm_gamma: [hidden_size]
///   ffn_norm_gamma: [hidden_size]
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
    pub dim: usize,
    pub d_head: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub kv_dim: usize,
    pub ffn_dim: usize,
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
        let kv_dim = config.kv_dim();
        let ffn_dim = config.intermediate_size;
        let prefix = format!("L{}", layer_idx);

        Ok(Self {
            // Q projection: [hidden_size, hidden_size]
            wq: Linear::new(runtime, dim, dim, &format!("{}_wq", prefix))?,
            // K projection: [hidden_size, kv_dim]  (GQA: kv_dim < dim when kv_heads < heads)
            wk: Linear::new(runtime, dim, kv_dim, &format!("{}_wk", prefix))?,
            // V projection: [hidden_size, kv_dim]
            wv: Linear::new(runtime, dim, kv_dim, &format!("{}_wv", prefix))?,
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
            dim,
            d_head,
            n_heads,
            n_kv_heads,
            kv_dim,
            ffn_dim,
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
            dim,
            d_head,
            n_heads,
            n_kv_heads: n_heads, // no GQA: kv_heads == heads
            kv_dim: dim,
            ffn_dim,
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
        ]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        vec![
            &mut self.wq.weight, &mut self.wk.weight, &mut self.wv.weight, &mut self.wo.weight,
            &mut self.w_gate.weight, &mut self.w_up.weight, &mut self.w_down.weight,
            &mut self.attn_norm_gamma, &mut self.ffn_norm_gamma,
        ]
    }
}
