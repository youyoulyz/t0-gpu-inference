//! LanguageModel — Full model: Embedding → N×TransformerLayer → RMSNorm → LM Head.

#[cfg(feature = "rocm")]
use std::sync::Arc;
#[cfg(feature = "rocm")]
use super::Module;
#[cfg(feature = "rocm")]
use super::linear::Linear;
#[cfg(feature = "rocm")]
use super::embedding::Embedding;
#[cfg(feature = "rocm")]
use super::transformer::TransformerLayer;
#[cfg(feature = "rocm")]
use super::config::Qwen3Config;
#[cfg(feature = "rocm")]
use super::super::tensor::Tensor;
#[cfg(feature = "rocm")]
use super::super::gpu_context::GpuRuntime;
#[cfg(feature = "rocm")]
use super::super::ops;

/// Complete language model.
#[cfg(feature = "rocm")]
pub struct LanguageModel {
    pub embedding: Embedding,
    pub layers: Vec<TransformerLayer>,
    pub final_norm_gamma: Tensor,
    pub lm_head: Linear,
    pub config: Qwen3Config,
    runtime: Arc<GpuRuntime>,
}

#[cfg(feature = "rocm")]
impl LanguageModel {
    /// Create from Qwen3Config (preferred).
    pub fn from_config(
        runtime: &Arc<GpuRuntime>,
        config: &Qwen3Config,
    ) -> Result<Self, String> {
        config.validate()?;

        let embedding = Embedding::new(runtime, config.vocab_size, config.hidden_size, "emb")?;

        let mut layers = Vec::new();
        for i in 0..config.num_layers {
            layers.push(TransformerLayer::from_config(runtime, config, i)?);
        }

        let mut final_norm_gamma = Tensor::from_f32(
            runtime, &vec![1.0f32; config.hidden_size], &[config.hidden_size], "final_norm",
        )?;
        final_norm_gamma.set_requires_grad(true);

        // LM head: always [hidden_size, vocab_size]
        // If tie_word_embeddings, lm_head.weight will be replaced by embedding.weight
        // during weight loading (they share the same tensor).
        let lm_head = Linear::new(runtime, config.hidden_size, config.vocab_size, "lm_head")?;

        Ok(Self {
            embedding, layers, final_norm_gamma, lm_head,
            config: config.clone(),
            runtime: runtime.clone(),
        })
    }

    /// Legacy constructor (no GQA, no config).
    pub fn new(
        runtime: &Arc<GpuRuntime>,
        vocab_size: usize,
        dim: usize,
        n_layers: usize,
        n_heads: usize,
        ffn_mult: usize,
    ) -> Result<Self, String> {
        let config = Qwen3Config {
            hidden_size: dim,
            num_layers: n_layers,
            num_attention_heads: n_heads,
            num_key_value_heads: n_heads,
            intermediate_size: dim * ffn_mult,
            vocab_size,
            max_position_embeddings: 40960,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            tie_word_embeddings: false,
        };
        Self::from_config(runtime, &config)
    }

    /// Forward pass: token_ids → logits
    pub fn forward_ids(&self, ids: &[u32]) -> Result<Tensor, String> {
        let device = &self.runtime.device;

        // Embedding
        let mut h = self.embedding.forward_cpu(ids)?;

        // Transformer layers
        for layer in &self.layers {
            h = layer.forward(&h)?;
        }

        // Final RMSNorm
        h = ops::rmsnorm::rmsnorm(&h, &self.final_norm_gamma, device)?;

        // LM head → logits
        self.lm_head.forward(&h)
    }

    /// Tie lm_head.weight to embedding.weight (weight sharing).
    ///
    /// Call after loading weights if config.tie_word_embeddings is true.
    pub fn tie_lm_head(&mut self) {
        self.lm_head.weight = self.embedding.weight.clone();
    }

    /// Get all parameters for optimizer.
    pub fn all_parameters(&self) -> Vec<&Tensor> {
        let mut params = vec![&self.embedding.weight];
        for layer in &self.layers {
            params.extend(layer.parameters());
        }
        params.push(&self.final_norm_gamma);
        // Only include lm_head if not tied (to avoid double-counting)
        if !self.config.tie_word_embeddings {
            params.push(&self.lm_head.weight);
        }
        params
    }

    /// Get all mutable parameters.
    pub fn all_parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut params: Vec<&mut Tensor> = vec![&mut self.embedding.weight];
        for layer in &mut self.layers {
            params.extend(layer.parameters_mut());
        }
        params.push(&mut self.final_norm_gamma);
        if !self.config.tie_word_embeddings {
            params.push(&mut self.lm_head.weight);
        }
        params
    }

    /// Total number of parameters.
    pub fn param_count(&self) -> usize {
        self.all_parameters().iter().map(|t| t.numel()).sum()
    }
}
