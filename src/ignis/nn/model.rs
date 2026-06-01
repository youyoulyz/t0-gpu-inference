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
            head_dim: dim / n_heads,
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

    /// Prefill: process all prompt tokens at once, filling the KV cache.
    ///
    /// Returns logits for the last token position: [vocab_size] f32.
    pub fn forward_prefill(
        &mut self,
        ids: &[u32],
        kv_cache: &mut super::super::kv_cache::KvCache,
    ) -> Result<Tensor, String> {
        let device = &self.runtime.device;
        let seq_len = ids.len();

        // Embed all tokens at once via CPU (reads weight table once)
        let mut h = self.embedding.forward_cpu(ids)?;

        // Run through all transformer layers with KV cache
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            h = layer.forward_inference(&h, 0, layer_idx, kv_cache)?;
        }

        // Advance KV cache position once after all layers have written
        kv_cache.advance_by(seq_len);

        // Final RMSNorm
        h = ops::rmsnorm::rmsnorm(&h, &self.final_norm_gamma, device)?;

        // Take last token's hidden state: [1, hidden_size]
        let hidden = self.config.hidden_size;
        let last_hidden_data = {
            let full = h.to_f32_vec();
            let start = (seq_len - 1) * hidden;
            full[start..start + hidden].to_vec()
        };
        let last_hidden = Tensor::from_f32(&self.runtime, &last_hidden_data, &[1, hidden], "last_hidden")?;

        // LM head → logits [1, vocab_size]
        self.lm_head.forward(&last_hidden)
    }

    /// Decode: process a single new token with KV cache, return logits.
    ///
    /// Returns logits: [vocab_size] f32.
    pub fn forward_decode(
        &mut self,
        token_id: u32,
        pos: usize,
        kv_cache: &mut super::super::kv_cache::KvCache,
    ) -> Result<Tensor, String> {
        let device = &self.runtime.device;

        // Embed single token via CPU
        let mut h = self.embedding.forward_cpu(&[token_id])?;

        // Run through all transformer layers
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            h = layer.forward_inference(&h, pos, layer_idx, kv_cache)?;
        }

        // Advance KV cache position once after all layers have written
        kv_cache.advance();

        // Final RMSNorm
        h = ops::rmsnorm::rmsnorm(&h, &self.final_norm_gamma, device)?;

        // LM head → logits [1, vocab_size]
        self.lm_head.forward(&h)
    }

    /// Generate text autoregressively.
    ///
    /// # Arguments
    /// - `prompt_ids`: tokenized prompt
    /// - `max_tokens`: maximum new tokens to generate
    /// - `temperature`: sampling temperature (<=0 for greedy)
    /// - `top_p`: nucleus sampling threshold
    /// - `eos_id`: end-of-sequence token ID
    /// - `kv_cache`: pre-allocated KV cache
    ///
    /// # Returns
    /// Generated token IDs (excluding prompt).
    pub fn generate(
        &mut self,
        prompt_ids: &[u32],
        max_tokens: usize,
        temperature: f32,
        top_p: f32,
        eos_id: u32,
        kv_cache: &mut super::super::kv_cache::KvCache,
    ) -> Result<Vec<u32>, String> {
        kv_cache.reset();

        // Prefill phase: process entire prompt
        eprintln!("[Generate] Prefilling {} prompt tokens...", prompt_ids.len());
        let t_prefill = std::time::Instant::now();
        let logits = self.forward_prefill(prompt_ids, kv_cache)?;
        eprintln!("[Generate] Prefill done in {:.1}s", t_prefill.elapsed().as_secs_f64());
        let mut generated = Vec::new();

        // Sample first token from prefill logits
        let next_token = ops::argmax::sample_token(&logits, temperature, top_p, &self.runtime)?;
        generated.push(next_token);

        if next_token == eos_id {
            return Ok(generated);
        }

        // Decode phase: generate tokens one at a time
        let start_pos = prompt_ids.len();
        for step in 0..max_tokens - 1 {
            let t0 = std::time::Instant::now();
            let pos = start_pos + step;
            let logits = self.forward_decode(next_token, pos, kv_cache)?;
            let decode_ms = t0.elapsed().as_millis();

            // Debug: show top token and logit stats
            let logits_data = logits.to_f32_vec();
            let max_logit = logits_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let max_idx = logits_data.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
            let mean_logit: f32 = logits_data.iter().sum::<f32>() / logits_data.len() as f32;

            let next_token = ops::argmax::sample_token(&logits, temperature, top_p, &self.runtime)?;
            generated.push(next_token);

            if next_token == eos_id {
                eprintln!("[Generate] EOS at step {}", step + 1);
                break;
            }

            eprint!("[tok {}] {}ms id={} max_id={} max={:.2} mean={:.2}  ",
                step + 1, decode_ms, next_token, max_idx, max_logit, mean_logit);
            if (step + 1) % 5 == 0 { eprintln!(); }
        }
        eprintln!("[Generate] Done. {} tokens generated.", generated.len());

        Ok(generated)
    }
}

/// Upload u32 slice to GPU buffer.
fn upload_u32(runtime: &Arc<GpuRuntime>, data: &[u32]) -> Result<crate::kfd::GpuBuffer, String> {
    let bytes = data.len() * 4;
    let buf = runtime.device.alloc_vram(bytes)?;
    let byte_slice = unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, bytes)
    };
    buf.write_bytes(0, byte_slice);
    Ok(buf)
}
