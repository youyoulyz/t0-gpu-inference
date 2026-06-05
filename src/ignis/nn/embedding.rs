//! Embedding layer — token ID → dense vector lookup.

#[cfg(feature = "rocm")]
use std::sync::Arc;
#[cfg(feature = "rocm")]
use crate::kfd::GpuBuffer;
#[cfg(feature = "rocm")]
use super::Module;
#[cfg(feature = "rocm")]
use super::super::tensor::Tensor;
#[cfg(feature = "rocm")]
use super::super::gpu_context::GpuRuntime;
#[cfg(feature = "rocm")]
use super::super::ops;

/// Embedding layer: lookup table [vocab_size, dim]
#[cfg(feature = "rocm")]
pub struct Embedding {
    pub weight: Tensor,
    pub vocab_size: usize,
    pub dim: usize,
    runtime: Arc<GpuRuntime>,
    /// CPU-side cached weight table (lazily populated on first forward_cpu call)
    cached_table: std::sync::RwLock<Option<std::sync::Arc<Vec<f32>>>>,
}

#[cfg(feature = "rocm")]
impl Embedding {
    pub fn new(
        runtime: &Arc<GpuRuntime>,
        vocab_size: usize,
        dim: usize,
        name: &str,
    ) -> Result<Self, String> {
        // Normal initialization scaled by 1/sqrt(dim)
        let scale = 1.0 / (dim as f64).sqrt() as f32;
        let n = vocab_size * dim;
        let mut rng_state = 123u64;
        let data: Vec<f32> = (0..n).map(|_| {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let u = (rng_state >> 33) as f32 / (1u64 << 31) as f32 - 1.0;
            u * scale
        }).collect();

        let mut weight = Tensor::from_f32(
            runtime, &data, &[vocab_size, dim],
            &format!("{}_table", name),
        )?;
        weight.set_requires_grad(true);

        Ok(Self { weight, vocab_size, dim, runtime: runtime.clone(), cached_table: std::sync::RwLock::new(None) })
    }

    /// Forward: gather rows by token IDs.
    ///
    /// ids: GpuBuffer containing [seq_len] u32 token indices
    pub fn forward_ids(
        &self,
        ids: &GpuBuffer,
        seq_len: usize,
    ) -> Result<Tensor, String> {
        ops::embedding::embedding_forward(
            &self.weight, ids, seq_len, self.dim, &self.runtime,
        )
    }

    /// GPU-based forward: gather rows by token IDs on GPU.
    /// For decode (1 token), uses BF16→F32 gather kernel (1 dispatch).
    /// For prefill (>1 tokens), falls back to CPU (embedding gather kernel
    /// has SSA compile issue with scalar program_id indices).
    pub fn forward_gpu(&self, ids: &[u32]) -> Result<Tensor, String> {
        if ids.len() == 1 {
            let token_id = ids[0];
            let out_buf = self.runtime.alloc_f32(self.dim)?;

            let kernel = self.runtime.ensure_kernel_blockdsl("emb_gather_bf16", || {
                crate::t0::embedding_kernels::build_embedding_gather_bf16()
            })?;

            let ka = crate::kernargs![
                self.weight.gpu_addr() => u64,
                out_buf.gpu_addr() => u64,
                token_id => u32,
                self.dim as u32 => u32
            ];
            self.runtime.dispatch(&kernel, [256, 1, 1], &ka)?;

            let out_arc = std::sync::Arc::new(out_buf);
            Ok(Tensor::from_buffer(out_arc, &self.runtime, &[1, self.dim], super::super::tensor::DType::F32, "emb_out"))
        } else {
            self.forward_cpu(ids)
        }
    }

    /// Simple CPU-based forward for testing.
    /// Records backward on tape so embedding gets gradients.
    pub fn forward_cpu(&self, ids: &[u32]) -> Result<Tensor, String> {
        use super::super::tape::Tape;
        use super::super::tensor::DType;

        // Use cached weight table if available, otherwise read from GPU and cache
        let table = {
            let cached = self.cached_table.read().unwrap();
            if let Some(ref t) = *cached {
                t.clone() // Arc clone — only increments refcount, no data copy
            } else {
                drop(cached);
                let t = std::sync::Arc::new(self.weight.to_f32_vec());
                let mut cached = self.cached_table.write().unwrap();
                cached.get_or_insert(t.clone());
                t
            }
        };
        let seq_len = ids.len();
        let mut out = vec![0f32; seq_len * self.dim];
        for (i, &id) in ids.iter().enumerate() {
            let src = &table[(id as usize) * self.dim..(id as usize + 1) * self.dim];
            out[i * self.dim..(i + 1) * self.dim].copy_from_slice(src);
        }
        let mut result = Tensor::from_f32(&self.runtime, &out, &[seq_len, self.dim], "emb_out")?;

        if self.weight.requires_grad() {
            result.set_requires_grad(true);
        }

        // Record backward on tape for gradient flow
        if Tape::is_recording() && self.weight.requires_grad() {
            let weight_id = Some(self.weight.id());
            let ids_owned = ids.to_vec();
            let dim = self.dim;
            let vocab_size = self.vocab_size;
            let weight_buf_saved = self.weight.buffer_arc().clone();

            let node_id = Tape::record(
                "embedding",
                result.id(),
                vec![weight_id],
                vec![true],
                vec![weight_buf_saved],
                Box::new(move |grad_output, _saved, runtime| {
                    // Backward: scatter-add dY into dW
                    // dW[token_id, :] += dY[position, :]
                    let seq = ids_owned.len();
                    let mut dy_data = vec![0f32; seq * dim];
                    grad_output.read(unsafe {
                        std::slice::from_raw_parts_mut(
                            dy_data.as_mut_ptr() as *mut u8, seq * dim * 4
                        )
                    });

                    let mut dw = vec![0f32; vocab_size * dim];
                    for (i, &id) in ids_owned.iter().enumerate() {
                        let id = id as usize;
                        for d in 0..dim {
                            dw[id * dim + d] += dy_data[i * dim + d];
                        }
                    }

                    let dw_buf = runtime.alloc_f32(vocab_size * dim)?;
                    runtime.write_f32(&dw_buf, &dw);
                    Ok(vec![Some(Arc::new(dw_buf))])
                }),
            );
            result.set_tape_node(node_id);
        }

        Ok(result)
    }
}

#[cfg(feature = "rocm")]
impl Module for Embedding {
    fn forward(&self, _input: &Tensor) -> Result<Tensor, String> {
        Err("Embedding.forward() requires IDs — use forward_ids() or forward_cpu()".to_string())
    }

    fn parameters(&self) -> Vec<&Tensor> {
        vec![&self.weight]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        vec![&mut self.weight]
    }
}
