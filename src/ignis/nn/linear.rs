//! Linear layer — f32 master weight + bf16 WMMA GEMM path.
//!
//! Stores weights in f32, converts to bf16 for WMMA matmul on the fly.
//! The bf16 cache in GpuRuntime avoids re-conversion every forward.

#[cfg(feature = "rocm")]
use std::sync::Arc;
#[cfg(feature = "rocm")]
use super::Module;
#[cfg(feature = "rocm")]
use super::super::tensor::Tensor;
#[cfg(feature = "rocm")]
use super::super::gpu_context::GpuRuntime;

/// Linear layer: Y = X @ W (no bias)
///
/// Weights: [in_features, out_features] f32
#[cfg(feature = "rocm")]
pub struct Linear {
    pub weight: Tensor,
    pub in_features: usize,
    pub out_features: usize,
    runtime: Arc<GpuRuntime>,
    /// Cached bf16 transposed weight [out_features, in_features] for GEMM
    cached_wt_bf16: std::sync::OnceLock<Option<crate::kfd::GpuBuffer>>,
}

#[cfg(feature = "rocm")]
impl Linear {
    /// Create with random initialization (scaled normal).
    pub fn new(
        runtime: &Arc<GpuRuntime>,
        in_features: usize,
        out_features: usize,
        name: &str,
    ) -> Result<Self, String> {
        // Xavier/He initialization: scale = sqrt(2 / fan_in)
        let scale = (2.0 / in_features as f64).sqrt() as f32;
        let n = in_features * out_features;
        let mut rng_state = 42u64; // simple LCG
        let data: Vec<f32> = (0..n).map(|_| {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u = (rng_state >> 33) as f32 / (1u64 << 31) as f32 - 1.0;
            u * scale
        }).collect();

        let mut weight = Tensor::from_f32(
            runtime, &data, &[in_features, out_features],
            &format!("{}_weight", name),
        )?;
        weight.set_requires_grad(true);

        Ok(Self {
            weight,
            in_features,
            out_features,
            runtime: runtime.clone(),
            cached_wt_bf16: std::sync::OnceLock::new(),
        })
    }

    /// Create from existing weight tensor.
    pub fn from_weight(weight: Tensor, runtime: &Arc<GpuRuntime>) -> Self {
        let shape = weight.shape().to_vec();
        assert_eq!(shape.len(), 2);
        Self {
            in_features: shape[0],
            out_features: shape[1],
            weight,
            runtime: runtime.clone(),
            cached_wt_bf16: std::sync::OnceLock::new(),
        }
    }

    /// Pre-set the cached bf16 transposed weight (for fast loading).
    /// Use this to avoid the f32→bf16 conversion on first forward pass.
    pub fn set_cached_wt_bf16(&self, wt_bf16: crate::kfd::GpuBuffer) {
        let _ = self.cached_wt_bf16.set(Some(wt_bf16));
    }
}

#[cfg(feature = "rocm")]
impl Module for Linear {
    fn forward(&self, input: &Tensor) -> Result<Tensor, String> {
        // Y = input @ weight using WMMA GEMM with cached bf16 weight
        let m = input.shape()[0];
        let k = self.in_features;
        let n = self.out_features;

        // CPU f32 GEMM path (for debugging precision issues)
        // Set T0_F32_GEMM=1 to enable
        if std::env::var("T0_F32_GEMM").ok().as_deref() == Some("1") {
            let x_data = input.to_f32_vec();
            let w_raw = self.weight.to_f32_vec();
            // Weight is stored as [in_features, out_features] = [K, N]
            // CPU GEMM expects [K, N] layout: w[kk * N + j]
            // If weight shape matches [K, N], use directly; otherwise transpose
            let w_shape = self.weight.shape();
            let w_data = if w_shape.len() == 2 && w_shape[0] == k && w_shape[1] == n {
                w_raw
            } else {
                // Transpose [N, K] → [K, N]
                let mut transposed = vec![0.0f32; k * n];
                for i in 0..k {
                    for j in 0..n {
                        transposed[i * n + j] = w_raw[j * k + i];
                    }
                }
                transposed
            };
            let y_data = super::super::ops::bf16_matmul::gemm_f32_cpu(&x_data, &w_data, m, k, n);
            return super::super::tensor::Tensor::from_f32(&self.runtime, &y_data, &[m, n], "cpu_gemm_out");
        }

        // Get or compute cached bf16 transposed weight
        let wt_bf16 = self.cached_wt_bf16.get_or_init(|| {
            // Convert weight f32 → bf16 and transpose to [N, K]
            let wt = super::super::ops::bf16_matmul::precompute_wt_bf16(
                &self.runtime, self.weight.buffer(), k, n,
            );
            Some(wt.ok()).flatten()
        });

        if let Some(wt) = wt_bf16 {
            super::super::ops::bf16_matmul::matmul_with_wt_bf16(input, wt, m, k, n, &self.runtime)
        } else {
            super::super::ops::bf16_matmul::matmul(input, &self.weight, &self.runtime.device)
        }
    }

    fn parameters(&self) -> Vec<&Tensor> {
        vec![&self.weight]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        vec![&mut self.weight]
    }
}
