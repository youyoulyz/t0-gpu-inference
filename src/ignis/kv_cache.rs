//! KV Cache — pre-allocated key-value cache for autoregressive LLM inference.
//!
//! Layout: single contiguous VRAM buffer `[num_layers, 2, max_seq_len, num_kv_heads, head_dim]`
//! stored as f32. Index 0 = K cache, index 1 = V cache per layer.
//!
//! ## Operations
//! - **append**: copy K/V tensors into cache at current sequence position
//! - **get**: return GPU address + shape for K/V slice used in attention
//! - **reset**: set current sequence position back to 0
//!
//! ## Prefill vs Decode
//! - **Prefill**: `append_many()` copies entire prompt K/V in one shot, advancing position by `seq_len`
//! - **Decode**: `append()` copies single-token K/V, advancing position by 1
//!
//! ## Memory layout (flat f32 buffer)
//! ```text
//! Layer 0:
//!   K[0..max_seq, head0..headN]  (contiguous)
//!   V[0..max_seq, head0..headN]  (contiguous)
//! Layer 1:
//!   K[0..max_seq, head0..headN]
//!   V[0..max_seq, head0..headN]
//! ...
//! ```
//!
//! Element offset for layer `l`, kv_index `kv` (0=K, 1=V), seq `s`, head `h`:
//! ```text
//! offset = ((l * 2 + kv) * max_seq + s) * (num_kv_heads * head_dim) + h * head_dim
//! ```

#[cfg(feature = "rocm")]
use std::sync::Arc;
#[cfg(feature = "rocm")]
use std::sync::atomic::{AtomicU32, Ordering};

#[cfg(feature = "rocm")]
use crate::kfd::GpuBuffer;
#[cfg(feature = "rocm")]
use super::gpu_context::GpuRuntime;
#[cfg(feature = "rocm")]
use super::tensor::Tensor;

/// Pre-allocated KV cache for autoregressive decoding.
///
/// Thread-safe: uses atomic for position tracking. Suitable for
/// single-threaded inference (the common case for LLM serving).
#[cfg(feature = "rocm")]
pub struct KvCache {
    /// Single contiguous VRAM buffer for all layers, K+V
    buf: GpuBuffer,
    /// Configuration
    config: KvCacheConfig,
    /// Current sequence position (atomic for interior mutability)
    position: AtomicU32,
}

#[cfg(feature = "rocm")]
unsafe impl Send for KvCache {}
#[cfg(feature = "rocm")]
unsafe impl Sync for KvCache {}

/// KV cache configuration.
#[derive(Clone, Debug)]
pub struct KvCacheConfig {
    /// Number of transformer layers
    pub num_layers: usize,
    /// Number of KV heads (for GQA/MQA, this < num_q_heads)
    pub num_kv_heads: usize,
    /// Head dimension (usually 64, 128, or 256)
    pub head_dim: usize,
    /// Maximum sequence length the cache can hold
    pub max_seq_len: usize,
}

impl KvCacheConfig {
    /// Total number of f32 elements in the cache.
    pub fn total_elements(&self) -> usize {
        self.num_layers * 2 * self.max_seq_len * self.num_kv_heads * self.head_dim
    }

    /// Total bytes (f32 = 4 bytes per element).
    pub fn total_bytes(&self) -> usize {
        self.total_elements() * 4
    }

    /// Bytes per layer (K + V for all positions).
    pub fn bytes_per_layer(&self) -> usize {
        2 * self.max_seq_len * self.num_kv_heads * self.head_dim * 4
    }

    /// Bytes per token (K + V for all heads at one position).
    pub fn bytes_per_token(&self) -> usize {
        2 * self.num_kv_heads * self.head_dim * 4
    }

    /// Bytes per head per token.
    pub fn bytes_per_head_token(&self) -> usize {
        self.head_dim * 4
    }
}

/// GPU address + shape info for a KV slice returned by `get()`.
#[derive(Clone, Copy, Debug)]
pub struct KvSlice {
    /// GPU virtual address of the slice
    pub gpu_addr: u64,
    /// Shape: [seq_len, num_kv_heads, head_dim]
    pub seq_len: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
}

#[cfg(feature = "rocm")]
impl KvCache {
    /// Create a new KV cache with pre-allocated VRAM.
    ///
    /// # Arguments
    /// - `runtime`: GPU runtime for VRAM allocation
    /// - `config`: cache dimensions
    pub fn new(runtime: &Arc<GpuRuntime>, config: KvCacheConfig) -> Result<Self, String> {
        let total_bytes = config.total_bytes();
        eprintln!(
            "[KvCache] Allocating {} MB for {} layers, {} kv_heads, head_dim={}, max_seq={}",
            total_bytes / (1024 * 1024),
            config.num_layers, config.num_kv_heads, config.head_dim, config.max_seq_len,
        );

        // Allocate contiguous VRAM buffer
        let buf = runtime.device.alloc_vram(total_bytes)?;
        buf.zero();

        Ok(Self {
            buf,
            config,
            position: AtomicU32::new(0),
        })
    }

    /// Reset the cache position to 0 (for new sequence).
    /// Does NOT zero the buffer — new data will overwrite.
    pub fn reset(&self) {
        self.position.store(0, Ordering::Release);
    }

    /// Get the current sequence position.
    pub fn position(&self) -> usize {
        self.position.load(Ordering::Acquire) as usize
    }

    /// Get the configuration.
    pub fn config(&self) -> &KvCacheConfig {
        &self.config
    }

    /// Get the GPU virtual address of the underlying buffer.
    pub fn gpu_addr(&self) -> u64 {
        self.buf.gpu_addr()
    }

    /// Get remaining capacity (max_seq_len - position).
    pub fn remaining(&self) -> usize {
        self.config.max_seq_len - self.position()
    }

    // ── Single-token append (decode path) ──

    /// Append a single token's K and V to the cache at the current position.
    /// Does NOT advance position — call `advance()` after writing all layers.
    ///
    /// # Arguments
    /// - `layer`: transformer layer index (0..num_layers)
    /// - `key`: tensor of shape `[num_kv_heads, head_dim]` — K for this token
    /// - `value`: tensor of shape `[num_kv_heads, head_dim]` — V for this token
    ///
    /// Typical decode loop:
    /// ```ignore
    /// for layer in 0..num_layers {
    ///     cache.append(&r, layer, &k, &v)?;
    /// }
    /// cache.advance();  // advance once after all layers
    /// ```
    pub fn append(
        &self,
        runtime: &Arc<GpuRuntime>,
        layer: usize,
        key: &Tensor,
        value: &Tensor,
    ) -> Result<(), String> {
        let pos = self.position.load(Ordering::Acquire) as usize;
        self.append_at(runtime, layer, pos, key, value)
    }

    /// Advance the sequence position by 1. Call after writing all layers.
    pub fn advance(&self) {
        let pos = self.position.load(Ordering::Acquire);
        self.position.store(pos + 1, Ordering::Release);
    }

    /// Advance the sequence position by N. Call after `append_many` for all layers.
    pub fn advance_by(&self, n: usize) {
        let pos = self.position.load(Ordering::Acquire);
        self.position.store(pos + n as u32, Ordering::Release);
    }

    /// Batch decode: submit K+V copies for all layers async, then wait once.
    ///
    /// This is the optimal decode path: N layer copies are submitted as async
    /// dispatches, then synchronized with a single `wait_idle()` at the end.
    ///
    /// # Arguments
    /// - `layer_kvs`: iterator of `(layer_index, key_tensor, value_tensor)`
    ///   where each tensor has shape `[num_kv_heads, head_dim]`
    ///
    /// Typical usage:
    /// ```ignore
    /// let layer_data: Vec<(usize, &Tensor, &Tensor)> = (0..num_layers)
    ///     .map(|l| (l, &layer_keys[l], &layer_values[l]))
    ///     .collect();
    /// cache.append_batch(&r, &layer_data)?;
    /// cache.advance();
    /// ```
    pub fn append_batch(
        &self,
        runtime: &Arc<GpuRuntime>,
        layer_kvs: &[(usize, &Tensor, &Tensor)],
    ) -> Result<(), String> {
        let pos = self.position.load(Ordering::Acquire) as usize;

        // Phase 1: Submit all layer copies async (no waiting)
        for &(layer, key, value) in layer_kvs {
            assert!(layer < self.config.num_layers, "layer {} >= num_layers", layer);
            assert!(pos < self.config.max_seq_len,
                "KV cache overflow: pos={} >= max_seq_len={}", pos, self.config.max_seq_len);

            let head_elements = self.config.num_kv_heads * self.config.head_dim;
            let head_bytes = head_elements * 4;

            assert_eq!(key.numel(), head_elements,
                "key numel mismatch at layer {}", layer);
            assert_eq!(value.numel(), head_elements,
                "value numel mismatch at layer {}", layer);

            let k_dst = self.buf.gpu_addr() + self.k_offset(layer, pos) as u64;
            let v_dst = self.buf.gpu_addr() + self.v_offset(layer, pos) as u64;

            gpu_memcpy_kv_async(runtime, key.gpu_addr(), value.gpu_addr(), k_dst, v_dst, head_bytes)?;
        }

        // Phase 2: Single synchronization waits for all submissions
        runtime.synchronize()
    }

    /// Append a single token's K and V at a specific position (without advancing).
    ///
    /// Used internally by `append()` and `append_many()`.
    fn append_at(
        &self,
        runtime: &Arc<GpuRuntime>,
        layer: usize,
        pos: usize,
        key: &Tensor,
        value: &Tensor,
    ) -> Result<(), String> {
        assert!(layer < self.config.num_layers, "layer {} >= num_layers {}", layer, self.config.num_layers);
        assert!(pos < self.config.max_seq_len,
            "KV cache overflow: pos={} >= max_seq_len={}", pos, self.config.max_seq_len);

        let head_elements = self.config.num_kv_heads * self.config.head_dim;
        let head_bytes = head_elements * 4;

        // Validate input tensor shapes
        assert_eq!(key.numel(), head_elements,
            "key numel {} != expected {} (kv_heads={} * head_dim={})",
            key.numel(), head_elements, self.config.num_kv_heads, self.config.head_dim);
        assert_eq!(value.numel(), head_elements,
            "value numel {} != expected {}", value.numel(), head_elements);

        // Compute destination offsets
        let k_offset = self.k_offset(layer, pos);
        let v_offset = self.v_offset(layer, pos);

        // K+V copy: separate dispatches (fused kernel has issues on small BAR)
        let k_dst = self.buf.gpu_addr() + k_offset as u64;
        let v_dst = self.buf.gpu_addr() + v_offset as u64;
        gpu_memcpy(runtime, key.gpu_addr(), k_dst, head_bytes)?;
        gpu_memcpy(runtime, value.gpu_addr(), v_dst, head_bytes)
    }

    // ── Multi-token append (prefill path) ──

    /// Append a batch of tokens' K and V to the cache (prefill path).
    /// Does NOT advance position — call `advance_by(seq_len)` after writing all layers.
    ///
    /// Uses async dispatch: submits all layer copies without waiting, then
    /// synchronizes once at the end. This reduces dispatch overhead from
    /// 2×N layer round-trips to 1.
    ///
    /// # Arguments
    /// - `layer`: transformer layer index
    /// - `keys`: tensor of shape `[seq_len, num_kv_heads, head_dim]` — K for all tokens
    /// - `values`: tensor of shape `[seq_len, num_kv_heads, head_dim]` — V for all tokens
    ///
    /// Typical prefill:
    /// ```ignore
    /// for layer in 0..num_layers {
    ///     cache.append_many(&r, layer, &layer_keys[layer], &layer_values[layer])?;
    /// }
    /// cache.advance_by(seq_len);  // advance once after all layers
    /// ```
    pub fn append_many(
        &self,
        runtime: &Arc<GpuRuntime>,
        layer: usize,
        keys: &Tensor,
        values: &Tensor,
    ) -> Result<(), String> {
        let pos = self.position.load(Ordering::Acquire) as usize;
        let seq_len = keys.shape()[0];

        assert!(keys.shape().len() == 3, "keys must be 3D: [seq_len, kv_heads, head_dim]");
        assert!(values.shape().len() == 3, "values must be 3D: [seq_len, kv_heads, head_dim]");
        assert_eq!(keys.shape()[1], self.config.num_kv_heads,
            "keys kv_heads {} != config {}", keys.shape()[1], self.config.num_kv_heads);
        assert_eq!(keys.shape()[2], self.config.head_dim,
            "keys head_dim {} != config {}", keys.shape()[2], self.config.head_dim);
        assert_eq!(keys.shape(), values.shape(), "keys and values shape mismatch");
        assert!(pos + seq_len <= self.config.max_seq_len,
            "KV cache overflow: pos={} + seq_len={} > max_seq_len={}",
            pos, seq_len, self.config.max_seq_len);

        let head_elements = self.config.num_kv_heads * self.config.head_dim;
        let head_bytes = head_elements * 4;
        let tokens_bytes = seq_len * head_bytes;

        let keys_addr = keys.gpu_addr();
        let values_addr = values.gpu_addr();
        let k_dst = self.buf.gpu_addr() + self.k_offset(layer, pos) as u64;
        let v_dst = self.buf.gpu_addr() + self.v_offset(layer, pos) as u64;

        // Fused K+V async copy: submit without waiting
        gpu_memcpy_kv_async(runtime, keys_addr, values_addr, k_dst, v_dst, tokens_bytes)?;

        // Synchronize after this layer's submission
        // For multi-layer prefill, the last layer's sync will wait for all prior submissions
        runtime.synchronize()
    }

    /// Batch append: submit K+V copies for all layers async, then wait once.
    ///
    /// This is the optimal prefill path: N layer copies are submitted as async
    /// dispatches, then synchronized with a single `wait_idle()` at the end.
    ///
    /// Compared to calling `append_many()` per layer (which syncs after each),
    /// this reduces dispatch overhead from N round-trips to 1.
    ///
    /// # Arguments
    /// - `layer_kvs`: iterator of `(layer_index, keys_tensor, values_tensor)`
    ///   where each tensor has shape `[seq_len, num_kv_heads, head_dim]`
    ///
    /// Typical usage:
    /// ```ignore
    /// let layer_data: Vec<(usize, &Tensor, &Tensor)> = (0..num_layers)
    ///     .map(|l| (l, &layer_keys[l], &layer_values[l]))
    ///     .collect();
    /// cache.append_many_batch(&r, &layer_data)?;
    /// cache.advance_by(seq_len);
    /// ```
    pub fn append_many_batch(
        &self,
        runtime: &Arc<GpuRuntime>,
        layer_kvs: &[(usize, &Tensor, &Tensor)],
    ) -> Result<(), String> {
        let pos = self.position.load(Ordering::Acquire) as usize;

        // Phase 1: Submit all layer copies async (no waiting)
        for &(layer, keys, values) in layer_kvs {
            let seq_len = keys.shape()[0];
            assert!(keys.shape().len() == 3, "keys must be 3D");
            assert!(values.shape().len() == 3, "values must be 3D");
            assert_eq!(keys.shape()[1], self.config.num_kv_heads,
                "keys kv_heads mismatch at layer {}", layer);
            assert_eq!(keys.shape()[2], self.config.head_dim,
                "keys head_dim mismatch at layer {}", layer);
            assert!(pos + seq_len <= self.config.max_seq_len,
                "KV cache overflow at layer {}", layer);

            let head_elements = self.config.num_kv_heads * self.config.head_dim;
            let head_bytes = head_elements * 4;
            let tokens_bytes = seq_len * head_bytes;

            let k_dst = self.buf.gpu_addr() + self.k_offset(layer, pos) as u64;
            let v_dst = self.buf.gpu_addr() + self.v_offset(layer, pos) as u64;

            gpu_memcpy_kv_async(runtime, keys.gpu_addr(), values.gpu_addr(), k_dst, v_dst, tokens_bytes)?;
        }

        // Phase 2: Single synchronization waits for all submissions
        runtime.synchronize()
    }

    // ── Get (for attention computation) ──

    /// Get K cache slice for attention at a given layer.
    ///
    /// Returns GPU address and shape for `K[layer, 0..pos, :, :]`.
    /// This is the full key history used in attention: Q @ K^T.
    pub fn get_k(&self, layer: usize) -> KvSlice {
        let pos = self.position.load(Ordering::Acquire) as usize;
        assert!(layer < self.config.num_layers, "layer {} >= num_layers", layer);
        assert!(pos > 0, "KV cache is empty (position=0)");

        KvSlice {
            gpu_addr: self.buf.gpu_addr() + self.k_offset(layer, 0) as u64,
            seq_len: pos,
            num_kv_heads: self.config.num_kv_heads,
            head_dim: self.config.head_dim,
        }
    }

    /// Get V cache slice for attention at a given layer.
    ///
    /// Returns GPU address and shape for `V[layer, 0..pos, :, :]`.
    /// This is used in attention: softmax(Q @ K^T) @ V.
    pub fn get_v(&self, layer: usize) -> KvSlice {
        let pos = self.position.load(Ordering::Acquire) as usize;
        assert!(layer < self.config.num_layers, "layer {} >= num_layers", layer);
        assert!(pos > 0, "KV cache is empty (position=0)");

        KvSlice {
            gpu_addr: self.buf.gpu_addr() + self.v_offset(layer, 0) as u64,
            seq_len: pos,
            num_kv_heads: self.config.num_kv_heads,
            head_dim: self.config.head_dim,
        }
    }

    /// Get both K and V slices for a layer.
    pub fn get_kv(&self, layer: usize) -> (KvSlice, KvSlice) {
        (self.get_k(layer), self.get_v(layer))
    }

    // ── Offset calculations ──

    /// Byte offset to K data for layer at position.
    pub(crate) fn k_offset(&self, layer: usize, pos: usize) -> usize {
        let head_elements = self.config.num_kv_heads * self.config.head_dim;
        let elements_per_kv = self.config.max_seq_len * head_elements;
        let elements_per_layer = 2 * elements_per_kv;
        let offset_elements = layer * elements_per_layer + pos * head_elements;
        offset_elements * 4
    }

    /// Byte offset to V data for layer at position.
    pub(crate) fn v_offset(&self, layer: usize, pos: usize) -> usize {
        let head_elements = self.config.num_kv_heads * self.config.head_dim;
        let elements_per_kv = self.config.max_seq_len * head_elements;
        let elements_per_layer = 2 * elements_per_kv;
        // V starts after K within the layer
        let offset_elements = layer * elements_per_layer + elements_per_kv + pos * head_elements;
        offset_elements * 4
    }

    // ── Read helpers for attention ──

    /// Read K data for a layer into a flat f32 vector.
    /// Returns [pos * kv_heads * head_dim] elements (all cached positions).
    pub fn read_k_layer(&self, runtime: &Arc<GpuRuntime>, layer: usize) -> Vec<f32> {
        let _ = runtime.wait_idle();
        let pos = self.position.load(Ordering::Acquire) as usize;
        let n = pos * self.config.num_kv_heads * self.config.head_dim;
        let offset = self.k_offset(layer, 0);
        let mut data = vec![0f32; n];
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.buf.host_ptr.add(offset) as *const f32,
                data.as_mut_ptr(),
                n,
            );
        }
        data
    }

    /// Read V data for a layer into a flat f32 vector.
    /// Returns [pos * kv_heads * head_dim] elements (all cached positions).
    pub fn read_v_layer(&self, runtime: &Arc<GpuRuntime>, layer: usize) -> Vec<f32> {
        let _ = runtime.wait_idle();
        let pos = self.position.load(Ordering::Acquire) as usize;
        let n = pos * self.config.num_kv_heads * self.config.head_dim;
        let offset = self.v_offset(layer, 0);
        let mut data = vec![0f32; n];
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.buf.host_ptr.add(offset) as *const f32,
                data.as_mut_ptr(),
                n,
            );
        }
        data
    }

    // ── Debug helpers ──

    /// Read back the entire KV cache to CPU (for testing).
    /// WARNING: very slow for large caches — use only for debugging.
    pub fn to_cpu_vec(&self, runtime: &Arc<GpuRuntime>) -> Vec<f32> {
        let _ = runtime.wait_idle();
        let n = self.config.total_elements();
        let mut data = vec![0f32; n];
        self.buf.read(unsafe {
            std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut u8, n * 4)
        });
        data
    }

    /// Read back a single token's K from the cache.
    pub fn read_k_token(&self, runtime: &Arc<GpuRuntime>, layer: usize, pos: usize) -> Vec<f32> {
        let _ = runtime.queue.synchronize();
        let offset = self.k_offset(layer, pos);
        let n = self.config.num_kv_heads * self.config.head_dim;
        let mut data = vec![0f32; n];
        self.buf.read_bytes(offset, n * 4);
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.buf.host_ptr.add(offset) as *const f32,
                data.as_mut_ptr(),
                n,
            );
        }
        data
    }

    /// Read back a single token's V from the cache.
    pub fn read_v_token(&self, runtime: &Arc<GpuRuntime>, layer: usize, pos: usize) -> Vec<f32> {
        let _ = runtime.queue.synchronize();
        let offset = self.v_offset(layer, pos);
        let n = self.config.num_kv_heads * self.config.head_dim;
        let mut data = vec![0f32; n];
        self.buf.read_bytes(offset, n * 4);
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.buf.host_ptr.add(offset) as *const f32,
                data.as_mut_ptr(),
                n,
            );
        }
        data
    }
}

/// GPU memory copy using vectorized (f32x4) elementwise memcpy kernel.
///
/// Copies `n_bytes` from `src_addr` to `dst_addr` on the GPU.
/// Both addresses must be valid GPU VRAM addresses.
#[cfg(feature = "rocm")]
fn gpu_memcpy(
    runtime: &Arc<GpuRuntime>,
    src_addr: u64,
    dst_addr: u64,
    n_bytes: usize,
) -> Result<(), String> {
    let n_elems = n_bytes / 4;
    let n_4elems = (n_elems + 3) / 4; // ceil(n_elems / 4)
    assert_eq!(n_bytes % 4, 0, "gpu_memcpy requires 4-byte aligned size");

    let kernel = runtime.ensure_kernel_blockdsl(
        "gpu_memcpy_x4",
        || crate::t0::elementwise_kernels::build_memcpy_x4(),
    )?;

    let ka = crate::kernargs![
        src_addr => u64,
        dst_addr => u64,
        n_4elems as u32 => u32,
    ];

    let grid_x = crate::t0::elementwise_kernels::elementwise_grid_x4(n_elems as u32);
    runtime.dispatch(&kernel, [grid_x, 1, 1], &ka)
}

/// Fused K+V memory copy — copies both K and V in a single dispatch.
/// Uses vectorized (f32x4) loads/stores for 4× bandwidth per thread.
///
/// Copies `n_bytes` from `k_src → k_dst` AND `v_src → v_dst` in one kernel.
/// n_bytes must be the size of K (or V), which are assumed equal.
#[cfg(feature = "rocm")]
fn gpu_memcpy_kv(
    runtime: &Arc<GpuRuntime>,
    k_src: u64,
    v_src: u64,
    k_dst: u64,
    v_dst: u64,
    n_bytes: usize,
) -> Result<(), String> {
    let n_elems = n_bytes / 4;
    assert_eq!(n_bytes % 4, 0, "gpu_memcpy_kv requires 4-byte aligned size");

    let kernel = runtime.ensure_kernel_blockdsl(
        "gpu_memcpy_kv_x4",
        || crate::t0::elementwise_kernels::build_memcpy_kv_x4(),
    )?;

    let ka = crate::kernargs![
        k_src => u64,
        v_src => u64,
        k_dst => u64,
        v_dst => u64,
        n_elems as u32 => u32,
    ];

    let grid_x = crate::t0::elementwise_kernels::elementwise_grid_x4(n_elems as u32);
    runtime.dispatch(&kernel, [grid_x, 1, 1], &ka)
}

/// Async fused K+V memory copy — submits without waiting.
/// Caller must synchronize via `runtime.wait_idle()` or `runtime.synchronize()`.
#[cfg(feature = "rocm")]
fn gpu_memcpy_kv_async(
    runtime: &Arc<GpuRuntime>,
    k_src: u64,
    v_src: u64,
    k_dst: u64,
    v_dst: u64,
    n_bytes: usize,
) -> Result<usize, String> {
    let n_elems = n_bytes / 4;
    assert_eq!(n_bytes % 4, 0, "gpu_memcpy_kv_async requires 4-byte aligned size");

    let kernel = runtime.ensure_kernel_blockdsl(
        "gpu_memcpy_kv_x4",
        || crate::t0::elementwise_kernels::build_memcpy_kv_x4(),
    )?;

    let ka = crate::kernargs![
        k_src => u64,
        v_src => u64,
        k_dst => u64,
        v_dst => u64,
        n_elems as u32 => u32,
    ];

    let grid_x = crate::t0::elementwise_kernels::elementwise_grid_x4(n_elems as u32);
    let slot = runtime.dispatch_async(&kernel, [grid_x, 1, 1], &ka);
    Ok(slot)
}

// ═══════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════

#[cfg(all(test, feature = "rocm"))]
mod kv_cache_tests {
    use std::sync::{Arc, OnceLock};
    use crate::ignis::gpu_context::GpuRuntime;
    use crate::ignis::tensor::Tensor;
    use crate::ignis::kv_cache::{KvCache, KvCacheConfig};

    struct SyncRt(Arc<GpuRuntime>);
    unsafe impl Sync for SyncRt {}
    unsafe impl Send for SyncRt {}
    static GPU_RT: OnceLock<SyncRt> = OnceLock::new();

    fn rt() -> Arc<GpuRuntime> {
        GPU_RT.get_or_init(|| {
            SyncRt(GpuRuntime::new().expect("Failed to create GpuRuntime"))
        }).0.clone()
    }

    fn small_config() -> KvCacheConfig {
        KvCacheConfig {
            num_layers: 2,
            num_kv_heads: 4,
            head_dim: 8,
            max_seq_len: 16,
        }
    }

    // ── Test 1: allocation and zeroing ──

    #[test]
    fn test_kv_cache_alloc() {
        let r = rt();
        let cfg = small_config();
        let cache = KvCache::new(&r, cfg.clone()).unwrap();

        // Check size
        let expected_elements = cfg.num_layers * 2 * cfg.max_seq_len * cfg.num_kv_heads * cfg.head_dim;
        assert_eq!(cache.config().total_elements(), expected_elements);

        // Check position starts at 0
        assert_eq!(cache.position(), 0);
        assert_eq!(cache.remaining(), cfg.max_seq_len);

        // Verify buffer is zeroed
        let data = cache.to_cpu_vec(&r);
        assert!(data.iter().all(|&x| x == 0.0), "cache should be zero-initialized");
    }

    // ── Test 2: single-token append (decode path) ──

    #[test]
    fn test_kv_cache_append_single() {
        let r = rt();
        let cfg = small_config();
        let cache = KvCache::new(&r, cfg.clone()).unwrap();

        let head_elements = cfg.num_kv_heads * cfg.head_dim; // 4 * 8 = 32

        // Create K/V tensors for layer 0, token 0
        let k_data: Vec<f32> = (0..head_elements).map(|i| (i + 1) as f32 * 0.1).collect();
        let v_data: Vec<f32> = (0..head_elements).map(|i| (i + 1) as f32 * 0.2).collect();

        let key = Tensor::from_f32(&r, &k_data, &[cfg.num_kv_heads, cfg.head_dim], "k0").unwrap();
        let val = Tensor::from_f32(&r, &v_data, &[cfg.num_kv_heads, cfg.head_dim], "v0").unwrap();

        cache.append(&r, 0, &key, &val).unwrap();
        cache.advance();

        // Position should advance by 1
        assert_eq!(cache.position(), 1);

        // Verify K at position 0
        let k_read = cache.read_k_token(&r, 0, 0);
        for i in 0..head_elements {
            assert!((k_read[i] - k_data[i]).abs() < 1e-5,
                "K mismatch at [{}]: got {}, expected {}", i, k_read[i], k_data[i]);
        }

        // Verify V at position 0
        let v_read = cache.read_v_token(&r, 0, 0);
        for i in 0..head_elements {
            assert!((v_read[i] - v_data[i]).abs() < 1e-5,
                "V mismatch at [{}]: got {}, expected {}", i, v_read[i], v_data[i]);
        }
    }

    // ── Test 3: multi-token append (prefill path) ──

    #[test]
    fn test_kv_cache_append_many() {
        let r = rt();
        let cfg = small_config();
        let cache = KvCache::new(&r, cfg.clone()).unwrap();

        let seq_len = 5usize;
        let head_elements = cfg.num_kv_heads * cfg.head_dim;

        // Create [seq_len, kv_heads, head_dim] K/V tensors
        let mut k_data = Vec::with_capacity(seq_len * head_elements);
        let mut v_data = Vec::with_capacity(seq_len * head_elements);
        for s in 0..seq_len {
            for i in 0..head_elements {
                k_data.push((s * head_elements + i + 1) as f32 * 0.01);
                v_data.push((s * head_elements + i + 1) as f32 * 0.02);
            }
        }

        let keys = Tensor::from_f32(&r, &k_data, &[seq_len, cfg.num_kv_heads, cfg.head_dim], "keys").unwrap();
        let vals = Tensor::from_f32(&r, &v_data, &[seq_len, cfg.num_kv_heads, cfg.head_dim], "vals").unwrap();

        cache.append_many(&r, 0, &keys, &vals).unwrap();
        cache.advance_by(seq_len);

        // Position should advance by seq_len
        assert_eq!(cache.position(), seq_len);

        // Verify each token
        for s in 0..seq_len {
            let k_read = cache.read_k_token(&r, 0, s);
            let v_read = cache.read_v_token(&r, 0, s);
            let k_start = s * head_elements;
            let v_start = s * head_elements;
            for i in 0..head_elements {
                assert!((k_read[i] - k_data[k_start + i]).abs() < 1e-5,
                    "K mismatch at seq={}, element={}: got {}, expected {}",
                    s, i, k_read[i], k_data[k_start + i]);
                assert!((v_read[i] - v_data[v_start + i]).abs() < 1e-5,
                    "V mismatch at seq={}, element={}: got {}, expected {}",
                    s, i, v_read[i], v_data[v_start + i]);
            }
        }
    }

    // ── Test 4: multiple layers (all share same position) ──

    #[test]
    fn test_kv_cache_multi_layer() {
        let r = rt();
        let cfg = small_config(); // 2 layers
        let cache = KvCache::new(&r, cfg.clone()).unwrap();

        let head_elements = cfg.num_kv_heads * cfg.head_dim;

        // In real LLM usage, all layers process the same token at the same position.
        // So we append K/V for all layers at one position, then advance together.
        let k_data: Vec<f32> = (0..head_elements).map(|i| (i + 1) as f32 * 0.1).collect();
        let v_data: Vec<f32> = (0..head_elements).map(|i| (i + 1) as f32 * 0.2).collect();

        // Append for both layers at position 0
        for layer in 0..cfg.num_layers {
            let key = Tensor::from_f32(&r, &k_data, &[cfg.num_kv_heads, cfg.head_dim], "k").unwrap();
            let val = Tensor::from_f32(&r, &v_data, &[cfg.num_kv_heads, cfg.head_dim], "v").unwrap();
            cache.append(&r, layer, &key, &val).unwrap();
        }
        // Advance once after all layers
        cache.advance();

        // Position is now 1 (one token processed across all layers)
        assert_eq!(cache.position(), 1);

        // Verify: data was written to position 0 of each layer
        for layer in 0..cfg.num_layers {
            let k_read = cache.read_k_token(&r, layer, 0);
            let v_read = cache.read_v_token(&r, layer, 0);
            for i in 0..head_elements {
                let expected_k = k_data[i];
                let expected_v = v_data[i];
                assert!((k_read[i] - expected_k).abs() < 1e-5,
                    "Layer {} pos 0 K[{}] mismatch: got {}, expected {}",
                    layer, i, k_read[i], expected_k);
                assert!((v_read[i] - expected_v).abs() < 1e-5,
                    "Layer {} pos 0 V[{}] mismatch: got {}, expected {}",
                    layer, i, v_read[i], expected_v);
            }
        }
    }

    // ── Test 5: get_k / get_v return correct slices ──

    #[test]
    fn test_kv_cache_get_slices() {
        let r = rt();
        let cfg = small_config();
        let cache = KvCache::new(&r, cfg.clone()).unwrap();

        let head_elements = cfg.num_kv_heads * cfg.head_dim;

        // Append 3 tokens to layer 0
        for s in 0..3 {
            let k_data: Vec<f32> = (0..head_elements).map(|i| (s * 100 + i) as f32).collect();
            let v_data: Vec<f32> = (0..head_elements).map(|i| (s * 200 + i) as f32).collect();
            let key = Tensor::from_f32(&r, &k_data, &[cfg.num_kv_heads, cfg.head_dim], "k").unwrap();
            let val = Tensor::from_f32(&r, &v_data, &[cfg.num_kv_heads, cfg.head_dim], "v").unwrap();
            cache.append(&r, 0, &key, &val).unwrap();
            cache.advance();
        }

        // get_k should return slice with seq_len=3
        let k_slice = cache.get_k(0);
        assert_eq!(k_slice.seq_len, 3);
        assert_eq!(k_slice.num_kv_heads, cfg.num_kv_heads);
        assert_eq!(k_slice.head_dim, cfg.head_dim);

        // Verify GPU address points to the right place (start of layer 0 K)
        assert_eq!(k_slice.gpu_addr, cache.buf.gpu_addr() + cache.k_offset(0, 0) as u64);

        let v_slice = cache.get_v(0);
        assert_eq!(v_slice.seq_len, 3);
        assert_eq!(v_slice.gpu_addr, cache.buf.gpu_addr() + cache.v_offset(0, 0) as u64);
    }

    // ── Test 6: reset clears position ──

    #[test]
    fn test_kv_cache_reset() {
        let r = rt();
        let cfg = small_config();
        let cache = KvCache::new(&r, cfg.clone()).unwrap();

        let head_elements = cfg.num_kv_heads * cfg.head_dim;

        // Fill 5 positions
        for _ in 0..5 {
            let k_data = vec![1.0f32; head_elements];
            let v_data = vec![2.0f32; head_elements];
            let key = Tensor::from_f32(&r, &k_data, &[cfg.num_kv_heads, cfg.head_dim], "k").unwrap();
            let val = Tensor::from_f32(&r, &v_data, &[cfg.num_kv_heads, cfg.head_dim], "v").unwrap();
            cache.append(&r, 0, &key, &val).unwrap();
            cache.advance();
        }
        assert_eq!(cache.position(), 5);

        // Reset and verify position is 0
        cache.reset();
        assert_eq!(cache.position(), 0);
        assert_eq!(cache.remaining(), cfg.max_seq_len);

        // Write new data at position 0 (overwrites old)
        let k_new = vec![99.0f32; head_elements];
        let v_new = vec![88.0f32; head_elements];
        let key = Tensor::from_f32(&r, &k_new, &[cfg.num_kv_heads, cfg.head_dim], "k").unwrap();
        let val = Tensor::from_f32(&r, &v_new, &[cfg.num_kv_heads, cfg.head_dim], "v").unwrap();
        cache.append(&r, 0, &key, &val).unwrap();
        cache.advance();

        let k_read = cache.read_k_token(&r, 0, 0);
        for i in 0..head_elements {
            assert!((k_read[i] - 99.0).abs() < 1e-5,
                "After reset+append, K[0][{}] should be 99.0, got {}", i, k_read[i]);
        }
    }

    // ── Test 7: prefill + decode combined workflow ──

    #[test]
    fn test_kv_cache_prefill_then_decode() {
        let r = rt();
        let cfg = small_config();
        let cache = KvCache::new(&r, cfg.clone()).unwrap();

        let head_elements = cfg.num_kv_heads * cfg.head_dim;

        // Phase 1: Prefill with prompt of 4 tokens
        let prefill_seq = 4usize;
        let mut k_prefill = Vec::with_capacity(prefill_seq * head_elements);
        let mut v_prefill = Vec::with_capacity(prefill_seq * head_elements);
        for s in 0..prefill_seq {
            for i in 0..head_elements {
                k_prefill.push((s * 1000 + i) as f32);
                v_prefill.push((s * 2000 + i) as f32);
            }
        }
        let keys = Tensor::from_f32(&r, &k_prefill, &[prefill_seq, cfg.num_kv_heads, cfg.head_dim], "pk").unwrap();
        let vals = Tensor::from_f32(&r, &v_prefill, &[prefill_seq, cfg.num_kv_heads, cfg.head_dim], "pv").unwrap();
        cache.append_many(&r, 0, &keys, &vals).unwrap();
        cache.advance_by(prefill_seq);

        assert_eq!(cache.position(), prefill_seq);

        // Phase 2: Decode 2 more tokens one at a time
        for step in 0..2 {
            let pos = prefill_seq + step;
            let k_data: Vec<f32> = (0..head_elements).map(|i| (pos * 1000 + i) as f32).collect();
            let v_data: Vec<f32> = (0..head_elements).map(|i| (pos * 2000 + i) as f32).collect();
            let key = Tensor::from_f32(&r, &k_data, &[cfg.num_kv_heads, cfg.head_dim], "dk").unwrap();
            let val = Tensor::from_f32(&r, &v_data, &[cfg.num_kv_heads, cfg.head_dim], "dv").unwrap();
            cache.append(&r, 0, &key, &val).unwrap();
            cache.advance();
        }

        assert_eq!(cache.position(), prefill_seq + 2);

        // Verify all 6 tokens are correct
        for s in 0..6 {
            let k_read = cache.read_k_token(&r, 0, s);
            let v_read = cache.read_v_token(&r, 0, s);
            for i in 0..head_elements {
                let expected_k = (s * 1000 + i) as f32;
                let expected_v = (s * 2000 + i) as f32;
                assert!((k_read[i] - expected_k).abs() < 1e-5,
                    "Token {} K[{}] mismatch: got {}, expected {}", s, i, k_read[i], expected_k);
                assert!((v_read[i] - expected_v).abs() < 1e-5,
                    "Token {} V[{}] mismatch: got {}, expected {}", s, i, v_read[i], expected_v);
            }
        }

        // Verify get_k returns the full 6-token slice
        let k_slice = cache.get_k(0);
        assert_eq!(k_slice.seq_len, 6);
    }

    // ── Test 8: get_kv helper ──

    #[test]
    fn test_kv_cache_get_kv() {
        let r = rt();
        let cfg = small_config();
        let cache = KvCache::new(&r, cfg.clone()).unwrap();

        let head_elements = cfg.num_kv_heads * cfg.head_dim;
        let k_data = vec![7.0f32; head_elements];
        let v_data = vec![8.0f32; head_elements];
        let key = Tensor::from_f32(&r, &k_data, &[cfg.num_kv_heads, cfg.head_dim], "k").unwrap();
        let val = Tensor::from_f32(&r, &v_data, &[cfg.num_kv_heads, cfg.head_dim], "v").unwrap();
        cache.append(&r, 0, &key, &val).unwrap();
        cache.advance();

        let (k_slice, v_slice) = cache.get_kv(0);
        assert_eq!(k_slice.seq_len, 1);
        assert_eq!(v_slice.seq_len, 1);
        assert_eq!(k_slice.num_kv_heads, cfg.num_kv_heads);
        assert_eq!(v_slice.num_kv_heads, cfg.num_kv_heads);
        // K and V should be at different addresses
        assert_ne!(k_slice.gpu_addr, v_slice.gpu_addr);
    }

    // ── Test 9: offset calculations ──

    #[test]
    fn test_kv_cache_offsets() {
        let r = rt();
        let cfg = KvCacheConfig {
            num_layers: 3,
            num_kv_heads: 2,
            head_dim: 4,
            max_seq_len: 10,
        };
        let cache = KvCache::new(&r, cfg.clone()).unwrap();

        let head_elements = cfg.num_kv_heads * cfg.head_dim; // 8
        let elements_per_kv = cfg.max_seq_len * head_elements; // 80
        let elements_per_layer = 2 * elements_per_kv; // 160

        // K offset for layer 1, pos 3
        let k_off = cache.k_offset(1, 3);
        let expected_k_off = (1 * elements_per_layer + 3 * head_elements) * 4; // (160 + 24) * 4 = 736
        assert_eq!(k_off, expected_k_off);

        // V offset for layer 1, pos 3
        let v_off = cache.v_offset(1, 3);
        let expected_v_off = (1 * elements_per_layer + elements_per_kv + 3 * head_elements) * 4; // (160 + 80 + 24) * 4 = 1056
        assert_eq!(v_off, expected_v_off);

        // V offset should be after K offset within the same layer+pos
        assert!(cache.v_offset(0, 0) > cache.k_offset(0, 0));
    }

    // ── Test 10: large cache allocation (realistic Qwen3 config) ──

    #[test]
    fn test_kv_cache_qwen3_config() {
        let r = rt();
        // Qwen3-8B-ish config (scaled down for speed)
        let cfg = KvCacheConfig {
            num_layers: 4,       // full model has 36
            num_kv_heads: 4,     // GQA: 4 KV heads (full model has 8)
            head_dim: 128,       // standard
            max_seq_len: 512,    // typical context
        };

        let cache = KvCache::new(&r, cfg.clone()).unwrap();
        let expected_mb = cfg.total_bytes() / (1024 * 1024);
        eprintln!("  Qwen3-scaled KV cache: {} MB", expected_mb);

        // Should be able to prefill a reasonable sequence
        let seq_len = 64;
        let head_elements = cfg.num_kv_heads * cfg.head_dim;
        let k_data = vec![0.5f32; seq_len * head_elements];
        let v_data = vec![0.5f32; seq_len * head_elements];
        let keys = Tensor::from_f32(&r, &k_data, &[seq_len, cfg.num_kv_heads, cfg.head_dim], "k").unwrap();
        let vals = Tensor::from_f32(&r, &v_data, &[seq_len, cfg.num_kv_heads, cfg.head_dim], "v").unwrap();

        cache.append_many(&r, 0, &keys, &vals).unwrap();
        cache.advance_by(seq_len);
        assert_eq!(cache.position(), seq_len);
    }

    // ── Test 11: cache overflow protection ──

    #[test]
    #[should_panic(expected = "KV cache overflow")]
    fn test_kv_cache_overflow() {
        let r = rt();
        let cfg = KvCacheConfig {
            num_layers: 1,
            num_kv_heads: 2,
            head_dim: 4,
            max_seq_len: 2, // only 2 tokens fit
        };
        let cache = KvCache::new(&r, cfg.clone()).unwrap();
        let head_elements = 8;

        // Fill up the cache (2 tokens)
        for _ in 0..2 {
            let k_data = vec![1.0f32; head_elements];
            let v_data = vec![1.0f32; head_elements];
            let key = Tensor::from_f32(&r, &k_data, &[cfg.num_kv_heads, cfg.head_dim], "k").unwrap();
            let val = Tensor::from_f32(&r, &v_data, &[cfg.num_kv_heads, cfg.head_dim], "v").unwrap();
            cache.append(&r, 0, &key, &val).unwrap();
            cache.advance();
        }

        // This should panic on the 3rd append
        let k_data = vec![1.0f32; head_elements];
        let v_data = vec![1.0f32; head_elements];
        let key = Tensor::from_f32(&r, &k_data, &[cfg.num_kv_heads, cfg.head_dim], "k").unwrap();
        let val = Tensor::from_f32(&r, &v_data, &[cfg.num_kv_heads, cfg.head_dim], "v").unwrap();
        cache.append(&r, 0, &key, &val).unwrap();
    }

    // ── Test 12: read_k_layer / read_v_layer ──

    #[test]
    fn test_kv_cache_read_layers() {
        let r = rt();
        let cfg = small_config();
        let cache = KvCache::new(&r, cfg.clone()).unwrap();

        let head_elements = cfg.num_kv_heads * cfg.head_dim;
        let kv_dim = cfg.num_kv_heads * cfg.head_dim;

        // Prefill 3 tokens with distinct data
        for s in 0..3 {
            let k_data: Vec<f32> = (0..head_elements).map(|i| (s * 100 + i) as f32).collect();
            let v_data: Vec<f32> = (0..head_elements).map(|i| (s * 200 + i) as f32).collect();
            let key = Tensor::from_f32(&r, &k_data, &[cfg.num_kv_heads, cfg.head_dim], "k").unwrap();
            let val = Tensor::from_f32(&r, &v_data, &[cfg.num_kv_heads, cfg.head_dim], "v").unwrap();
            cache.append(&r, 0, &key, &val).unwrap();
            cache.advance();
        }

        // read_k_layer should return all 3 tokens' K data
        let k_all = cache.read_k_layer(&r, 0);
        assert_eq!(k_all.len(), 3 * kv_dim, "k_all len mismatch");

        // Verify each token's K data
        for s in 0..3 {
            for i in 0..head_elements {
                let expected = (s * 100 + i) as f32;
                let got = k_all[s * kv_dim + i];
                assert!((got - expected).abs() < 1e-5,
                    "read_k_layer[{}][{}]: got {} expected {}", s, i, got, expected);
            }
        }

        // read_v_layer should return all 3 tokens' V data
        let v_all = cache.read_v_layer(&r, 0);
        assert_eq!(v_all.len(), 3 * kv_dim);

        for s in 0..3 {
            for i in 0..head_elements {
                let expected = (s * 200 + i) as f32;
                let got = v_all[s * kv_dim + i];
                assert!((got - expected).abs() < 1e-5,
                    "read_v_layer[{}][{}]: got {} expected {}", s, i, got, expected);
            }
        }
    }

    #[test]
    fn test_kv_cache_read_layers_multiple_layers() {
        let r = rt();
        let cfg = small_config(); // 2 layers
        let cache = KvCache::new(&r, cfg.clone()).unwrap();

        let head_elements = cfg.num_kv_heads * cfg.head_dim;

        // Write different data to each layer
        for layer in 0..cfg.num_layers {
            let k_data: Vec<f32> = (0..head_elements).map(|i| (layer * 1000 + i) as f32).collect();
            let v_data: Vec<f32> = (0..head_elements).map(|i| (layer * 2000 + i) as f32).collect();
            let key = Tensor::from_f32(&r, &k_data, &[cfg.num_kv_heads, cfg.head_dim], "k").unwrap();
            let val = Tensor::from_f32(&r, &v_data, &[cfg.num_kv_heads, cfg.head_dim], "v").unwrap();
            cache.append(&r, layer, &key, &val).unwrap();
        }
        cache.advance();

        // Each layer should return its own data
        for layer in 0..cfg.num_layers {
            let k_all = cache.read_k_layer(&r, layer);
            for i in 0..head_elements {
                let expected = (layer * 1000 + i) as f32;
                assert!((k_all[i] - expected).abs() < 1e-5,
                    "layer {} K[{}]: got {} expected {}", layer, i, k_all[i], expected);
            }
        }
    }

    #[test]
    fn test_kv_cache_read_layers_empty() {
        let r = rt();
        let cfg = small_config();
        let cache = KvCache::new(&r, cfg.clone()).unwrap();

        // Position=0 → empty read
        let k_all = cache.read_k_layer(&r, 0);
        assert_eq!(k_all.len(), 0, "empty cache should return empty vec");
    }

    // ── Test 13: remaining capacity tracking ──

    #[test]
    fn test_kv_cache_remaining() {
        let r = rt();
        let cfg = KvCacheConfig {
            num_layers: 1,
            num_kv_heads: 2,
            head_dim: 4,
            max_seq_len: 10,
        };
        let cache = KvCache::new(&r, cfg.clone()).unwrap();
        let head_elements = 8;

        assert_eq!(cache.remaining(), 10);

        // Append 3 tokens
        for _ in 0..3 {
            let k_data = vec![0.0f32; head_elements];
            let v_data = vec![0.0f32; head_elements];
            let key = Tensor::from_f32(&r, &k_data, &[cfg.num_kv_heads, cfg.head_dim], "k").unwrap();
            let val = Tensor::from_f32(&r, &v_data, &[cfg.num_kv_heads, cfg.head_dim], "v").unwrap();
            cache.append(&r, 0, &key, &val).unwrap();
            cache.advance();
        }
        assert_eq!(cache.remaining(), 7);

        // Prefill 4 more tokens
        let k_data = vec![0.0f32; 4 * head_elements];
        let v_data = vec![0.0f32; 4 * head_elements];
        let keys = Tensor::from_f32(&r, &k_data, &[4, cfg.num_kv_heads, cfg.head_dim], "k").unwrap();
        let vals = Tensor::from_f32(&r, &v_data, &[4, cfg.num_kv_heads, cfg.head_dim], "v").unwrap();
        cache.append_many(&r, 0, &keys, &vals).unwrap();
        cache.advance_by(4);
        assert_eq!(cache.remaining(), 3);

        // Reset
        cache.reset();
        assert_eq!(cache.remaining(), 10);
    }
}
