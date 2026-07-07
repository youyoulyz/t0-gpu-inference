//! GpuRuntime — High-level GPU context for Ignis autodiff.
//!
//! Wraps the low-level KFD bare-metal runtime into a convenient API:
//! - `GpuRuntime` owns device, queue, dispatch pool
//! - `ensure_kernel()` compiles and caches GPU kernels by name
//! - `dispatch()` / `dispatch_fused()` for convenient kernel launch
//! - `kernargs!` macro for building kernarg byte arrays

#[cfg(feature = "rocm")]
use std::sync::Arc;
#[cfg(feature = "rocm")]
use std::path::PathBuf;
#[cfg(feature = "rocm")]
use std::io::{Read, Write};
#[cfg(feature = "rocm")]
use std::fs;
#[cfg(feature = "rocm")]
use std::collections::HashMap;
#[cfg(feature = "rocm")]
use std::sync::Mutex;

#[cfg(feature = "rocm")]
use crate::kfd::{KfdDevice, AqlQueue, GpuBuffer, GpuKernel, KernelLoadConfig, DispatchPool};


// =============================================================================
// BufferPool — LRU cache for GPU VRAM buffers
// =============================================================================
//
// Eliminates KFD VA-reuse race condition: freed buffers are cached by
// aligned_size instead of being unmapped/freed. Next allocation of the
// same size pops from cache (zero syscalls, same VA + mapping).

/// GPU buffer pool with size-keyed caching.
#[cfg(feature = "rocm")]
pub struct BufferPool {
    cache: Mutex<HashMap<usize, Vec<GpuBuffer>>>,
    device: Arc<KfdDevice>,
    cached_bytes: std::sync::atomic::AtomicUsize,
}

#[cfg(feature = "rocm")]
impl BufferPool {
    pub fn new(device: &Arc<KfdDevice>) -> Self {
        Self {
            cache: Mutex::new(HashMap::new()),
            device: Arc::clone(device),
            cached_bytes: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    /// Allocate a VRAM buffer. Checks cache first, falls back to device alloc.
    pub fn alloc(&self, n_bytes: usize) -> Result<GpuBuffer, String> {
        let aligned = ((n_bytes + 4095) / 4096) * 4096;
        let mut cache = self.cache.lock().unwrap();
        if let Some(bufs) = cache.get_mut(&aligned) {
            if let Some(buf) = bufs.pop() {
                self.cached_bytes.fetch_sub(aligned, std::sync::atomic::Ordering::Relaxed);
                return Ok(buf);
            }
        }
        drop(cache);
        self.device.alloc_vram(n_bytes)
    }

    /// Return a buffer to the pool instead of freeing it.
    pub fn recycle(&self, buf: GpuBuffer) {
        let size = buf.size;
        let mut cache = self.cache.lock().unwrap();
        cache.entry(size).or_default().push(buf);
        self.cached_bytes.fetch_add(size, std::sync::atomic::Ordering::Relaxed);
    }

    /// Free all cached buffers.
    pub fn clear(&self) {
        let mut cache = self.cache.lock().unwrap();
        cache.clear();
        self.cached_bytes.store(0, std::sync::atomic::Ordering::Relaxed);
    }

    /// Total bytes currently cached.
    pub fn cached_bytes(&self) -> usize {
        self.cached_bytes.load(std::sync::atomic::Ordering::Relaxed)
    }
}

/// High-level GPU runtime context.
///
/// Owns the device, queue, dispatch pool, and a compile cache for kernels.
/// Shared via `Arc<GpuRuntime>` across tensors and ops.
#[cfg(feature = "rocm")]
pub struct GpuRuntime {
    /// KFD device handle
    pub device: Arc<KfdDevice>,
    /// AQL hardware queue
    pub queue: AqlQueue,
    /// Dispatch pool for kernarg memory
    pub pool: DispatchPool,
    /// Buffer pool — LRU cache for VRAM buffers (eliminates VA reuse race)
    pub buffer_pool: BufferPool,
    /// Kernel compile cache: name → loaded GpuKernel (in-memory)
    kernel_cache: Mutex<HashMap<String, Arc<GpuKernel>>>,
    /// Persistent kernel cache directory (~/.t0/kernel_cache).
    kernel_cache_dir: PathBuf,
    /// Next kernarg slot (monotonically increasing, wraps around pool size)
    slot_counter: Mutex<usize>,
    /// BF16 weight cache for GEMM ops: tensor_id → bf16 buffer
    pub bf16_cache: Mutex<HashMap<u64, Arc<GpuBuffer>>>,
    /// Kernel args metadata cache: name → (args, kernarg_size, wg_size)
    args_cache: Mutex<HashMap<String, CachedKernelInfo>>,
    /// Queue poisoned flag: set after GPU timeout/reset to prevent further dispatches
    /// that would cause cascading hangs on the already-corrupted queue.
    poisoned: std::sync::atomic::AtomicBool,
    /// Deferred synchronization mode: when true, dispatch() uses submit_batch()
    /// (no doorbell ring) instead of submit() + wait_idle(). Caller must call
    /// end_defer_sync() to ring doorbell once and synchronize.
    defer_sync: std::sync::atomic::AtomicBool,
    /// Number of dispatches submitted in defer_sync mode (for batch doorbell).
    defer_count: std::sync::atomic::AtomicU32,
}

#[cfg(feature = "rocm")]
impl GpuRuntime {
    /// Create a new GpuRuntime.
    ///
    /// Opens the first KFD GPU device, creates a queue and dispatch pool.
    fn cache_dir() -> PathBuf {
        let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
        PathBuf::from(home).join(".t0").join("kernel_cache")
    }

    pub fn new() -> Result<Arc<Self>, String> {
        let device = KfdDevice::open()?;
        let queue = device.create_queue()?;
        let pool = DispatchPool::new(&device, 64)?;
        let cache_dir = Self::cache_dir();
        let _ = fs::create_dir_all(&cache_dir);

        Ok(Arc::new(Self {
            buffer_pool: BufferPool::new(&device),
            device,
            queue,
            pool,
            kernel_cache: Mutex::new(HashMap::new()),
            kernel_cache_dir: cache_dir,
            slot_counter: Mutex::new(0),
            bf16_cache: Mutex::new(HashMap::new()),
            args_cache: Mutex::new(HashMap::new()),
            poisoned: std::sync::atomic::AtomicBool::new(false),
            defer_sync: std::sync::atomic::AtomicBool::new(false),
            defer_count: std::sync::atomic::AtomicU32::new(0),
        }))
    }

    /// Create with a specific device.
    pub fn with_device(device: Arc<KfdDevice>) -> Result<Arc<Self>, String> {
        let queue = device.create_queue()?;
        let pool = DispatchPool::new(&device, 64)?;
        let cache_dir = Self::cache_dir();
        let _ = fs::create_dir_all(&cache_dir);

        Ok(Arc::new(Self {
            buffer_pool: BufferPool::new(&device),
            device,
            queue,
            pool,
            kernel_cache: Mutex::new(HashMap::new()),
            kernel_cache_dir: cache_dir,
            slot_counter: Mutex::new(0),
            bf16_cache: Mutex::new(HashMap::new()),
            args_cache: Mutex::new(HashMap::new()),
            poisoned: std::sync::atomic::AtomicBool::new(false),
            defer_sync: std::sync::atomic::AtomicBool::new(false),
            defer_count: std::sync::atomic::AtomicU32::new(0),
        }))
    }

    /// Check if queue is poisoned (GPU timeout/reset occurred)
    pub fn is_poisoned(&self) -> bool {
        self.poisoned.load(std::sync::atomic::Ordering::Relaxed)
    }

    fn mark_poisoned(&self) {
        self.poisoned.store(true, std::sync::atomic::Ordering::Relaxed);
        eprintln!("[KFD] ⚠️  Queue POISONED — all subsequent dispatches will fail fast");
    }

    // ── Kernel compilation cache ──

    /// Get a cached kernel by name (returns None if not compiled yet).
    pub fn get_kernel(&self, name: &str) -> Option<Arc<GpuKernel>> {
        let cache = self.kernel_cache.lock().unwrap();
        cache.get(name).cloned()
    }

    /// Ensure a kernel is compiled from a T0Kernel, with automatic ELF compilation.
    ///
    /// Bridges T0 compiler output → Ignis dispatch:
    ///   T0Kernel → .compile(GFX1100) → ELF → GpuKernel::load → cache
    pub fn ensure_kernel_t0(
        &self,
        name: &str,
        builder: impl FnOnce() -> crate::t0::compile::T0Kernel,
        wg_size: [u32; 3],
        lds_override: u32,
    ) -> Result<Arc<GpuKernel>, String> {
        // Check in-memory cache
        {
            let cache = self.kernel_cache.lock().unwrap();
            if let Some(k) = cache.get(name) {
                return Ok(k.clone());
            }
        }
        // Check disk cache
        if let Some(k) = self.try_load_disk_cache(name) {
            return Ok(k);
        }

        let t0k = builder();
        let wg_actual = [t0k.wg_size(), 1, 1];
        let elf = t0k.compile(crate::t0::ir::Target::GFX1100)?;
        let lds = if lds_override > 0 { lds_override } else { t0k.lds_size() };
        let wg = if wg_size[0] > 0 { wg_size } else { wg_actual };
        let config = KernelLoadConfig { workgroup_size: wg, lds_size: lds };

        // Auto-cache args metadata
        let args_meta: Vec<crate::t0::dsl::KernArgMeta> = t0k.args().iter().map(|a| {
            crate::t0::dsl::KernArgMeta {
                name: a.name.clone(),
                kind: match a.kind {
                    crate::t0::ir::ArgKind::Ptr => crate::t0::dsl::KernArgType::Ptr,
                    crate::t0::ir::ArgKind::U32 => crate::t0::dsl::KernArgType::U32,
                    crate::t0::ir::ArgKind::F32 => crate::t0::dsl::KernArgType::F32,
                },
                offset: a.offset as usize,
            }
        }).collect();

        {
            let mut ac = self.args_cache.lock().unwrap();
            ac.insert(name.to_string(), CachedKernelInfo {
                args: args_meta,
                kernarg_size: t0k.kernarg_size() as usize,
                workgroup_size: config.workgroup_size,
            });
        }

        // Save to disk cache
        self.save_kernel_cache(name, &elf, wg, lds);

        let kernel = GpuKernel::load(&self.device, &elf, &config)?;
        let kernel_arc = Arc::new(kernel);
        {
            let mut cache = self.kernel_cache.lock().unwrap();
            cache.insert(name.to_string(), kernel_arc.clone());
        }
        Ok(kernel_arc)
    }

    /// Ensure a kernel is compiled from a BlockDSL BlockKernel, with SSA compilation.
    ///
    /// Bridges T0 BlockDSL pipeline → Ignis dispatch:
    ///   BlockKernel → compile_via_ssa(GFX1100) → ELF → GpuKernel::load → cache
    ///
    /// This is the preferred path for new kernels (replaces ensure_kernel_t0 for non-legacy ops).
    pub fn ensure_kernel_blockdsl(
        &self,
        name: &str,
        builder: impl FnOnce() -> crate::t0::block_dsl::BlockKernel,
    ) -> Result<Arc<GpuKernel>, String> {
        // Check in-memory cache
        {
            let cache = self.kernel_cache.lock().unwrap();
            if let Some(k) = cache.get(name) {
                return Ok(k.clone());
            }
        }
        // Check disk cache
        if let Some(k) = self.try_load_disk_cache(name) {
            return Ok(k);
        }

        let kb = builder();
        let ck = kb.compile_via_ssa(crate::t0::ir::Target::GFX1100)
            .map_err(|e| format!("BlockDSL compile '{}': {}", name, e))?;

        let config = KernelLoadConfig {
            workgroup_size: ck.workgroup_size,
            lds_size: ck.lds_size,
        };

        // Save to disk cache
        self.save_kernel_cache(name, &ck.elf, ck.workgroup_size, ck.lds_size);

        let kernel = GpuKernel::load(&self.device, &ck.elf, &config)?;
        let kernel_arc = Arc::new(kernel);
        {
            let mut cache = self.kernel_cache.lock().unwrap();
            cache.insert(name.to_string(), kernel_arc.clone());
        }
        Ok(kernel_arc)
    }

    /// Load a pre-compiled kernel from ELF bytes and cache it.
    /// Used for kernels compiled via TileSSA directly (not BlockDSL).
    ///
    /// The `builder` closure returns `(elf_bytes, workgroup_size, lds_size)`.
    pub fn ensure_kernel_precompiled<F>(
        &self,
        name: &str,
        builder: F,
    ) -> Result<Arc<GpuKernel>, String>
    where F: FnOnce() -> Result<(Vec<u8>, [u32; 3], u32), String> {
        // Check in-memory cache
        {
            let cache = self.kernel_cache.lock().unwrap();
            if let Some(k) = cache.get(name) {
                return Ok(k.clone());
            }
        }
        // Check disk cache
        if let Some(k) = self.try_load_disk_cache(name) {
            return Ok(k);
        }

        let (elf, wg_size, lds_size) = builder()?;
        let config = KernelLoadConfig {
            workgroup_size: wg_size,
            lds_size,
        };

        // Save to disk cache
        self.save_kernel_cache(name, &elf, wg_size, lds_size);

        let kernel = GpuKernel::load(&self.device, &elf, &config)?;
        let kernel_arc = Arc::new(kernel);
        {
            let mut cache = self.kernel_cache.lock().unwrap();
            cache.insert(name.to_string(), kernel_arc.clone());
        }
        Ok(kernel_arc)
    }

    // ── Dispatch helpers ──

    /// Allocate a kernarg slot and return its index.
    fn next_slot(&self) -> usize {
        let mut counter = self.slot_counter.lock().unwrap();
        let slot = *counter;
        *counter = (*counter + 1) % 256; // wrap around, pool grows as needed
        slot
    }

    /// Dispatch a kernel with the given grid size and kernarg data.
    ///
    /// When `defer_sync` mode is active (via `begin_defer_sync()`), this uses
    /// `submit()` (write AQL packet + ring doorbell) without `wait_idle()`.
    /// The GPU starts processing immediately, but the CPU doesn't block.
    /// Call `end_defer_sync()` to wait for all batched dispatches.
    ///
    /// Without defer_sync: synchronous dispatch (submit + wait_idle).
    /// With defer_sync: fire-and-forget dispatch (submit, no wait).
    pub fn dispatch(
        &self,
        kernel: &GpuKernel,
        grid: [u32; 3],
        kernargs: &[u8],
    ) -> Result<(), String> {
        if self.is_poisoned() {
            return Err("[KFD] Queue poisoned after GPU hang — refusing dispatch to prevent system hang".into());
        }

        if std::env::var("T0_VALIDATE_KA").is_ok() {
            Self::validate_kernarg_pointers(kernargs);
        }

        let slot = self.next_slot();
        let ka_buf = self.pool.write_kernargs(slot, kernargs);

        if self.defer_sync.load(std::sync::atomic::Ordering::Relaxed) {
            self.queue.submit(kernel, grid, ka_buf);
            self.defer_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Ok(())
        } else {
            self.queue.submit(kernel, grid, ka_buf);
            self.queue.wait_idle().map_err(|e| {
                self.mark_poisoned();
                e
            })
        }
    }

    /// Benchmark-optimized dispatch: AGENT fence scope + spin-wait.
    ///
    /// Uses `submit_fast` (FENCE_SCOPE_AGENT, no ring overflow check) and
    /// `wait_idle_spin` (tight spin, no timeout). Measures pure kernel
    /// execution time without PCIe writeback or Mutex overhead.
    ///
    /// **Only for benchmarks!** Not safe for production (no timeout, no poison check).
    pub fn dispatch_bench(
        &self,
        kernel: &GpuKernel,
        grid: [u32; 3],
        kernargs: &[u8],
    ) {
        let slot = self.next_slot();
        let ka_buf = self.pool.write_kernargs(slot, kernargs);
        self.queue.submit_fast(kernel, grid, ka_buf);
        self.queue.wait_idle_spin();
    }

    /// Validate GPU pointers in kernarg buffer.
    /// Scans for 8-byte aligned u64 values that look like GPU addresses.
    /// Logs warnings for zero pointers and out-of-range addresses.
    fn validate_kernarg_pointers(kernargs: &[u8]) {
        let mut offset = 0;
        while offset + 8 <= kernargs.len() {
            let val = u64::from_le_bytes(kernargs[offset..offset+8].try_into().unwrap());
            // Heuristic: values > 0x1_0000 and non-scalar are likely GPU pointers
            if val > 0x1_0000 && val < u32::MAX as u64 {
                // Likely a pair of u32 scalars, skip
            } else if val >= 0x1_0000_0000 {
                // Looks like a GPU address — validate range
                if val < 0x1_0000_0000 || val > 0x100_0000_0000 {
                    eprintln!("[KA VALIDATE] ⚠️  Suspicious GPU VA at offset {}: 0x{:X} (outside expected VRAM range)", offset, val);
                }
            }
            offset += 8;
        }
    }

    /// Dispatch a kernel asynchronously (no wait).
    ///
    /// Uses `submit` (with ring space safety check + SYSTEM fence) to ensure
    /// the ring buffer doesn't overflow. The caller is responsible for
    /// final synchronization via `wait_idle()` or `synchronize()`.
    ///
    /// Returns the kernarg slot index (for debugging).
    pub fn dispatch_async(
        &self,
        kernel: &GpuKernel,
        grid: [u32; 3],
        kernargs: &[u8],
    ) -> usize {
        let slot = self.next_slot();
        let ka_buf = self.pool.write_kernargs(slot, kernargs);
        self.queue.submit(kernel, grid, ka_buf);
        slot
    }

    /// Dispatch a fused elementwise kernel.
    ///
    /// Used for gradient accumulation (add two buffers in-place).
    ///
    /// # Arguments
    /// - `plan`: the compiled kernel for the operation
    /// - `inputs`: GPU addresses of input buffers
    /// - `output`: GPU address of output buffer (can alias input for in-place)
    /// - `n_elems`: number of f32 elements to process
    pub fn dispatch_fused(
        &self,
        plan: &GpuKernel,
        inputs: &[u64],
        output: u64,
        n_elems: usize,
    ) -> Result<(), String> {
        // Standard elementwise kernarg layout: [input0_ptr, input1_ptr, output_ptr, n_elems]
        let mut ka = [0u8; 32]; // up to 4 args × 8 bytes
        let mut offset = 0;
        for &addr in inputs {
            ka[offset..offset+8].copy_from_slice(&addr.to_le_bytes());
            offset += 8;
        }
        ka[offset..offset+8].copy_from_slice(&output.to_le_bytes());
        offset += 8;
        ka[offset..offset+4].copy_from_slice(&(n_elems as u32).to_le_bytes());

        let wg_size = 256u32;
        let grid_x = ((n_elems as u32 + wg_size - 1) / wg_size) * wg_size;
        self.dispatch(plan, [grid_x, 1, 1], &ka[..offset+4])
    }

    /// Synchronize — wait for all pending GPU work to complete.
    pub fn synchronize(&self) -> Result<(), String> {
        self.queue.synchronize()
    }

    /// Wait for GPU to be idle.
    pub fn wait_idle(&self) -> Result<(), String> {
        self.queue.wait_idle()
    }

    // ── Deferred synchronization (async dispatch batching) ──

    /// Begin deferred synchronization mode.
    ///
    /// After calling this, all `dispatch()` calls will submit AQL packets
    /// (with doorbell ring) but skip `wait_idle()`. The GPU processes
    /// dispatches immediately, but the CPU doesn't block on each one.
    ///
    /// Call `end_defer_sync()` to wait for all submitted dispatches.
    ///
    /// This eliminates per-dispatch `wait_idle()` polling overhead, allowing
    /// the CPU to prepare the next dispatch while the GPU executes the current
    /// one (CPU/GPU overlap).
    ///
    /// Typical usage (one transformer layer):
    /// ```ignore
    /// runtime.begin_defer_sync();
    /// // ... ~19 dispatches via ops that call runtime.dispatch() ...
    /// runtime.end_defer_sync()?;  // single wait_idle for all 19
    /// ```
    pub fn begin_defer_sync(&self) {
        self.defer_sync.store(true, std::sync::atomic::Ordering::Relaxed);
        self.defer_count.store(0, std::sync::atomic::Ordering::Relaxed);
    }

    /// End deferred synchronization mode.
    ///
    /// Waits for all dispatches submitted during defer_sync mode to complete.
    /// If no dispatches were submitted, this is a no-op.
    ///
    /// After this call, `dispatch()` reverts to synchronous behavior
    /// (submit + wait_idle per dispatch).
    pub fn end_defer_sync(&self) -> Result<(), String> {
        let count = self.defer_count.load(std::sync::atomic::Ordering::Relaxed);
        self.defer_sync.store(false, std::sync::atomic::Ordering::Relaxed);
        self.defer_count.store(0, std::sync::atomic::Ordering::Relaxed);

        if count > 0 {
            self.queue.wait_idle().map_err(|e| {
                self.mark_poisoned();
                e
            })
        } else {
            Ok(())
        }
    }

    // ── BF16 weight cache ──

    /// Get or create a BF16 copy of an f32 weight buffer.
    ///
    /// WMMA GEMM requires bf16 inputs. This caches the conversion
    /// to avoid re-converting every forward pass.
    pub fn get_or_create_bf16(
        &self,
        tensor_id: u64,
        f32_buf: &GpuBuffer,
        n_elems: usize,
        convert_kernel: &GpuKernel,
    ) -> Result<Arc<GpuBuffer>, String> {
        let mut cache = self.bf16_cache.lock().unwrap();

        if let Some(bf16) = cache.get(&tensor_id) {
            return Ok(bf16.clone());
        }

        // Allocate bf16 buffer (2 bytes per element, padded to 256)
        let bf16_bytes = ((n_elems * 2) + 255) & !255;
        let bf16_buf = self.device.alloc_vram(bf16_bytes)?;

        // Dispatch f32→bf16 conversion
        let epl = ((n_elems + 255) / 256) as u32;
        let mut ka = [0u8; 24];
        ka[0..8].copy_from_slice(&f32_buf.gpu_addr().to_le_bytes());
        ka[8..16].copy_from_slice(&bf16_buf.gpu_addr().to_le_bytes());
        ka[16..20].copy_from_slice(&epl.to_le_bytes());
        self.dispatch(convert_kernel, [256, 1, 1], &ka[..20])?;

        let bf16_arc = Arc::new(bf16_buf);
        cache.insert(tensor_id, bf16_arc.clone());
        Ok(bf16_arc)
    }

    /// Invalidate BF16 cache for a tensor (call after weight update).
    pub fn invalidate_bf16(&self, tensor_id: u64) {
        let mut cache = self.bf16_cache.lock().unwrap();
        cache.remove(&tensor_id);
    }

    /// Clear entire BF16 cache.
    pub fn clear_bf16_cache(&self) {
        let mut cache = self.bf16_cache.lock().unwrap();
        cache.clear();
    }

    // ── Buffer allocation convenience ──

    /// Allocate a VRAM buffer of `n_bytes` bytes (from buffer pool).
    pub fn alloc(&self, n_bytes: usize) -> Result<GpuBuffer, String> {
        self.buffer_pool.alloc(n_bytes)
    }

    /// Allocate a zero-initialized VRAM buffer (from buffer pool).
    pub fn alloc_zero(&self, n_bytes: usize) -> Result<GpuBuffer, String> {
        let buf = self.buffer_pool.alloc(n_bytes)?;
        buf.zero();
        Ok(buf)
    }

    /// Allocate an f32 buffer for `n` elements, padded to 512 bytes minimum (from buffer pool).
    pub fn alloc_f32(&self, n: usize) -> Result<GpuBuffer, String> {
        let bytes = ((n * 4).max(512) + 511) & !511;
        let buf = self.buffer_pool.alloc(bytes)?;
        buf.zero();
        Ok(buf)
    }

    /// Return a buffer to the pool for reuse (instead of dropping/freeing).
    pub fn recycle(&self, buf: GpuBuffer) {
        self.buffer_pool.recycle(buf);
    }

    // ── Debug helpers ──

    /// Read f32 values from a GPU buffer.
    pub fn read_f32(&self, buf: &GpuBuffer, n: usize) -> Vec<f32> {
        let _ = self.queue.wait_idle();
        let mut data = vec![0f32; n];
        buf.read(unsafe {
            std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut u8, n * 4)
        });
        data
    }

    /// Write f32 values to a GPU buffer.
    pub fn write_f32(&self, buf: &GpuBuffer, data: &[f32]) {
        buf.write(unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4)
        });
    }

    /// Allocate a GPU buffer and upload f32 data in one step.
    ///
    /// Equivalent to `alloc_f32(n) + write_f32(buf, data)`.
    pub fn upload_f32(&self, data: &[f32]) -> Result<GpuBuffer, String> {
        let buf = self.alloc_f32(data.len())?;
        self.write_f32(&buf, data);
        Ok(buf)
    }

    /// Compile and load a DSL-produced CompiledKernel (T0 compiler output).
    ///
    /// Caches by name. Used for DSL pipeline: `dsl_lower::lower()` → `CompiledKernel` → `compile_dsl()`.
    pub fn compile_dsl(
        &self,
        kernel: crate::t0::dsl::CompiledKernel,
    ) -> Result<Arc<GpuKernel>, String> {
        let mut cache = self.kernel_cache.lock().unwrap();
        if let Some(k) = cache.get(&kernel.name) {
            return Ok(k.clone());
        }

        let config = KernelLoadConfig {
            workgroup_size: kernel.workgroup_size,
            lds_size: kernel.lds_size,
        };
        let gpu_kernel = GpuKernel::load(&self.device, &kernel.elf, &config)?;
        let arc = Arc::new(gpu_kernel);
        cache.insert(kernel.name.clone(), arc.clone());

        // Cache args metadata for build_kernargs
        let mut args_cache = self.args_cache.lock().unwrap();
        args_cache.insert(kernel.name.clone(), CachedKernelInfo {
            args: kernel.args,
            kernarg_size: kernel.kernarg_size,
            workgroup_size: kernel.workgroup_size,
        });

        Ok(arc)
    }


    /// Get cached args metadata for a kernel (for manual build_kernargs).
    /// Load a compiled kernel from disk cache. Returns (elf, wg_size, lds_size).
    fn load_kernel_cache(&self, name: &str) -> Option<(Vec<u8>, [u32; 3], u32)> {
        let safe_name = name.replace(|c: char| !c.is_alphanumeric() && c != '_', "_");
        let path = self.kernel_cache_dir.join(&safe_name);
        let mut f = fs::File::open(&path).ok()?;
        let mut header = [0u8; 24];
        f.read_exact(&mut header).ok()?;
        if &header[0..4] != b"T0KC" { return None; }
        let elf_len = u64::from_le_bytes(header[4..12].try_into().ok()?) as usize;
        let wg_x = u32::from_le_bytes(header[12..16].try_into().ok()?);
        let wg_y = u32::from_le_bytes(header[16..20].try_into().ok()?);
        let wg_z = u32::from_le_bytes(header[20..24].try_into().ok()?);
        // lds_size follows
        let mut lds_bytes = [0u8; 4];
        f.read_exact(&mut lds_bytes).ok()?;
        let lds = u32::from_le_bytes(lds_bytes);
        let mut elf = vec![0u8; elf_len];
        f.read_exact(&mut elf).ok()?;
        Some((elf, [wg_x, wg_y, wg_z], lds))
    }

    /// Save a compiled kernel to disk cache.
    fn save_kernel_cache(&self, name: &str, elf: &[u8], wg_size: [u32; 3], lds: u32) {
        let safe_name = name.replace(|c: char| !c.is_alphanumeric() && c != '_', "_");
        let path = self.kernel_cache_dir.join(&safe_name);
        if let Ok(mut f) = fs::File::create(&path) {
            let _ = f.write_all(b"T0KC");
            let _ = f.write_all(&(elf.len() as u64).to_le_bytes());
            let _ = f.write_all(&wg_size[0].to_le_bytes());
            let _ = f.write_all(&wg_size[1].to_le_bytes());
            let _ = f.write_all(&wg_size[2].to_le_bytes());
            let _ = f.write_all(&lds.to_le_bytes());
            let _ = f.write_all(elf);
        }
    }

    /// Try to load a kernel from disk cache into in-memory cache. Returns true if loaded.
    fn try_load_disk_cache(&self, name: &str) -> Option<Arc<GpuKernel>> {
        let (elf, wg, lds) = self.load_kernel_cache(name)?;
        let config = KernelLoadConfig { workgroup_size: wg, lds_size: lds };
        let kernel = GpuKernel::load(&self.device, &elf, &config).ok()?;
        let arc = Arc::new(kernel);
        let mut cache = self.kernel_cache.lock().unwrap();
        cache.insert(name.to_string(), arc.clone());
        Some(arc)
    }

    pub fn get_kernel_info(&self, name: &str) -> Option<CachedKernelInfo> {
        let cache = self.args_cache.lock().unwrap();
        cache.get(name).cloned()
    }
}

/// Cached kernel metadata for type-safe dispatch.
#[cfg(feature = "rocm")]
#[derive(Clone, Debug)]
pub struct CachedKernelInfo {
    pub args: Vec<crate::t0::dsl::KernArgMeta>,
    pub kernarg_size: usize,
    pub workgroup_size: [u32; 3],
}

// ── kernargs! macro ──

/// Build a kernarg byte array from typed values.
///
/// Usage:
/// ```rust
/// let ka = kernargs![
///     input_ptr => u64,
///     output_ptr => u64,
///     n_elems => u32,
///     scale => f32,
/// ];
/// ```
///
/// Supports u32, u64, f32, i32 types.
#[macro_export]
macro_rules! kernargs {
    ($($val:expr => $ty:ty),* $(,)?) => {{
        let mut _ka = Vec::new();
        $(
            _ka.extend_from_slice(&<$ty>::to_le_bytes($val as $ty));
        )*
        _ka
    }};
}

/// Build a fixed-size kernarg byte array (stack-allocated).
///
/// Usage:
/// ```rust
/// let ka = kernargs_fixed!(size=32;
///     0 => input_addr: u64,
///     8 => output_addr: u64,
///     16 => n_elems: u32,
///     20 => scale: f32,
/// );
/// ```
#[macro_export]
macro_rules! kernargs_fixed {
    (size=$size:expr; $($offset:expr => $val:expr => $ty:ty),* $(,)?) => {{
        let mut _ka = [0u8; $size];
        $(
            _ka[$offset..$offset + std::mem::size_of::<$ty>()]
                .copy_from_slice(&<$ty>::to_le_bytes($val as $ty));
        )*
        _ka
    }};
}

