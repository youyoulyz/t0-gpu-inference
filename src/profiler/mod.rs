//! GPU Profiler — CPU/GPU interaction timeline with I/O shape tracking.
//!
//! Enabled via `--features profile`. Zero-cost when disabled.
//!
//! Chrome tracing output has two tracks:
//! - tid=0: CPU (dispatch + wait)
//! - tid=1: GPU (kernel execution, estimated or measured)
//!
//! # Usage
//!
//! ```ignore
//! use t0_gpu::profiler;
//!
//! profiler::begin("attention_qk");
//! profiler::set_shapes(
//!     vec![profiler::ShapeInfo::new(&[32, 128, 128])],
//!     vec![profiler::ShapeInfo::new(&[32, 128, 128])],
//! );
//! // ... dispatch kernels ...
//! profiler::end("attention_qk");
//!
//! profiler::report();        // human-readable table
//! profiler::to_json();       // Chrome tracing JSON
//! ```

mod op_profiler;
mod gpu_timestamp;

pub use op_profiler::{OpProfiler, OpRecord, ShapeInfo};
pub use gpu_timestamp::GpuTimestamp;

use std::sync::{LazyLock, Mutex};

static GLOBAL_PROFILER: LazyLock<Mutex<OpProfiler>> = LazyLock::new(|| Mutex::new(OpProfiler::new()));

fn with_profiler<F: FnOnce(&mut OpProfiler)>(f: F) {
    if let Ok(mut p) = GLOBAL_PROFILER.lock() {
        f(&mut p);
    }
}

/// Begin recording an op. No-op if profiling is disabled.
#[inline]
pub fn begin(name: &'static str) {
    #[cfg(feature = "profile")]
    with_profiler(|p| p.begin(name));
}

/// Set input/output shapes for the currently open op.
#[inline]
pub fn set_shapes(inputs: Vec<ShapeInfo>, outputs: Vec<ShapeInfo>) {
    #[cfg(feature = "profile")]
    with_profiler(|p| p.set_shapes(inputs, outputs));
}

/// End recording an op. No-op if profiling is disabled.
#[inline]
pub fn end(name: &'static str) {
    #[cfg(feature = "profile")]
    with_profiler(|p| p.end(name));
}

/// Record a completed kernel dispatch (single-call variant).
#[inline]
pub fn record_kernel(name: &'static str, cpu_ns: u64) {
    #[cfg(feature = "profile")]
    with_profiler(|p| p.record_kernel(name, cpu_ns));
}

/// Set GPU execution time for the most recent record with the given name.
#[inline]
pub fn record_gpu_timing(name: &'static str, gpu_ns: u64) {
    #[cfg(feature = "profile")]
    with_profiler(|p| p.record_gpu_timing(name, gpu_ns));
}

/// Print human-readable profiling report to stderr.
pub fn report() {
    with_profiler(|p| p.report());
}

/// Export profiling data as Chrome tracing JSON with CPU/GPU dual tracks.
pub fn to_json() -> String {
    GLOBAL_PROFILER.lock().map(|p| p.to_json()).unwrap_or_default()
}

/// Reset all profiling data.
pub fn reset() {
    with_profiler(|p| p.reset());
}

/// Check if profiling is compiled in.
#[inline]
pub fn is_enabled() -> bool {
    cfg!(feature = "profile")
}

/// RAII guard that automatically calls `end()` on drop.
pub struct ProfileGuard {
    name: &'static str,
}

impl ProfileGuard {
    pub fn new(name: &'static str) -> Self {
        begin(name);
        Self { name }
    }
}

impl Drop for ProfileGuard {
    fn drop(&mut self) {
        end(self.name);
    }
}

/// Convenience macro for profiling a scope.
#[macro_export]
macro_rules! profile_scope {
    ($name:expr) => {
        let _guard = $crate::profiler::ProfileGuard::new($name);
    };
}
