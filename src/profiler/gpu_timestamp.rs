//! GPU timestamp infrastructure.
//!
//! Currently provides CPU-side timing with dispatch overhead estimation.
//! GPU hardware timestamps (s_getreg SHADER_CYCLES) can be added later
//! as an enhancement when SMEM store instructions are available in rdna3_asm.rs.

/// Estimated AQL dispatch overhead per kernel (ns).
///
/// Measured: submit + doorbell + wait_idle polling ≈ 2μs on RX 7900 XTX.
/// This is subtracted from CPU wall time to estimate GPU execution time.
pub const DISPATCH_OVERHEAD_NS: u64 = 2_000;

/// Placeholder for future GPU hardware timestamp support.
#[cfg(feature = "rocm")]
pub struct GpuTimestamp {
    // Future: VRAM buffer for s_getreg SHADER_CYCLES writes
}

#[cfg(feature = "rocm")]
impl GpuTimestamp {
    pub fn new(_runtime: &crate::ignis::gpu_context::GpuRuntime) -> Result<Self, String> {
        Ok(Self {})
    }
}

/// Estimate GPU execution time from CPU wall time.
///
/// Subtracts estimated dispatch overhead (submit + wait_idle).
/// Returns 0 if cpu_ns < overhead (very short kernels).
pub fn estimate_gpu_ns(cpu_ns: u64) -> u64 {
    cpu_ns.saturating_sub(DISPATCH_OVERHEAD_NS)
}
