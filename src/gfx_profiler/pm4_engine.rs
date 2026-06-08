//! PM4 counter engine — programs hardware performance counters and reads back values.
//!
//! Uses `Pm4CmdBuilder` (from kfd) to construct PM4 sequences:
//! 1. SET_UCONFIG_REG(GRBM_GFX_INDEX) — target specific CU or broadcast
//! 2. SET_SH_REG(SQ_PERFCOUNTER*_SELECT) — select events
//! 3. SET_SH_REG(SQ_PERFCOUNTER_CTRL) — enable CS counters
//! 4. Dispatch kernel via AQL
//! 5. EVENT_WRITE(CS_PARTIAL_FLUSH) — wait for completion
//! 6. RELEASE_MEM(data_sel=1/2) — read counter 0/1 into GTT buffer

use std::sync::Arc;
use crate::kfd::{KfdDevice, AqlQueue, GpuBuffer, GpuKernel, Pm4CmdBuilder};
use super::counter_config::{
    ProfilePass,
    SQ_PERFCOUNTER0_SELECT, SQ_PERFCOUNTER1_SELECT,
    SQ_PERFCOUNTER_CTRL,
    GRBM_PERFCOUNTER0_SELECT, GRBM_PERFCOUNTER1_SELECT,
    TCC_PERFCOUNTER0_SELECT, TCC_PERFCOUNTER1_SELECT,
};
use crate::kfd::{GRBM_GFX_INDEX, CS_PARTIAL_FLUSH, EVENT_INDEX_PARTIAL_FLUSH};

/// Completion signal magic value.
const COMPLETION_MAGIC: u64 = 0xDEADBEEF_CAFEBABE;

/// Number of u64 slots in the readback buffer per pass.
/// Max 6 counters (2 SQ + 2 GRBM + 2 TCC) + 1 completion signal = 7 slots.
const MAX_SLOTS_PER_PASS: usize = 7;

/// PM4 counter programming and readback engine.
pub struct Pm4CounterEngine {
    device: Arc<KfdDevice>,
    /// GTT buffer for counter readback (host-visible, GPU-writable)
    readback_buf: GpuBuffer,
    /// Signal buffer for completion detection
    signal_buf: GpuBuffer,
}

impl Pm4CounterEngine {
    pub fn new(device: &Arc<KfdDevice>) -> Result<Self, String> {
        let readback_buf = device.alloc_uncached(MAX_SLOTS_PER_PASS * 8)?;
        let signal_buf = device.alloc_uncached(8)?;
        Ok(Self {
            device: Arc::clone(device),
            readback_buf,
            signal_buf,
        })
    }

    /// Build PM4 commands to configure performance counters for a single pass.
    ///
    /// All perf counter SELECT registers (SQ, GRBM, TCC) are in the GRBM register
    /// space (0xD000+), NOT the compute SH space (0x2C00). They must be written
    /// via SET_UCONFIG_REG, not SET_SH_REG.
    pub fn build_counter_config_cmds(&self, pass: &ProfilePass, cu_index: Option<u32>) -> Vec<u32> {
        // TODO: PM4 counter config causes GPU hangs on GFX1100.
        // The register addresses or SET_UCONFIG_REG base needs verification.
        // For now, return empty — profiler reports timing-only metrics.
        let _ = (pass, cu_index);
        Vec::new()
    }

    /// Build PM4 commands to read back counter values after kernel execution.
    ///
    /// Uses RELEASE_MEM with data_sel=1 (perfcounter0_lo) and data_sel=2 (perfcounter1_lo).
    /// Layout in readback buffer:
    ///   [0] SQ counter 0   [1] SQ counter 1
    ///   [2] GRBM counter 0 [3] GRBM counter 1
    ///   [4] TCC counter 0  [5] TCC counter 1
    ///   [6] completion signal
    pub fn build_counter_readback_cmds(&self, pass: &ProfilePass) -> Vec<u32> {
        // TODO: counter readback disabled (no counter config).
        let _ = pass;
        Vec::new()
    }

    /// Execute a single profiling pass: configure → dispatch → readback → wait.
    ///
    /// Returns the raw counter values in pass order (SQ0, SQ1, GRBM0, GRBM1, TCC0, TCC1).
    pub fn execute_pass(
        &self,
        queue: &mut AqlQueue,
        kernel: &GpuKernel,
        grid: [u32; 3],
        kernargs: &GpuBuffer,
        pass: &ProfilePass,
        cu_index: Option<u32>,
    ) -> Result<Vec<u64>, String> {
        // Clear readback buffer
        unsafe {
            let ptr = self.readback_buf.host_ptr as *mut u64;
            for i in 0..MAX_SLOTS_PER_PASS {
                std::ptr::write_volatile(ptr.add(i), 0);
            }
        }
        std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);

        // 1. Configure counters
        let config_cmds = self.build_counter_config_cmds(pass, cu_index);
        if !config_cmds.is_empty() {
            queue.submit_pm4(&config_cmds)?;
        }

        // 2. Dispatch kernel (using submit which handles barrier + ring management)
        queue.submit(kernel, grid, kernargs);

        // Wait for kernel completion
        queue.wait_idle()?;

        // 3. Read back counters
        let readback_cmds = self.build_counter_readback_cmds(pass);
        if !readback_cmds.is_empty() {
            queue.submit_pm4(&readback_cmds)?;
            queue.wait_idle()?;
        }

        // 4. Parse readback buffer
        let mut results = Vec::with_capacity(pass.num_counters());
        let ptr = self.readback_buf.host_ptr as *const u32;
        for i in 0..pass.num_counters() {
            let val = unsafe { std::ptr::read_volatile(ptr.add(i * 2)) };
            results.push(val as u64);
        }

        Ok(results)
    }
}
