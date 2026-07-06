//! PM4 counter engine — programs hardware performance counters and reads back values.
//!
//! PM4 sequence for each profiling pass:
//! 1. SET_UCONFIG_REG(SQ_PERFCOUNTER_CTRL, 0) — disable counters (reset)
//! 2. SET_UCONFIG_REG(SQ_PERFCOUNTER*_SELECT, event_id) — select events
//! 3. SET_UCONFIG_REG(GRBM_PERFCOUNTER*_SELECT, event_id) — select GRBM events
//! 4. SET_UCONFIG_REG(TCC_PERFCOUNTER*_SELECT, event_id) — select TCC events
//! 5. SET_UCONFIG_REG(SQ_PERFCOUNTER_CTRL, CS_EN) — enable CS counters
//! 6. Dispatch kernel via AQL
//! 7. RELEASE_MEM(data_sel=4/5) — read counter 0/1 (64-bit) into GTT buffer
//!
//! NOTE: We do NOT use ACQUIRE_MEM. Inside an INDIRECT_BUFFER, ACQUIRE_MEM can
//! deadlock the CP because the CP may treat the IB fetch itself as a pending
//! memory operation. Counter config is pure register writes; no cache invalidation
//! is needed.
//!
//! NOTE: GRBM_GFX_INDEX is used for per-CU SQ targeting. After SQ config and
//! TCC config, GRBM_GFX_INDEX is reset to broadcast (0xFFFFFFFF) to avoid
//! leaking instance state.
//!
//! NOTE: We use CS_EN only (bit 6), NOT SPM_EN (bit 0). SPM_EN requires the
//! kernel's COMPUTE_PGM_RSRC1 register to have PERF_CNT_EN set, which external
//! HIP kernels may not set — using SPM_EN can cause GPU hangs.
//!
//! Safety: set T0_GFX_PROFILER_NO_COUNTERS=1 to skip counter config/readback
//! (profiler will only measure timing).

use std::sync::Arc;
use crate::kfd::{KfdDevice, AqlQueue, GpuBuffer, GpuKernel, Pm4CmdBuilder};
use super::counter_config::{ProfilePass, SQ_PERFCOUNTER_CTRL};
use crate::kfd::GRBM_GFX_INDEX;

const MAX_SLOTS_PER_PASS: usize = 12;

pub struct Pm4CounterEngine {
    readback_buf: GpuBuffer,
    counters_enabled: bool,
}

impl Pm4CounterEngine {
    pub fn new(device: &Arc<KfdDevice>) -> Result<Self, String> {
        let readback_buf = device.alloc_uncached(MAX_SLOTS_PER_PASS * 8)?;
        let counters_enabled = std::env::var("T0_GFX_PROFILER_NO_COUNTERS").is_err();
        if !counters_enabled {
            eprintln!("[profiler] T0_GFX_PROFILER_NO_COUNTERS=1 — hardware counters disabled, timing only");
        }
        Ok(Self { readback_buf, counters_enabled })
    }

    pub fn build_counter_config_cmds(&self, pass: &ProfilePass, cu_index: Option<u32>) -> Vec<u32> {
        if !self.counters_enabled {
            return Vec::new();
        }

        let mut builder = Pm4CmdBuilder::new();
        let cu = cu_index.unwrap_or(0);

        let has_sq = pass.sq_events[0].is_some() || pass.sq_events[1].is_some();
        let has_grbm = pass.grbm_events[0].is_some() || pass.grbm_events[1].is_some();
        let has_tcc = pass.tcc_events[0].is_some() || pass.tcc_events[1].is_some();

        let any_counters = has_sq || has_grbm || has_tcc;
        if !any_counters {
            return Vec::new();
        }

        // 1. Disable all counters first (reset state)
        builder.set_uconfig_reg(SQ_PERFCOUNTER_CTRL, &[0u32]);

        // 2. SQ counters: per-CU via GRBM_GFX_INDEX targeting
        if has_sq {
            builder.set_uconfig_reg(GRBM_GFX_INDEX, &[cu]);
            if let Some(evt) = pass.sq_events[0] {
                builder.set_uconfig_reg(evt.select_reg, &[evt.event_id]);
            }
            if let Some(evt) = pass.sq_events[1] {
                builder.set_uconfig_reg(evt.select_reg, &[evt.event_id]);
            }
            builder.set_uconfig_reg(GRBM_GFX_INDEX, &[0xFFFFFFFFu32]);
        }

        // 3. GRBM counters: global, no CU targeting needed
        if has_grbm {
            if let Some(evt) = pass.grbm_events[0] {
                builder.set_uconfig_reg(evt.select_reg, &[evt.event_id]);
            }
            if let Some(evt) = pass.grbm_events[1] {
                builder.set_uconfig_reg(evt.select_reg, &[evt.event_id]);
            }
        }

        // 4. TCC counters: per-channel
        if has_tcc {
            builder.set_uconfig_reg(GRBM_GFX_INDEX, &[0x00000000u32]);
            if let Some(evt) = pass.tcc_events[0] {
                builder.set_uconfig_reg(evt.select_reg, &[evt.event_id]);
            }
            if let Some(evt) = pass.tcc_events[1] {
                builder.set_uconfig_reg(evt.select_reg, &[evt.event_id]);
            }
            builder.set_uconfig_reg(GRBM_GFX_INDEX, &[0xFFFFFFFFu32]);
        }

        // 5. Enable CS counters: bit 6 (CS_EN) only
        builder.set_uconfig_reg(SQ_PERFCOUNTER_CTRL, &[1u32 << 6]);

        builder.finish()
    }

    pub fn build_counter_readback_cmds(&self, pass: &ProfilePass) -> Vec<u32> {
        if !self.counters_enabled {
            return Vec::new();
        }

        let mut builder = Pm4CmdBuilder::new();
        let base = self.readback_buf.gpu_addr();
        let mut slot: u64 = 0;

        if pass.sq_events[0].is_some() || pass.sq_events[1].is_some() {
            builder.set_uconfig_reg(GRBM_GFX_INDEX, &[0xFFFFFFFFu32]);
            for i in 0..2 {
                if pass.sq_events[i].is_some() {
                    builder.release_mem(base + slot * 8, 0, if i == 0 { 4 } else { 5 }, 0, false);
                    slot += 1;
                }
            }
        }

        if pass.grbm_events[0].is_some() || pass.grbm_events[1].is_some() {
            for i in 0..2 {
                if let Some(evt) = pass.grbm_events[i] {
                    builder.copy_data_reg(evt.value_lo_reg, base + slot * 8);
                    builder.copy_data_reg(evt.value_lo_reg + 4, base + slot * 8 + 4);
                    slot += 1;
                }
            }
        }

        if pass.tcc_events[0].is_some() || pass.tcc_events[1].is_some() {
            builder.set_uconfig_reg(GRBM_GFX_INDEX, &[0x00000000u32]);
            for i in 0..2 {
                if let Some(evt) = pass.tcc_events[i] {
                    builder.copy_data_reg(evt.value_lo_reg, base + slot * 8);
                    builder.copy_data_reg(evt.value_lo_reg + 4, base + slot * 8 + 4);
                    slot += 1;
                }
            }
            builder.set_uconfig_reg(GRBM_GFX_INDEX, &[0xFFFFFFFFu32]);
        }

        builder.finish()
    }

    pub fn execute_pass(
        &self,
        queue: &mut AqlQueue,
        kernel: &GpuKernel,
        grid: [u32; 3],
        kernargs: &GpuBuffer,
        pass: &ProfilePass,
        cu_index: Option<u32>,
    ) -> Result<Vec<u64>, String> {
        unsafe {
            let ptr = self.readback_buf.host_ptr as *mut u64;
            for i in 0..MAX_SLOTS_PER_PASS {
                std::ptr::write_volatile(ptr.add(i), 0);
            }
        }
        std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);

        let num_counters = pass.num_counters();

        let config_cmds = self.build_counter_config_cmds(pass, cu_index);
        if !config_cmds.is_empty() {
            queue.submit_pm4(&config_cmds)?;
        }

        queue.submit(kernel, grid, kernargs);
        queue.wait_idle()?;

        let readback_cmds = self.build_counter_readback_cmds(pass);
        if !readback_cmds.is_empty() {
            queue.submit_pm4(&readback_cmds)?;
            queue.wait_idle()?;
        }

        let mut results = Vec::with_capacity(num_counters);
        let ptr = self.readback_buf.host_ptr as *const u64;
        for i in 0..num_counters {
            let val = unsafe { std::ptr::read_volatile(ptr.add(i)) };
            results.push(val);
        }

        Ok(results)
    }
}