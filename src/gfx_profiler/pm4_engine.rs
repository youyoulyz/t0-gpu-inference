//! PM4 counter engine — programs hardware performance counters and reads back values.
//!
//! PM4 sequence for each profiling pass:
//! 1. SET_UCONFIG_REG(SQ_PERFCOUNTER_CTRL, 0) — disable counters (reset)
//! 2. ACQUIRE_MEM — invalidate caches for clean counter start
//! 3. SET_UCONFIG_REG(GRBM_GFX_INDEX, cu) — target specific CU (for SQ counters)
//! 4. SET_UCONFIG_REG(SQ_PERFCOUNTER*_SELECT, event_id) — select SQ events
//! 5. SET_UCONFIG_REG(GRBM_PERFCOUNTER*_SELECT, event_id) — select GRBM events
//! 6. SET_UCONFIG_REG(TCC_PERFCOUNTER*_SELECT, event_id) — select TCC events
//! 7. SET_UCONFIG_REG(SQ_PERFCOUNTER_CTRL, CS_EN | SPM_EN) — enable CS counters
//! 8. Dispatch kernel via AQL
//! 9. RELEASE_MEM(data_sel=4/5) — read counter 0/1 (64-bit) into GTT buffer
//!
//! All perf counter registers (SQ/GRBM/TCC SELECT, CTRL) are in the GRBM uconfig
//! space (0xD000+), accessed via SET_UCONFIG_REG with raw dword offset (reg_addr >> 2).
//!
//! Safety: set T0_GFX_PROFILER_NO_COUNTERS=1 to skip counter config/readback
//! (profiler will only measure timing). Useful for debugging GPU hangs.

use std::sync::Arc;
use crate::kfd::{KfdDevice, AqlQueue, GpuBuffer, GpuKernel, Pm4CmdBuilder};
use super::counter_config::{
    ProfilePass,
    SQ_PERFCOUNTER_CTRL,
};
use crate::kfd::GRBM_GFX_INDEX;

const MAX_SLOTS_PER_PASS: usize = 12;

pub struct Pm4CounterEngine {
    device: Arc<KfdDevice>,
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
        Ok(Self {
            device: Arc::clone(device),
            readback_buf,
            counters_enabled,
        })
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

        // 2. ACQUIRE_MEM to invalidate caches for clean counter start
        builder.acquire_mem_gfx10();

        // 3. SQ counters: per-CU via GRBM_GFX_INDEX targeting
        if has_sq {
            builder.set_uconfig_reg(GRBM_GFX_INDEX, &[cu]);
            if let Some(evt) = pass.sq_events[0] {
                builder.set_uconfig_reg(evt.select_reg, &[evt.event_id]);
            }
            if let Some(evt) = pass.sq_events[1] {
                builder.set_uconfig_reg(evt.select_reg, &[evt.event_id]);
            }
        }

        // 4. GRBM counters: global, no CU targeting needed
        if has_grbm {
            if let Some(evt) = pass.grbm_events[0] {
                builder.set_uconfig_reg(evt.select_reg, &[evt.event_id]);
            }
            if let Some(evt) = pass.grbm_events[1] {
                builder.set_uconfig_reg(evt.select_reg, &[evt.event_id]);
            }
        }

        // 5. TCC counters: per-channel
        if has_tcc {
            if let Some(evt) = pass.tcc_events[0] {
                builder.set_uconfig_reg(evt.select_reg, &[evt.event_id]);
            }
            if let Some(evt) = pass.tcc_events[1] {
                builder.set_uconfig_reg(evt.select_reg, &[evt.event_id]);
            }
        }

        // 6. Enable CS counters: bit 6 (CS_EN) | bit 0 (SPM_EN)
        builder.set_uconfig_reg(SQ_PERFCOUNTER_CTRL, &[(1u32 << 6) | (1u32 << 0)]);

        builder.finish()
    }

    pub fn build_counter_readback_cmds(&self, pass: &ProfilePass) -> Vec<u32> {
        if !self.counters_enabled {
            return Vec::new();
        }

        let mut builder = Pm4CmdBuilder::new();
        let base = self.readback_buf.gpu_addr();
        let mut slot: u64 = 0;

        // SQ readback: requires GRBM_GFX_INDEX targeting
        if pass.sq_events[0].is_some() || pass.sq_events[1].is_some() {
            let cu = 0xFFFFFFFFu32; // broadcast
            builder.set_uconfig_reg(GRBM_GFX_INDEX, &[cu]);
            for i in 0..2 {
                if pass.sq_events[i].is_some() {
                    let data_sel = if i == 0 { 4u32 } else { 5u32 };
                    builder.release_mem(base + slot * 8, 0, data_sel, 0, false);
                    slot += 1;
                }
            }
        }

        // GRBM readback: global counters
        if pass.grbm_events[0].is_some() || pass.grbm_events[1].is_some() {
            for i in 0..2 {
                if pass.grbm_events[i].is_some() {
                    let data_sel = if i == 0 { 4u32 } else { 5u32 };
                    builder.release_mem(base + slot * 8, 0, data_sel, 0, false);
                    slot += 1;
                }
            }
        }

        // TCC readback: per-channel counters
        if pass.tcc_events[0].is_some() || pass.tcc_events[1].is_some() {
            for i in 0..2 {
                if pass.tcc_events[i].is_some() {
                    let data_sel = if i == 0 { 4u32 } else { 5u32 };
                    builder.release_mem(base + slot * 8, 0, data_sel, 0, false);
                    slot += 1;
                }
            }
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

        // 1. Configure counters
        let config_cmds = self.build_counter_config_cmds(pass, cu_index);
        if !config_cmds.is_empty() {
            queue.submit_pm4(&config_cmds)?;
        }

        // 2. Dispatch kernel
        queue.submit(kernel, grid, kernargs);
        queue.wait_idle()?;

        // 3. Read back counters
        let readback_cmds = self.build_counter_readback_cmds(pass);
        if !readback_cmds.is_empty() {
            queue.submit_pm4(&readback_cmds)?;
            queue.wait_idle()?;
        }

        // 4. Parse readback buffer
        let mut results = Vec::with_capacity(num_counters);
        let ptr = self.readback_buf.host_ptr as *const u64;
        for i in 0..num_counters {
            let val = unsafe { std::ptr::read_volatile(ptr.add(i)) };
            results.push(val);
        }

        Ok(results)
    }
}