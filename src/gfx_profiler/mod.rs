//! Bare-metal GPU hardware counter profiler for GFX1100 (RDNA3).
//!
//! Programs GPU performance counters via PM4 commands through the KFD runtime,
//! collects SQ/GRBM/TCC counter values across multiple profiling passes,
//! and generates NCU-equivalent reports with optimization suggestions.
//!
//! # Usage
//!
//! ```ignore
//! use t0_gpu::gfx_profiler::{GfxProfiler, OutputFormat};
//!
//! let mut profiler = GfxProfiler::new()?;
//! let result = profiler.profile_t0_kernel(&kernel, grid, &kernargs, "softmax", None)?;
//! profiler.report(&result, OutputFormat::Text);
//! ```

pub mod counter_config;
pub mod pm4_engine;
pub mod metrics;
pub mod report;
pub mod suggestions;

use std::sync::Arc;
use crate::kfd::{KfdDevice, AqlQueue, GpuBuffer, GpuKernel, DispatchPool};

pub use counter_config::{CounterEvent, ProfilePass, schedule_passes, standard_events};
pub use pm4_engine::Pm4CounterEngine;
pub use metrics::{RawCounters, ProfileMetrics, Bottleneck, HwLimits, compute_metrics};
pub use report::{OutputFormat, generate_report};
pub use suggestions::{OptimizationSuggestion, Severity, generate_suggestions};

/// Complete profiling result for one kernel.
#[derive(Clone, Debug)]
pub struct ProfileResult {
    pub kernel_name: String,
    pub elapsed_ns: u64,
    pub raw_counters: RawCounters,
    pub metrics: ProfileMetrics,
    pub suggestions: Vec<OptimizationSuggestion>,
    pub num_passes: u32,
}

/// Bare-metal GPU hardware counter profiler.
pub struct GfxProfiler {
    device: Arc<KfdDevice>,
    queue: AqlQueue,
    pool: DispatchPool,
    pm4_engine: Pm4CounterEngine,
    hw_limits: HwLimits,
}

impl GfxProfiler {
    /// Access the underlying KFD device (for buffer allocation, etc.)
    pub fn device(&self) -> &Arc<KfdDevice> {
        &self.device
    }

    /// Create a new profiler. Opens a fresh KFD device and PM4 counter engine.
    pub fn new() -> Result<Self, String> {
        let device = KfdDevice::open()?;
        let queue = device.create_queue()?;
        let pool = DispatchPool::new_sized(&device, 32, 512)?;
        let pm4_engine = Pm4CounterEngine::new(&device)?;
        Ok(Self {
            device,
            queue,
            pool,
            pm4_engine,
            hw_limits: HwLimits::default(),
        })
    }

    /// Profile a T0-compiled kernel with the standard event set.
    ///
    /// `kernargs_data` is the raw kernarg bytes (from `build_kernargs`).
    /// The profiler uses DispatchPool to avoid kernargs buffer reuse issues.
    pub fn profile_t0_kernel(
        &mut self,
        kernel: &GpuKernel,
        grid: [u32; 3],
        kernargs_data: &[u8],
        kernel_name: &str,
        total_flops: Option<u64>,
        cu_index: Option<u32>,
    ) -> Result<ProfileResult, String> {
        let events = standard_events();
        self.profile_with_events(kernel, grid, kernargs_data, &events, kernel_name, total_flops, cu_index)
    }

    /// Profile with a custom set of events.
    pub fn profile_with_events(
        &mut self,
        kernel: &GpuKernel,
        grid: [u32; 3],
        kernargs_data: &[u8],
        events: &[CounterEvent],
        kernel_name: &str,
        total_flops: Option<u64>,
        cu_index: Option<u32>,
    ) -> Result<ProfileResult, String> {
        // Schedule events into passes
        let passes = schedule_passes(events);

        // Warmup + timing: single dispatch
        let ka = self.pool.write_kernargs(0, kernargs_data);
        let start = std::time::Instant::now();
        self.queue.submit(kernel, grid, ka);
        self.queue.wait_idle().map_err(|e| format!("dispatch failed: {}", e))?;
        let elapsed_ns = start.elapsed().as_nanos() as u64;

        // Execute profiling passes
        let mut raw = RawCounters::default();
        for (pass_idx, pass) in passes.iter().enumerate() {
            let ka = self.pool.write_kernargs(1 + pass_idx, kernargs_data);
            let values = self.pm4_engine.execute_pass(
                &mut self.queue, kernel, grid, ka, pass, cu_index,
            )?;
            self.map_counter_values(&mut raw, pass, &values);
        }

        // Compute derived metrics
        let metrics = compute_metrics(&raw, elapsed_ns, total_flops, &self.hw_limits);
        let suggestions = generate_suggestions(&metrics);

        Ok(ProfileResult {
            kernel_name: kernel_name.to_string(),
            elapsed_ns,
            raw_counters: raw,
            metrics,
            suggestions,
            num_passes: passes.len() as u32,
        })
    }

    /// Map readback values from a pass into the RawCounters struct.
    fn map_counter_values(&self, raw: &mut RawCounters, pass: &ProfilePass, values: &[u64]) {
        let mut idx = 0;

        // SQ counters
        for slot in 0..2 {
            if pass.sq_events[slot].is_some() {
                if let Some(val) = values.get(idx) {
                    let name = pass.sq_events[slot].unwrap().name;
                    match name {
                        "SQ_WAVES"          => raw.sq_waves = *val,
                        "SQ_INSTS"          => raw.sq_insts = *val,
                        "SQ_INSTS_VALU"     => raw.sq_insts_valu = *val,
                        "SQ_INSTS_SALU"     => raw.sq_insts_salu = *val,
                        "SQ_INSTS_SMEM"     => raw.sq_insts_smem = *val,
                        "SQ_INSTS_FLAT"     => raw.sq_insts_flat = *val,
                        "SQ_INSTS_LDS"      => raw.sq_insts_lds = *val,
                        "SQ_INSTS_VMEM"     => raw.sq_insts_vmem = *val,
                        "SQ_INSTS_MFMA"     => raw.sq_insts_mfma = *val,
                        "SQ_THREAD_CYCLES"  => raw.sq_thread_cycles = *val,
                        "SQ_WAIT_INSTS"     => raw.sq_wait_insts = *val,
                        "SQ_ACTIVE_INSTS"   => raw.sq_active_insts = *val,
                        _ => {}
                    }
                }
                idx += 1;
            }
        }

        // GRBM counters
        for slot in 0..2 {
            if pass.grbm_events[slot].is_some() {
                if let Some(val) = values.get(idx) {
                    let name = pass.grbm_events[slot].unwrap().name;
                    match name {
                        "GRBM_COUNT"       => raw.grbm_count = *val,
                        "GRBM_GUI_ACTIVE"  => raw.grbm_gui_active = *val,
                        _ => {}
                    }
                }
                idx += 1;
            }
        }

        // TCC counters
        for slot in 0..2 {
            if pass.tcc_events[slot].is_some() {
                if let Some(val) = values.get(idx) {
                    let name = pass.tcc_events[slot].unwrap().name;
                    match name {
                        "TCC_REQ"          => raw.tcc_req = *val,
                        "TCC_HIT"          => raw.tcc_hit = *val,
                        "TCC_MISS"         => raw.tcc_miss = *val,
                        "TCC_MC_WRITEREQ"  => raw.tcc_mc_writereq = *val,
                        "TCC_MC_READREQ"   => raw.tcc_mc_readreq = *val,
                        _ => {}
                    }
                }
                idx += 1;
            }
        }
    }

    /// Generate and print a profiling report.
    pub fn report(&self, result: &ProfileResult, format: OutputFormat) {
        let output = generate_report(
            &result.kernel_name,
            &result.metrics,
            &result.suggestions,
            format,
        );
        eprint!("{}", output);
    }
}
