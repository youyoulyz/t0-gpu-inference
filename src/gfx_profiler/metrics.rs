//! Derived metrics computation from raw hardware counter values.
//!
//! Transforms raw SQ/GRBM/TCC counter values into NCU-equivalent metrics:
//! IPC, occupancy, cache hit rates, memory bandwidth, roofline analysis.

/// Raw counter values aggregated from all profiling passes.
#[derive(Clone, Debug, Default)]
pub struct RawCounters {
    // SQ per-CU counters
    pub sq_waves: u64,
    pub sq_insts: u64,
    pub sq_insts_valu: u64,
    pub sq_insts_salu: u64,
    pub sq_insts_smem: u64,
    pub sq_insts_flat: u64,
    pub sq_insts_lds: u64,
    pub sq_insts_vmem: u64,
    pub sq_insts_mfma: u64,
    pub sq_thread_cycles: u64,
    pub sq_wait_insts: u64,
    pub sq_active_insts: u64,
    // GRBM global counters
    pub grbm_count: u64,
    pub grbm_gui_active: u64,
    // TCC L2 cache counters
    pub tcc_req: u64,
    pub tcc_hit: u64,
    pub tcc_miss: u64,
    pub tcc_mc_writereq: u64,
    pub tcc_mc_readreq: u64,
}

/// Bottleneck classification.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Bottleneck {
    Compute,
    Memory,
    Latency,
    Unknown,
}

/// NCU-equivalent derived metrics.
#[derive(Clone, Debug)]
pub struct ProfileMetrics {
    // Timing
    pub elapsed_ns: u64,
    pub achieved_tflops: f64,
    pub peak_tflops: f64,
    pub compute_util_pct: f64,

    // Instruction mix (percentages)
    pub total_insts: u64,
    pub valu_pct: f64,
    pub salu_pct: f64,
    pub smem_pct: f64,
    pub flat_pct: f64,
    pub lds_pct: f64,
    pub vmem_pct: f64,
    pub mfma_pct: f64,

    // Throughput
    pub ipc: f64,                    // sq_insts / sq_thread_cycles
    pub active_ipc: f64,             // sq_insts / sq_active_insts (when waves are active)

    // Occupancy
    pub achieved_occupancy: f64,     // sq_waves / max_waves_per_cu (0.0-1.0)
    pub max_waves_per_cu: u32,

    // Cache
    pub l2_hit_rate: f64,            // tcc_hit / (tcc_hit + tcc_miss)
    pub l2_total_requests: u64,

    // Memory bandwidth
    pub dram_read_bytes: u64,        // tcc_mc_readreq * 64
    pub dram_write_bytes: u64,       // tcc_mc_writereq * 64
    pub dram_total_bytes: u64,
    pub memory_bandwidth_gbps: f64,
    pub memory_bandwidth_util_pct: f64, // % of 960 GB/s peak

    // Roofline
    pub arithmetic_intensity: f64,   // FLOP / byte
    pub bottleneck: Bottleneck,

    // Pipeline
    pub cu_busy_pct: f64,            // grbm_gui_active / grbm_count
    pub wait_ratio: f64,             // sq_wait_insts / sq_insts
}

/// GFX1100 hardware limits (RX 7900 XTX).
pub struct HwLimits {
    pub n_cus: u32,
    pub max_vgprs: u32,
    pub simds_per_cu: u32,
    pub max_waves_per_simd: u32,
    pub peak_bandwidth_gbps: f64,
    pub peak_tflops_bf16: f64,
    pub l2_cache_bytes: u64,
    pub clock_ghz: f64,
}

impl Default for HwLimits {
    fn default() -> Self {
        Self {
            n_cus: 96,
            max_vgprs: 256,
            simds_per_cu: 2,
            max_waves_per_simd: 16,   // 256 VGPRs / 16 VGPRs per wave = 16 (theoretical max)
            peak_bandwidth_gbps: 960.0,
            peak_tflops_bf16: 123.0,
            l2_cache_bytes: 6 * 1024 * 1024,
            clock_ghz: 2.5,
        }
    }
}

impl HwLimits {
    /// Maximum waves per CU = simds_per_cu * max_waves_per_simd.
    pub fn max_waves_per_cu(&self) -> u32 {
        self.simds_per_cu * self.max_waves_per_simd
    }
}

/// Compute derived metrics from raw counters.
///
/// `total_flops` is optional — if provided, enables roofline analysis.
/// `elapsed_ns` is the kernel wall-clock time.
pub fn compute_metrics(
    raw: &RawCounters,
    elapsed_ns: u64,
    total_flops: Option<u64>,
    limits: &HwLimits,
) -> ProfileMetrics {
    let max_waves = limits.max_waves_per_cu();

    // Instruction mix
    let total_insts = raw.sq_insts;
    let pct = |n: u64| -> f64 {
        if total_insts == 0 { 0.0 } else { n as f64 / total_insts as f64 * 100.0 }
    };
    let valu_pct = pct(raw.sq_insts_valu);
    let salu_pct = pct(raw.sq_insts_salu);
    let smem_pct = pct(raw.sq_insts_smem);
    let flat_pct = pct(raw.sq_insts_flat);
    let lds_pct = pct(raw.sq_insts_lds);
    let vmem_pct = pct(raw.sq_insts_vmem);
    let mfma_pct = pct(raw.sq_insts_mfma);

    // IPC
    let ipc = if raw.sq_thread_cycles > 0 {
        raw.sq_insts as f64 / raw.sq_thread_cycles as f64
    } else { 0.0 };
    let active_ipc = if raw.sq_active_insts > 0 {
        raw.sq_insts as f64 / raw.sq_active_insts as f64
    } else { 0.0 };

    // Occupancy
    // sq_waves is per-CU sampled (broadcast mode: same value for all CUs).
    // achieved_occupancy = sq_waves / max_waves_per_cu
    let achieved_occupancy = if max_waves > 0 && raw.sq_waves > 0 {
        (raw.sq_waves as f64 / max_waves as f64).min(1.0)
    } else { 0.0 };

    // L2 cache
    let l2_total = raw.tcc_hit + raw.tcc_miss;
    let l2_hit_rate = if l2_total > 0 {
        raw.tcc_hit as f64 / l2_total as f64
    } else { 0.0 };

    // Memory bandwidth
    let dram_read = raw.tcc_mc_readreq * 64;   // 64 bytes per request
    let dram_write = raw.tcc_mc_writereq * 64;
    let dram_total = dram_read + dram_write;
    let elapsed_s = elapsed_ns as f64 / 1e9;
    let bw_gbps = if elapsed_s > 0.0 {
        dram_total as f64 / elapsed_s / 1e9
    } else { 0.0 };
    let bw_util = if limits.peak_bandwidth_gbps > 0.0 {
        bw_gbps / limits.peak_bandwidth_gbps * 100.0
    } else { 0.0 };

    // Compute utilization
    let achieved_tflops = if let Some(flops) = total_flops {
        if elapsed_ns > 0 {
            flops as f64 / (elapsed_ns as f64 / 1e9) / 1e12
        } else { 0.0 }
    } else { 0.0 };
    let compute_util = if limits.peak_tflops_bf16 > 0.0 {
        achieved_tflops / limits.peak_tflops_bf16 * 100.0
    } else { 0.0 };

    // Roofline bottleneck
    let ai = if dram_total > 0 {
        total_flops.unwrap_or(0) as f64 / dram_total as f64
    } else { 0.0 };
    let peak_ai = limits.peak_tflops_bf16 * 1e3 / limits.peak_bandwidth_gbps; // FLOP/byte
    let bottleneck = if total_flops.is_none() {
        Bottleneck::Unknown
    } else if ai < peak_ai * 0.8 {
        Bottleneck::Memory
    } else if achieved_occupancy < 0.5 {
        Bottleneck::Latency
    } else {
        Bottleneck::Compute
    };

    // Pipeline
    let cu_busy = if raw.grbm_count > 0 {
        raw.grbm_gui_active as f64 / raw.grbm_count as f64 * 100.0
    } else { 0.0 };
    let wait_ratio = if total_insts > 0 {
        raw.sq_wait_insts as f64 / total_insts as f64 * 100.0
    } else { 0.0 };

    ProfileMetrics {
        elapsed_ns,
        achieved_tflops,
        peak_tflops: limits.peak_tflops_bf16,
        compute_util_pct: compute_util,
        total_insts,
        valu_pct, salu_pct, smem_pct, flat_pct, lds_pct, vmem_pct, mfma_pct,
        ipc, active_ipc,
        achieved_occupancy,
        max_waves_per_cu: max_waves,
        l2_hit_rate,
        l2_total_requests: l2_total,
        dram_read_bytes: dram_read,
        dram_write_bytes: dram_write,
        dram_total_bytes: dram_total,
        memory_bandwidth_gbps: bw_gbps,
        memory_bandwidth_util_pct: bw_util,
        arithmetic_intensity: ai,
        bottleneck,
        cu_busy_pct: cu_busy,
        wait_ratio,
    }
}
