//! Optimization suggestion engine.
//!
//! Applies heuristic rules to derived metrics and generates NCU-style
//! optimization suggestions with severity levels.

use super::metrics::{ProfileMetrics, Bottleneck};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Severity {
    Critical,
    Warning,
    Info,
}

#[derive(Clone, Debug)]
pub struct OptimizationSuggestion {
    pub severity: Severity,
    pub category: &'static str,
    pub metric_name: &'static str,
    pub value: f64,
    pub threshold: f64,
    pub message: String,
}

/// Generate optimization suggestions from profile metrics.
pub fn generate_suggestions(m: &ProfileMetrics) -> Vec<OptimizationSuggestion> {
    let mut out = Vec::new();

    // 1. Low occupancy
    if m.achieved_occupancy > 0.0 && m.achieved_occupancy < 0.5 {
        out.push(OptimizationSuggestion {
            severity: Severity::Warning,
            category: "Occupancy",
            metric_name: "achieved_occupancy",
            value: m.achieved_occupancy,
            threshold: 0.5,
            message: format!(
                "Occupancy is {:.0}%. Consider: (1) reduce VGPR usage, \
                 (2) use smaller tile sizes, (3) use bf16 to halve register pressure.",
                m.achieved_occupancy * 100.0
            ),
        });
    }

    // 2. Memory bound
    if m.bottleneck == Bottleneck::Memory {
        out.push(OptimizationSuggestion {
            severity: Severity::Warning,
            category: "Memory",
            metric_name: "arithmetic_intensity",
            value: m.arithmetic_intensity,
            threshold: 0.0,
            message: format!(
                "Kernel is memory bound (AI={:.1} FLOP/byte). \
                 Consider: (1) increase tile size for more data reuse, \
                 (2) use WMMA to increase compute density, \
                 (3) use vectorized loads (global_load_dwordx4).",
                m.arithmetic_intensity
            ),
        });
    }

    // 3. Low L2 hit rate
    if m.l2_total_requests > 0 && m.l2_hit_rate < 0.3 {
        out.push(OptimizationSuggestion {
            severity: Severity::Warning,
            category: "Cache",
            metric_name: "l2_hit_rate",
            value: m.l2_hit_rate,
            threshold: 0.3,
            message: format!(
                "L2 hit rate is {:.0}%. Consider: (1) increase tile size for \
                 better data reuse, (2) adjust access pattern for spatial locality, \
                 (3) use LDS for within-tile reuse.",
                m.l2_hit_rate * 100.0
            ),
        });
    }

    // 4. Low IPC
    if m.ipc > 0.0 && m.ipc < 1.0 {
        out.push(OptimizationSuggestion {
            severity: Severity::Info,
            category: "Latency",
            metric_name: "ipc",
            value: m.ipc,
            threshold: 1.0,
            message: format!(
                "IPC is {:.2}. Consider: (1) software pipelining to hide VMEM latency \
                 (~500 cycles), (2) increase ILP by interleaving independent instructions, \
                 (3) use double-buffered LDS loads.",
                m.ipc
            ),
        });
    }

    // 5. Low WMMA utilization for compute-bound kernels
    if m.mfma_pct < 30.0 && m.total_insts > 1000 && m.bottleneck == Bottleneck::Compute {
        out.push(OptimizationSuggestion {
            severity: Severity::Warning,
            category: "Compute",
            metric_name: "mfma_pct",
            value: m.mfma_pct,
            threshold: 30.0,
            message: format!(
                "WMMA/MFMA instructions are only {:.0}% of total. \
                 Restructure K-loop to maximize v_wmma usage. \
                 Scalar FMA is ~16x slower than WMMA.",
                m.mfma_pct
            ),
        });
    }

    // 6. High memory BW utilization (well-optimized)
    if m.memory_bandwidth_util_pct > 80.0 && m.bottleneck == Bottleneck::Memory {
        out.push(OptimizationSuggestion {
            severity: Severity::Info,
            category: "Memory",
            metric_name: "memory_bandwidth_util",
            value: m.memory_bandwidth_util_pct,
            threshold: 80.0,
            message: "Memory bandwidth near peak. Kernel is well-optimized \
                      for its arithmetic intensity. To improve further, \
                      increase data reuse (larger tiles, WMMA).".to_string(),
        });
    }

    // 7. High control overhead
    let ctrl_pct = 100.0 - m.valu_pct - m.salu_pct - m.smem_pct
        - m.flat_pct - m.lds_pct - m.vmem_pct - m.mfma_pct;
    if ctrl_pct > 10.0 && m.total_insts > 1000 {
        out.push(OptimizationSuggestion {
            severity: Severity::Info,
            category: "Control",
            metric_name: "ctrl_overhead",
            value: ctrl_pct,
            threshold: 10.0,
            message: format!(
                "Control flow overhead is {:.0}%. Consider: (1) unroll loops, \
                 (2) reduce branch divergence, (3) merge basic blocks.",
                ctrl_pct
            ),
        });
    }

    // 8. High wait ratio
    if m.wait_ratio > 30.0 {
        out.push(OptimizationSuggestion {
            severity: Severity::Warning,
            category: "Latency",
            metric_name: "wait_ratio",
            value: m.wait_ratio,
            threshold: 30.0,
            message: format!(
                "Wait instructions are {:.0}% of total. Waves are stalling. \
                 Consider: (1) software pipelining, (2) increase workgroup size \
                 to improve latency hiding, (3) reduce register pressure.",
                m.wait_ratio
            ),
        });
    }

    out
}
