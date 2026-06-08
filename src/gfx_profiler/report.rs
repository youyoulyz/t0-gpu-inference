//! NCU-style report generation.
//!
//! Produces human-readable text reports with metric tables, instruction mix bars,
//! roofline analysis, and optimization suggestions.

use super::metrics::{ProfileMetrics, Bottleneck};
use super::suggestions::{OptimizationSuggestion, Severity};

#[derive(Clone, Copy, Debug)]
pub enum OutputFormat {
    Text,
    Json,
}

/// Generate a profiling report.
pub fn generate_report(
    kernel_name: &str,
    result: &ProfileMetrics,
    suggestions: &[OptimizationSuggestion],
    format: OutputFormat,
) -> String {
    match format {
        OutputFormat::Text => generate_text_report(kernel_name, result, suggestions),
        OutputFormat::Json => generate_json_report(kernel_name, result, suggestions),
    }
}

fn generate_text_report(
    kernel_name: &str,
    m: &ProfileMetrics,
    suggestions: &[OptimizationSuggestion],
) -> String {
    use std::fmt::Write;
    let mut out = String::with_capacity(4096);

    let sep = "=".repeat(64);
    let thin = "-".repeat(64);

    // Header
    writeln!(out, "{}", sep).unwrap();
    writeln!(out, "  GFX1100 Profile Report: {}", kernel_name).unwrap();
    writeln!(out, "  Target: AMD RX 7900 XTX (GFX1100, 96 CU, Wave32)").unwrap();
    writeln!(out, "{}", sep).unwrap();
    writeln!(out).unwrap();

    // Timing
    let elapsed_str = if m.elapsed_ns < 1_000 {
        format!("{} ns", m.elapsed_ns)
    } else if m.elapsed_ns < 1_000_000 {
        format!("{:.2} us", m.elapsed_ns as f64 / 1e3)
    } else {
        format!("{:.3} ms", m.elapsed_ns as f64 / 1e6)
    };
    writeln!(out, "  Duration:           {}", elapsed_str).unwrap();
    if m.achieved_tflops > 0.0 {
        writeln!(out, "  Achieved TFLOPS:    {:.1}  ({:.1}% of {:.0} peak)",
            m.achieved_tflops, m.compute_util_pct, m.peak_tflops).unwrap();
    }
    writeln!(out).unwrap();

    // Metrics table
    writeln!(out, "  {:<24} {:>10}  {:<24}", "Metric", "Value", "Status").unwrap();
    writeln!(out, "  {}", thin).unwrap();

    let bar = |pct: f64| -> String {
        let filled = (pct / 5.0).round() as usize;
        let filled = filled.min(20);
        let empty = 20 - filled;
        format!("{}{}", "█".repeat(filled), "░".repeat(empty))
    };

    writeln!(out, "  {:<24} {:>9.1}%  {}",
        "Achieved Occupancy", m.achieved_occupancy * 100.0, bar(m.achieved_occupancy * 100.0)).unwrap();
    writeln!(out, "  {:<24} {:>9.1}%  {}",
        "L2 Hit Rate", m.l2_hit_rate * 100.0, bar(m.l2_hit_rate * 100.0)).unwrap();
    if m.ipc > 0.0 {
        writeln!(out, "  {:<24} {:>9.2}  {}",
            "IPC", m.ipc, bar((m.ipc / 4.0 * 100.0).min(100.0))).unwrap();
    }
    writeln!(out, "  {:<24} {:>9.1}%  {}",
        "Mem BW Utilization", m.memory_bandwidth_util_pct,
        bar(m.memory_bandwidth_util_pct)).unwrap();
    writeln!(out, "  {:<24} {:>9.1}%  {}",
        "CU Busy", m.cu_busy_pct, bar(m.cu_busy_pct)).unwrap();
    writeln!(out, "  {:<24} {:>9.1}%  {}",
        "Wait Ratio", m.wait_ratio, bar(m.wait_ratio)).unwrap();

    if m.memory_bandwidth_gbps > 0.0 {
        writeln!(out).unwrap();
        writeln!(out, "  Memory Bandwidth:     {:.1} GB/s  (peak: 960 GB/s)",
            m.memory_bandwidth_gbps).unwrap();
        writeln!(out, "  DRAM Read:            {:.2} MB", m.dram_read_bytes as f64 / 1e6).unwrap();
        writeln!(out, "  DRAM Write:           {:.2} MB", m.dram_write_bytes as f64 / 1e6).unwrap();
    }
    writeln!(out).unwrap();

    // Instruction mix
    if m.total_insts > 0 {
        writeln!(out, "  {}", thin).unwrap();
        writeln!(out, "  INSTRUCTION MIX (total: {} insts)", m.total_insts).unwrap();
        writeln!(out, "  {}", thin).unwrap();

        let mix = [
            ("WMMA", m.mfma_pct),
            ("VALU", m.valu_pct),
            ("SALU", m.salu_pct),
            ("VMEM", m.vmem_pct),
            ("SMEM", m.smem_pct),
            ("LDS",  m.lds_pct),
            ("FLAT", m.flat_pct),
        ];
        for (name, pct) in &mix {
            let filled = (*pct / 2.5).round() as usize;
            let filled = filled.min(40);
            let bar_str: String = "█".repeat(filled) + &"░".repeat(40 - filled);
            writeln!(out, "  {:<6} {} {:>5.1}%", name, bar_str, pct).unwrap();
        }
        writeln!(out).unwrap();
    }

    // Bottleneck
    let bn = match m.bottleneck {
        Bottleneck::Compute => "COMPUTE BOUND",
        Bottleneck::Memory => "MEMORY BOUND",
        Bottleneck::Latency => "LATENCY BOUND",
        Bottleneck::Unknown => "UNKNOWN",
    };
    writeln!(out, "  Bottleneck:           {}", bn).unwrap();
    if m.arithmetic_intensity > 0.0 {
        writeln!(out, "  Arithmetic Intensity: {:.1} FLOP/byte", m.arithmetic_intensity).unwrap();
    }
    writeln!(out).unwrap();

    // Suggestions
    if !suggestions.is_empty() {
        writeln!(out, "  {}", thin).unwrap();
        writeln!(out, "  OPTIMIZATION SUGGESTIONS").unwrap();
        writeln!(out, "  {}", thin).unwrap();
        for s in suggestions {
            let sev = match s.severity {
                Severity::Critical => "CRIT",
                Severity::Warning => "WARN",
                Severity::Info => "INFO",
            };
            // Word-wrap message at ~56 chars
            let words: Vec<&str> = s.message.split_whitespace().collect();
            let mut line = format!("  [{}] {}: ", sev, s.category);
            let indent = " ".repeat(line.len());
            let mut _first = true;
            for word in &words {
                if !_first && line.len() + word.len() > 62 {
                    writeln!(out, "{}", line).unwrap();
                    line = indent.clone();
                }
                if !line.ends_with(' ') { line.push(' '); }
                line.push_str(word);
                _first = false;
            }
            if line.len() > indent.len() {
                writeln!(out, "{}", line).unwrap();
            }
        }
        writeln!(out).unwrap();
    }

    writeln!(out, "{}", sep).unwrap();
    out
}

fn generate_json_report(
    kernel_name: &str,
    m: &ProfileMetrics,
    suggestions: &[OptimizationSuggestion],
) -> String {
    use std::fmt::Write;
    let mut out = String::with_capacity(2048);
    writeln!(out, "{{").unwrap();
    writeln!(out, "  \"kernel\": \"{}\",", kernel_name).unwrap();
    writeln!(out, "  \"elapsed_ns\": {},", m.elapsed_ns).unwrap();
    writeln!(out, "  \"achieved_tflops\": {:.2},", m.achieved_tflops).unwrap();
    writeln!(out, "  \"peak_tflops\": {:.1},", m.peak_tflops).unwrap();
    writeln!(out, "  \"compute_util_pct\": {:.1},", m.compute_util_pct).unwrap();
    writeln!(out, "  \"ipc\": {:.3},", m.ipc).unwrap();
    writeln!(out, "  \"achieved_occupancy\": {:.3},", m.achieved_occupancy).unwrap();
    writeln!(out, "  \"l2_hit_rate\": {:.3},", m.l2_hit_rate).unwrap();
    writeln!(out, "  \"memory_bandwidth_gbps\": {:.1},", m.memory_bandwidth_gbps).unwrap();
    writeln!(out, "  \"memory_bandwidth_util_pct\": {:.1},", m.memory_bandwidth_util_pct).unwrap();
    writeln!(out, "  \"arithmetic_intensity\": {:.2},", m.arithmetic_intensity).unwrap();
    writeln!(out, "  \"bottleneck\": \"{:?}\",", m.bottleneck).unwrap();
    writeln!(out, "  \"total_insts\": {},", m.total_insts).unwrap();
    writeln!(out, "  \"instruction_mix\": {{").unwrap();
    writeln!(out, "    \"mfma_pct\": {:.1},", m.mfma_pct).unwrap();
    writeln!(out, "    \"valu_pct\": {:.1},", m.valu_pct).unwrap();
    writeln!(out, "    \"salu_pct\": {:.1},", m.salu_pct).unwrap();
    writeln!(out, "    \"vmem_pct\": {:.1},", m.vmem_pct).unwrap();
    writeln!(out, "    \"smem_pct\": {:.1},", m.smem_pct).unwrap();
    writeln!(out, "    \"lds_pct\": {:.1},", m.lds_pct).unwrap();
    writeln!(out, "    \"flat_pct\": {:.1}", m.flat_pct).unwrap();
    writeln!(out, "  }},").unwrap();
    writeln!(out, "  \"dram_read_bytes\": {},", m.dram_read_bytes).unwrap();
    writeln!(out, "  \"dram_write_bytes\": {},", m.dram_write_bytes).unwrap();
    writeln!(out, "  \"cu_busy_pct\": {:.1},", m.cu_busy_pct).unwrap();
    writeln!(out, "  \"wait_ratio\": {:.1},", m.wait_ratio).unwrap();
    writeln!(out, "  \"suggestions\": [").unwrap();
    for (i, s) in suggestions.iter().enumerate() {
        let comma = if i < suggestions.len() - 1 { "," } else { "" };
        writeln!(out, "    {{").unwrap();
        writeln!(out, "      \"severity\": \"{:?}\",", s.severity).unwrap();
        writeln!(out, "      \"category\": \"{}\",", s.category).unwrap();
        writeln!(out, "      \"metric\": \"{}\",", s.metric_name).unwrap();
        writeln!(out, "      \"value\": {:.3},", s.value).unwrap();
        writeln!(out, "      \"threshold\": {:.3},", s.threshold).unwrap();
        // Escape quotes in message
        let msg = s.message.replace('\\', "\\\\").replace('"', "\\\"");
        writeln!(out, "      \"message\": \"{}\"", msg).unwrap();
        writeln!(out, "    }}{}", comma).unwrap();
    }
    writeln!(out, "  ]").unwrap();
    writeln!(out, "}}").unwrap();
    out
}
