//! Op-level profiler — CPU/GPU interaction timeline with I/O shape tracking.

use std::collections::HashMap;
use std::time::Instant;

/// Shape info for a single tensor.
#[derive(Clone, Debug, Default)]
pub struct ShapeInfo {
    pub dims: Vec<usize>,
}

impl ShapeInfo {
    pub fn new(dims: &[usize]) -> Self {
        Self { dims: dims.to_vec() }
    }

    pub fn numel(&self) -> usize {
        self.dims.iter().product()
    }

    pub fn fmt_short(&self) -> String {
        if self.dims.is_empty() {
            return "scalar".to_string();
        }
        self.dims.iter().map(|d| d.to_string()).collect::<Vec<_>>().join("x")
    }
}

/// A single profiling record for one op invocation.
#[derive(Clone, Debug)]
pub struct OpRecord {
    pub name: &'static str,
    /// CPU wall time (ns) — includes submit + wait_idle
    pub cpu_ns: u64,
    /// GPU execution time (ns) — 0 = use estimate
    pub gpu_ns: u64,
    /// Absolute start time (ns) from session start
    pub start_ns: u64,
    pub kernel_count: u32,
    pub depth: u32,
    pub input_shapes: Vec<ShapeInfo>,
    pub output_shapes: Vec<ShapeInfo>,
}

/// Aggregated stats for an op across all invocations.
#[derive(Clone, Debug)]
pub struct OpStats {
    pub name: &'static str,
    pub total_cpu_ns: u64,
    pub total_gpu_ns: u64,
    pub call_count: u64,
    pub total_kernel_count: u64,
    pub input_shapes: Vec<ShapeInfo>,
    pub output_shapes: Vec<ShapeInfo>,
}

pub struct OpProfiler {
    records: Vec<OpRecord>,
    open_stack: Vec<OpenOp>,
    recording: bool,
    session_start: Instant,
}

struct OpenOp {
    name: &'static str,
    start: Instant,
    depth: u32,
    input_shapes: Vec<ShapeInfo>,
    output_shapes: Vec<ShapeInfo>,
}

impl OpProfiler {
    pub fn new() -> Self {
        Self {
            records: Vec::new(),
            open_stack: Vec::new(),
            recording: true,
            session_start: Instant::now(),
        }
    }

    fn elapsed_ns(&self) -> u64 {
        self.session_start.elapsed().as_nanos() as u64
    }

    pub fn begin(&mut self, name: &'static str) {
        if !self.recording {
            return;
        }
        let depth = self.open_stack.len() as u32;
        self.open_stack.push(OpenOp {
            name,
            start: Instant::now(),
            depth,
            input_shapes: Vec::new(),
            output_shapes: Vec::new(),
        });
    }

    pub fn set_shapes(
        &mut self,
        inputs: Vec<ShapeInfo>,
        outputs: Vec<ShapeInfo>,
    ) {
        if let Some(open) = self.open_stack.last_mut() {
            open.input_shapes = inputs;
            open.output_shapes = outputs;
        }
    }

    pub fn end(&mut self, name: &'static str) {
        if !self.recording {
            return;
        }
        let idx = self.open_stack.iter().rposition(|o| o.name == name);
        if let Some(idx) = idx {
            let open = self.open_stack.remove(idx);
            let cpu_ns = open.start.elapsed().as_nanos() as u64;
            let start_ns = self.elapsed_ns() - cpu_ns;
            self.records.push(OpRecord {
                name,
                cpu_ns,
                gpu_ns: 0,
                start_ns,
                kernel_count: 1,
                depth: open.depth,
                input_shapes: open.input_shapes,
                output_shapes: open.output_shapes,
            });
        }
    }

    /// Set GPU execution time for the most recent record with the given name.
    pub fn record_gpu_timing(&mut self, name: &'static str, gpu_ns: u64) {
        if let Some(r) = self.records.iter_mut().rev().find(|r| r.name == name) {
            r.gpu_ns = gpu_ns;
        }
    }

    pub fn record_kernel(&mut self, name: &'static str, cpu_ns: u64) {
        if !self.recording {
            return;
        }
        let depth = self.open_stack.len() as u32;
        let start_ns = self.elapsed_ns() - cpu_ns;
        self.records.push(OpRecord {
            name,
            cpu_ns,
            gpu_ns: 0,
            start_ns,
            kernel_count: 1,
            depth,
            input_shapes: Vec::new(),
            output_shapes: Vec::new(),
        });
    }

    /// Print human-readable summary table.
    pub fn report(&self) {
        let stats = self.aggregate();

        eprintln!();
        eprintln!("=== GPU Profiler Report ===");
        eprintln!("{:<24} {:>10} {:>10} {:>10} {:>6}  {}",
            "Op", "CPU(ms)", "GPU(ms)", "Avg(μs)", "Calls", "Shapes");
        eprintln!("{}", "-".repeat(95));

        let mut total_cpu: u64 = 0;
        let mut total_gpu: u64 = 0;
        let mut total_calls: u64 = 0;

        for s in &stats {
            let avg_us = if s.call_count > 0 {
                (s.total_cpu_ns / s.call_count) as f64 / 1000.0
            } else {
                0.0
            };

            let shapes = Self::fmt_shapes(&s.input_shapes, &s.output_shapes);
            let gpu_ms = if s.total_gpu_ns > 0 {
                format!("{:>10.3}", s.total_gpu_ns as f64 / 1_000_000.0)
            } else {
                format!("{:>10}", "-")
            };

            eprintln!("{:<24} {:>10.3} {} {:>10.1} {:>6}  {}",
                s.name,
                s.total_cpu_ns as f64 / 1_000_000.0,
                gpu_ms,
                avg_us,
                s.call_count,
                shapes,
            );
            total_cpu += s.total_cpu_ns;
            total_gpu += s.total_gpu_ns;
            total_calls += s.call_count;
        }

        eprintln!("{}", "-".repeat(95));
        eprintln!("{:<24} {:>10.3} {:>10.3} {:>10} {:>6}",
            "TOTAL",
            total_cpu as f64 / 1_000_000.0,
            if total_gpu > 0 { total_gpu as f64 / 1_000_000.0 } else { 0.0 },
            "",
            total_calls,
        );

        if total_gpu > 0 {
            let overhead = total_cpu.saturating_sub(total_gpu);
            eprintln!();
            eprintln!("CPU-GPU overhead: {:.3} ms ({:.1}%)",
                overhead as f64 / 1_000_000.0,
                overhead as f64 / total_cpu as f64 * 100.0,
            );
        }
        eprintln!();
    }

    fn fmt_shapes(inputs: &[ShapeInfo], outputs: &[ShapeInfo]) -> String {
        let mut parts = Vec::new();
        if !inputs.is_empty() {
            let in_strs: Vec<String> = inputs.iter().map(|s| s.fmt_short()).collect();
            parts.push(format!("in:{}", in_strs.join(",")));
        }
        if !outputs.is_empty() {
            let out_strs: Vec<String> = outputs.iter().map(|s| s.fmt_short()).collect();
            parts.push(format!("out:{}", out_strs.join(",")));
        }
        parts.join(" ")
    }

    /// Export as Chrome tracing JSON with CPU/GPU dual tracks.
    ///
    /// - tid=0: CPU track (dispatch + wait)
    /// - tid=1: GPU track (kernel execution, estimated)
    ///
    /// Load at `chrome://tracing` or `ui.perfetto.dev`.
    pub fn to_json(&self) -> String {
        let mut events = Vec::new();

        for r in &self.records {
            let shapes = Self::fmt_shapes(&r.input_shapes, &r.output_shapes);
            let label = if shapes.is_empty() {
                r.name.to_string()
            } else {
                format!("{} [{}]", r.name, shapes)
            };

            // CPU track (tid=0): full dispatch time
            events.push(format!(
                r#"{{"name":"{}","ph":"X","ts":{},"dur":{},"pid":1,"tid":0,"cat":"cpu"}}"#,
                label,
                r.start_ns / 1000,
                r.cpu_ns / 1000,
            ));

            // GPU track (tid=1): kernel execution time
            let gpu_ns = if r.gpu_ns > 0 {
                r.gpu_ns
            } else {
                super::gpu_timestamp::estimate_gpu_ns(r.cpu_ns)
            };
            if gpu_ns > 0 {
                let gpu_start = r.start_ns + 2000; // ~2μs AQL dispatch latency
                events.push(format!(
                    r#"{{"name":"{}","ph":"X","ts":{},"dur":{},"pid":1,"tid":1,"cat":"gpu"}}"#,
                    label,
                    gpu_start / 1000,
                    gpu_ns / 1000,
                ));
            }
        }

        format!(r#"{{"traceEvents":[{}]}}"#, events.join(","))
    }

    fn aggregate(&self) -> Vec<OpStats> {
        let mut map: HashMap<&str, OpStats> = HashMap::new();
        for r in &self.records {
            let entry = map.entry(r.name).or_insert(OpStats {
                name: r.name,
                total_cpu_ns: 0,
                total_gpu_ns: 0,
                call_count: 0,
                total_kernel_count: 0,
                input_shapes: r.input_shapes.clone(),
                output_shapes: r.output_shapes.clone(),
            });
            entry.total_cpu_ns += r.cpu_ns;
            entry.total_gpu_ns += r.gpu_ns;
            entry.call_count += 1;
            entry.total_kernel_count += r.kernel_count as u64;
        }

        let mut seen = Vec::new();
        for r in &self.records {
            if !seen.contains(&r.name) {
                seen.push(r.name);
            }
        }

        seen.into_iter().filter_map(|n| map.remove(n)).collect()
    }

    pub fn reset(&mut self) {
        self.records.clear();
        self.open_stack.clear();
        self.session_start = Instant::now();
    }

    pub fn records(&self) -> &[OpRecord] {
        &self.records
    }

    pub fn total_cpu_ns(&self) -> u64 {
        self.records.iter().map(|r| r.cpu_ns).sum()
    }
}
