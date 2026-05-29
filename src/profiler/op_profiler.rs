//! Op-level profiler — tracks per-op CPU/GPU timing, kernel counts, nesting, and I/O shapes.

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

    /// Format as "MxNxK" or "4096" for 1D.
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
    pub cpu_ns: u64,
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
    pub call_count: u64,
    pub total_kernel_count: u64,
    /// Representative input shapes (from first invocation).
    pub input_shapes: Vec<ShapeInfo>,
    /// Representative output shapes (from first invocation).
    pub output_shapes: Vec<ShapeInfo>,
}

/// Op-level profiler.
pub struct OpProfiler {
    records: Vec<OpRecord>,
    open_stack: Vec<OpenOp>,
    recording: bool,
}

struct OpenOp {
    name: &'static str,
    start: Instant,
    depth: u32,
    input_shapes: Vec<ShapeInfo>,
    output_shapes: Vec<ShapeInfo>,
}

impl OpProfiler {
    pub const fn new() -> Self {
        Self {
            records: Vec::new(),
            open_stack: Vec::new(),
            recording: true,
        }
    }

    /// Begin recording an op.
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

    /// Set input/output shapes for the currently open op.
    ///
    /// Call after `begin()` and before `end()`.
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

    /// End recording an op.
    pub fn end(&mut self, name: &'static str) {
        if !self.recording {
            return;
        }
        let idx = self.open_stack.iter().rposition(|o| o.name == name);
        if let Some(idx) = idx {
            let open = self.open_stack.remove(idx);
            let cpu_ns = open.start.elapsed().as_nanos() as u64;
            self.records.push(OpRecord {
                name,
                cpu_ns,
                kernel_count: 1,
                depth: open.depth,
                input_shapes: open.input_shapes,
                output_shapes: open.output_shapes,
            });
        }
    }

    /// Record a completed kernel dispatch in one call.
    pub fn record_kernel(&mut self, name: &'static str, cpu_ns: u64) {
        if !self.recording {
            return;
        }
        let depth = self.open_stack.len() as u32;
        self.records.push(OpRecord {
            name,
            cpu_ns,
            kernel_count: 1,
            depth,
            input_shapes: Vec::new(),
            output_shapes: Vec::new(),
        });
    }

    /// Print a human-readable summary table to stderr.
    pub fn report(&self) {
        let stats = self.aggregate();

        eprintln!();
        eprintln!("=== GPU Profiler Report ===");
        eprintln!("{:<24} {:>10} {:>10} {:>6} {:>6}  {}",
            "Op", "Total(ms)", "Avg(μs)", "Calls", "Kerns", "Shapes");
        eprintln!("{}", "-".repeat(90));

        let mut total_ns: u64 = 0;
        let mut total_calls: u64 = 0;
        let mut total_kernels: u64 = 0;

        for s in &stats {
            let avg_us = if s.call_count > 0 {
                (s.total_cpu_ns / s.call_count) as f64 / 1000.0
            } else {
                0.0
            };

            let shapes = Self::fmt_shapes(&s.input_shapes, &s.output_shapes);

            eprintln!("{:<24} {:>10.3} {:>10.1} {:>6} {:>6}  {}",
                s.name,
                s.total_cpu_ns as f64 / 1_000_000.0,
                avg_us,
                s.call_count,
                s.total_kernel_count,
                shapes,
            );
            total_ns += s.total_cpu_ns;
            total_calls += s.call_count;
            total_kernels += s.total_kernel_count;
        }

        eprintln!("{}", "-".repeat(90));
        eprintln!("{:<24} {:>10.3} {:>10} {:>6} {:>6}",
            "TOTAL",
            total_ns as f64 / 1_000_000.0,
            "",
            total_calls,
            total_kernels,
        );
        eprintln!();
    }

    /// Format input/output shapes for the report.
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

    /// Export as Chrome tracing JSON format.
    pub fn to_json(&self) -> String {
        let mut events = Vec::new();
        let mut offset_ns: u64 = 0;

        for r in &self.records {
            let shapes = Self::fmt_shapes(&r.input_shapes, &r.output_shapes);
            let name = if shapes.is_empty() {
                r.name.to_string()
            } else {
                format!("{} [{}]", r.name, shapes)
            };
            events.push(format!(
                r#"{{"name":"{}","ph":"X","ts":{},"dur":{},"pid":1,"tid":{}}}"#,
                name,
                offset_ns / 1000,
                r.cpu_ns / 1000,
                r.depth,
            ));
            offset_ns += r.cpu_ns;
        }

        format!(r#"{{"traceEvents":[{}]}}"#, events.join(","))
    }

    /// Aggregate records by op name.
    fn aggregate(&self) -> Vec<OpStats> {
        let mut map: HashMap<&str, OpStats> = HashMap::new();
        for r in &self.records {
            let entry = map.entry(r.name).or_insert(OpStats {
                name: r.name,
                total_cpu_ns: 0,
                call_count: 0,
                total_kernel_count: 0,
                input_shapes: r.input_shapes.clone(),
                output_shapes: r.output_shapes.clone(),
            });
            entry.total_cpu_ns += r.cpu_ns;
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
    }

    pub fn records(&self) -> &[OpRecord] {
        &self.records
    }

    pub fn total_cpu_ns(&self) -> u64 {
        self.records.iter().map(|r| r.cpu_ns).sum()
    }
}
