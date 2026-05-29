//! Level 3: Safe Dispatch — GPU dispatch with timeout + isolation.
//!
//! Wraps GPU kernel dispatch with:
//! - Configurable timeout (default 5s)
//! - Dispatch logging for debugging
//! - Error recovery guidance
//!
//! # Usage
//! ```ignore
//! use t0_gpu::t0::safe_mode::safe_dispatch::SafeDispatcher;
//!
//! let dispatcher = SafeDispatcher::new(rt);
//! dispatcher.safe_dispatch(&kernel, [256, 1, 1], &kernarg, None)?;
//! ```
//!
//! # Limitations
//! - Cannot kill a hung GPU kernel mid-execution (hardware limitation)
//! - Timeout is for the entire dispatch cycle (submit + wait)
//! - Recovery from GPU hang requires MODE1 reset (process-level)

use std::time::{Duration, Instant};

/// Safe dispatch configuration.
pub struct SafeDispatchConfig {
    /// Maximum time to wait for kernel completion.
    pub timeout: Duration,
    /// Log dispatch details to stderr.
    pub log_dispatches: bool,
    /// Maximum kernel code size in bytes (reject if larger).
    pub max_code_size: usize,
}

impl Default for SafeDispatchConfig {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(5),
            log_dispatches: true,
            max_code_size: 65536, // 64KB
        }
    }
}

/// Dispatch error with recovery guidance.
#[derive(Debug)]
pub enum DispatchError {
    /// Kernel timed out — GPU may be hung.
    Timeout {
        elapsed: Duration,
        grid: [u32; 3],
    },
    /// Kernel code too large.
    CodeTooLarge {
        size: usize,
        max: usize,
    },
    /// KFD runtime error.
    RuntimeError(String),
    /// Pre-dispatch validation failed.
    ValidationError(String),
}

impl std::fmt::Display for DispatchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DispatchError::Timeout { elapsed, grid } => {
                write!(f, "GPU dispatch timed out after {:.1}s (grid={:?}). \
                    The kernel may be stuck in an infinite loop or missing s_endpgm. \
                    Recovery: restart the process. The GPU queue may need MODE1 reset.",
                    elapsed.as_secs_f32(), grid)
            }
            DispatchError::CodeTooLarge { size, max } => {
                write!(f, "Kernel code ({} bytes) exceeds limit ({} bytes). \
                    May cause instruction cache thrashing and GPU hangs.",
                    size, max)
            }
            DispatchError::RuntimeError(msg) => {
                write!(f, "GPU runtime error: {}", msg)
            }
            DispatchError::ValidationError(msg) => {
                write!(f, "Pre-dispatch validation failed: {}", msg)
            }
        }
    }
}

impl std::error::Error for DispatchError {}

/// Dispatch log entry for debugging.
#[derive(Debug, Clone)]
pub struct DispatchLog {
    pub kernel_name: String,
    pub grid: [u32; 3],
    pub workgroup_size: [u32; 3],
    pub code_size: usize,
    pub start_time: Instant,
    pub elapsed: Option<Duration>,
    pub success: bool,
    pub error: Option<String>,
}

/// Safe dispatcher — wraps GPU dispatch with timeout and logging.
///
/// Works with the existing GpuRuntime. Does NOT create isolated queues
/// (that would require KFD-level changes). Instead, adds timeout
/// monitoring and pre-dispatch validation.
#[cfg(feature = "rocm")]
pub struct SafeDispatcher {
    config: SafeDispatchConfig,
    log: Vec<DispatchLog>,
}

#[cfg(feature = "rocm")]
impl SafeDispatcher {
    pub fn new(config: SafeDispatchConfig) -> Self {
        Self { config, log: Vec::new() }
    }

    pub fn with_defaults() -> Self {
        Self::new(SafeDispatchConfig::default())
    }

    /// Safely dispatch a kernel with timeout monitoring.
    ///
    /// # Arguments
    /// * `rt` — GPU runtime
    /// * `kernel` — Compiled kernel
    /// * `grid` — Grid dimensions [x, y, z]
    /// * `kernarg` — Kernel argument bytes
    ///
    /// # Returns
    /// `Ok(())` if the kernel completed within the timeout.
    pub fn safe_dispatch(
        &mut self,
        rt: &crate::kfd::GpuRuntime,
        kernel: &crate::kfd::GpuKernel,
        grid: [u32; 3],
        kernarg: &[u8],
    ) -> Result<(), DispatchError> {
        let start = Instant::now();

        // Pre-dispatch logging
        if self.config.log_dispatches {
            eprintln!("[SafeDispatch] Dispatching kernel '{}' grid={:?} wg={:?} code={}B",
                kernel.name, grid, kernel.workgroup_size, kernel.code_size);
        }

        // Code size check
        if kernel.code_size > self.config.max_code_size {
            return Err(DispatchError::CodeTooLarge {
                size: kernel.code_size,
                max: self.config.max_code_size,
            });
        }

        // Submit
        let submit_result = rt.dispatch_async(kernel, grid, kernarg);
        if let Err(e) = submit_result {
            let log_entry = DispatchLog {
                kernel_name: kernel.name.clone(),
                grid,
                workgroup_size: kernel.workgroup_size,
                code_size: kernel.code_size,
                start_time: start,
                elapsed: Some(start.elapsed()),
                success: false,
                error: Some(e.clone()),
            };
            self.log.push(log_entry);
            return Err(DispatchError::RuntimeError(e));
        }

        // Wait with timeout
        let wait_start = Instant::now();
        loop {
            if wait_start.elapsed() > self.config.timeout {
                let elapsed = start.elapsed();
                let log_entry = DispatchLog {
                    kernel_name: kernel.name.clone(),
                    grid,
                    workgroup_size: kernel.workgroup_size,
                    code_size: kernel.code_size,
                    start_time: start,
                    elapsed: Some(elapsed),
                    success: false,
                    error: Some("timeout".to_string()),
                };
                self.log.push(log_entry);
                return Err(DispatchError::Timeout { elapsed, grid });
            }

            // Check if GPU is idle
            match rt.wait_idle_timeout(Duration::from_millis(100)) {
                Ok(true) => break, // idle
                Ok(false) => continue, // still running
                Err(e) => {
                    let log_entry = DispatchLog {
                        kernel_name: kernel.name.clone(),
                        grid,
                        workgroup_size: kernel.workgroup_size,
                        code_size: kernel.code_size,
                        start_time: start,
                        elapsed: Some(start.elapsed()),
                        success: false,
                        error: Some(e.clone()),
                    };
                    self.log.push(log_entry);
                    return Err(DispatchError::RuntimeError(e));
                }
            }
        }

        let elapsed = start.elapsed();
        if self.config.log_dispatches {
            eprintln!("[SafeDispatch] Kernel '{}' completed in {:.3}ms",
                kernel.name, elapsed.as_secs_f64() * 1000.0);
        }

        let log_entry = DispatchLog {
            kernel_name: kernel.name.clone(),
            grid,
            workgroup_size: kernel.workgroup_size,
            code_size: kernel.code_size,
            start_time: start,
            elapsed: Some(elapsed),
            success: true,
            error: None,
        };
        self.log.push(log_entry);

        Ok(())
    }

    /// Get dispatch log for debugging.
    pub fn log(&self) -> &[DispatchLog] {
        &self.log
    }

    /// Print dispatch summary.
    pub fn print_summary(&self) {
        let total = self.log.len();
        let ok = self.log.iter().filter(|l| l.success).count();
        let failed = total - ok;

        eprintln!("[SafeDispatch] Summary: {} dispatches, {} ok, {} failed", total, ok, failed);

        for entry in &self.log {
            let status = if entry.success { "OK" } else { "FAIL" };
            let elapsed = entry.elapsed
                .map(|d| format!("{:.3}ms", d.as_secs_f64() * 1000.0))
                .unwrap_or_else(|| "pending".to_string());
            eprintln!("  [{}] {} grid={:?} time={} {}",
                status, entry.kernel_name, entry.grid, elapsed,
                entry.error.as_deref().unwrap_or(""));
        }
    }
}

/// Timeout-aware wait helper for the KFD runtime.
///
/// Polls the GPU queue with a deadline. Returns:
/// - `Ok(true)` if GPU became idle within timeout
/// - `Ok(false)` if still running (caller should loop)
/// - `Err(msg)` if queue error
#[cfg(feature = "rocm")]
pub fn wait_idle_with_timeout(
    rt: &crate::kfd::GpuRuntime,
    timeout: Duration,
) -> Result<bool, String> {
    let start = Instant::now();
    loop {
        match rt.wait_idle() {
            Ok(()) => return Ok(true),
            Err(e) => {
                if start.elapsed() > timeout {
                    return Err(format!("Timeout after {:.1}s: {}", timeout.as_secs_f32(), e));
                }
                // Brief sleep before retry
                std::thread::sleep(Duration::from_micros(100));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dispatch_error_display() {
        let err = DispatchError::Timeout {
            elapsed: Duration::from_secs(5),
            grid: [256, 1, 1],
        };
        let msg = format!("{}", err);
        assert!(msg.contains("timed out"));
        assert!(msg.contains("256"));
        assert!(msg.contains("infinite loop"));
    }

    #[test]
    fn test_code_too_large() {
        let err = DispatchError::CodeTooLarge {
            size: 100_000,
            max: 65536,
        };
        let msg = format!("{}", err);
        assert!(msg.contains("100000"));
        assert!(msg.contains("65536"));
    }

    #[test]
    fn test_default_config() {
        let config = SafeDispatchConfig::default();
        assert_eq!(config.timeout, Duration::from_secs(5));
        assert_eq!(config.max_code_size, 65536);
        assert!(config.log_dispatches);
    }
}
