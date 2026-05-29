//! Level 3: Safe Dispatch — GPU dispatch with timeout + isolation.
//!
//! Wraps GPU kernel dispatch with:
//! - Configurable timeout (default 5s)
//! - Dispatch logging for debugging
//! - Error recovery guidance
//!
//! # Usage
//! ```ignore
//! use t0_gpu::t0::safe_mode::safe_dispatch::{SafeDispatcher, SafeDispatchConfig};
//!
//! let mut dispatcher = SafeDispatcher::with_defaults();
//! dispatcher.safe_dispatch(&queue, &kernel, [256, 1, 1], &kernarg_buf)?;
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
/// Works with AqlQueue. Does NOT create isolated queues
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
    /// * `queue` — AQL compute queue
    /// * `kernel` — Compiled kernel
    /// * `grid` — Grid dimensions [x, y, z]
    /// * `kernarg` — Kernel argument buffer (GPU-resident)
    ///
    /// # Returns
    /// `Ok(())` if the kernel completed within the timeout.
    pub fn safe_dispatch(
        &mut self,
        queue: &crate::kfd::AqlQueue,
        kernel: &crate::kfd::GpuKernel,
        grid: [u32; 3],
        kernarg: &crate::kfd::GpuBuffer,
    ) -> Result<(), DispatchError> {
        let start = Instant::now();
        let kernel_label = format!("kernel@{:#x}", kernel.descriptor_va);
        let code_size = kernel.code_buffer.size;

        // Pre-dispatch logging
        if self.config.log_dispatches {
            eprintln!("[SafeDispatch] Dispatching {} grid={:?} wg={:?} code={}B",
                kernel_label, grid, kernel.workgroup_size, code_size);
        }

        // Code size check
        if code_size > self.config.max_code_size {
            return Err(DispatchError::CodeTooLarge {
                size: code_size,
                max: self.config.max_code_size,
            });
        }

        // Submit
        let submit_result = queue.dispatch(kernel, grid, kernarg);
        if let Err(e) = submit_result {
            let log_entry = DispatchLog {
                kernel_name: kernel_label.clone(),
                grid,
                workgroup_size: kernel.workgroup_size,
                code_size,
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
                    kernel_name: kernel_label,
                    grid,
                    workgroup_size: kernel.workgroup_size,
                    code_size,
                    start_time: start,
                    elapsed: Some(elapsed),
                    success: false,
                    error: Some("timeout".to_string()),
                };
                self.log.push(log_entry);
                return Err(DispatchError::Timeout { elapsed, grid });
            }

            // Check if GPU is idle
            match queue.wait_idle() {
                Ok(()) => break,
                Err(e) => {
                    // wait_idle returned error — retry until timeout
                    if wait_start.elapsed() > self.config.timeout {
                        let log_entry = DispatchLog {
                            kernel_name: kernel_label,
                            grid,
                            workgroup_size: kernel.workgroup_size,
                            code_size,
                            start_time: start,
                            elapsed: Some(start.elapsed()),
                            success: false,
                            error: Some(e),
                        };
                        self.log.push(log_entry);
                        return Err(DispatchError::Timeout { elapsed: start.elapsed(), grid });
                    }
                    std::thread::sleep(Duration::from_micros(100));
                }
            }
        }

        let elapsed = start.elapsed();
        if self.config.log_dispatches {
            eprintln!("[SafeDispatch] {} completed in {:.3}ms",
                kernel_label, elapsed.as_secs_f64() * 1000.0);
        }

        let log_entry = DispatchLog {
            kernel_name: kernel_label,
            grid,
            workgroup_size: kernel.workgroup_size,
            code_size,
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

/// Timeout-aware wait helper for the AQL queue.
///
/// Polls the GPU queue with a deadline. Returns:
/// - `Ok(())` if GPU became idle within timeout
/// - `Err(msg)` if timeout or queue error
#[cfg(feature = "rocm")]
pub fn wait_idle_with_timeout(
    queue: &crate::kfd::AqlQueue,
    timeout: Duration,
) -> Result<(), String> {
    let start = Instant::now();
    loop {
        match queue.wait_idle() {
            Ok(()) => return Ok(()),
            Err(e) => {
                if start.elapsed() > timeout {
                    return Err(format!("Timeout after {:.1}s: {}", timeout.as_secs_f32(), e));
                }
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
