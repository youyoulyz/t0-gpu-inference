//! Safe Mode — Three-level safety net for GPU kernel development.
//!
//! # Level 1: IR Interpreter (CPU)
//! Execute TileSSA kernels on CPU without GPU hardware.
//! Catches algorithm errors, OOB access, numerical issues.
//!
//! # Level 2: Binary Verification
//! Validate compiled ELF using llvm-mc disassembly before GPU dispatch.
//! Catches encoding errors, missing endpgm, waitcnt issues.
//!
//! # Level 3: Safe Dispatch
//! GPU dispatch with timeout + isolated queue + recovery.
//! Catches infinite loops, GPU hangs.

pub mod interpreter;
pub mod verifier;
pub mod safe_dispatch;

/// Result of a safe mode check.
#[derive(Debug)]
pub enum SafeResult<T> {
    Ok(T),
    CpuError(String),
    VerifyError(String),
    DispatchError(String),
}

impl<T> SafeResult<T> {
    pub fn is_ok(&self) -> bool { matches!(self, SafeResult::Ok(_)) }

    pub fn unwrap(self) -> T {
        match self {
            SafeResult::Ok(v) => v,
            SafeResult::CpuError(e) => panic!("CPU interpreter error: {}", e),
            SafeResult::VerifyError(e) => panic!("Binary verification error: {}", e),
            SafeResult::DispatchError(e) => panic!("GPU dispatch error: {}", e),
        }
    }

    pub fn report(&self) -> &str {
        match self {
            SafeResult::Ok(_) => "OK",
            SafeResult::CpuError(e) | SafeResult::VerifyError(e) | SafeResult::DispatchError(e) => e,
        }
    }
}
