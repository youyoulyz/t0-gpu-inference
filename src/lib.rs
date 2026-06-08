#![allow(dead_code)]           // Reserved hardware constants/fields
#![allow(non_camel_case_types)] // WmmaFormat: BF16_F32 etc.
#![allow(unreachable_patterns)] // Redundant Op match arms in asm_emitter

//! # T0-GPU
//!
//! RDNA3 (GFX1100) 裸金属 GPU 内核编译器 & KFD 运行时。
//!
//! - **DSL**: 声明式内核定义 → 自动编译到 GFX1100 ISA
//! - **T0 编译器**: 数学 IR → GFX1100 ISA → AMD HSA ELF
//! - **KFD 运行时**: 直接通过 /dev/kfd 驱动接口与 GPU 通信
//!
//! ## 示例 (DSL API)
//! ```ignore
//! use t0_gpu::prelude::*;
//!
//! // GEMM — 自动选择最优配置
//! let kernel = gemm(1024, 1024, 4096).compile()?;
//!
//! // 融合操作
//! let fused = KernelBuilder::new(Target::GFX1100)
//!     .op(Op::SiLU)
//!     .op(Op::Mul)
//!     .compile()?;
//! ```

// ── T0 编译器 ──
pub mod t0;

// ── 便捷导入 ──
pub mod prelude;

// ── ISA 编码器 ──
pub mod rdna3_asm;

// ── Code Object (ELF) 生成器 ──
pub mod rdna3_code_object;

// ── KFD 裸金属运行时 ──
#[cfg(feature = "rocm")]
pub mod kfd;

// ── Ignis — GPU-native autodiff framework ──
#[cfg(feature = "rocm")]
pub mod ignis;

// ── GPU Profiler ──
pub mod profiler;

// ── Bare-Metal GPU Hardware Counter Profiler ──
#[cfg(all(feature = "rocm", feature = "gfx-profiler"))]
pub mod gfx_profiler;

/// Check if T0_DEBUG environment variable is set (DEBUG mode).
/// In RELEASE mode (default), all debug readbacks and verbose logging are disabled.
pub fn t0_debug() -> bool {
    use std::sync::atomic::{AtomicI8, Ordering};
    static FLAG: AtomicI8 = AtomicI8::new(-1);
    let val = FLAG.load(Ordering::Relaxed);
    if val >= 0 {
        return val != 0;
    }
    let enabled = std::env::var("T0_DEBUG").ok().as_deref() == Some("1");
    FLAG.store(if enabled { 1 } else { 0 }, Ordering::Relaxed);
    enabled
}
