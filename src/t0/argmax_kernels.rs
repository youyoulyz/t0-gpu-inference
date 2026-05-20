//! Argmax GPU kernel — find the index of the maximum value per row.
//!
//! # Algorithm
//! Each workgroup processes one row. Each thread holds one element.
//! A two-phase approach:
//!
//! Phase 1: Use wg_reduce_max to find the maximum value per row
//! Phase 2: Each thread checks if its value == max_val; if so, candidate = tid,
//!          else candidate = 255 (invalid). Then wg_reduce_min finds the
//!          smallest thread index that has the max value (tie-breaking to
//!          the first occurrence).
//!
//! Constraint: cols ≤ WG_SIZE (256).
//! For larger cols, a multi-pass chunked argmax is needed.

use super::block_dsl::*;
use super::ir::Target;

/// Workgroup size for argmax kernel.
const WG_SIZE: u32 = 256;

/// Build an argmax kernel using BlockDSL.
///
/// Kernarg layout: [input_ptr:u64, output_ptr:u64, cols:u32, n_rows:u32]
/// Grid: (n_rows * WG_SIZE, 1, 1) — one WG per row
///
/// Output: [n_rows] of f32 (indices 0..255, can be cast to u32 on host)
///
/// Constraint: cols ≤ WG_SIZE (256)
pub fn build_argmax() -> BlockKernel {
    let mut kb = BlockKernel::new("argmax", WG_SIZE);

    // Kernargs
    let input_ptr = kb.arg_ptr("input");
    let output_ptr = kb.arg_ptr("output");
    let cols = kb.arg_u32("cols");

    // Thread/WG IDs
    let tid = kb.thread_id();
    let row_idx = kb.program_id(0); // row index

    // Offset into input: row_idx * cols + tid
    let row_base = row_idx.mul(&mut kb, cols);
    let offset = row_base.add(&mut kb, tid);
    let mask = tid.lt(&mut kb, cols); // OOB lanes masked

    // Load element (OOB threads get -inf)
    let x = kb.load(input_ptr, offset, mask);
    let neg_inf = kb.const_f32(f32::NEG_INFINITY);
    let val = mask.select(&mut kb, x, neg_inf);

    // Phase 1: WG-level max reduction
    let max_val = kb.wg_reduce_max(val);

    // Phase 2: Find the first thread whose value equals the max
    // candidate = (val == max_val) ? tid : 255
    // Use nested selects: if val < max_val → invalid, if val > max_val → invalid, else → tid
    let is_less = val.lt_f32(&mut kb, max_val);
    let is_greater = val.gt_f32(&mut kb, max_val);
    let invalid = kb.const_f32(255.0);
    let tid_f = kb.thread_id().to_f32(&mut kb);
    let after_lt = is_less.select(&mut kb, invalid, tid_f);
    let candidate = is_greater.select(&mut kb, invalid, after_lt);

    // WG-level min reduction to find the smallest tid with max_val
    let argmax_idx = kb.wg_reduce_min(candidate);

    // All threads write the result (same value), masked by row validity
    // Only thread 0 actually writes to avoid redundant stores
    let one = kb.const_u32(1);
    let is_zero = kb.thread_id().lt(&mut kb, one);
    kb.store(output_ptr, row_idx, argmax_idx, is_zero);

    kb
}

/// Convenience: compute grid dimensions for argmax.
pub fn argmax_grid(n_rows: u32) -> (u32, u32) {
    (n_rows * WG_SIZE, 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_argmax_kernel_compiles() {
        let kb = build_argmax();
        let ck = kb.compile_via_ssa(Target::GFX1100).expect("argmax should compile");
        assert!(!ck.elf.is_empty());
        eprintln!("✓ Argmax: {} bytes ELF, wg={:?}, lds={}",
            ck.elf.len(), ck.workgroup_size, ck.lds_size);
    }

    #[test]
    fn test_argmax_grid_dims() {
        let (gx, gy) = argmax_grid(1);
        assert_eq!(gx, 256);
        assert_eq!(gy, 1);

        let (gx, gy) = argmax_grid(32);
        assert_eq!(gx, 8192);
        assert_eq!(gy, 1);
    }
}
