//! Argmax GPU kernel — find the index of the maximum value per row.
//!
//! # Algorithm
//! Three variants:
//!
//! **Single-pass** (cols ≤ 256): One WG per row, each thread holds one element.
//! WG reduce_max + reduce_min to find argmax.
//!
//! **Chunked large** (cols > 256): Each WG processes one row. Each thread scans
//! multiple elements by iterating over chunks of WG_SIZE. We unroll 8 chunks
//! per "iteration" and repeat 8 times, covering up to 8*8*256 = 16384 elements.
//! For larger sizes, we chain multiple dispatches.
//!
//! **Reduce** (any size): One WG per 256-element chunk, finds local max + global
//! index. CPU reduces partial results. Single dispatch covers entire vocab.

use super::block_dsl::*;
use super::ir::Target;

const WG_SIZE: u32 = 256;

pub fn build_argmax() -> BlockKernel {
    let mut kb = BlockKernel::new("argmax", WG_SIZE);

    let input_ptr = kb.arg_ptr("input");
    let output_ptr = kb.arg_ptr("output");
    let cols = kb.arg_u32("cols");

    let tid = kb.thread_id();
    let row_idx = kb.program_id(0);

    let row_base = row_idx.mul(&mut kb, cols);
    let offset = row_base.add(&mut kb, tid);
    let mask = tid.lt(&mut kb, cols);

    let x = kb.load(input_ptr, offset, mask);
    let neg_inf = kb.const_f32(f32::NEG_INFINITY);
    let val = mask.select(&mut kb, x, neg_inf);

    let max_val = kb.wg_reduce_max(val);

    let is_less = val.lt_f32(&mut kb, max_val);
    let is_greater = val.gt_f32(&mut kb, max_val);
    let invalid = kb.const_f32(255.0);
    let tid_f = kb.thread_id().to_f32(&mut kb);
    let after_lt = is_less.select(&mut kb, invalid, tid_f);
    let candidate = is_greater.select(&mut kb, invalid, after_lt);

    let argmax_idx = kb.wg_reduce_min(candidate);

    let one = kb.const_u32(1);
    let is_zero = kb.thread_id().lt(&mut kb, one);
    kb.store(output_ptr, row_idx, argmax_idx, is_zero);

    kb
}

/// Build chunked argmax kernel for large cols.
///
/// Each thread scans multiple chunks of WG_SIZE elements, tracking the
/// local max and its index. After scanning, WG-level reduction finds
/// the global argmax.
///
/// Unrolls 32 chunks (covers 32*256 = 8192 elements per dispatch).
/// For vocab_size > 8192, chain multiple dispatches with partial results.
///
/// Kernarg layout: [input_ptr:u64, output_ptr:u64, cols:u32, _pad:u32]
/// Grid: (n_rows * WG_SIZE, 1, 1)
pub fn build_argmax_large() -> BlockKernel {
    let mut kb = BlockKernel::new("argmax_large", WG_SIZE);

    let input_ptr = kb.arg_ptr("input");
    let output_ptr = kb.arg_ptr("output");
    let cols = kb.arg_u32("cols");

    let tid = kb.thread_id();
    let row_idx = kb.program_id(0);

    let row_base = row_idx.mul(&mut kb, cols);

    let neg_inf = kb.const_f32(f32::NEG_INFINITY);
    let zero_u = kb.const_u32(0);

    let mut best_val = neg_inf;
    let mut best_idx = zero_u;

    let mut chunk_start = zero_u;

    // Unroll 32 chunks: covers 32 * 256 = 8192 elements
    for _chunk in 0..32 {
        let offset_in_chunk = chunk_start.add(&mut kb, tid);
        let offset = row_base.add(&mut kb, offset_in_chunk);

        let valid = offset_in_chunk.lt(&mut kb, cols);

        let x = kb.load(input_ptr, offset, valid);
        let val = valid.select(&mut kb, x, neg_inf);

        let is_better = val.gt_f32(&mut kb, best_val);
        best_val = is_better.select(&mut kb, val, best_val);
        best_idx = is_better.select(&mut kb, offset_in_chunk, best_idx);

        let wg_size_val = kb.const_u32(WG_SIZE);
        chunk_start = chunk_start.add(&mut kb, wg_size_val);
    }

    let global_max = kb.wg_reduce_max(best_val);

    let is_max = best_val.eq_f32(&mut kb, global_max);
    let invalid_idx = kb.const_f32(f32::MAX);
    let best_idx_f = best_idx.to_f32(&mut kb);
    let candidate = is_max.select(&mut kb, best_idx_f, invalid_idx);
    let argmax_f = kb.wg_reduce_min(candidate);

    let one = kb.const_u32(1);
    let is_zero = kb.thread_id().lt(&mut kb, one);
    kb.store(output_ptr, row_idx, argmax_f, is_zero);

    kb
}

pub fn argmax_grid(n_rows: u32) -> (u32, u32) {
    (n_rows * WG_SIZE, 1)
}

/// Build chunked argmax reduce kernel for large vocab sizes.
///
/// Each workgroup processes one chunk of WG_SIZE (256) elements.
/// Finds the local max value and its global index within the chunk.
/// Writes (max_val, global_idx) to two output buffers.
/// CPU then reduces the partial results to find the global argmax.
///
/// Kernarg layout: [input_ptr:u64, out_vals_ptr:u64, out_idxs_ptr:u64, total_cols:u32, _pad:u32]
/// Grid: (n_chunks * WG_SIZE, 1, 1)
pub fn build_argmax_reduce() -> BlockKernel {
    let mut kb = BlockKernel::new("argmax_reduce", WG_SIZE);

    let input_ptr = kb.arg_ptr("input");
    let out_vals_ptr = kb.arg_ptr("out_vals");
    let out_idxs_ptr = kb.arg_ptr("out_idxs");
    let total_cols = kb.arg_u32("total_cols");

    let tid = kb.thread_id();
    let wg_id = kb.program_id(0);
    let chunk_size_val = kb.const_u32(WG_SIZE);
    let chunk_start = wg_id.mul(&mut kb, chunk_size_val);

    let offset = chunk_start.add(&mut kb, tid);
    let valid = offset.lt(&mut kb, total_cols);
    let x = kb.load(input_ptr, offset, valid);
    let neg_inf = kb.const_f32(f32::NEG_INFINITY);
    let val = valid.select(&mut kb, x, neg_inf);

    let max_val = kb.wg_reduce_max(val);

    let is_max = val.eq_f32(&mut kb, max_val);
    let invalid_idx = kb.const_f32(f32::MAX);
    let offset_f = offset.to_f32(&mut kb);
    let candidate = is_max.select(&mut kb, offset_f, invalid_idx);
    let argmax_f = kb.wg_reduce_min(candidate);

    let one = kb.const_u32(1);
    let is_zero = tid.lt(&mut kb, one);
    kb.store(out_vals_ptr, wg_id, max_val, is_zero);
    kb.store(out_idxs_ptr, wg_id, argmax_f, is_zero);

    kb
}

pub fn argmax_reduce_grid(n_chunks: u32) -> (u32, u32) {
    (n_chunks * WG_SIZE, 1)
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
    fn test_argmax_large_kernel_compiles() {
        let kb = build_argmax_large();
        let ck = kb.compile_via_ssa(Target::GFX1100).expect("argmax_large should compile");
        assert!(!ck.elf.is_empty());
        eprintln!("✓ Argmax_large: {} bytes ELF, wg={:?}, lds={}",
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

    #[test]
    fn test_argmax_reduce_kernel_compiles() {
        let kb = build_argmax_reduce();
        let ck = kb.compile_via_ssa(Target::GFX1100).expect("argmax_reduce should compile");
        assert!(!ck.elf.is_empty());
        eprintln!("✓ Argmax_reduce: {} bytes ELF, wg={:?}, lds={}",
            ck.elf.len(), ck.workgroup_size, ck.lds_size);
    }

    #[test]
    fn test_argmax_reduce_grid_dims() {
        let (gx, gy) = argmax_reduce_grid(1);
        assert_eq!(gx, 256);
        assert_eq!(gy, 1);

        let (gx, gy) = argmax_reduce_grid(594);
        assert_eq!(gx, 594 * 256);
        assert_eq!(gy, 1);
    }
}
