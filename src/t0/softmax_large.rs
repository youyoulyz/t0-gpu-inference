//! Large-row Softmax GPU kernel — chunked online softmax for vocab projection.
//!
//! Handles rows with cols > WG_SIZE (256), up to 65536 or more.
//!
//! # Algorithm (Three-Pass Chunked Softmax)
//!
//! Each workgroup processes one row. Each thread iterates through chunks of 256 elements:
//!
//! Pass 1: Each thread finds max over its elements (for_range_acc), then wg_reduce_max → global_max
//! Pass 2: Each thread finds sum of exp(x - global_max) over its elements (for_range_acc), then wg_reduce_sum → global_sum
//! Pass 3: Each thread writes output = exp(x - global_max) / global_sum
//!
//! # Kernarg layout
//! [input_ptr:u64, output_ptr:u64, cols:u32, n_chunks:u32]
//!
//! Grid: (rows * WG_SIZE, 1, 1) — one WG per row

use super::block_dsl::*;

/// Workgroup size for softmax kernels.
const WG_SIZE: u32 = 256;

/// Build a chunked softmax forward kernel for large columns.
///
/// Each workgroup processes one row. Each thread handles multiple chunks of 256 elements.
///
/// # Kernarg layout
/// [input_ptr:u64, output_ptr:u64, cols:u32, n_chunks:u32]
///
/// Grid: (rows * WG_SIZE, 1, 1)
///
/// Where n_chunks = ceil(cols / WG_SIZE)
pub fn build_softmax_large() -> BlockKernel {
    let mut kb = BlockKernel::new("softmax_large", WG_SIZE);

    // Kernargs
    let input_ptr = kb.arg_ptr("input");
    let output_ptr = kb.arg_ptr("output");
    let cols = kb.arg_u32("cols");
    let n_chunks = kb.arg_u32("n_chunks");

    // Thread/WG IDs
    let tid = kb.thread_id();
    let pid = kb.program_id(0); // row index

    // Row base offset
    let row_base = pid.mul(&mut kb, cols);

    // Constants
    let neg_inf = kb.const_f32(f32::NEG_INFINITY);
    let zero_f = kb.const_f32(0.0);
    let chunk_size = kb.const_u32(WG_SIZE);
    let zero_u = kb.const_u32(0);

    // ════════════════════════════════════════════
    // Pass 1: Compute max value in the row
    //
    // Each thread computes max over its chunk elements via for_range_acc,
    // then wg_reduce_max gives global row max.
    // ════════════════════════════════════════════

    // for chunk_idx in range(0, n_chunks, 1):
    //   offset = row_base + chunk_idx * WG_SIZE + tid
    //   if offset < cols: local_max = max(local_max, input[offset])
    let (chunk_iter, local_max) = kb.for_range_acc(zero_u, n_chunks, 1, neg_inf);
    // chunk_start = chunk_idx * WG_SIZE
    let chunk_base = chunk_iter.mul(&mut kb, chunk_size);
    let offset = row_base.add(&mut kb, chunk_base).add(&mut kb, tid);
    // Check if offset is within bounds
    let in_bounds = offset.lt(&mut kb, cols);
    let x = kb.load(input_ptr, offset, in_bounds);
    let x_safe = in_bounds.select(&mut kb, x, neg_inf);
    // Update max: new_max = max(local_max, x_safe)
    let x_is_bigger = x_safe.gt_f32(&mut kb, local_max);
    let new_max = x_is_bigger.select(&mut kb, x_safe, local_max);
    let row_max = kb.end_for_acc(chunk_iter, new_max);

    // WG-level max reduction → global row max (broadcast to all lanes)
    let global_max = kb.wg_reduce_max(row_max);

    // ════════════════════════════════════════════
    // Pass 2: Compute sum of exp(x - global_max)
    //
    // Each thread computes sum over its chunk elements via for_range_acc,
    // then wg_reduce_sum gives global row sum.
    // ════════════════════════════════════════════

    // for chunk_idx in range(0, n_chunks, 1):
    //   offset = row_base + chunk_idx * WG_SIZE + tid
    //   if offset < cols: local_sum += exp(input[offset] - global_max)
    let (chunk_iter2, local_sum) = kb.for_range_acc(zero_u, n_chunks, 1, zero_f);
    let chunk_base2 = chunk_iter2.mul(&mut kb, chunk_size);
    let offset2 = row_base.add(&mut kb, chunk_base2).add(&mut kb, tid);
    let in_bounds2 = offset2.lt(&mut kb, cols);
    let x2 = kb.load(input_ptr, offset2, in_bounds2);
    // exp(x - global_max), masked to 0 if OOB
    let shifted = x2.sub(&mut kb, global_max);
    let exp_val = shifted.exp(&mut kb);
    let exp_safe = in_bounds2.select(&mut kb, exp_val, zero_f);
    let new_sum = local_sum.add(&mut kb, exp_safe);
    let row_sum = kb.end_for_acc(chunk_iter2, new_sum);

    // WG-level sum reduction → global row sum (broadcast to all lanes)
    let global_sum = kb.wg_reduce_sum(row_sum);
    let inv_sum = global_sum.rcp(&mut kb);

    // ════════════════════════════════════════════
    // Pass 3: Write output = exp(x - global_max) / sum
    // ════════════════════════════════════════════

    let chunk_iter3 = kb.for_range(zero_u, n_chunks, 1);
    let chunk_base3 = chunk_iter3.mul(&mut kb, chunk_size);
    let offset3 = row_base.add(&mut kb, chunk_base3).add(&mut kb, tid);
    let in_bounds3 = offset3.lt(&mut kb, cols);
    let x3 = kb.load(input_ptr, offset3, in_bounds3);
    let shifted3 = x3.sub(&mut kb, global_max);
    let exp_val3 = shifted3.exp(&mut kb);
    let result = exp_val3.mul(&mut kb, inv_sum);
    // Only write if in bounds
    kb.store(output_ptr, offset3, result, in_bounds3);
    kb.end_for(chunk_iter3);

    kb
}

/// Get grid dimensions for large softmax dispatch.
pub fn softmax_large_grid(rows: u32) -> (u32, u32) {
    (rows * WG_SIZE, 1)
}

/// Compute number of chunks for given cols.
pub fn softmax_n_chunks(cols: u32) -> u32 {
    (cols + WG_SIZE - 1) / WG_SIZE
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax_large_compiles() {
        let kb = build_softmax_large();
        let ck = kb.compile_via_ssa(crate::t0::ir::Target::GFX1100)
            .expect("softmax_large should compile");
        assert!(!ck.elf.is_empty());
        eprintln!("✓ Softmax large: {} bytes ELF, wg={:?}, lds={}",
            ck.elf.len(), ck.workgroup_size, ck.lds_size);
    }

    #[test]
    fn test_softmax_n_chunks() {
        assert_eq!(softmax_n_chunks(256), 1);
        assert_eq!(softmax_n_chunks(257), 2);
        assert_eq!(softmax_n_chunks(512), 2);
        assert_eq!(softmax_n_chunks(1024), 4);
        assert_eq!(softmax_n_chunks(150_000), 586); // ~150K vocab
    }
}
