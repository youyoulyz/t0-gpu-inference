//! Large-row Softmax GPU kernel — chunked online softmax for vocab projection.
//!
//! Handles rows with cols > WG_SIZE (256), up to 65536 or more.
//!
//! # Algorithm (Three-Pass Chunked Softmax)
//!
//! Each workgroup processes one row. Each thread iterates through chunks of 256 elements:
//!
//! Pass 1: Each thread finds max over its elements, then wg_reduce_max → global_max
//! Pass 2: Each thread finds sum of exp(x - global_max), then wg_reduce_add → global_sum
//! Pass 3: Each thread writes output = exp(x - global_max) / global_sum
//!
//! # Key Design: Scalar VReg Accumulators
//!
//! BlockDSL's `for_range_acc` uses a scalar SGPR accumulator, which loses per-thread
//! data when copied back via `v_readfirstlane`. This TileSSA version uses a Scalar(F32)
//! VReg accumulator. The TileSSA lowering keeps the accumulator in a VReg during the
//! loop body and uses proper VReg→VReg copy on the back-edge (no v_readfirstlane).
//!
//! # Kernarg layout
//! [input_ptr:u64, output_ptr:u64, cols:u32, n_chunks:u32]
//!
//! Grid: (rows * WG_SIZE, 1, 1) — one WG per row

use super::tile_ssa::{TileFunc, TileType, ScalarDType};
use super::tile_ssa_lower::lower_elementwise_1d;
use super::dsl::CompiledKernel;
use super::ir::Target;

const WG_SIZE: u32 = 256;

/// Build a chunked softmax kernel using TileSSA with per-thread scalar VReg accumulators.
pub fn build_softmax_large() -> TileFunc {
    let mut f = TileFunc::new("softmax_large");

    // Kernargs
    let input_ptr = f.arg_ptr("input");
    let output_ptr = f.arg_ptr("output");
    let cols = f.arg_u32("cols");
    let n_chunks = f.arg_u32("n_chunks");

    // Thread/WG IDs
    let lane_id = f.thread_id_x();
    let pid = f.program_id(0); // row index

    // Constants
    let neg_inf = f.const_f32(f32::NEG_INFINITY);
    let zero_f = f.const_f32(0.0);
    let zero_u = f.const_u32(0);
    let wg_size = f.const_u32(WG_SIZE);
    let log2e = f.const_f32(1.4426950408889634); // log2(e)

    // Row base offset = pid * cols
    let row_base = f.mul(pid, cols);
    // Row end = row_base + cols (one-past-last index for this row)
    let row_end = f.add(row_base, cols);

    // ════════════════════════════════════════════
    // Pass 1: Compute max value in the row
    // ════════════════════════════════════════════

    // Scalar accumulator: per-thread max across chunks
    let lp1 = f.for_range_with_acc_runtime(zero_u, n_chunks, 1,
        neg_inf, TileType::Scalar(ScalarDType::F32));

    // idx = row_base + chunk_idx * WG_SIZE + lane_id
    let chunk_base = f.mul(lp1.iv, wg_size);
    let tmp = f.add(row_base, chunk_base);
    let idx = f.add(tmp, lane_id);

    // Mask: idx < row_end (within this row's bounds)
    let mask = f.cmp_lt(idx, row_end);

    // Load with mask, -inf for OOB
    let x = f.load(input_ptr, idx, ScalarDType::F32);
    let x_safe = f.select(mask, x, neg_inf);

    // Update max: new_max = max(acc, x_safe)
    let new_max = f.max(lp1.acc, x_safe);

    f.end_for_acc(&lp1, new_max);

    // WG-level max reduction → global row max (broadcast to all lanes)
    let global_max = f.wg_reduce_max(lp1.result, WG_SIZE);

    // ════════════════════════════════════════════
    // Pass 2: Compute sum of exp(x - global_max)
    // ════════════════════════════════════════════

    // Scalar accumulator: per-thread sum across chunks
    let lp2 = f.for_range_with_acc_runtime(zero_u, n_chunks, 1,
        zero_f, TileType::Scalar(ScalarDType::F32));

    let chunk_base2 = f.mul(lp2.iv, wg_size);
    let tmp2 = f.add(row_base, chunk_base2);
    let idx2 = f.add(tmp2, lane_id);
    let mask2 = f.cmp_lt(idx2, row_end);

    let x2 = f.load(input_ptr, idx2, ScalarDType::F32);
    let x_safe2 = f.select(mask2, x2, zero_f);

    // exp2((x - global_max) * log2e)
    let shifted2 = f.sub(x_safe2, global_max);
    let prod2 = f.mul(shifted2, log2e);
    let exp_val2 = f.exp2(prod2);
    let exp_safe2 = f.select(mask2, exp_val2, zero_f);

    // Accumulate
    let new_sum = f.add(lp2.acc, exp_safe2);

    f.end_for_acc(&lp2, new_sum);

    // WG-level sum reduction → global row sum
    let global_sum = f.wg_reduce_add(lp2.result, WG_SIZE);

    // Inverse sum for division
    let inv_sum = f.rcp(global_sum);

    // ════════════════════════════════════════════
    // Pass 3: Write output = exp(x - global_max) * inv_sum
    // ════════════════════════════════════════════

    let lp3 = f.for_range_runtime(zero_u, n_chunks, 1);

    let chunk_base3 = f.mul(lp3.iv, wg_size);
    let tmp3 = f.add(row_base, chunk_base3);
    let idx3 = f.add(tmp3, lane_id);
    let mask3 = f.cmp_lt(idx3, row_end);

    let x3 = f.load(input_ptr, idx3, ScalarDType::F32);
    let x_safe3 = f.select(mask3, x3, zero_f);

    let shifted3 = f.sub(x_safe3, global_max);
    let prod3 = f.mul(shifted3, log2e);
    let exp_val3 = f.exp2(prod3);
    let result = f.mul(exp_val3, inv_sum);

    f.store_masked(output_ptr, idx3, result, mask3);

    f.end_for(&lp3);

    // Mark kernel end
    f.return_();

    f
}

/// Compile the softmax_large kernel to ELF.
pub fn compile_softmax_large() -> Result<CompiledKernel, String> {
    let func = build_softmax_large();
    let lowered = lower_elementwise_1d(&func, WG_SIZE, 1)?;
    let elf = lowered.kernel.compile(Target::GFX1100)?;
    Ok(CompiledKernel {
        elf,
        kernarg_size: 24, // 2*u64 (ptrs) + 2*u32 (cols, n_chunks)
        workgroup_size: [WG_SIZE, 1, 1],
        lds_size: lowered.kernel.lds_size() as u32,
        name: "softmax_large".to_string(),
        args: vec![],
    })
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
        let func = build_softmax_large();
        let lowered = lower_elementwise_1d(&func, WG_SIZE, 1)
            .expect("should lower");
        let elf = lowered.kernel.compile(Target::GFX1100)
            .expect("softmax_large should compile");
        assert!(!elf.is_empty());
        eprintln!("✓ Softmax large: {} bytes ELF, wg={}, lds={}",
            elf.len(), WG_SIZE, lowered.kernel.lds_size());
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
