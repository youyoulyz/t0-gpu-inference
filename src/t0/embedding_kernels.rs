//! Embedding GPU kernels — forward (lookup) and backward (scatter_add).
//!
//! # Algorithm
//!
//! Forward: table lookup
//! ```text
//! output[i, :] = embedding_table[indices[i], :]
//! ```
//!
//! Backward: scatter-add gradients back to embedding table
//! ```text
//! grad_table[indices[i], :] += grad_output[i, :]
//! ```
//!
//! # Design
//!
//! Forward: Each WG processes one token. Each thread copies one element of the
//! embedding dimension. WG_SIZE = 256, so dim ≤ 256 per thread (or multi-pass).
//! Backward: Uses atomic_add for scatter (multiple tokens may map to same row).

use super::block_dsl::*;
use super::ir::Target;

const WG_SIZE: u32 = 256;

/// Build embedding forward (lookup) kernel.
///
/// Kernarg layout: [table:u64, indices:u64, output:u64, dim:u32]
/// Grid: (num_tokens * WG_SIZE, 1, 1) — one WG per token
///
/// Constraint: dim ≤ WG_SIZE (256)
pub fn build_embedding_forward() -> BlockKernel {
    let mut kb = BlockKernel::new("embedding_fwd", WG_SIZE);

    let table_ptr = kb.arg_ptr("table");        // [vocab_size, dim] f32
    let indices_ptr = kb.arg_ptr("indices");     // [num_tokens] u32
    let output_ptr = kb.arg_ptr("output");       // [num_tokens, dim] f32
    let dim = kb.arg_u32("dim");

    let tid = kb.thread_id();
    let pid = kb.program_id(0); // token index

    let mask = tid.lt(&mut kb, dim);

    // Load index for this token: indices[pid]
    let zero_u = kb.const_u32(0);
    let pid_scalar = tid.mul(&mut kb, zero_u).add(&mut kb, pid);
    let idx = kb.load_u32(indices_ptr, pid_scalar, mask);

    // table_offset = idx * dim + tid
    let table_row = idx.mul(&mut kb, dim);
    let table_offset = table_row.add(&mut kb, tid);

    // output_offset = pid * dim + tid
    let out_row = pid.mul(&mut kb, dim);
    let out_offset = out_row.add(&mut kb, tid);

    // Copy: output[pid, tid] = table[idx, tid]
    let val = kb.load(table_ptr, table_offset, mask);
    kb.store(output_ptr, out_offset, val, mask);

    kb
}

/// Build embedding backward (scatter_add) kernel.
///
/// Kernarg layout: [grad_table:u64, indices:u64, grad_output:u64, dim:u32]
/// Grid: (num_tokens * WG_SIZE, 1, 1) — one WG per token
///
/// grad_table[indices[pid], tid] += grad_output[pid, tid]
///
/// Uses atomic_add since multiple tokens may share the same index.
pub fn build_embedding_backward() -> BlockKernel {
    let mut kb = BlockKernel::new("embedding_bwd", WG_SIZE);

    let grad_table_ptr = kb.arg_ptr("grad_table");
    let indices_ptr = kb.arg_ptr("indices");
    let grad_output_ptr = kb.arg_ptr("grad_output");
    let dim = kb.arg_u32("dim");

    let tid = kb.thread_id();
    let pid = kb.program_id(0);

    let mask = tid.lt(&mut kb, dim);

    // Load index
    let zero_u = kb.const_u32(0);
    let pid_scalar = tid.mul(&mut kb, zero_u).add(&mut kb, pid);
    let idx = kb.load_u32(indices_ptr, pid_scalar, mask);

    // grad_output[pid, tid]
    let out_row = pid.mul(&mut kb, dim);
    let out_offset = out_row.add(&mut kb, tid);
    let grad_val = kb.load(grad_output_ptr, out_offset, mask);

    // atomic_add to grad_table[idx, tid]
    let table_row = idx.mul(&mut kb, dim);
    let table_offset = table_row.add(&mut kb, tid);
    kb.atomic_add_f32(grad_table_ptr, table_offset, grad_val, mask);

    kb
}

/// CPU reference: embedding forward
pub fn cpu_embedding_forward(
    table: &[f32], indices: &[u32], output: &mut [f32],
    num_tokens: usize, dim: usize,
) {
    for t in 0..num_tokens {
        let idx = indices[t] as usize;
        for d in 0..dim {
            output[t * dim + d] = table[idx * dim + d];
        }
    }
}

/// CPU reference: embedding backward
pub fn cpu_embedding_backward(
    grad_table: &mut [f32], indices: &[u32], grad_output: &[f32],
    num_tokens: usize, dim: usize,
) {
    for t in 0..num_tokens {
        let idx = indices[t] as usize;
        for d in 0..dim {
            grad_table[idx * dim + d] += grad_output[t * dim + d];
        }
    }
}

pub fn embedding_grid(num_tokens: u32) -> (u32, u32) { (num_tokens * WG_SIZE, 1) }
pub fn embedding_wg_size() -> u32 { WG_SIZE }

/// Build embedding gather kernel for decode: BF16 table → F32 output.
///
/// Each WG processes one token. Each thread loads one BF16 element
/// from the weight table row and stores it as F32.
/// Supports dim > WG_SIZE via multi-pass (for_range).
///
/// Kernarg layout: [table:u64, output:u64, token_id:u32, dim:u32]
/// Grid: (1 * WG_SIZE, 1, 1) — one WG for one token
pub fn build_embedding_gather_bf16() -> BlockKernel {
    let mut kb = BlockKernel::new("emb_gather_bf16", WG_SIZE);

    let table_ptr = kb.arg_ptr("table");
    let output_ptr = kb.arg_ptr("output");
    let token_id = kb.arg_u32("token_id");
    let dim = kb.arg_u32("dim");

    let tid = kb.thread_id();
    let zero = kb.const_u32(0);

    let row_offset = token_id.mul(&mut kb, dim);

    let iter = kb.for_range(zero, dim, WG_SIZE);
    {
        let col = iter.add(&mut kb, tid);
        let mask = col.lt(&mut kb, dim);
        let src_offset = row_offset.add(&mut kb, col);
        let val = kb.load_bf16(table_ptr, src_offset, mask);
        kb.store(output_ptr, col, val, mask);
    }
    kb.end_for(iter);

    kb
}

/// CPU reference: embedding gather BF16→F32 for single token
/// Input is raw BF16 bytes (2 bytes per element, little-endian).
pub fn cpu_embedding_gather_bf16(
    table_bf16_bytes: &[u8], token_id: u32, dim: usize,
) -> Vec<f32> {
    let row_start = (token_id as usize) * dim * 2;
    let mut out = Vec::with_capacity(dim);
    for i in 0..dim {
        let offset = row_start + i * 2;
        let bits = u16::from_le_bytes([table_bf16_bytes[offset], table_bf16_bytes[offset + 1]]);
        let f32_bits = (bits as u32) << 16;
        out.push(f32::from_bits(f32_bits));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_embedding_forward() {
        let table = vec![
            1.0, 2.0, 3.0,  // row 0
            4.0, 5.0, 6.0,  // row 1
            7.0, 8.0, 9.0,  // row 2
        ];
        let indices = vec![2u32, 0, 1];
        let mut output = vec![0.0; 9];
        cpu_embedding_forward(&table, &indices, &mut output, 3, 3);
        assert_eq!(&output[0..3], &[7.0, 8.0, 9.0]); // row 2
        assert_eq!(&output[3..6], &[1.0, 2.0, 3.0]); // row 0
        assert_eq!(&output[6..9], &[4.0, 5.0, 6.0]); // row 1
    }

    #[test]
    fn test_cpu_embedding_backward() {
        let mut grad_table = vec![0.0; 6]; // 2 rows × 3 cols
        let indices = vec![1u32, 0, 1]; // token 0→row1, token 1→row0, token 2→row1
        let grad_output = vec![
            1.0, 1.0, 1.0, // token 0
            2.0, 2.0, 2.0, // token 1
            3.0, 3.0, 3.0, // token 2
        ];
        cpu_embedding_backward(&mut grad_table, &indices, &grad_output, 3, 3);
        assert_eq!(&grad_table[0..3], &[2.0, 2.0, 2.0]); // row 0 = token 1
        assert_eq!(&grad_table[3..6], &[4.0, 4.0, 4.0]); // row 1 = token 0 + token 2
    }

    #[test]
    fn test_embedding_fwd_compiles() {
        let kb = build_embedding_forward();
        let ck = kb.compile_via_ssa(Target::GFX1100).expect("embedding fwd compile");
        assert!(!ck.elf.is_empty());
        eprintln!("✓ Embedding forward: {} bytes ELF, wg={:?}, lds={}",
            ck.elf.len(), ck.workgroup_size, ck.lds_size);
    }

    #[test]
    fn test_embedding_bwd_compiles() {
        let kb = build_embedding_backward();
        let ck = kb.compile_via_ssa(Target::GFX1100).expect("embedding bwd compile");
        assert!(!ck.elf.is_empty());
        eprintln!("✓ Embedding backward: {} bytes ELF, wg={:?}, lds={}",
            ck.elf.len(), ck.workgroup_size, ck.lds_size);
    }

    #[test]
    fn test_embedding_gather_bf16_compiles() {
        let kb = build_embedding_gather_bf16();
        let ck = kb.compile_via_ssa(Target::GFX1100).expect("embedding gather bf16 compile");
        assert!(!ck.elf.is_empty());
        eprintln!("✓ Embedding gather BF16: {} bytes ELF, wg={:?}, lds={}",
            ck.elf.len(), ck.workgroup_size, ck.lds_size);
    }

    #[cfg(feature = "rocm")]
    #[test]
    fn test_embedding_gather_bf16_gpu() {
        use crate::ignis::gpu_context::GpuRuntime;
        use crate::kfd::{GpuKernel, KernelLoadConfig};
        use std::sync::{Arc, OnceLock};

        struct SyncRt(Arc<GpuRuntime>);
        unsafe impl Sync for SyncRt {}
        unsafe impl Send for SyncRt {}
        static GPU_RT: OnceLock<SyncRt> = OnceLock::new();

        let rt = GPU_RT.get_or_init(|| {
            SyncRt(GpuRuntime::new().expect("GPU runtime"))
        }).0.clone();
        let _ = rt.wait_idle();

        let vocab_size: u32 = 256;
        let dim: u32 = 128;
        let token_id: u32 = 42;

        let table_f32: Vec<f32> = (0..vocab_size * dim).map(|i| (i as f32 * 0.01)).collect();

        let mut table_bf16_bytes = Vec::with_capacity((vocab_size * dim) as usize * 2);
        for &v in &table_f32 {
            let bits = v.to_bits();
            let bf16_bits = (bits >> 16) as u16;
            table_bf16_bytes.extend_from_slice(&bf16_bits.to_le_bytes());
        }

        let expected = cpu_embedding_gather_bf16(&table_bf16_bytes, token_id, dim as usize);

        let table_buf = rt.device.alloc_vram(table_bf16_bytes.len()).unwrap();
        table_buf.write_bytes(0, &table_bf16_bytes);

        let output_buf = rt.alloc_f32(dim as usize).unwrap();

        let kb = build_embedding_gather_bf16();
        let ck = kb.compile_via_ssa(crate::t0::ir::Target::GFX1100).expect("compile");
        let kernel = GpuKernel::load(&rt.device, &ck.elf, &KernelLoadConfig {
            workgroup_size: ck.workgroup_size, lds_size: ck.lds_size,
        }).expect("load");

        let ka = crate::kernargs![
            table_buf.gpu_addr() => u64,
            output_buf.gpu_addr() => u64,
            token_id => u32,
            dim => u32
        ];
        rt.dispatch(&kernel, [256, 1, 1], &ka).expect("dispatch");

        let gpu_output = rt.read_f32(&output_buf, dim as usize);
        let mut max_err: f32 = 0.0;
        for i in 0..dim as usize {
            let err = (gpu_output[i] - expected[i]).abs();
            max_err = max_err.max(err);
        }
        assert!(max_err < 0.01, "Embedding gather BF16 max_err={}", max_err);
        let _ = rt.wait_idle();
        eprintln!("✓ Embedding gather BF16 GPU: token_id={}, dim={}, max_err={:.2e}", token_id, dim, max_err);
    }

    #[cfg(feature = "rocm")]
    #[test]
    fn test_embedding_fwd_gpu() {
        use crate::ignis::gpu_context::GpuRuntime;
        use crate::kfd::{GpuKernel, KernelLoadConfig};
        use std::sync::{Arc, OnceLock};

        struct SyncRt(Arc<GpuRuntime>);
        unsafe impl Sync for SyncRt {}
        unsafe impl Send for SyncRt {}
        static GPU_RT: OnceLock<SyncRt> = OnceLock::new();

        let rt = GPU_RT.get_or_init(|| {
            SyncRt(GpuRuntime::new().expect("GPU runtime"))
        }).0.clone();
        let _ = rt.wait_idle();

        let vocab_size: u32 = 16;
        let dim: u32 = 64;
        let num_tokens: u32 = 4;

        let table: Vec<f32> = (0..vocab_size * dim).map(|i| (i as f32 * 0.01)).collect();
        let indices: Vec<u32> = vec![3, 7, 0, 12];
        let mut expected = vec![0.0f32; (num_tokens * dim) as usize];
        cpu_embedding_forward(&table, &indices, &mut expected, num_tokens as usize, dim as usize);

        let table_buf = rt.upload_f32(&table).unwrap();
        let indices_f32: Vec<f32> = indices.iter().map(|&i| f32::from_bits(i)).collect();
        let indices_buf = rt.upload_f32(&indices_f32).unwrap();
        let output_buf = rt.alloc_f32((num_tokens * dim) as usize).unwrap();

        let kb = build_embedding_forward();
        let ck = kb.compile_via_ssa(crate::t0::ir::Target::GFX1100).expect("compile");
        let kernel = GpuKernel::load(&rt.device, &ck.elf, &KernelLoadConfig {
            workgroup_size: ck.workgroup_size, lds_size: ck.lds_size,
        }).expect("load");

        let ka = crate::kernargs![
            table_buf.gpu_addr() => u64,
            indices_buf.gpu_addr() => u64,
            output_buf.gpu_addr() => u64,
            dim => u32
        ];
        let (grid_x, _) = embedding_grid(num_tokens);
        rt.dispatch(&kernel, [grid_x, 1, 1], &ka).expect("dispatch");

        let gpu_output = rt.read_f32(&output_buf, (num_tokens * dim) as usize);
        let mut max_err: f32 = 0.0;
        for i in 0..(num_tokens * dim) as usize {
            let err = (gpu_output[i] - expected[i]).abs();
            max_err = max_err.max(err);
        }
        assert!(max_err < 1e-6, "Embedding fwd max_err={}", max_err);
        let _ = rt.wait_idle();
        eprintln!("✓ Embedding forward GPU: {}×{}, max_err={:.2e}", num_tokens, dim, max_err);
    }
}
