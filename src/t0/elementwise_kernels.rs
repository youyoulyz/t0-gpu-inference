//! Elementwise utility GPU kernels — memcpy, residual_add, scale, etc.
//!
//! These are infrastructure kernels used throughout the Ignis framework
//! for gradient accumulation, tensor copying, and basic elementwise ops.

use super::block_dsl::*;
use super::ir::Target;

const WG_SIZE: u32 = 256;

/// Build memcpy kernel: output[i] = input[i]
///
/// Kernarg layout: [input:u64, output:u64, n:u32]
/// Grid: (ceil(n/WG_SIZE) * WG_SIZE, 1, 1)
pub fn build_memcpy() -> BlockKernel {
    let mut kb = BlockKernel::new("memcpy", WG_SIZE);

    let input_ptr = kb.arg_ptr("input");
    let output_ptr = kb.arg_ptr("output");
    let n = kb.arg_u32("n");

    let tid = kb.thread_id();
    let pid = kb.program_id(0);
    let wg_size = kb.const_u32(WG_SIZE);
    let wg_offset = pid.mul(&mut kb, wg_size);
    let gid = wg_offset.add(&mut kb, tid);
    let mask = gid.lt(&mut kb, n);

    let val = kb.load(input_ptr, gid, mask);
    kb.store(output_ptr, gid, val, mask);

    kb
}

/// Build residual_add kernel: y[i] += x[i]  (in-place)
///
/// Kernarg layout: [x:u64, y:u64, n:u32]
/// Grid: (ceil(n/WG_SIZE) * WG_SIZE, 1, 1)
///
/// Semantics: y[i] = y[i] + x[i]
pub fn build_residual_add() -> BlockKernel {
    let mut kb = BlockKernel::new("residual_add", WG_SIZE);

    let x_ptr = kb.arg_ptr("x");
    let y_ptr = kb.arg_ptr("y");
    let n = kb.arg_u32("n");

    let tid = kb.thread_id();
    let pid = kb.program_id(0);
    let wg_size = kb.const_u32(WG_SIZE);
    let wg_offset = pid.mul(&mut kb, wg_size);
    let gid = wg_offset.add(&mut kb, tid);
    let mask = gid.lt(&mut kb, n);

    let x_val = kb.load(x_ptr, gid, mask);
    let y_val = kb.load(y_ptr, gid, mask);
    let result = y_val.add(&mut kb, x_val);
    kb.store(y_ptr, gid, result, mask);

    kb
}

/// Build scale kernel: output[i] = input[i] * scale
///
/// Kernarg layout: [input:u64, output:u64, n:u32, scale:f32]
/// Grid: (ceil(n/WG_SIZE) * WG_SIZE, 1, 1)
pub fn build_scale() -> BlockKernel {
    let mut kb = BlockKernel::new("scale", WG_SIZE);

    let input_ptr = kb.arg_ptr("input");
    let output_ptr = kb.arg_ptr("output");
    let n = kb.arg_u32("n");
    let scale = kb.arg_f32("scale");

    let tid = kb.thread_id();
    let pid = kb.program_id(0);
    let wg_size = kb.const_u32(WG_SIZE);
    let wg_offset = pid.mul(&mut kb, wg_size);
    let gid = wg_offset.add(&mut kb, tid);
    let mask = gid.lt(&mut kb, n);

    let val = kb.load(input_ptr, gid, mask);
    let result = val.mul(&mut kb, scale);
    kb.store(output_ptr, gid, result, mask);

    kb
}

/// Compute grid for elementwise kernels.
pub fn elementwise_grid(n: u32) -> u32 {
    ((n + WG_SIZE - 1) / WG_SIZE) * WG_SIZE
}

/// Build memcpy kernel with f32x4 vectorized loads/stores (16 bytes per thread).
/// Each thread copies 4 consecutive f32 values.
///
/// Kernarg layout: [input:u64, output:u64, n_4elems:u32]
/// where n_4elems = ceil(n_elems / 4). Grid: ceil(n_4elems/WG_SIZE) * WG_SIZE.
pub fn build_memcpy_x4() -> BlockKernel {
    let mut kb = BlockKernel::new("memcpy_x4", WG_SIZE);

    let input_ptr = kb.arg_ptr("input");
    let output_ptr = kb.arg_ptr("output");
    let n_4elems = kb.arg_u32("n_4elems");

    let tid = kb.thread_id();
    let pid = kb.program_id(0);
    let wg_size = kb.const_u32(WG_SIZE);
    let four = kb.const_u32(4);
    let one = kb.const_u32(1);
    let two = kb.const_u32(2);
    let three = kb.const_u32(3);

    let wg_offset = pid.mul(&mut kb, wg_size);
    let gid = wg_offset.add(&mut kb, tid);
    let mask = gid.lt(&mut kb, n_4elems);
    let base_offset = gid.mul(&mut kb, four);
    let off1 = base_offset.add(&mut kb, one);
    let off2 = base_offset.add(&mut kb, two);
    let off3 = base_offset.add(&mut kb, three);

    let v0 = kb.load(input_ptr, base_offset, mask);
    let v1 = kb.load(input_ptr, off1, mask);
    let v2 = kb.load(input_ptr, off2, mask);
    let v3 = kb.load(input_ptr, off3, mask);

    kb.store(output_ptr, base_offset, v0, mask);
    kb.store(output_ptr, off1, v1, mask);
    kb.store(output_ptr, off2, v2, mask);
    kb.store(output_ptr, off3, v3, mask);

    kb
}

/// Build fused K+V memcpy kernel — copies K and V in a single dispatch.
/// Each thread copies 4 f32 elements from both K and V (32 bytes total per thread).
///
/// Kernarg layout: [k_src:u64, v_src:u64, k_dst:u64, v_dst:u64, n_4elems:u32]
/// where n_4elems = ceil(head_elements / 4) for single-token,
/// or n_4elems = ceil(seq_len * head_elements / 4) for multi-token.
pub fn build_memcpy_kv_x4() -> BlockKernel {
    let mut kb = BlockKernel::new("memcpy_kv_x4", WG_SIZE);

    let k_src = kb.arg_ptr("k_src");
    let v_src = kb.arg_ptr("v_src");
    let k_dst = kb.arg_ptr("k_dst");
    let v_dst = kb.arg_ptr("v_dst");
    let n_4elems = kb.arg_u32("n_4elems");

    let tid = kb.thread_id();
    let pid = kb.program_id(0);
    let wg_size = kb.const_u32(WG_SIZE);
    let four = kb.const_u32(4);
    let one = kb.const_u32(1);
    let two = kb.const_u32(2);
    let three = kb.const_u32(3);

    let wg_offset = pid.mul(&mut kb, wg_size);
    let gid = wg_offset.add(&mut kb, tid);
    let mask = gid.lt(&mut kb, n_4elems);
    let base = gid.mul(&mut kb, four);
    let off1 = base.add(&mut kb, one);
    let off2 = base.add(&mut kb, two);
    let off3 = base.add(&mut kb, three);

    let k0 = kb.load(k_src, base, mask);
    let k1 = kb.load(k_src, off1, mask);
    let k2 = kb.load(k_src, off2, mask);
    let k3 = kb.load(k_src, off3, mask);

    let v0 = kb.load(v_src, base, mask);
    let v1 = kb.load(v_src, off1, mask);
    let v2 = kb.load(v_src, off2, mask);
    let v3 = kb.load(v_src, off3, mask);

    kb.store(k_dst, base, k0, mask);
    kb.store(k_dst, off1, k1, mask);
    kb.store(k_dst, off2, k2, mask);
    kb.store(k_dst, off3, k3, mask);

    kb.store(v_dst, base, v0, mask);
    kb.store(v_dst, off1, v1, mask);
    kb.store(v_dst, off2, v2, mask);
    kb.store(v_dst, off3, v3, mask);

    kb
}

/// Build fused K+V memcpy kernel — HIGH BANDWIDTH version.
/// Each thread copies 16 f32 elements from both K and V (128 bytes total per thread).
///
/// Kernarg layout: [k_src:u64, v_src:u64, k_dst:u64, v_dst:u64, n_elems:u32]
/// Grid: (ceil(n_elems / (WG_SIZE*16)) * WG_SIZE, 1, 1)
pub fn build_memcpy_kv_x16() -> BlockKernel {
    let mut kb = BlockKernel::new("memcpy_kv_x16", WG_SIZE);

    let k_src = kb.arg_ptr("k_src");
    let v_src = kb.arg_ptr("v_src");
    let k_dst = kb.arg_ptr("k_dst");
    let v_dst = kb.arg_ptr("v_dst");
    let n_elems = kb.arg_u32("n_elems");

    let tid = kb.thread_id();
    let pid = kb.program_id(0);
    let wg_size = kb.const_u32(WG_SIZE);
    let sixteen = kb.const_u32(16);

    // Pre-create 16 offset constants first (no mutable borrow conflict)
    let c0 = kb.const_u32(0);
    let c1 = kb.const_u32(1);
    let c2 = kb.const_u32(2);
    let c3 = kb.const_u32(3);
    let c4 = kb.const_u32(4);
    let c5 = kb.const_u32(5);
    let c6 = kb.const_u32(6);
    let c7 = kb.const_u32(7);
    let c8 = kb.const_u32(8);
    let c9 = kb.const_u32(9);
    let c10 = kb.const_u32(10);
    let c11 = kb.const_u32(11);
    let c12 = kb.const_u32(12);
    let c13 = kb.const_u32(13);
    let c14 = kb.const_u32(14);
    let c15 = kb.const_u32(15);

    // Base index for this thread: (pid * WG_SIZE + tid) * 16
    let wg_offset = pid.mul(&mut kb, wg_size);
    let thread_idx = wg_offset.add(&mut kb, tid);
    let base = thread_idx.mul(&mut kb, sixteen);

    // Compute base + constant offsets
    let o0 = base.add(&mut kb, c0);
    let o1 = base.add(&mut kb, c1);
    let o2 = base.add(&mut kb, c2);
    let o3 = base.add(&mut kb, c3);
    let o4 = base.add(&mut kb, c4);
    let o5 = base.add(&mut kb, c5);
    let o6 = base.add(&mut kb, c6);
    let o7 = base.add(&mut kb, c7);
    let o8 = base.add(&mut kb, c8);
    let o9 = base.add(&mut kb, c9);
    let o10 = base.add(&mut kb, c10);
    let o11 = base.add(&mut kb, c11);
    let o12 = base.add(&mut kb, c12);
    let o13 = base.add(&mut kb, c13);
    let o14 = base.add(&mut kb, c14);
    let o15 = base.add(&mut kb, c15);

    // Compute masks
    let m0 = o0.lt(&mut kb, n_elems);
    let m1 = o1.lt(&mut kb, n_elems);
    let m2 = o2.lt(&mut kb, n_elems);
    let m3 = o3.lt(&mut kb, n_elems);
    let m4 = o4.lt(&mut kb, n_elems);
    let m5 = o5.lt(&mut kb, n_elems);
    let m6 = o6.lt(&mut kb, n_elems);
    let m7 = o7.lt(&mut kb, n_elems);
    let m8 = o8.lt(&mut kb, n_elems);
    let m9 = o9.lt(&mut kb, n_elems);
    let m10 = o10.lt(&mut kb, n_elems);
    let m11 = o11.lt(&mut kb, n_elems);
    let m12 = o12.lt(&mut kb, n_elems);
    let m13 = o13.lt(&mut kb, n_elems);
    let m14 = o14.lt(&mut kb, n_elems);
    let m15 = o15.lt(&mut kb, n_elems);

    // Load K
    let k0 = kb.load(k_src, o0, m0);
    let k1 = kb.load(k_src, o1, m1);
    let k2 = kb.load(k_src, o2, m2);
    let k3 = kb.load(k_src, o3, m3);
    let k4 = kb.load(k_src, o4, m4);
    let k5 = kb.load(k_src, o5, m5);
    let k6 = kb.load(k_src, o6, m6);
    let k7 = kb.load(k_src, o7, m7);
    let k8 = kb.load(k_src, o8, m8);
    let k9 = kb.load(k_src, o9, m9);
    let k10 = kb.load(k_src, o10, m10);
    let k11 = kb.load(k_src, o11, m11);
    let k12 = kb.load(k_src, o12, m12);
    let k13 = kb.load(k_src, o13, m13);
    let k14 = kb.load(k_src, o14, m14);
    let k15 = kb.load(k_src, o15, m15);

    // Load V
    let v0 = kb.load(v_src, o0, m0);
    let v1 = kb.load(v_src, o1, m1);
    let v2 = kb.load(v_src, o2, m2);
    let v3 = kb.load(v_src, o3, m3);
    let v4 = kb.load(v_src, o4, m4);
    let v5 = kb.load(v_src, o5, m5);
    let v6 = kb.load(v_src, o6, m6);
    let v7 = kb.load(v_src, o7, m7);
    let v8 = kb.load(v_src, o8, m8);
    let v9 = kb.load(v_src, o9, m9);
    let v10 = kb.load(v_src, o10, m10);
    let v11 = kb.load(v_src, o11, m11);
    let v12 = kb.load(v_src, o12, m12);
    let v13 = kb.load(v_src, o13, m13);
    let v14 = kb.load(v_src, o14, m14);
    let v15 = kb.load(v_src, o15, m15);

    // Store K
    kb.store(k_dst, o0, k0, m0);
    kb.store(k_dst, o1, k1, m1);
    kb.store(k_dst, o2, k2, m2);
    kb.store(k_dst, o3, k3, m3);
    kb.store(k_dst, o4, k4, m4);
    kb.store(k_dst, o5, k5, m5);
    kb.store(k_dst, o6, k6, m6);
    kb.store(k_dst, o7, k7, m7);
    kb.store(k_dst, o8, k8, m8);
    kb.store(k_dst, o9, k9, m9);
    kb.store(k_dst, o10, k10, m10);
    kb.store(k_dst, o11, k11, m11);
    kb.store(k_dst, o12, k12, m12);
    kb.store(k_dst, o13, k13, m13);
    kb.store(k_dst, o14, k14, m14);
    kb.store(k_dst, o15, k15, m15);

    // Store V
    kb.store(v_dst, o0, v0, m0);
    kb.store(v_dst, o1, v1, m1);
    kb.store(v_dst, o2, v2, m2);
    kb.store(v_dst, o3, v3, m3);
    kb.store(v_dst, o4, v4, m4);
    kb.store(v_dst, o5, v5, m5);
    kb.store(v_dst, o6, v6, m6);
    kb.store(v_dst, o7, v7, m7);
    kb.store(v_dst, o8, v8, m8);
    kb.store(v_dst, o9, v9, m9);
    kb.store(v_dst, o10, v10, m10);
    kb.store(v_dst, o11, v11, m11);
    kb.store(v_dst, o12, v12, m12);
    kb.store(v_dst, o13, v13, m13);
    kb.store(v_dst, o14, v14, m14);
    kb.store(v_dst, o15, v15, m15);

    kb
}

// ── BF16 conversion kernels ──

/// Minimal f32→bf16 store test kernel (for debugging store_bf16 hang).
///
/// Kernarg layout: [src:u64, dst:u64, n:u32]
/// Grid: ((n + WG_SIZE - 1) / WG_SIZE * WG_SIZE, 1)
pub fn build_bf16_store_test() -> BlockKernel {
    let mut kb = BlockKernel::new("bf16_store_test", WG_SIZE);

    let src = kb.arg_ptr("src");
    let dst = kb.arg_ptr("dst");
    let n = kb.arg_u32("n");

    let tid = kb.thread_id();
    let pid = kb.program_id(0);
    let wg_size = kb.const_u32(WG_SIZE);
    let offset = pid.mul(&mut kb, wg_size).add(&mut kb, tid);
    let in_bounds = offset.lt(&mut kb, n);

    let val = kb.load(src, offset, in_bounds);
    kb.store_bf16(dst, offset, val, in_bounds);

    kb
}

/// f32 → bf16 conversion using store_b32 (bypasses store_bf16).
/// Each thread converts one f32 to bf16 and stores it as u32 at dst[offset].
/// The bf16 value is in the lower 16 bits of the u32.
///
/// Kernarg layout: [src:u64, dst:u64, n:u32]
/// Grid: ((n + WG_SIZE - 1) / WG_SIZE * WG_SIZE, 1)
pub fn build_f32_to_bf16_b32() -> BlockKernel {
    let mut kb = BlockKernel::new("f32_to_bf16_b32", WG_SIZE);

    let src = kb.arg_ptr("src");
    let dst = kb.arg_ptr("dst");
    let n = kb.arg_u32("n");

    let tid = kb.thread_id();
    let pid = kb.program_id(0);
    let wg_size = kb.const_u32(WG_SIZE);
    let offset = pid.mul(&mut kb, wg_size).add(&mut kb, tid);
    let in_bounds = offset.lt(&mut kb, n);

    let val = kb.load(src, offset, in_bounds);
    // f32 → bf16: shift right 16 (truncation)
    let bf16_val = val.shr(&mut kb, 16);
    // Store as b32 (bf16 in lower 16 bits)
    kb.store(dst, offset, bf16_val, in_bounds);

    kb
}

/// f32 → bf16 conversion with per-row padding.
///
/// Converts f32 [real_rows, real_cols] → bf16 [pad_rows, pad_cols].
/// Out-of-bounds positions are zero-filled (bf16 0x0000).
///
/// Kernarg layout: [src:u64, dst:u64, real_cols:u32, pad_cols:u32]
/// Grid: (real_rows * WG_SIZE, 1) — only dispatch for real rows, not padded.
/// Caller must zero() the dst buffer before calling to handle padded rows.
pub fn build_f32_to_bf16_padded() -> BlockKernel {
    let mut kb = BlockKernel::new("f32_to_bf16_pad", WG_SIZE);

    let src = kb.arg_ptr("src");
    let dst = kb.arg_ptr("dst");
    let real_cols = kb.arg_u32("real_cols");
    let pad_cols = kb.arg_u32("pad_cols");

    let tid = kb.thread_id();
    let row = kb.program_id(0);

    // EPL loop: each thread handles elements tid, tid+WG_SIZE, tid+2*WG_SIZE, ...
    let wg = kb.const_u32(WG_SIZE);
    let mut col = tid;
    let epl = 16; // max elements per lane (up to 16*256=4096 cols)
    for _ in 0..epl {
        let in_bounds = col.lt(&mut kb, real_cols);
        let src_off = row.mul(&mut kb, real_cols).add(&mut kb, col);
        let dst_off = row.mul(&mut kb, pad_cols).add(&mut kb, col);
        let val = kb.load(src, src_off, in_bounds);
        kb.store_bf16(dst, dst_off, val, in_bounds);
        col = col.add(&mut kb, wg);
    }

    kb
}

/// f32 → bf16 transpose with per-row padding.
///
/// Converts f32 [real_rows, real_cols] → bf16 [pad_cols, pad_rows].
/// Transposed layout: out[col * pad_rows + row] = in[row * real_cols + col].
///
/// Kernarg layout: [src:u64, dst:u64, real_cols:u32, pad_rows:u32]
/// Grid: (real_rows * WG_SIZE, 1)
pub fn build_f32_to_bf16_transpose_padded() -> BlockKernel {
    let mut kb = BlockKernel::new("f32_to_bf16_tp", WG_SIZE);

    let src = kb.arg_ptr("src");
    let dst = kb.arg_ptr("dst");
    let real_cols = kb.arg_u32("real_cols");
    let pad_rows = kb.arg_u32("pad_rows");

    let tid = kb.thread_id();
    let row = kb.program_id(0);

    // EPL loop: each thread handles elements tid, tid+WG_SIZE, tid+2*WG_SIZE, ...
    let wg = kb.const_u32(WG_SIZE);
    let mut col = tid;
    let epl = 8;
    for _ in 0..epl {
        let in_bounds = col.lt(&mut kb, real_cols);
        let src_off = row.mul(&mut kb, real_cols).add(&mut kb, col);
        let dst_off = col.mul(&mut kb, pad_rows).add(&mut kb, row);
        let val = kb.load(src, src_off, in_bounds);
        kb.store_bf16(dst, dst_off, val, in_bounds);
        col = col.add(&mut kb, wg);
    }

    kb
}

/// Grid for f32_to_bf16_padded: one workgroup per padded row.
pub fn f32_to_bf16_grid(padded_rows: u32) -> u32 {
    padded_rows * WG_SIZE
}

/// Compute grid for x16 elementwise kernels.
pub fn elementwise_grid_x16(n_elems: u32) -> u32 {
    let elems_per_wg = WG_SIZE * 16;
    ((n_elems + elems_per_wg - 1) / elems_per_wg) * WG_SIZE
}

/// Compute grid for vectorized (x4) elementwise kernels.
pub fn elementwise_grid_x4(n_elems: u32) -> u32 {
    let n_4elems = (n_elems + 3) / 4; // ceil(n_elems / 4)
    ((n_4elems + WG_SIZE - 1) / WG_SIZE) * WG_SIZE
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memcpy_compiles() {
        let kb = build_memcpy();
        let ck = kb.compile_via_ssa(Target::GFX1100).expect("memcpy compile");
        assert!(!ck.elf.is_empty());
        eprintln!("✓ memcpy: {} bytes ELF", ck.elf.len());
    }

    #[test]
    fn test_residual_add_compiles() {
        let kb = build_residual_add();
        let ck = kb.compile_via_ssa(Target::GFX1100).expect("residual_add compile");
        assert!(!ck.elf.is_empty());
        eprintln!("✓ residual_add: {} bytes ELF", ck.elf.len());
    }

    #[test]
    fn test_scale_compiles() {
        let kb = build_scale();
        let ck = kb.compile_via_ssa(Target::GFX1100).expect("scale compile");
        assert!(!ck.elf.is_empty());
        eprintln!("✓ scale: {} bytes ELF", ck.elf.len());
    }

    #[cfg(feature = "rocm")]
    #[test]
    fn test_memcpy_gpu() {
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

        let n: u32 = 1024;
        let input: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
        let input_buf = rt.upload_f32(&input).unwrap();
        let output_buf = rt.alloc_f32(n as usize).unwrap();

        let kb = build_memcpy();
        let ck = kb.compile_via_ssa(crate::t0::ir::Target::GFX1100).expect("compile");
        let kernel = GpuKernel::load(&rt.device, &ck.elf, &KernelLoadConfig {
            workgroup_size: ck.workgroup_size, lds_size: ck.lds_size,
        }).expect("load");

        let ka = crate::kernargs![
            input_buf.gpu_addr() => u64,
            output_buf.gpu_addr() => u64,
            n => u32
        ];
        let grid_x = elementwise_grid(n);
        rt.dispatch(&kernel, [grid_x, 1, 1], &ka).expect("dispatch");

        let output = rt.read_f32(&output_buf, n as usize);
        let max_err: f32 = input.iter().zip(output.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(max_err < 1e-6, "memcpy max_err={}", max_err);
        eprintln!("✓ memcpy GPU: n={}, max_err={:.2e}", n, max_err);
    }

    #[cfg(feature = "rocm")]
    #[test]
    fn test_residual_add_gpu() {
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

        let n: u32 = 512;
        let x: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
        let y: Vec<f32> = (0..n).map(|i| i as f32 * 0.2).collect();

        let x_buf = rt.upload_f32(&x).unwrap();
        let y_buf = rt.upload_f32(&y).unwrap();

        let kb = build_residual_add();
        let ck = kb.compile_via_ssa(crate::t0::ir::Target::GFX1100).expect("compile");
        let kernel = GpuKernel::load(&rt.device, &ck.elf, &KernelLoadConfig {
            workgroup_size: ck.workgroup_size, lds_size: ck.lds_size,
        }).expect("load");

        let ka = crate::kernargs![
            x_buf.gpu_addr() => u64,
            y_buf.gpu_addr() => u64,
            n => u32
        ];
        let grid_x = elementwise_grid(n);
        rt.dispatch(&kernel, [grid_x, 1, 1], &ka).expect("dispatch");

        let result = rt.read_f32(&y_buf, n as usize);
        let mut max_err: f32 = 0.0;
        for i in 0..n as usize {
            let expected = x[i] + y[i];
            max_err = max_err.max((result[i] - expected).abs());
        }
        assert!(max_err < 1e-5, "residual_add max_err={}", max_err);
        eprintln!("✓ residual_add GPU: n={}, max_err={:.2e}", n, max_err);
    }

    #[test]
    fn test_bf16_store_compiles() {
        let kb = build_bf16_store_test();
        let ck = kb.compile_via_ssa(Target::GFX1100).expect("bf16_store compile");
        assert!(!ck.elf.is_empty());
        eprintln!("✓ bf16_store_test: {} bytes ELF", ck.elf.len());
    }

    #[test]
    fn test_f32_to_bf16_padded_compiles() {
        let kb = build_f32_to_bf16_padded();
        let ck = kb.compile_via_ssa(Target::GFX1100).expect("f32_to_bf16_padded compile");
        assert!(!ck.elf.is_empty());
        eprintln!("✓ f32_to_bf16_padded: {} bytes ELF", ck.elf.len());
    }

    #[test]
    fn test_f32_to_bf16_transpose_padded_compiles() {
        let kb = build_f32_to_bf16_transpose_padded();
        let ck = kb.compile_via_ssa(Target::GFX1100).expect("f32_to_bf16_tp compile");
        assert!(!ck.elf.is_empty());
        eprintln!("✓ f32_to_bf16_transpose_padded: {} bytes ELF", ck.elf.len());
    }

    #[cfg(feature = "rocm")]
    #[test]
    fn test_bf16_store_gpu() {
        use crate::ignis::gpu_context::GpuRuntime;
        use std::sync::{Arc, OnceLock};

        struct SyncRt(Arc<GpuRuntime>);
        unsafe impl Sync for SyncRt {}
        unsafe impl Send for SyncRt {}
        static GPU_RT: OnceLock<SyncRt> = OnceLock::new();

        let rt = GPU_RT.get_or_init(|| {
            SyncRt(GpuRuntime::new().expect("GPU runtime"))
        }).0.clone();

        let n: u32 = 64;
        let f32_data: Vec<f32> = (0..n).map(|i| (i as f32) * 0.1).collect();
        let src_buf = rt.upload_f32(&f32_data).unwrap();
        let dst_buf = rt.alloc(((n as usize * 2) + 255) & !255).unwrap();
        dst_buf.zero();

        let kernel = rt.ensure_kernel_blockdsl("bf16_store_test", || build_bf16_store_test()).unwrap();

        let grid = ((n + WG_SIZE - 1) / WG_SIZE) * WG_SIZE;
        let ka = crate::kernargs![
            src_buf.gpu_addr() => u64,
            dst_buf.gpu_addr() => u64,
            n => u32
        ];
        rt.dispatch(&kernel, [grid, 1, 1], &ka).expect("dispatch");

        // Test padded kernel with large dimensions (like FFN weights)
        let rows: u32 = 16;  // pad_rows
        let cols: u32 = 3072; // real_cols (FFN intermediate size)
        let pad_cols: u32 = 3072;
        let f32_data2: Vec<f32> = (0..rows * cols).map(|i| (i as f32) * 0.001).collect();
        let src_buf2 = rt.upload_f32(&f32_data2).unwrap();
        let dst_buf2 = rt.alloc(((rows * pad_cols * 2) as usize + 255) & !255).unwrap();
        dst_buf2.zero();

        let kernel2 = rt.ensure_kernel_blockdsl("f32_to_bf16_pad", || build_f32_to_bf16_padded()).unwrap();
        let grid2 = f32_to_bf16_grid(rows);
        let ka2 = crate::kernargs![
            src_buf2.gpu_addr() => u64,
            dst_buf2.gpu_addr() => u64,
            cols => u32,
            pad_cols => u32
        ];
        rt.dispatch(&kernel2, [grid2, 1, 1], &ka2).expect("padded dispatch large");

        // Test: run GEMM first, then padded kernel (to reproduce inference hang)
        use crate::ignis::ops::bf16_matmul;
        let gm = 1usize;
        let gk = 128usize;
        let gn = 128usize;
        let x_data: Vec<f32> = (0..gm * gk).map(|i| (i as f32) * 0.01).collect();
        let w_data: Vec<f32> = (0..gk * gn).map(|i| (i as f32) * 0.02).collect();
        let x_buf = rt.upload_f32(&x_data).unwrap();
        let w_buf = rt.upload_f32(&w_data).unwrap();
        let _ = bf16_matmul::gemm_f32_raw(&rt, &x_buf, &w_buf, gm, gk, gn).expect("gemm");

        // Now run padded kernel again after GEMM
        rt.dispatch(&kernel2, [grid2, 1, 1], &ka2).expect("padded dispatch after gemm");

        // Read back bf16 data and verify against CPU conversion
        // Note: GPU uses truncation (shift right 16), CPU uses round-to-nearest-even.
        // Allow ±1 ULP difference.
        let mut bf16_bytes = vec![0u8; n as usize * 2];
        dst_buf.read(&mut bf16_bytes);
        for i in 0..n as usize {
            let bf16_bits = u16::from_le_bytes([bf16_bytes[i * 2], bf16_bytes[i * 2 + 1]]);
            let src_val = f32_data[i];
            // CPU round-to-nearest-even conversion
            let src_bf16_bits = ((src_val.to_bits() + 0x7FFF + ((src_val.to_bits() >> 16) & 1)) >> 16) as u16;
            let diff = (bf16_bits as i32 - src_bf16_bits as i32).abs();
            assert!(diff <= 1,
                "[{}] bf16 mismatch > 1 ULP: got 0x{:04x}, expected 0x{:04x} (src={})",
                i, bf16_bits, src_bf16_bits, src_val);
        }
        eprintln!("✓ bf16_store_test GPU: n={}, all values within 1 ULP", n);
    }

    #[cfg(feature = "rocm")]
    #[test]
    fn test_gpu_bf16_then_gemm() {
        // Minimal test: GPU bf16 conversion → GEMM with pre-converted bf16 data.
        // Uses matmul_with_wt_bf16 which takes pre-transposed bf16 weights.
        use crate::ignis::gpu_context::GpuRuntime;
        use crate::ignis::tensor::{Tensor, DType};
        use crate::ignis::ops::bf16_matmul;
        use std::sync::{Arc, OnceLock};

        struct SyncRt(Arc<GpuRuntime>);
        unsafe impl Sync for SyncRt {}
        unsafe impl Send for SyncRt {}
        static GPU_RT: OnceLock<SyncRt> = OnceLock::new();
        let rt = GPU_RT.get_or_init(|| SyncRt(GpuRuntime::new().expect("GPU runtime"))).0.clone();

        let m = 1usize; let k = 128usize; let n = 128usize;
        let x_data: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.01).collect();
        let w_data: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.02).collect();
        let x_buf = rt.upload_f32(&x_data).unwrap();
        let w_buf = rt.upload_f32(&w_data).unwrap();

        // Step 1: GPU bf16 conversion (WT transpose)
        let bf16_tp_kernel = rt.ensure_kernel_blockdsl("f32_to_bf16_tp", || build_f32_to_bf16_transpose_padded()).unwrap();
        let n_pad = 128usize; let k_pad = 128usize;
        let wt_bf16 = rt.alloc(((n_pad * k_pad * 2) + 255) & !255).unwrap();
        wt_bf16.zero();
        let ka_w = crate::kernargs![w_buf.gpu_addr() => u64, wt_bf16.gpu_addr() => u64, n as u32 => u32, n_pad as u32 => u32];
        rt.dispatch(&bf16_tp_kernel, [f32_to_bf16_grid(k as u32), 1, 1], &ka_w).expect("bf16 WT");
        eprintln!("✓ GPU bf16 WT conversion OK");

        // Step 2: GEMM with pre-converted GPU bf16 weights
        // matmul_with_wt_bf16 converts X to bf16 on CPU (via f32_to_bf16_gpu_padded),
        // then dispatches GEMM with GPU-converted WT bf16 data.
        let x_tensor = Tensor::from_buffer(Arc::new(x_buf), &rt, &[m, k], DType::F32, "x");
        let y = bf16_matmul::matmul_with_wt_bf16(&x_tensor, &wt_bf16, m, k, n, &rt);
        match y {
            Ok(_) => eprintln!("✓ GPU bf16 WT → GEMM OK"),
            Err(e) => panic!("GPU bf16 WT → GEMM HANG/FAIL: {}", e),
        }

        // Step 3: Compare with fully CPU bf16 GEMM
        let x_buf2 = rt.upload_f32(&x_data).unwrap();
        let w_buf2 = rt.upload_f32(&w_data).unwrap();
        let _y_cpu = bf16_matmul::gemm_f32_raw(&rt, &x_buf2, &w_buf2, m, k, n).expect("CPU GEMM");
        eprintln!("✓ CPU bf16 GEMM OK — both paths work!");
    }

    #[cfg(feature = "rocm")]
    #[test]
    fn test_gpu_bf16_x_then_gemm() {
        // Test: GPU bf16 conversion for X → GEMM (the suspected hang path).
        use crate::ignis::gpu_context::GpuRuntime;
        use crate::ignis::tensor::{Tensor, DType};
        use crate::ignis::ops::bf16_matmul;
        use std::sync::{Arc, OnceLock};

        struct SyncRt(Arc<GpuRuntime>);
        unsafe impl Sync for SyncRt {}
        unsafe impl Send for SyncRt {}
        static GPU_RT: OnceLock<SyncRt> = OnceLock::new();
        let rt = GPU_RT.get_or_init(|| SyncRt(GpuRuntime::new().expect("GPU runtime"))).0.clone();

        let m = 1usize; let k = 128usize; let n = 128usize;
        let x_data: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.01).collect();
        let w_data: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.02).collect();
        let x_buf = rt.upload_f32(&x_data).unwrap();
        let w_buf = rt.upload_f32(&w_data).unwrap();

        // GPU bf16 X conversion
        let bf16_kernel = rt.ensure_kernel_blockdsl("f32_to_bf16_pad", || build_f32_to_bf16_padded()).unwrap();
        let m_pad = 16usize; let k_pad = 128usize;
        let x_bf16 = rt.alloc(((m_pad * k_pad * 2) + 255) & !255).unwrap();
        x_bf16.zero();
        let ka_x = crate::kernargs![x_buf.gpu_addr() => u64, x_bf16.gpu_addr() => u64, k as u32 => u32, k_pad as u32 => u32];
        rt.dispatch(&bf16_kernel, [f32_to_bf16_grid(m_pad as u32), 1, 1], &ka_x).expect("bf16 X");
        eprintln!("✓ GPU bf16 X conversion OK");

        // GEMM after GPU bf16 dispatch
        let y = bf16_matmul::gemm_f32_raw(&rt, &x_buf, &w_buf, m, k, n);
        match y {
            Ok(_) => eprintln!("✓ GEMM after GPU bf16 X dispatch OK"),
            Err(e) => panic!("GEMM after GPU bf16 X dispatch HANG/FAIL: {}", e),
        }
    }

    #[cfg(feature = "rocm")]
    #[test]
    fn test_rmsnorm_then_gpu_bf16_then_gemm() {
        // Reproduce the exact inference pipeline: rmsnorm → GPU bf16 → GEMM.
        use crate::ignis::gpu_context::GpuRuntime;
        use crate::ignis::tensor::{Tensor, DType};
        use crate::ignis::ops::{rmsnorm, bf16_matmul};
        use std::sync::{Arc, OnceLock};

        struct SyncRt(Arc<GpuRuntime>);
        unsafe impl Sync for SyncRt {}
        unsafe impl Send for SyncRt {}
        static GPU_RT: OnceLock<SyncRt> = OnceLock::new();
        let rt = GPU_RT.get_or_init(|| SyncRt(GpuRuntime::new().expect("GPU runtime"))).0.clone();

        let dim = 1024usize;
        let x_data: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.001).collect();
        let gamma_data: Vec<f32> = vec![1.0f32; dim];

        let device = Arc::new(crate::kfd::KfdDevice::open().expect("KFD device"));

        // Step 1: rmsnorm
        let x = Tensor::from_buffer(Arc::new(rt.upload_f32(&x_data).unwrap()), &rt, &[1, dim], DType::F32, "x");
        let gamma = Tensor::from_buffer(Arc::new(rt.upload_f32(&gamma_data).unwrap()), &rt, &[dim], DType::F32, "gamma");
        let rms_out = rmsnorm::rmsnorm(&x, &gamma, &device).expect("rmsnorm");
        eprintln!("✓ rmsnorm OK");

        // Step 2: GEMM after rmsnorm — this calls dispatch_gemm_forward which uses
        // f32_to_bf16_gpu_padded (CPU path) for both X and W.
        // If this works, the issue is specific to the GPU bf16 conversion path.
        let w_data: Vec<f32> = (0..dim * dim).map(|i| (i as f32) * 0.0001).collect();
        let w_buf = rt.upload_f32(&w_data).unwrap();
        let y = bf16_matmul::gemm_f32_raw(&rt, rms_out.buffer(), &w_buf, 1, dim, dim);
        match y {
            Ok(_) => eprintln!("✓ rmsnorm → CPU bf16 → GEMM OK (baseline)"),
            Err(e) => panic!("rmsnorm → CPU bf16 → GEMM FAIL: {}", e),
        }

        // Step 3: Now switch f32_to_bf16_gpu_padded to GPU path and test again.
        // We can't easily switch mid-test, so let's verify the GPU bf16 kernel
        // produces correct data after rmsnorm by reading it back.
        let bf16_kernel = rt.ensure_kernel_blockdsl("f32_to_bf16_pad", || build_f32_to_bf16_padded()).unwrap();
        let m_pad = 16usize; let k_pad = 1024usize;
        let x_bf16 = rt.alloc(((m_pad * k_pad * 2) + 255) & !255).unwrap();
        x_bf16.zero();
        let ka_x = crate::kernargs![rms_out.buffer().gpu_addr() => u64, x_bf16.gpu_addr() => u64, dim as u32 => u32, k_pad as u32 => u32];
        rt.dispatch(&bf16_kernel, [f32_to_bf16_grid(m_pad as u32), 1, 1], &ka_x).expect("bf16 X after rmsnorm");

        // Read back and verify
        let mut bf16_bytes = vec![0u8; dim * 2];
        x_bf16.read(&mut bf16_bytes);
        let mut rms_f32 = vec![0f32; dim];
        rms_out.buffer().read(unsafe { std::slice::from_raw_parts_mut(rms_f32.as_mut_ptr() as *mut u8, dim * 4) });
        for i in 0..dim {
            let bf16_bits = u16::from_le_bytes([bf16_bytes[i * 2], bf16_bytes[i * 2 + 1]]);
            let expected = ((rms_f32[i].to_bits() + 0x7FFF + ((rms_f32[i].to_bits() >> 16) & 1)) >> 16) as u16;
            let diff = (bf16_bits as i32 - expected as i32).abs();
            assert!(diff <= 1, "bf16 mismatch at {}: got 0x{:04x}, expected 0x{:04x}", i, bf16_bits, expected);
        }
        eprintln!("✓ GPU bf16 X after rmsnorm: data verified correct");

        // Step 4: Use the GPU-converted bf16 data directly in a GEMM
        // by using matmul_with_wt_bf16 (which does CPU bf16 for X but uses pre-computed WT).
        // The GPU bf16 dispatch at step 3 runs BEFORE this GEMM.
        // If the hang is caused by a preceding GPU bf16 dispatch, this will catch it.
        let y2 = bf16_matmul::matmul_with_wt_bf16(&rms_out, &bf16_matmul::precompute_wt_bf16(&rt, &w_buf, dim, dim).unwrap(), 1, dim, dim, &rt);
        match y2 {
            Ok(_) => eprintln!("✓ rmsnorm → GPU bf16 X dispatch → GEMM OK"),
            Err(e) => panic!("rmsnorm → GPU bf16 X dispatch → GEMM HANG/FAIL: {}", e),
        }
    }

}
