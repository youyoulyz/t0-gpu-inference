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
}
