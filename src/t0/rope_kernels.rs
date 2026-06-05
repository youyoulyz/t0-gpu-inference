//! RoPE (Rotary Position Embedding) GPU kernels — forward and backward.
//!
//! # Algorithm
//!
//! RoPE applies position-dependent rotation using the "rotate_half" convention
//! (matching HuggingFace / Qwen3):
//! ```text
//! For each pair (x[i], x[i + d/2]) at position pos:
//!   θ = pos / rope_theta^(2i / d_model)
//!   x'[i]       = x[i] * cos(θ) - x[i + d/2] * sin(θ)
//!   x'[i + d/2] = x[i] * sin(θ) + x[i + d/2] * cos(θ)
//! ```
//!
//! # Design
//!
//! Each thread processes one pair of features.
//! WG processes one row (one token's embedding vector).
//! WG_SIZE = 128 (handles d_model ≤ 256 since each thread does 2 elements).
//!
//! Grid: (n_tokens * WG_SIZE, 1, 1)

use super::block_dsl::*;
use super::ir::Target;

const WG_SIZE: u32 = 128;

/// Build RoPE forward kernel.
///
/// Kernarg layout: [x:u64, out:u64, d_model:u32, _reserved:u32, pos_base:u32, rope_theta:f32]
///
/// The kernel applies RoPE to each row. Position = pos_base + pid >> n_heads_shift.
/// n_heads_shift is baked in at kernel build time (compile-time constant).
///
/// Grid: (n_rows * WG_SIZE, 1, 1)
pub fn build_rope_forward(n_heads_shift: u32) -> BlockKernel {
    let mut kb = BlockKernel::new(&format!("rope_fwd_s{}", n_heads_shift), WG_SIZE);

    let x_ptr = kb.arg_ptr("x");
    let out_ptr = kb.arg_ptr("out");
    let d_model = kb.arg_u32("d_model");
    let _reserved = kb.arg_u32("_reserved");
    let pos_base = kb.arg_u32("pos_base");
    let rope_theta = kb.arg_f32("rope_theta");

    let tid = kb.thread_id();
    let pid = kb.program_id(0);

    // rotate_half style: pair is (x[tid], x[tid + d_model/2])
    let half_d = d_model.shr(&mut kb, 1);

    // Mask: tid < d_model / 2
    let pair_mask = tid.lt(&mut kb, half_d);

    // Row base offset
    let row_base = pid.mul(&mut kb, d_model);

    // First half index and second half index
    let first_off = row_base.add(&mut kb, tid);
    let second_off = row_base.add(&mut kb, half_d).add(&mut kb, tid);

    // Load first and second half elements
    let x_first = kb.load(x_ptr, first_off, pair_mask);
    let x_second = kb.load(x_ptr, second_off, pair_mask);

    // Compute frequency: θ = pos * (1 / rope_theta^(2i/d_model))
    // = pos * exp(-ln(rope_theta) * 2i/d_model)
    let tid_f32 = tid.to_f32(&mut kb);
    let two_f = kb.const_f32(2.0);
    let two_tid_f = tid_f32.mul(&mut kb, two_f);
    let d_model_f = d_model.to_f32(&mut kb);
    let inv_d_model = d_model_f.rcp(&mut kb);
    let ratio = two_tid_f.mul(&mut kb, inv_d_model);

    let ln_theta = rope_theta.log(&mut kb);
    let neg_ln_theta = ln_theta.neg(&mut kb);
    let log_freq = ratio.mul(&mut kb, neg_ln_theta);
    let freq = log_freq.exp(&mut kb);

    // Position = pos_base + (pid >> n_heads_shift)
    let token_idx = if n_heads_shift > 0 {
        pid.shr(&mut kb, n_heads_shift as u8)
    } else {
        pid
    };
    let pos = pos_base.add(&mut kb, token_idx);
    let pos_f = pos.to_f32(&mut kb);
    let theta = pos_f.mul(&mut kb, freq);

    let cos_theta = theta.cos(&mut kb);
    let sin_theta = theta.sin(&mut kb);

    // Apply rotation (rotate_half style):
    // out[i]       = x[i] * cos - x[i+d/2] * sin
    // out[i + d/2] = x[i] * sin + x[i+d/2] * cos
    let xc = x_first.mul(&mut kb, cos_theta);
    let xs = x_second.mul(&mut kb, sin_theta);
    let out_first = xc.sub(&mut kb, xs);

    let xe_sin = x_first.mul(&mut kb, sin_theta);
    let xo_cos = x_second.mul(&mut kb, cos_theta);
    let out_second = xe_sin.add(&mut kb, xo_cos);

    kb.store(out_ptr, first_off, out_first, pair_mask);
    kb.store(out_ptr, second_off, out_second, pair_mask);

    kb
}

/// Build RoPE backward kernel.
///
/// The backward of RoPE is the inverse rotation (transpose of rotation matrix):
/// ```text
/// dx[i]       = dout[i] * cos(θ) + dout[i + d/2] * sin(θ)
/// dx[i + d/2] = -dout[i] * sin(θ) + dout[i + d/2] * cos(θ)
/// ```
///
/// Kernarg layout: [dout:u64, dx:u64, d_model:u32, n_tokens:u32, pos_base:u32, rope_theta:f32]
pub fn build_rope_backward() -> BlockKernel {
    let mut kb = BlockKernel::new("rope_bwd", WG_SIZE);

    let dout_ptr = kb.arg_ptr("dout");
    let dx_ptr = kb.arg_ptr("dx");
    let d_model = kb.arg_u32("d_model");
    let _n_tokens = kb.arg_u32("n_tokens");
    let pos_base = kb.arg_u32("pos_base");
    let rope_theta = kb.arg_f32("rope_theta");

    let tid = kb.thread_id();
    let pid = kb.program_id(0);

    let half_d = d_model.shr(&mut kb, 1);
    let pair_mask = tid.lt(&mut kb, half_d);

    let row_base = pid.mul(&mut kb, d_model);
    let first_off = row_base.add(&mut kb, tid);
    let second_off = row_base.add(&mut kb, half_d).add(&mut kb, tid);

    let dout_first = kb.load(dout_ptr, first_off, pair_mask);
    let dout_second = kb.load(dout_ptr, second_off, pair_mask);

    let tid_f32 = tid.to_f32(&mut kb);
    let two_f = kb.const_f32(2.0);
    let two_tid_f = tid_f32.mul(&mut kb, two_f);
    let d_model_f = d_model.to_f32(&mut kb);
    let inv_d_model = d_model_f.rcp(&mut kb);
    let ratio = two_tid_f.mul(&mut kb, inv_d_model);
    let ln_theta = rope_theta.log(&mut kb);
    let neg_ln_theta = ln_theta.neg(&mut kb);
    let log_freq = ratio.mul(&mut kb, neg_ln_theta);
    let freq = log_freq.exp(&mut kb);
    let pos = pos_base.add(&mut kb, pid);
    let pos_f = pos.to_f32(&mut kb);
    let theta = pos_f.mul(&mut kb, freq);
    let cos_theta = theta.cos(&mut kb);
    let sin_theta = theta.sin(&mut kb);

    // Inverse rotation (rotate_half style):
    // dx[i]       = dout[i] * cos + dout[i+d/2] * sin
    // dx[i + d/2] = -dout[i] * sin + dout[i+d/2] * cos
    let dc = dout_first.mul(&mut kb, cos_theta);
    let ds = dout_second.mul(&mut kb, sin_theta);
    let dx_first = dc.add(&mut kb, ds);

    let neg_de_sin = dout_first.mul(&mut kb, sin_theta).neg(&mut kb);
    let do_cos = dout_second.mul(&mut kb, cos_theta);
    let dx_second = neg_de_sin.add(&mut kb, do_cos);

    kb.store(dx_ptr, first_off, dx_first, pair_mask);
    kb.store(dx_ptr, second_off, dx_second, pair_mask);

    kb
}

/// CPU reference: RoPE forward (rotate_half style)
pub fn cpu_rope_forward(x: &[f32], out: &mut [f32], n_tokens: usize, d_model: usize, base: f32) {
    let half_d = d_model / 2;
    for t in 0..n_tokens {
        for i in 0..half_d {
            let freq = 1.0 / base.powf(2.0 * i as f32 / d_model as f32);
            let theta = t as f32 * freq;
            let cos_t = theta.cos();
            let sin_t = theta.sin();

            let first = t * d_model + i;
            let second = first + half_d;
            out[first]  = x[first] * cos_t - x[second] * sin_t;
            out[second] = x[first] * sin_t + x[second] * cos_t;
        }
    }
}

/// CPU reference: RoPE backward (rotate_half style)
pub fn cpu_rope_backward(dout: &[f32], dx: &mut [f32], n_tokens: usize, d_model: usize, base: f32) {
    let half_d = d_model / 2;
    for t in 0..n_tokens {
        for i in 0..half_d {
            let freq = 1.0 / base.powf(2.0 * i as f32 / d_model as f32);
            let theta = t as f32 * freq;
            let cos_t = theta.cos();
            let sin_t = theta.sin();

            let first = t * d_model + i;
            let second = first + half_d;
            dx[first]  =  dout[first] * cos_t + dout[second] * sin_t;
            dx[second] = -dout[first] * sin_t + dout[second] * cos_t;
        }
    }
}

pub fn rope_grid(n_tokens: u32) -> (u32, u32) { (n_tokens * WG_SIZE, 1) }
pub fn rope_wg_size() -> u32 { WG_SIZE }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_rope() {
        let d = 8; // 4 pairs
        let n = 2; // 2 tokens
        let x: Vec<f32> = (0..n*d).map(|i| (i as f32 + 1.0) * 0.1).collect();
        let mut out = vec![0.0f32; n * d];
        let mut dx = vec![0.0f32; n * d];

        cpu_rope_forward(&x, &mut out, n, d, 10000.0);

        // Token 0 (pos=0): θ=0 → cos=1, sin=0 → identity
        for i in 0..d {
            assert!((out[i] - x[i]).abs() < 1e-5,
                "pos=0: out[{}]={} expected {}", i, out[i], x[i]);
        }

        // Backward should be inverse
        cpu_rope_backward(&out, &mut dx, n, d, 10000.0);
        for i in 0..n*d {
            assert!((dx[i] - x[i]).abs() < 1e-4,
                "roundtrip: dx[{}]={} expected {}", i, dx[i], x[i]);
        }
    }

    #[test]
    fn test_rope_fwd_compiles() {
        let kb = build_rope_forward(0);
        let ck = kb.compile_via_ssa(Target::GFX1100).expect("RoPE fwd compile");
        assert!(!ck.elf.is_empty());
        eprintln!("✓ RoPE fwd: {} bytes ELF, wg={:?}", ck.elf.len(), ck.workgroup_size);
    }

    #[test]
    fn test_rope_bwd_compiles() {
        let kb = build_rope_backward();
        let ck = kb.compile_via_ssa(Target::GFX1100).expect("RoPE bwd compile");
        assert!(!ck.elf.is_empty());
        eprintln!("✓ RoPE bwd: {} bytes ELF, wg={:?}", ck.elf.len(), ck.workgroup_size);
    }

    #[cfg(feature = "rocm")]
    #[test]
    fn test_rope_fwd_gpu() {
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

        let n_tokens: u32 = 4;
        let d_model: u32 = 64;
        let n = (n_tokens * d_model) as usize;

        let x: Vec<f32> = (0..n).map(|i| ((i as f32 * 0.17).sin() * 2.0)).collect();
        let mut expected = vec![0.0f32; n];
        cpu_rope_forward(&x, &mut expected, n_tokens as usize, d_model as usize, 10000.0);

        let x_buf = rt.upload_f32(&x).unwrap();
        let out_buf = rt.alloc_f32(n).unwrap();

        let kb = build_rope_forward(0);
        let ck = kb.compile_via_ssa(crate::t0::ir::Target::GFX1100).expect("compile");
        let config = KernelLoadConfig {
            workgroup_size: ck.workgroup_size,
            lds_size: ck.lds_size,
        };
        let kernel = GpuKernel::load(&rt.device, &ck.elf, &config).expect("load");

        let ka = crate::kernargs![
            x_buf.gpu_addr() => u64,
            out_buf.gpu_addr() => u64,
            d_model => u32,
            n_tokens => u32,
            0u32 => u32  // pos_base
        ];
        let (grid_x, _) = rope_grid(n_tokens);
        rt.dispatch(&kernel, [grid_x, 1, 1], &ka).expect("dispatch");

        let gpu_out = rt.read_f32(&out_buf, n);

        let mut max_err: f32 = 0.0;
        for i in 0..n {
            let err = (gpu_out[i] - expected[i]).abs();
            max_err = max_err.max(err);
            assert!(err < 1e-2,
                "RoPE[{}]: gpu={:.6} cpu={:.6} err={:.6}",
                i, gpu_out[i], expected[i], err);
        }

        let _ = rt.wait_idle();
        eprintln!("✓ RoPE fwd GPU: {}×{}, max_err={:.2e}", n_tokens, d_model, max_err);
    }
}
