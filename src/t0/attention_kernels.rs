//! Attention helper GPU kernels — gather, transpose, scale+mask, scatter.
//!
//! Used by `ignis::ops::attention::standard_attention` to keep all
//! per-head data on GPU and eliminate PCIe round-trips.

use super::block_dsl::*;

const WG_SIZE: u32 = 128;

// ── attn_gather ──

/// Per-head strided gather: extract `[rows, head_dim]` from
/// interleaved `[rows, n_heads * head_dim]` layout.
///
/// out[row * head_dim + col] = in[row * stride + h * head_dim + col]
///
/// Kernarg layout: [in_ptr:u64, out_ptr:u64, head_dim:u32, stride:u32, h:u32, rows:u32]
/// Grid: (rows * WG_SIZE, 1)
pub fn build_attn_gather() -> BlockKernel {
    let mut kb = BlockKernel::new("attn_gather", WG_SIZE);

    let in_ptr = kb.arg_ptr("in");
    let out_ptr = kb.arg_ptr("out");
    let head_dim = kb.arg_u32("head_dim");
    let stride = kb.arg_u32("stride"); // n_heads * head_dim
    let h = kb.arg_u32("h");
    let _rows = kb.arg_u32("rows");

    let tid = kb.thread_id();
    let pid = kb.program_id(0); // row index (pid / WG_SIZE not needed — pid IS the row)

    let in_bounds = tid.lt(&mut kb, head_dim);

    // in_offset = pid * stride + h * head_dim + tid
    let row_base = pid.mul(&mut kb, stride);
    let h_base = h.mul(&mut kb, head_dim);
    let in_offset = row_base.add(&mut kb, h_base).add(&mut kb, tid);

    // out_offset = pid * head_dim + tid
    let out_offset = pid.mul(&mut kb, head_dim).add(&mut kb, tid);

    let val = kb.load(in_ptr, in_offset, in_bounds);
    kb.store(out_ptr, out_offset, val, in_bounds);

    kb
}

pub fn attn_gather_grid(rows: u32) -> (u32, u32) { (rows * WG_SIZE, 1) }

// ── attn_transpose ──

/// Matrix transpose: `[rows, cols]` → `[cols, rows]`.
///
/// out[col * rows + row] = in[row * cols + col]
///
/// Kernarg layout: [in_ptr:u64, out_ptr:u64, cols:u32, rows:u32]
/// Grid: (rows * WG_SIZE, 1)
pub fn build_attn_transpose() -> BlockKernel {
    let mut kb = BlockKernel::new("attn_transpose", WG_SIZE);

    let in_ptr = kb.arg_ptr("in");
    let out_ptr = kb.arg_ptr("out");
    let cols = kb.arg_u32("cols");
    let _rows = kb.arg_u32("rows");

    let tid = kb.thread_id();
    let pid = kb.program_id(0); // row index

    let in_bounds = tid.lt(&mut kb, cols);

    // in_offset = pid * cols + tid
    let in_offset = pid.mul(&mut kb, cols).add(&mut kb, tid);

    // out_offset = tid * rows + pid
    let out_offset = tid.mul(&mut kb, _rows).add(&mut kb, pid);

    let val = kb.load(in_ptr, in_offset, in_bounds);
    kb.store(out_ptr, out_offset, val, in_bounds);

    kb
}

pub fn attn_transpose_grid(rows: u32) -> (u32, u32) { (rows * WG_SIZE, 1) }

// ── attn_scale (scale only, no mask) ──

/// Scale attention scores by 1/sqrt(head_dim).
///
/// Kernarg layout: [in_ptr:u64, out_ptr:u64, cols:u32, scale:f32]
/// Grid: (rows * WG_SIZE, 1)
pub fn build_attn_scale() -> BlockKernel {
    let mut kb = BlockKernel::new("attn_scale", WG_SIZE);

    let in_ptr = kb.arg_ptr("in");
    let out_ptr = kb.arg_ptr("out");
    let cols = kb.arg_u32("cols");
    let scale = kb.arg_f32("scale");

    let tid = kb.thread_id();
    let pid = kb.program_id(0);

    let row_base = pid.mul(&mut kb, cols);
    let offset = row_base.add(&mut kb, tid);
    let in_bounds = tid.lt(&mut kb, cols);

    let val = kb.load(in_ptr, offset, in_bounds);
    let scaled = val.mul(&mut kb, scale);
    kb.store(out_ptr, offset, scaled, in_bounds);

    kb
}

// ── attn_scale_causal (scale + causal mask) ──

/// Scale + causal mask for attention scores (prefill).
///
/// out[row, col] = scores[row, col] * scale    if col <= row
///               = -inf                         if col > row
///
/// Kernarg layout: [in_ptr:u64, out_ptr:u64, cols:u32, scale:f32]
/// Grid: (rows * WG_SIZE, 1)
pub fn build_attn_scale_causal() -> BlockKernel {
    let mut kb = BlockKernel::new("attn_scale_causal", WG_SIZE);

    let in_ptr = kb.arg_ptr("in");
    let out_ptr = kb.arg_ptr("out");
    let cols = kb.arg_u32("cols");
    let scale = kb.arg_f32("scale");

    let tid = kb.thread_id();
    let pid = kb.program_id(0);

    let row_base = pid.mul(&mut kb, cols);
    let offset = row_base.add(&mut kb, tid);
    let in_bounds = tid.lt(&mut kb, cols);

    let val = kb.load(in_ptr, offset, in_bounds);
    let scaled = val.mul(&mut kb, scale);

    // Causal mask: col <= row → tid < pid + 1
    let one_u = kb.const_u32(1);
    let pid_plus_one = pid.add(&mut kb, one_u);
    let is_valid = tid.lt(&mut kb, pid_plus_one);

    let neg_inf = kb.const_f32(f32::NEG_INFINITY);
    let masked = is_valid.select(&mut kb, scaled, neg_inf);
    kb.store(out_ptr, offset, masked, in_bounds);

    kb
}

pub fn attn_scale_grid(rows: u32) -> (u32, u32) { (rows * WG_SIZE, 1) }

// ── attn_scatter ──

/// Per-head output scatter: write `[rows, head_dim]` to the correct
/// position in `[rows, n_heads * head_dim]` output buffer.
///
/// out[row * stride + h * head_dim + col] = in[row * head_dim + col]
///
/// Kernarg layout: [in_ptr:u64, out_ptr:u64, head_dim:u32, stride:u32, h:u32, rows:u32]
/// Grid: (rows * WG_SIZE, 1)
pub fn build_attn_scatter() -> BlockKernel {
    let mut kb = BlockKernel::new("attn_scatter", WG_SIZE);

    let in_ptr = kb.arg_ptr("in");
    let out_ptr = kb.arg_ptr("out");
    let head_dim = kb.arg_u32("head_dim");
    let stride = kb.arg_u32("stride"); // n_heads * head_dim
    let h = kb.arg_u32("h");
    let _rows = kb.arg_u32("rows");

    let tid = kb.thread_id();
    let pid = kb.program_id(0); // row index

    let in_bounds = tid.lt(&mut kb, head_dim);

    // in_offset = pid * head_dim + tid
    let in_offset = pid.mul(&mut kb, head_dim).add(&mut kb, tid);

    // out_offset = pid * stride + h * head_dim + tid
    let row_base = pid.mul(&mut kb, stride);
    let h_base = h.mul(&mut kb, head_dim);
    let out_offset = row_base.add(&mut kb, h_base).add(&mut kb, tid);

    let val = kb.load(in_ptr, in_offset, in_bounds);
    kb.store(out_ptr, out_offset, val, in_bounds);

    kb
}

pub fn attn_scatter_grid(rows: u32) -> (u32, u32) { (rows * WG_SIZE, 1) }

// ── CPU reference functions ──

pub fn cpu_attn_gather(out: &mut [f32], inp: &[f32], head_dim: usize, stride: usize, h: usize, rows: usize) {
    for row in 0..rows {
        for col in 0..head_dim {
            out[row * head_dim + col] = inp[row * stride + h * head_dim + col];
        }
    }
}

pub fn cpu_attn_transpose(out: &mut [f32], inp: &[f32], cols: usize, rows: usize) {
    for row in 0..rows {
        for col in 0..cols {
            out[col * rows + row] = inp[row * cols + col];
        }
    }
}

pub fn cpu_attn_scale_mask(out: &mut [f32], inp: &[f32], cols: usize, scale: f32, apply_mask: bool, rows: usize) {
    for row in 0..rows {
        for col in 0..cols {
            let val = inp[row * cols + col] * scale;
            out[row * cols + col] = if apply_mask && col > row { f32::NEG_INFINITY } else { val };
        }
    }
}

pub fn cpu_attn_scatter(out: &mut [f32], inp: &[f32], head_dim: usize, stride: usize, h: usize, rows: usize) {
    for row in 0..rows {
        for col in 0..head_dim {
            out[row * stride + h * head_dim + col] = inp[row * head_dim + col];
        }
    }
}

// ── Tests ──

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::ir::Target;

    #[test]
    fn test_cpu_gather() {
        // 2 rows, 3 heads, head_dim=2 → stride=6
        let inp: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let mut out = vec![0.0f32; 4]; // 2 rows * head_dim=2
        cpu_attn_gather(&mut out, &inp, 2, 6, 1, 2);
        // row 0, head 1: inp[0*6+1*2+0]=2, inp[0*6+1*2+1]=3
        // row 1, head 1: inp[1*6+1*2+0]=8, inp[1*6+1*2+1]=9
        assert_eq!(out, vec![2.0, 3.0, 8.0, 9.0]);
    }

    #[test]
    fn test_cpu_transpose() {
        // [2, 3] → [3, 2]
        let inp = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut out = vec![0.0f32; 6];
        cpu_attn_transpose(&mut out, &inp, 3, 2);
        assert_eq!(out, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_cpu_scale_mask() {
        // 3x3 scores, scale=0.5, causal mask
        let inp = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let mut out = vec![0.0f32; 9];
        cpu_attn_scale_mask(&mut out, &inp, 3, 0.5, true, 3);
        // row 0: [0.5, -inf, -inf]
        assert_eq!(out[0], 0.5);
        assert!(out[1].is_infinite() && out[1].is_sign_negative());
        // row 1: [2.0, 2.5, -inf]
        assert_eq!(out[3], 2.0);
        assert_eq!(out[4], 2.5);
        assert!(out[5].is_infinite());
        // row 2: [3.5, 4.0, 4.5]
        assert_eq!(out[6], 3.5);
        assert_eq!(out[7], 4.0);
        assert_eq!(out[8], 4.5);
    }

    #[test]
    fn test_cpu_scale_no_mask() {
        let inp = vec![1.0, 2.0, 3.0, 4.0];
        let mut out = vec![0.0f32; 4];
        cpu_attn_scale_mask(&mut out, &inp, 2, 2.0, false, 2);
        assert_eq!(out, vec![2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn test_cpu_scatter() {
        // scatter head 1 of 3 heads, head_dim=2, 2 rows
        let inp = vec![10.0, 20.0, 30.0, 40.0];
        let mut out = vec![0.0f32; 12]; // 2 rows * stride=6
        cpu_attn_scatter(&mut out, &inp, 2, 6, 1, 2);
        // row 0: out[2]=10, out[3]=20
        // row 1: out[8]=30, out[9]=40
        assert_eq!(out[2], 10.0);
        assert_eq!(out[3], 20.0);
        assert_eq!(out[8], 30.0);
        assert_eq!(out[9], 40.0);
        // others remain 0
        assert_eq!(out[0], 0.0);
        assert_eq!(out[5], 0.0);
    }

    #[test]
    fn test_attn_gather_compiles() {
        let kb = build_attn_gather();
        let ck = kb.compile_via_ssa(Target::GFX1100).expect("attn_gather compile");
        assert!(!ck.elf.is_empty());
    }

    #[test]
    fn test_attn_transpose_compiles() {
        let kb = build_attn_transpose();
        let ck = kb.compile_via_ssa(Target::GFX1100).expect("attn_transpose compile");
        assert!(!ck.elf.is_empty());
    }

    #[test]
    fn test_attn_scale_compiles() {
        let kb = build_attn_scale();
        let ck = kb.compile_via_ssa(Target::GFX1100).expect("attn_scale compile");
        assert!(!ck.elf.is_empty());
    }

    #[test]
    fn test_attn_scale_causal_compiles() {
        let kb = build_attn_scale_causal();
        let ck = kb.compile_via_ssa(Target::GFX1100).expect("attn_scale_causal compile");
        assert!(!ck.elf.is_empty());
    }

    #[test]
    fn test_attn_scatter_compiles() {
        let kb = build_attn_scatter();
        let ck = kb.compile_via_ssa(Target::GFX1100).expect("attn_scatter compile");
        assert!(!ck.elf.is_empty());
    }
}
