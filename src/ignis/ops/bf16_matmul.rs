//! BF16 MatMul — GEMM forward and backward via T0 gemm_gen.
//!
//! Forward:  Y[M,N] = X[M,K] @ W[K,N]   (f32 in → bf16 convert → WMMA → f32 out)
//! Backward: dX[M,K] = dY[M,N] @ WT      (transpose WT, then NT GEMM)
//!           dW[K,N] = X^T @ dY           (transpose X + dY, then NT GEMM)
//!
//! Uses T0's gemm_gen for auto-selected tile configs. All bf16 buffers are
//! padded to tile-aligned sizes to prevent GPU page faults.

#[cfg(feature = "rocm")]
use std::sync::Arc;
#[cfg(feature = "rocm")]
use crate::kfd::{GpuBuffer, KfdDevice};
#[cfg(feature = "rocm")]
use super::super::tensor::{Tensor, DType};
#[cfg(feature = "rocm")]
use super::super::tape::Tape;
#[cfg(feature = "rocm")]
use super::super::gpu_context::GpuRuntime;

/// Matrix multiply: Y = X @ W
///
/// Inputs:
///   x: [M, K] f32 — activations
///   w: [K, N] f32 — weight matrix
///
/// Output:
///   y: [M, N] f32
///
/// Internally converts to bf16 for WMMA, accumulates in f32.
/// Pads M/N to tile boundaries to handle any dimension.
#[cfg(feature = "rocm")]
pub fn matmul(x: &Tensor, w: &Tensor, _device: &Arc<KfdDevice>) -> Result<Tensor, String> {
    crate::profile_scope!("matmul");
    let x_shape = x.shape();
    let w_shape = w.shape();
    assert_eq!(x_shape.len(), 2, "matmul: X must be 2D, got {:?}", x_shape);
    assert_eq!(w_shape.len(), 2, "matmul: W must be 2D, got {:?}", w_shape);

    let m = x_shape[0];
    let k = x_shape[1];
    let n = w_shape[1];
    assert_eq!(w_shape[0], k, "matmul: K mismatch X[{},{}] @ W[{},{}]", m, k, w_shape[0], n);

    let runtime = x.runtime().clone();

    // Forward GEMM: Y = X @ W
    let y_buf = dispatch_gemm_forward(&runtime, x.buffer(), w.buffer(), m, k, n)?;

    let y_arc = Arc::new(y_buf);
    let mut output = Tensor::from_buffer(y_arc, &runtime, &[m, n], DType::F32, "matmul_out");
    // Propagate requires_grad: output needs grad if any input needs grad
    if x.requires_grad() || w.requires_grad() {
        output.set_requires_grad(true);
    }

    // Record backward on tape
    if Tape::is_recording() && (x.requires_grad() || w.requires_grad()) {
        let x_id = Some(x.id());
        let w_id = Some(w.id());
        let x_needs = x.requires_grad();
        let w_needs = w.requires_grad();
        let x_buf_saved = x.buffer_arc().clone();
        let w_buf_saved = w.buffer_arc().clone();
        let mm = m; let kk = k; let nn = n;

        let node_id = Tape::record(
            "matmul",
            output.id(),
            vec![x_id, w_id],
            vec![x_needs, w_needs],
            vec![x_buf_saved, w_buf_saved],
            Box::new(move |grad_output, saved, runtime| {
                let mut grads = Vec::new();

                // dX = dY @ WT
                if x_needs {
                    let dx = gemm_backward_data(runtime, grad_output, &saved[1], mm, kk, nn)?;
                    grads.push(Some(Arc::new(dx)));
                } else {
                    grads.push(None);
                }

                // dW = X^T @ dY
                if w_needs {
                    let dw = gemm_backward_weight(runtime, &saved[0], grad_output, mm, kk, nn)?;
                    grads.push(Some(Arc::new(dw)));
                } else {
                    grads.push(None);
                }

                Ok(grads)
            }),
        );
        output.set_tape_node(node_id);
    }

    Ok(output)
}

/// Pre-compute bf16 transposed weight buffer [N, K] from f32 weight [K, N].
///
/// Call once during initialization, cache the result, and pass to
/// `matmul_with_wt_bf16` for fast repeated forwards.
#[cfg(feature = "rocm")]
pub fn precompute_wt_bf16(
    runtime: &Arc<GpuRuntime>,
    w_f32: &GpuBuffer,
    k: usize,
    n: usize,
) -> Result<GpuBuffer, String> {
    use crate::t0::gemm_gen::GemmConfig;
    let cfg = select_config(1); // M=1 for inference
    let n_pad = pad_tile(n, cfg.tile_n);
    let k_pad = pad_tile(k, cfg.tile_k);
    f32_to_bf16_transpose_gpu_padded(runtime, w_f32, k, n, n_pad, k_pad)
}

/// Pre-compute bf16 weight buffer from raw bf16 data (already transposed [N, K]).
///
/// This is the fast path for loading bf16 weights from safetensors —
/// avoids the bf16→f32→bf16 double conversion.
/// Input: bf16 buffer [N, K] (row-major, no padding)
/// Output: bf16 buffer [N_pad, K_pad] (row-major, with zero padding)
#[cfg(feature = "rocm")]
pub fn precompute_wt_bf16_from_raw(
    runtime: &Arc<GpuRuntime>,
    raw_bf16: &GpuBuffer,
    n: usize,
    k: usize,
) -> Result<GpuBuffer, String> {
    use crate::t0::gemm_gen::GemmConfig;
    let cfg = select_config(1);
    let n_pad = pad_tile(n, cfg.tile_n);
    let k_pad = pad_tile(k, cfg.tile_k);

    // Pad: copy each row of k bf16 values into a row of k_pad bf16 values
    let dst_bytes = n_pad * k_pad * 2;
    let dst = runtime.alloc((dst_bytes + 255) & !255)?;
    dst.zero();

    // Read raw bf16 data
    let src_bytes = n * k * 2;
    let mut raw = vec![0u8; src_bytes];
    raw_bf16.read(&mut raw);

    // Write with per-row padding
    let mut padded = vec![0u8; dst_bytes];
    for row in 0..n {
        let src_off = row * k * 2;
        let dst_off = row * k_pad * 2;
        padded[dst_off..dst_off + k * 2].copy_from_slice(&raw[src_off..src_off + k * 2]);
    }
    dst.write(&padded);
    Ok(dst)
}

/// MatMul with pre-transposed bf16 weight (for inference/repeated forward).
#[cfg(feature = "rocm")]
pub fn matmul_with_wt_bf16(
    x: &Tensor,
    wt_bf16: &GpuBuffer,
    m: usize, k: usize, n: usize,
    runtime: &Arc<GpuRuntime>,
) -> Result<Tensor, String> {
    crate::profile_scope!("matmul_wt_bf16");
    use crate::t0::gemm_gen::{self, GemmConfig};

    let cfg = select_config(m);
    let m_pad = pad_tile(m, cfg.tile_m);
    let n_pad = pad_tile(n, cfg.tile_n);

    let x_bf16 = f32_to_bf16_gpu_padded(runtime, x.buffer(), m, k, m_pad, k)?;

    let kernel = runtime.ensure_kernel_t0(
        &cfg.name(),
        || gemm_gen::generate(&cfg),
        [cfg.wg_size, 1, 1],
        cfg.lds_total(),
    )?;

    let y_buf = runtime.alloc_f32(m_pad * n_pad)?;
    y_buf.zero();

    let ka = gemm_gen::build_kernargs(
        x_bf16.gpu_addr(), wt_bf16.gpu_addr(), y_buf.gpu_addr(),
        k as u32, n as u32, m as u32, &cfg,
    );
    let (gx, gy) = gemm_gen::compute_grid_auto(&cfg, m as u32, n as u32);
    runtime.dispatch(&kernel, [gx, gy, 1], &ka)?;

    Ok(Tensor::from_buffer(Arc::new(y_buf), runtime, &[m, n], DType::F32, "matmul_wt_out"))
}

/// Raw f32 GEMM using bf16 WMMA pipeline (for attention and other inference ops).
///
/// Computes Y[M,N] = A[M,K] @ B[K,N] on GPU. All buffers are f32 GpuBuffers.
/// Internally converts to bf16, dispatches WMMA kernel, returns f32 output.
#[cfg(feature = "rocm")]
pub fn gemm_f32_raw(
    runtime: &Arc<GpuRuntime>,
    a_f32: &GpuBuffer,  // [M, K]
    b_f32: &GpuBuffer,  // [K, N]
    m: usize, k: usize, n: usize,
) -> Result<GpuBuffer, String> {
    crate::profile_scope!("bf16_gemm");
    crate::profiler::set_shapes(
        vec![
            crate::profiler::ShapeInfo::new(&[m, k]),
            crate::profiler::ShapeInfo::new(&[k, n]),
        ],
        vec![crate::profiler::ShapeInfo::new(&[m, n])],
    );
    dispatch_gemm_forward(runtime, a_f32, b_f32, m, k, n)
}

// ── Config selection and padding ──

/// Select tile config based on M dimension
#[cfg(feature = "rocm")]
fn select_config(m: usize) -> crate::t0::gemm_gen::GemmConfig {
    use crate::t0::gemm_gen::GemmConfig;
    if m <= 16 {
        GemmConfig::tile_16x64_k16()
    } else if m <= 32 {
        GemmConfig::tile_32x64_k16()
    } else {
        GemmConfig::tile_64x64_k16()
    }
}

/// Padded size for tile alignment
fn pad_tile(size: usize, tile: u32) -> usize {
    let t = tile as usize;
    (size + t - 1) / t * t
}

// ── Core GEMM dispatch ──

/// Forward GEMM: Y[M,N] = X[M,K] @ W[K,N]
/// Accepts raw f32 buffers, handles bf16 conversion + padding internally.
#[cfg(feature = "rocm")]
fn dispatch_gemm_forward(
    runtime: &Arc<GpuRuntime>,
    x_f32: &GpuBuffer,    // [M, K] f32
    w_f32: &GpuBuffer,    // [K, N] f32
    m: usize, k: usize, n: usize,
) -> Result<GpuBuffer, String> {
    use crate::t0::gemm_gen::{self, GemmConfig};

    let cfg = select_config(m);
    let m_pad = pad_tile(m, cfg.tile_m);
    let n_pad = pad_tile(n, cfg.tile_n);

    let k_pad_contraction = pad_tile(k, cfg.tile_k);

    // Convert A: X[M,K] → bf16 with padding to [m_pad, k_pad_contraction]
    let x_bf16 = f32_to_bf16_gpu_padded(runtime, x_f32, m, k, m_pad, k_pad_contraction)?;
    // Convert B: W[K,N] → transpose → WT[N,K] → bf16 with padding to [n_pad, k_pad_contraction]
    let wt_bf16 = f32_to_bf16_transpose_gpu_padded(runtime, w_f32, k, n, n_pad, k_pad_contraction)?;

    let kernel = runtime.ensure_kernel_t0(
        &cfg.name(),
        || gemm_gen::generate(&cfg),
        [cfg.wg_size, 1, 1],
        cfg.lds_total(),
    )?;

    let y_buf = runtime.alloc_f32(m_pad * n_pad)?;
    y_buf.zero();

    // CRITICAL: pass n_pad as N (output stride) to prevent out-of-bounds column
    // writes from wrapping to the next row when N < tile_n.
    // Pass k_pad_contraction as K to match padded input buffers.
    let ka = gemm_gen::build_kernargs(
        x_bf16.gpu_addr(), wt_bf16.gpu_addr(), y_buf.gpu_addr(),
        k_pad_contraction as u32, n_pad as u32, m as u32, &cfg,
    );
    let (gx, gy) = gemm_gen::compute_grid_auto(&cfg, m as u32, n_pad as u32);
    runtime.dispatch(&kernel, [gx, gy, 1], &ka)?;

    // Unpad output: copy [m, n] from padded [m_pad, n_pad] with stride=n_pad
    if m == m_pad && n == n_pad {
        Ok(y_buf)
    } else {
        unpad_f32(runtime, &y_buf, m, n, m_pad, n_pad)
    }
}

// ── Backward helpers ──

/// dX = dY @ W  (backward data)
/// dY: [M, N] f32, W: [K, N] f32
/// NT GEMM: A=dY[M,N], B=W[K,N], compute A @ B^T → dX[M,K]
#[cfg(feature = "rocm")]
fn gemm_backward_data(
    runtime: &Arc<GpuRuntime>,
    grad_y: &GpuBuffer,      // [M, N] f32
    w_buf: &Arc<GpuBuffer>,   // [K, N] f32
    m: usize, k: usize, n: usize,
) -> Result<GpuBuffer, String> {
    use crate::t0::gemm_gen::{self, GemmConfig};

    let cfg = select_config(m);
    let m_pad = pad_tile(m, cfg.tile_m);
    let k_pad = pad_tile(k, cfg.tile_n);
    let n_pad_k = pad_tile(n, cfg.tile_k); // pad contraction dim to tile_k

    // A = dY[M,N] → bf16 padded to [m_pad, n_pad_k]
    let dy_bf16 = f32_to_bf16_gpu_padded(runtime, grad_y, m, n, m_pad, n_pad_k)?;
    // B = W[K,N] → bf16 padded to [k_pad, n_pad_k] (B will be transposed by WMMA: B^T)
    let w_bf16 = f32_to_bf16_gpu_padded(runtime, w_buf, k, n, k_pad, n_pad_k)?;

    let kernel = runtime.ensure_kernel_t0(
        &format!("gemm_bwd_data_{}", cfg.name()),
        || gemm_gen::generate(&cfg),
        [cfg.wg_size, 1, 1],
        cfg.lds_total(),
    )?;

    let dx_buf = runtime.alloc_f32(m_pad * k_pad)?;
    dx_buf.zero();

    // Use k_pad as N (output stride) to prevent column wraparound
    // Use n_pad_k as K (contraction dim) since inputs are padded to tile_k
    let ka = gemm_gen::build_kernargs(
        dy_bf16.gpu_addr(), w_bf16.gpu_addr(), dx_buf.gpu_addr(),
        n_pad_k as u32, k_pad as u32, m as u32, &cfg,
    );
    let (gx, gy) = gemm_gen::compute_grid_auto(&cfg, m as u32, k_pad as u32);
    runtime.dispatch(&kernel, [gx, gy, 1], &ka)?;

    // Unpad output: copy [m, k] from padded [m_pad, k_pad]
    if m == m_pad && k == k_pad {
        Ok(dx_buf)
    } else {
        unpad_f32(runtime, &dx_buf, m, k, m_pad, k_pad)
    }
}

/// dW = X^T @ dY  (backward weight)
/// X: [M, K] f32, dY: [M, N] f32 → dW: [K, N]
/// Steps: transpose X→X_T[K,M], transpose dY→dY_T[N,M]
/// NT GEMM: A=X_T[K,M], B=dY_T[N,M] → A @ B^T = dW[K,N]
#[cfg(feature = "rocm")]
fn gemm_backward_weight(
    runtime: &Arc<GpuRuntime>,
    x_buf: &Arc<GpuBuffer>,  // [M, K] f32
    grad_y: &GpuBuffer,      // [M, N] f32
    m: usize, k: usize, n: usize,
) -> Result<GpuBuffer, String> {
    use crate::t0::gemm_gen::{self, GemmConfig};

    let cfg = select_config(k);
    let k_pad = pad_tile(k, cfg.tile_m);
    let n_pad = pad_tile(n, cfg.tile_n);
    let m_pad_k = pad_tile(m, cfg.tile_k); // pad contraction dim to tile_k

    // A = X_T[K,M] bf16 with padding to [k_pad, m_pad_k]
    let xt_bf16 = f32_to_bf16_transpose_gpu_padded(runtime, x_buf, m, k, k_pad, m_pad_k)?;
    // B = dY_T[N,M] bf16 with padding to [n_pad, m_pad_k]
    let dyt_bf16 = f32_to_bf16_transpose_gpu_padded(runtime, grad_y, m, n, n_pad, m_pad_k)?;

    let kernel = runtime.ensure_kernel_t0(
        &format!("gemm_bwd_wt_{}", cfg.name()),
        || gemm_gen::generate(&cfg),
        [cfg.wg_size, 1, 1],
        cfg.lds_total(),
    )?;

    let dw_buf = runtime.alloc_f32(k_pad * n_pad)?;
    dw_buf.zero();

    // Use n_pad as N (output stride) to prevent column wraparound
    // Use m_pad_k as K (contraction dim) since inputs are padded to tile_k
    let ka = gemm_gen::build_kernargs(
        xt_bf16.gpu_addr(), dyt_bf16.gpu_addr(), dw_buf.gpu_addr(),
        m_pad_k as u32, n_pad as u32, k as u32, &cfg,
    );
    let (gx, gy) = gemm_gen::compute_grid_auto(&cfg, k as u32, n_pad as u32);
    runtime.dispatch(&kernel, [gx, gy, 1], &ka)?;

    // Unpad output: copy [k, n] from padded [k_pad, n_pad]
    if k == k_pad && n == n_pad {
        Ok(dw_buf)
    } else {
        unpad_f32(runtime, &dw_buf, k, n, k_pad, n_pad)
    }
}

// ── Output unpadding helper ──

/// Strip padding from a GEMM output buffer (GPU kernel).
///
/// The GEMM kernel writes to a padded [m_pad, n_pad] buffer (stride=n_pad).
/// This copies only the valid [m, n] portion into a contiguous buffer (stride=n).
/// Uses t0_unpad_2d GPU kernel — no CPU roundtrip.
#[cfg(feature = "rocm")]
fn unpad_f32(
    runtime: &Arc<GpuRuntime>,
    padded: &GpuBuffer,
    m: usize, n: usize,
    _m_pad: usize, n_pad: usize,
) -> Result<GpuBuffer, String> {
    use crate::t0::math;

    let kernel = runtime.ensure_kernel_t0(
        "t0_unpad_2d",
        || math::t0_unpad_2d(),
        [32, 1, 1],
        0,
    )?;

    let out_buf = runtime.alloc_f32(m * n)?;

    let ka = crate::kernargs![
        padded.gpu_addr() => u64,
        out_buf.gpu_addr() => u64,
        m as u32 => u32,
        n as u32 => u32,
        n_pad as u32 => u32,
        0u32 => u32  // padding to 32 bytes
    ];

    let total = m * n;
    let grid_x = ((total as u32 + 31) / 32) * 32;
    runtime.dispatch(&kernel, [grid_x, 1, 1], &ka)?;

    Ok(out_buf)
}

// ── BF16 conversion helpers with padding ──

/// Convert f32 [rows, cols] → bf16 [rows_padded, cols_padded] with per-row padding.
/// GPU path with epl loop for dim > WG_SIZE.
#[cfg(feature = "rocm")]
fn f32_to_bf16_gpu_padded(
    runtime: &Arc<GpuRuntime>,
    src: &GpuBuffer,
    rows: usize, cols: usize,
    rows_padded: usize, cols_padded: usize,
) -> Result<GpuBuffer, String> {
    use crate::t0::elementwise_kernels::{build_f32_to_bf16_padded, f32_to_bf16_grid};

    let bf16_bytes = rows_padded * cols_padded * 2;
    let alloc_bytes = (bf16_bytes + 255) & !255;
    let dst = runtime.alloc(alloc_bytes)?;
    dst.zero();

    let kernel = runtime.ensure_kernel_blockdsl("f32_to_bf16_pad", || build_f32_to_bf16_padded())?;
    let grid = f32_to_bf16_grid(rows as u32);  // only dispatch for real rows, padded rows stay zero
    let ka = crate::kernargs![
        src.gpu_addr() => u64,
        dst.gpu_addr() => u64,
        cols as u32 => u32,
        cols_padded as u32 => u32
    ];
    runtime.dispatch(&kernel, [grid, 1, 1], &ka)?;
    Ok(dst)
}

/// Convert f32 [rows, cols] → bf16 [cols_padded, rows_padded] transposed with padding.
/// GPU path with epl loop for dim > WG_SIZE.
#[cfg(feature = "rocm")]
fn f32_to_bf16_transpose_gpu_padded(
    runtime: &Arc<GpuRuntime>,
    src: &GpuBuffer,
    rows: usize, cols: usize,
    cols_padded: usize, rows_padded: usize,
) -> Result<GpuBuffer, String> {
    use crate::t0::elementwise_kernels::{build_f32_to_bf16_transpose_padded, f32_to_bf16_grid};

    let bf16_bytes = cols_padded * rows_padded * 2;
    let alloc_bytes = (bf16_bytes + 255) & !255;
    let dst = runtime.alloc(alloc_bytes)?;
    dst.zero();

    let kernel = runtime.ensure_kernel_blockdsl("f32_to_bf16_tp", || build_f32_to_bf16_transpose_padded())?;
    let grid = f32_to_bf16_grid(rows as u32);
    let ka = crate::kernargs![
        src.gpu_addr() => u64,
        dst.gpu_addr() => u64,
        cols as u32 => u32,
        rows_padded as u32 => u32
    ];
    runtime.dispatch(&kernel, [grid, 1, 1], &ka)?;
    Ok(dst)
}

#[cfg(all(test, feature = "rocm"))]
mod gemm_debug_tests {
    use super::*;
    use std::sync::{Arc, OnceLock};
    use crate::ignis::gpu_context::GpuRuntime;
    use crate::ignis::tensor::Tensor;

    struct SyncRt(Arc<GpuRuntime>);
    unsafe impl Sync for SyncRt {}
    unsafe impl Send for SyncRt {}
    static GPU_RT: OnceLock<SyncRt> = OnceLock::new();

    fn rt() -> Arc<GpuRuntime> {
        GPU_RT.get_or_init(|| {
            SyncRt(GpuRuntime::new().expect("GPU runtime"))
        }).0.clone()
    }

    #[test]
    fn test_gemm_m3_debug() {
        let r = rt();
        // X = [3, 4], W = [4, 3] → Y = [3, 3]
        let x_data: Vec<f32> = (0..12).map(|i| (i as f32 + 1.0) * 0.1).collect();
        let w_data: Vec<f32> = (0..12).map(|i| (i as f32 + 1.0) * 0.2).collect();

        let x = Tensor::from_f32(&r, &x_data, &[3, 4], "x").unwrap();
        let w = Tensor::from_f32(&r, &w_data, &[4, 3], "w").unwrap();

        // CPU reference
        let mut expected = vec![0.0f32; 9];
        for i in 0..3 {
            for j in 0..3 {
                let mut sum = 0.0f32;
                for k in 0..4 {
                    sum += x_data[i * 4 + k] * w_data[k * 3 + j];
                }
                expected[i * 3 + j] = sum;
            }
        }

        let y = matmul(&x, &w, &r.device).unwrap();
        let y_data = y.to_f32_vec();

        eprintln!("X: {:?}", x_data);
        eprintln!("W: {:?}", w_data);
        eprintln!("GPU Y: {:?}", y_data);
        eprintln!("CPU Y: {:?}", expected);

        for i in 0..9 {
            let err = (y_data[i] - expected[i]).abs();
            eprintln!("[{}] GPU={:.4} CPU={:.4} err={:.6}", i, y_data[i], expected[i], err);
            assert!(err < 0.05, "[{}] GPU={} CPU={} err={}", i, y_data[i], expected[i], err);
        }
    }

    #[test]
    fn test_gemm_m1_works() {
        let r = rt();
        // X = [1, 4], W = [4, 3] → Y = [1, 3]
        let x_data: Vec<f32> = (0..4).map(|i| (i as f32 + 1.0) * 0.1).collect();
        let w_data: Vec<f32> = (0..12).map(|i| (i as f32 + 1.0) * 0.2).collect();

        let x = Tensor::from_f32(&r, &x_data, &[1, 4], "x").unwrap();
        let w = Tensor::from_f32(&r, &w_data, &[4, 3], "w").unwrap();

        let mut expected = vec![0.0f32; 3];
        for j in 0..3 {
            let mut sum = 0.0f32;
            for k in 0..4 {
                sum += x_data[k] * w_data[k * 3 + j];
            }
            expected[j] = sum;
        }

        let y = matmul(&x, &w, &r.device).unwrap();
        let y_data = y.to_f32_vec();

        eprintln!("M=1 GPU: {:?}", y_data);
        eprintln!("M=1 CPU: {:?}", expected);

        for i in 0..3 {
            let err = (y_data[i] - expected[i]).abs();
            assert!(err < 0.01, "[{}] GPU={} CPU={} err={}", i, y_data[i], expected[i], err);
        }
    }

    #[test]
    fn test_gemm_m3_raw_output() {
        let r = rt();
        // X = [3, 4], W = [4, 3] → Y = [3, 3]
        let x_data: Vec<f32> = (0..12).map(|i| (i as f32 + 1.0) * 0.1).collect();
        let w_data: Vec<f32> = (0..12).map(|i| (i as f32 + 1.0) * 0.2).collect();

        let x = Tensor::from_f32(&r, &x_data, &[3, 4], "x").unwrap();
        let w = Tensor::from_f32(&r, &w_data, &[4, 3], "w").unwrap();

        // Manually call dispatch_gemm_forward to get raw padded output
        let cfg = select_config(3);
        let m_pad = pad_tile(3, cfg.tile_m);
        let n_pad = pad_tile(3, cfg.tile_n);
        let k_pad = pad_tile(4, cfg.tile_k);

        eprintln!("cfg: tile_m={} tile_n={} tile_k={}", cfg.tile_m, cfg.tile_n, cfg.tile_k);
        eprintln!("m_pad={} n_pad={} k_pad={}", m_pad, n_pad, k_pad);

        let x_bf16 = f32_to_bf16_gpu_padded(&r, x.buffer(), 3, 4, m_pad, k_pad).unwrap();
        let wt_bf16 = f32_to_bf16_transpose_gpu_padded(&r, w.buffer(), 4, 3, n_pad, k_pad).unwrap();

        let kernel = r.ensure_kernel_t0(
            &cfg.name(),
            || crate::t0::gemm_gen::generate(&cfg),
            [cfg.wg_size, 1, 1],
            cfg.lds_total(),
        ).unwrap();

        let y_buf = r.alloc_f32(m_pad * n_pad).unwrap();
        y_buf.zero();

        let ka = crate::t0::gemm_gen::build_kernargs(
            x_bf16.gpu_addr(), wt_bf16.gpu_addr(), y_buf.gpu_addr(),
            k_pad as u32, n_pad as u32, 3u32, &cfg,
        );

        eprintln!("kernarg[40..44] = {:?}", &ka[40..44]);
        let m_val = u32::from_le_bytes([ka[40], ka[41], ka[42], ka[43]]);
        eprintln!("M in kernarg = {}", m_val);

        let (gx, gy) = crate::t0::gemm_gen::compute_grid_auto(&cfg, 3, n_pad as u32);
        eprintln!("grid = ({}, {})", gx, gy);

        r.dispatch(&kernel, [gx, gy, 1], &ka).unwrap();

        // Read raw padded output
        let raw = r.read_f32(&y_buf, m_pad * n_pad);
        eprintln!("raw output (m_pad={}, n_pad={}):", m_pad, n_pad);
        for i in 0..3 {
            let row: Vec<String> = (0..3).map(|j| format!("{:.4}", raw[i * n_pad + j])).collect();
            eprintln!("  row {}: [{}]", i, row.join(", "));
        }
    }

    #[test]
    fn test_gemm_attention_size() {
        // Test GEMM with attention-like dimensions: M=1, K=128, N=128
        let r = rt();
        let m = 1;
        let k = 128;
        let n = 128;

        let mut rng_state = 42u64;
        let mut rand = || -> f32 {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((rng_state >> 33) as f32 / (1u64 << 31) as f32 - 1.0) * 0.5
        };

        let x_data: Vec<f32> = (0..m * k).map(|_| rand()).collect();
        let w_data: Vec<f32> = (0..k * n).map(|_| rand()).collect();

        let x = Tensor::from_f32(&r, &x_data, &[m, k], "x").unwrap();
        let w = Tensor::from_f32(&r, &w_data, &[k, n], "w").unwrap();

        let y = matmul(&x, &w, &r.device).unwrap();
        let y_data = y.to_f32_vec();

        // CPU reference
        let mut expected = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for kk in 0..k {
                    sum += x_data[i * k + kk] * w_data[kk * n + j];
                }
                expected[i * n + j] = sum;
            }
        }

        let mut max_err: f32 = 0.0;
        for i in 0..m * n {
            let err = (y_data[i] - expected[i]).abs();
            max_err = max_err.max(err);
        }
        eprintln!("GEMM {}x{}x{} max_err={:.6}", m, k, n, max_err);
        assert!(max_err < 0.1, "GEMM max_err={}", max_err);
    }

    #[test]
    fn test_gemm_down_proj_size() {
        // Test GEMM with down_proj dimensions: M=1, K=3072, N=1024
        let r = rt();
        let m = 1;
        let k = 3072;
        let n = 1024;

        let mut rng_state = 42u64;
        let mut rand = || -> f32 {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((rng_state >> 33) as f32 / (1u64 << 31) as f32 - 1.0) * 0.01
        };

        let x_data: Vec<f32> = (0..m * k).map(|_| rand()).collect();
        let w_data: Vec<f32> = (0..k * n).map(|_| rand()).collect();

        let x = Tensor::from_f32(&r, &x_data, &[m, k], "x").unwrap();
        let w = Tensor::from_f32(&r, &w_data, &[k, n], "w").unwrap();

        let y = matmul(&x, &w, &r.device).unwrap();
        let y_data = y.to_f32_vec();

        // CPU reference
        let mut expected = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for kk in 0..k {
                    sum += x_data[i * k + kk] * w_data[kk * n + j];
                }
                expected[i * n + j] = sum;
            }
        }

        let mut max_err: f32 = 0.0;
        let mut max_rel_err: f32 = 0.0;
        for i in 0..m * n {
            let err = (y_data[i] - expected[i]).abs();
            let rel = if expected[i].abs() > 1e-6 { err / expected[i].abs() } else { 0.0 };
            max_err = max_err.max(err);
            max_rel_err = max_rel_err.max(rel);
        }
        let y_norm: f32 = y_data.iter().map(|x| x*x).sum::<f32>().sqrt();
        let e_norm: f32 = expected.iter().map(|x| x*x).sum::<f32>().sqrt();
        eprintln!("GEMM {}x{}x{}: max_err={:.6} max_rel={:.2}% y_norm={:.4} e_norm={:.4}",
            m, k, n, max_err, max_rel_err * 100.0, y_norm, e_norm);
        eprintln!("  y[0..5]: {:?}", &y_data[..5]);
        eprintln!("  e[0..5]: {:?}", &expected[..5]);
        assert!(max_rel_err < 0.05, "GEMM max_rel_err={:.2}%", max_rel_err * 100.0);
    }
}
