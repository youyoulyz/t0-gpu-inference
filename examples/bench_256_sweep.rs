//! Sweep all GEMM configs for 256³ on RX 7900 XTX
//!
//! Bypasses Ignis bf16 conversion overhead by writing bf16 data directly to GPU.
//! Measures raw kernel dispatch + execution time.
//!
//! Run: cargo run --example bench_256_sweep --features rocm --release

use t0_gpu::t0::gemm_gen::GemmConfig;
use t0_gpu::t0::tile_ir::TileGemm;

fn main() -> Result<(), String> {
    let m: u32 = 256;
    let k: u32 = 256;
    let n: u32 = 256;

    eprintln!("╔══════════════════════════════════════════════════════════════╗");
    eprintln!("║  256³ GEMM Config Sweep — AMD RX 7900 XTX (GFX1100, 96 CU)║");
    eprintln!("║  bf16 input → WMMA f32 accumulate → f32 output             ║");
    eprintln!("║  Peak: 123 TFLOPS (bf16 WMMA)                             ║");
    eprintln!("╠══════════════════════════════════════════════════════════════╣");
    eprintln!("║  NOTE: Small matrices dominated by dispatch overhead.     ║");
    eprintln!("║  Theoretical min: 0.27μs (33.5M FLOPs / 123 TF)          ║");
    eprintln!("║  Measured: ~19-28μs (70-100× overhead)                   ║");
    eprintln!("╚══════════════════════════════════════════════════════════════╝");

    #[cfg(feature = "rocm")]
    {
        use t0_gpu::ignis::gpu_context::GpuRuntime;

        let rt = GpuRuntime::new().map_err(|e| format!("GpuRuntime: {e}"))?;

        // ── Allocate bf16 data buffers directly (no CPU conversion) ──
        let x_buf = rt.alloc((m as usize) * (k as usize) * 2)?;
        let w_buf = rt.alloc((n as usize) * (k as usize) * 2)?;

        // Fill with bf16 1.0 (0x3F80)
        let ones_bf16: Vec<u8> = vec![0x3F80u16; (m * k) as usize]
            .iter().flat_map(|v| v.to_le_bytes()).collect();
        x_buf.write(&ones_bf16);
        let w_ones_bf16: Vec<u8> = vec![0x3F80u16; (n * k) as usize]
            .iter().flat_map(|v| v.to_le_bytes()).collect();
        w_buf.write(&w_ones_bf16);

        let flops = 2.0 * (m as f64) * (k as f64) * (n as f64); // 33.55M

        // ── Config space to sweep ──
        let configs: Vec<(String, GemmConfig)> = vec![
            ("16x64_k16_db".into(), GemmConfig::tile_16x64_k16()),
            ("32x64_k16_db".into(), GemmConfig::tile_32x64_k16()),
            ("32x64_k32_db".into(), GemmConfig::tile_32x64_k32()),
            ("32x128_k16_db".into(), GemmConfig::tile_32x128_k16()),
            ("64x64_k16_db".into(), GemmConfig::tile_64x64_k16()),
            ("64x64_k32_db".into(), GemmConfig::tile_64x64_k32()),
            ("64x64_k64_db".into(), GemmConfig::tile_64x64_k64()),
            ("128x64_k16_db".into(), GemmConfig::tile_128x64_k16()),
            ("128x64_k32_db".into(), GemmConfig::tile_128x64_k32()),
            ("64x128_k16_db".into(), GemmConfig::tile_64x128_k16()),
            ("64x128_k32_db".into(), GemmConfig { tile_k: 32, ..GemmConfig::tile_64x128_k16() }),
            // Variants: no LDS (direct)
            ("32x64_direct".into(), GemmConfig::tile_32x64_direct()),
            // split-K variants
            ("32x64_k16_sk2".into(), GemmConfig { split_k: Some(2), ..GemmConfig::tile_32x64_k16() }),
            ("32x64_k16_sk4".into(), GemmConfig { split_k: Some(4), ..GemmConfig::tile_32x64_k16() }),
            ("32x64_k16_sk8".into(), GemmConfig { split_k: Some(8), ..GemmConfig::tile_32x64_k16() }),
            ("32x64_k32_sk2".into(), GemmConfig { split_k: Some(2), ..GemmConfig::tile_32x64_k32() }),
            ("32x64_k32_sk4".into(), GemmConfig { split_k: Some(4), ..GemmConfig::tile_32x64_k32() }),
            ("64x64_k16_sk2".into(), GemmConfig { split_k: Some(2), ..GemmConfig::tile_64x64_k16() }),
            ("64x64_k16_sk4".into(), GemmConfig { split_k: Some(4), ..GemmConfig::tile_64x64_k16() }),
            ("64x64_k32_sk2".into(), GemmConfig { split_k: Some(2), ..GemmConfig::tile_64x64_k32() }),
            ("64x64_k32_sk4".into(), GemmConfig { split_k: Some(4), ..GemmConfig::tile_64x64_k32() }),
            ("128x64_k32_sk2".into(), GemmConfig { split_k: Some(2), ..GemmConfig::tile_128x64_k32() }),
            // No WGP mode variants
            ("32x64_k16_nowgp".into(), GemmConfig { wgp_mode: false, ..GemmConfig::tile_32x64_k16() }),
            ("64x64_k16_nowgp".into(), GemmConfig { wgp_mode: false, ..GemmConfig::tile_64x64_k16() }),
            // Swap grid = false (M-on-X)
            ("32x64_k16_MonX".into(), GemmConfig { swap_grid: false, ..GemmConfig::tile_32x64_k16() }),
            ("64x64_k16_MonX".into(), GemmConfig { swap_grid: false, ..GemmConfig::tile_64x64_k16() }),
        ];

        eprintln!("\n{:>24} | {:>7} {:>6} {:>6} {:>6} | {:>8} {:>6}",
            "Config", "VGPRs", "LDS", "WGs", "Occ", "μs/iter", "TFLOPS");
        eprintln!("{}", "-".repeat(90));

        let mut best_tflops = 0.0f64;
        let mut best_name = String::new();

        for (name, cfg) in &configs {
            // Skip infeasible configs
            if !cfg.is_feasible() {
                eprintln!("{:>24} | {:>7} {:>6} {:>6} {:>6} | {:>8} {:>6} [INFEASIBLE]",
                    name, cfg.estimated_vgprs(), "-", "-", "-", "-", "-");
                continue;
            }

            // Check K divisibility for split-K
            let sk = cfg.split_k.unwrap_or(1);
            if k % (cfg.tile_k * sk) != 0 {
                eprintln!("{:>24} | {:>7} {:>6} {:>6} {:>6} | {:>8} {:>6} [K%sk!=0]",
                    name, cfg.estimated_vgprs(), "-", "-", "-", "-", "-");
                continue;
            }

            let result = match bench_gemm_gen(&rt, &cfg, m, k, n, &x_buf, &w_buf, flops) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("{:>24} | {:>7} {:>6} {:>6} {:>6} | {:>8} {:>6} [ERR: {}]",
                        name, cfg.estimated_vgprs(), cfg.lds_total()/1024, "-", "-", "-", "-", e);
                    continue;
                }
            };

            if result.tflops > best_tflops {
                best_tflops = result.tflops;
                best_name = name.clone();
            }

            eprintln!("{:>24} | {:>7} {:>4}K {:>6} {:>5}w | {:>6.1}μs {:>5.1} TF {}{}",
                name, cfg.estimated_vgprs(), cfg.lds_total()/1024,
                result.n_wgs, result.occupancy,
                result.us, result.tflops,
                if result.tflops == best_tflops { " ← BEST" } else { "" },
                if result.tflops >= best_tflops * 0.9 { " ★" } else { "" });
        }

        eprintln!("\n╔══════════════════════════════════════════════════════════════╗");
        eprintln!("║  Best: {:>44}  ║", format!("{} ({:.1} TF)", best_name, best_tflops));
        eprintln!("║  Peak utilization: {:.1}% of 123 TF                       ║", best_tflops / 123.0 * 100.0);
        eprintln!("╚══════════════════════════════════════════════════════════════╝");

        // ── tile_ir sweep ──
        eprintln!("\n--- tile_ir configs ---");
        let tile_configs = vec![
            ("tile_32x64_k16", TileGemm::tile_32x64_k16()),
            ("tile_32x64_k32", TileGemm { tile_k: 32, ..TileGemm::tile_32x64_k16() }),
            ("tile_64x64_k16", TileGemm::tile_64x64_k16()),
            ("tile_64x64_k32", TileGemm { tile_k: 32, ..TileGemm::tile_64x64_k16() }),
            ("tile_64x64_k64", TileGemm { tile_k: 64, ..TileGemm::tile_64x64_k16() }),
            ("tile_128x64_k16", TileGemm::tile_128x64_k16()),
            ("tile_128x64_k32", TileGemm { tile_k: 32, ..TileGemm::tile_128x64_k16() }),
            ("tile_64x128_k16", TileGemm::tile_64x128_k16()),
        ];

        for (name, spec) in &tile_configs {
            let result = match bench_tile_ir(&rt, &spec, m, k, n, &x_buf, &w_buf, flops) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("{:>24} | ERR: {}", name, e);
                    continue;
                }
            };
            eprintln!("{:>24} | {:>6.1}μs {:>5.1} TF", name, result.us, result.tflops);
        }

        // ── Fused dispatch benchmark ──
        eprintln!("\n--- Fused dispatch (async dispatch, single wait at end) ---");
        eprintln!("This amortizes dispatch overhead across multiple ops:");
        let flops = 2.0 * (m as f64) * (k as f64) * (n as f64);
        let best_cfg = GemmConfig { split_k: Some(4), ..GemmConfig::tile_64x64_k16() };
        for n_ops in [1, 4, 16, 36, 64] {
            let _ = bench_fused_gemm(&rt, &best_cfg, m, k, n, flops, n_ops);
        }
    }

    Ok(())
}

#[cfg(feature = "rocm")]
struct BenchResult {
    us: f64,
    tflops: f64,
    n_wgs: u32,
    occupancy: u32,
}

#[cfg(feature = "rocm")]
fn bench_gemm_gen(
    rt: &t0_gpu::ignis::gpu_context::GpuRuntime,
    cfg: &GemmConfig,
    m: u32, k: u32, n: u32,
    x_buf: &t0_gpu::kfd::GpuBuffer,
    w_buf: &t0_gpu::kfd::GpuBuffer,
    flops: f64,
) -> Result<BenchResult, String> {
    use t0_gpu::t0::gemm_gen;

    let kernel_name = format!("sweep_{}", cfg.name());
    let cfg_c = cfg.clone();
    let kernel = rt.ensure_kernel_t0(
        &kernel_name,
        || gemm_gen::generate(&cfg_c),
        [cfg.wg_size, 1, 1],
        cfg.lds_total(),
    )?;

    // Output buffer (f32)
    let sk = cfg.split_k.unwrap_or(1);
    let y_alloc = (m as usize) * (n as usize) * 4 * (sk as usize);
    let y_buf = rt.alloc(y_alloc)?;
    y_buf.zero();

    let ka = gemm_gen::build_kernargs(
        x_buf.gpu_addr(), w_buf.gpu_addr(), y_buf.gpu_addr(),
        k, n, m, cfg,
    );
    let (gx, gy) = gemm_gen::compute_grid_auto(cfg, m, n);

    // Warmup
    for _ in 0..3 {
        rt.dispatch(&kernel, [gx, gy, 1], &ka)?;
    }

    // Timed run
    let n_iters = 100;
    let start = std::time::Instant::now();
    for _ in 0..n_iters {
        rt.dispatch(&kernel, [gx, gy, 1], &ka)?;
    }
    let us = start.elapsed().as_micros() as f64 / n_iters as f64;
    let tflops = flops / (us * 1e6);

    // Occupancy estimate
    let vgprs = cfg.estimated_vgprs();
    let occ = if vgprs <= 64 { 16 } else if vgprs <= 96 { 10 } else if vgprs <= 128 { 8 } else if vgprs <= 192 { 4 } else { 2 };

    let n_wgs = gx * gy / cfg.wg_size;

    Ok(BenchResult { us, tflops, n_wgs, occupancy: occ })
}

#[cfg(feature = "rocm")]
fn bench_tile_ir(
    rt: &t0_gpu::ignis::gpu_context::GpuRuntime,
    spec: &TileGemm,
    m: u32, k: u32, n: u32,
    x_buf: &t0_gpu::kfd::GpuBuffer,
    w_buf: &t0_gpu::kfd::GpuBuffer,
    flops: f64,
) -> Result<BenchResult, String> {
    use t0_gpu::t0::tile_ir;

    let kernel_name = format!("tile_sweep_{}x{}", spec.tile_m, spec.tile_n);
    let spec_c = spec.clone();
    let kernel = rt.ensure_kernel_t0(
        &kernel_name,
        || tile_ir::lower_gemm(&spec_c),
        [spec.wg_size(), 1, 1],
        spec.lds_total(),
    )?;

    let y_bytes = (m as usize) * (n as usize) * 4;
    let y_buf = rt.alloc(y_bytes)?;
    y_buf.zero();

    let ka = tile_ir::build_kernargs(
        x_buf.gpu_addr(), w_buf.gpu_addr(), y_buf.gpu_addr(),
        k, n, spec,
    );
    let grid = tile_ir::compute_grid(spec, m, n);

    // Warmup
    for _ in 0..3 {
        rt.dispatch(&kernel, grid, &ka)?;
    }

    let n_iters = 100;
    let start = std::time::Instant::now();
    for _ in 0..n_iters {
        rt.dispatch(&kernel, grid, &ka)?;
    }
    let us = start.elapsed().as_micros() as f64 / n_iters as f64;
    let tflops = flops / (us * 1e6);

    Ok(BenchResult { us, tflops, n_wgs: grid[0] * grid[1] / spec.wg_size(), occupancy: 0 })
}

/// Benchmark: N fused GEMM operations with single sync.
/// Measures per-operation time when dispatch overhead is amortized.
#[cfg(feature = "rocm")]
fn bench_fused_gemm(
    rt: &t0_gpu::ignis::gpu_context::GpuRuntime,
    cfg: &GemmConfig,
    m: u32, k: u32, n: u32,
    flops: f64,
    n_ops: usize,
) -> Result<f64, String> {
    use t0_gpu::t0::gemm_gen;

    let cfg_c = cfg.clone();
    let kernel_name = format!("fused_{}", cfg.name());
    let kernel = rt.ensure_kernel_t0(
        &kernel_name,
        || gemm_gen::generate(&cfg_c),
        [cfg.wg_size, 1, 1],
        cfg.lds_total(),
    )?;

    // Allocate N pairs of input buffers + N output buffers
    let x_bytes = (m as usize) * (k as usize) * 2;
    let w_bytes = (n as usize) * (k as usize) * 2;
    let y_bytes = (m as usize) * (n as usize) * 4;

    // Use a single set of buffers (overwrite each time)
    let x_buf = rt.alloc(x_bytes)?;
    let w_buf = rt.alloc(w_bytes)?;
    let y_buf = rt.alloc(y_bytes)?;
    y_buf.zero();

    let ones_bf16: Vec<u8> = vec![0x3F80u16; (m * k) as usize]
        .iter().flat_map(|v| v.to_le_bytes()).collect();
    x_buf.write(&ones_bf16);
    let w_ones: Vec<u8> = vec![0x3F80u16; (n * k) as usize]
        .iter().flat_map(|v| v.to_le_bytes()).collect();
    w_buf.write(&w_ones);

    let ka = gemm_gen::build_kernargs(
        x_buf.gpu_addr(), w_buf.gpu_addr(), y_buf.gpu_addr(),
        k, n, m, cfg,
    );
    let (gx, gy) = gemm_gen::compute_grid_auto(cfg, m, n);

    // Warmup: run once
    rt.dispatch(&kernel, [gx, gy, 1], &ka)?;

    // Timed: N async dispatches, single wait at end
    let n_iters = 10; // outer loop iterations
    let start = std::time::Instant::now();
    for _ in 0..n_iters {
        for _ in 0..n_ops {
            // Async dispatch (no wait)
            rt.dispatch_async(&kernel, [gx, gy, 1], &ka);
        }
        // Single wait for all N ops
        rt.wait_idle()?;
    }
    let total_us = start.elapsed().as_micros() as f64;
    let per_op_us = total_us / (n_iters as f64 * n_ops as f64);
    let per_op_tflops = flops / (per_op_us * 1e6);

    eprintln!("  fused {} ops: total={:.0}μs, per_op={:.1}μs, {:.1} TF",
        n_ops, total_us / n_iters as f64, per_op_us, per_op_tflops);

    Ok(per_op_tflops)
}
