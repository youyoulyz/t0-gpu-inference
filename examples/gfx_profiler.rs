//! Bare-metal GPU hardware counter profiler CLI.
//!
//! Profiles T0-compiled kernels and external HSACO files on GFX1100 (RX 7900 XTX),
//! collecting hardware performance counters (SQ/GRBM/TCC) and generating
//! NCU-style reports with occupancy, IPC, cache hit rates, and optimization suggestions.
//!
//! # Usage
//!
//! ```bash
//! # Profile a T0 GEMM kernel
//! cargo run --release --features rocm,gfx-profiler --example gfx_profiler -- \
//!   --kernel gemm --m 4096 --n 4096 --k 4096
//!
//! # Profile softmax
//! cargo run --release --features rocm,gfx-profiler --example gfx_profiler -- \
//!   --kernel softmax --rows 1024 --cols 256
//!
//! # Profile RMSNorm
//! cargo run --release --features rocm,gfx-profiler --example gfx_profiler -- \
//!   --kernel rmsnorm --rows 1024 --cols 1024
//!
//! # Profile an external HSACO file
//! cargo run --release --features rocm,gfx-profiler --example gfx_profiler -- \
//!   --kernel elf:tests/hip_kernels/vector_add_gfx1100.hsaco --grid 4 --wg 256
//!
//! # JSON output
//! cargo run --release --features rocm,gfx-profiler --example gfx_profiler -- \
//!   --kernel gemm --m 256 --n 256 --k 256 --format json --output profile.json
//!
//! # Disable hardware counters (timing only, safe mode)
//! T0_GFX_PROFILER_NO_COUNTERS=1 cargo run --release --features rocm,gfx-profiler \
//!   --example gfx_profiler -- --kernel gemm --m 256 --n 256 --k 256
//! ```

use t0_gpu::gfx_profiler::{GfxProfiler, OutputFormat};
use t0_gpu::kfd::{GpuKernel, KernelLoadConfig};

fn main() -> Result<(), String> {
    let args = parse_args();

    eprintln!("═══════════════════════════════════════════════════");
    eprintln!("  GFX1100 Bare-Metal GPU Profiler");
    eprintln!("  Target: AMD RX 7900 XTX (96 CU, Wave32)");
    eprintln!("═══════════════════════════════════════════════════");
    eprintln!();

    let mut profiler = GfxProfiler::new()?;
    let device = profiler.device().clone();

    let kernels: Vec<&str> = args.kernel.split(',').map(|s| s.trim()).collect();

    for kernel_spec in &kernels {
        let (kernel_name, elf, grid, wg, lds, ka, flops) =
            prepare_kernel(&device, &args, kernel_spec)?;

        eprintln!("[2/3] Loading kernel onto GPU...");
        let gpu_kernel = GpuKernel::load(&device, &elf, &KernelLoadConfig {
            workgroup_size: [wg, 1, 1], lds_size: lds,
        })?;
        eprintln!("  Grid: {}x{}x{}, wg={}", grid[0], grid[1], grid[2], wg);

        eprintln!("[3/3] Profiling '{}'...", kernel_name);
        let result = profiler.profile_t0_kernel(
            &gpu_kernel, grid, &ka, &kernel_name, flops, args.cu,
        )?;
        eprintln!("  Done ({} passes, {} ns)\n", result.num_passes, result.elapsed_ns);

        let format = match args.format.as_str() {
            "json" => OutputFormat::Json,
            _ => OutputFormat::Text,
        };
        profiler.report(&result, format);

        if args.format == "json" {
            let path = args.output.clone().unwrap_or_else(|| {
                format!("profile_{}.json", kernel_name)
            });
            let json = t0_gpu::gfx_profiler::report::generate_report(
                &kernel_name, &result.metrics, &result.suggestions, OutputFormat::Json,
            );
            std::fs::write(&path, &json).map_err(|e| e.to_string())?;
            eprintln!("\nJSON written to {}", path);
        }
    }

    Ok(())
}

fn prepare_kernel(
    device: &std::sync::Arc<t0_gpu::kfd::KfdDevice>,
    args: &Args,
    kernel_spec: &str,
) -> Result<(String, Vec<u8>, [u32; 3], u32, u32, Vec<u8>, Option<u64>), String> {
    if kernel_spec.starts_with("elf:") {
        prepare_elf_kernel(device, args, kernel_spec)
    } else {
        match kernel_spec {
            "gemm" => prepare_gemm_kernel(device, args),
            "softmax" => prepare_softmax_kernel(device, args),
            "rmsnorm" => prepare_rmsnorm_kernel(device, args),
            other => Err(format!(
                "Unknown kernel: '{}'. Supported: gemm, softmax, rmsnorm, elf:<path>",
                other
            )),
        }
    }
}

fn prepare_elf_kernel(
    device: &std::sync::Arc<t0_gpu::kfd::KfdDevice>,
    args: &Args,
    kernel_spec: &str,
) -> Result<(String, Vec<u8>, [u32; 3], u32, u32, Vec<u8>, Option<u64>), String> {
    let path = &kernel_spec[4..];
    eprintln!("[1/3] Loading HSACO: {}", path);
    let elf = std::fs::read(path).map_err(|e| format!("Failed to read {}: {}", path, e))?;
    eprintln!("  ELF: {} bytes", elf.len());
    let g = args.grid.ok_or("--grid is required for elf: kernels")?;
    let wg = args.wg_size.unwrap_or(256);
    let ka_size = args.kernarg_size.unwrap_or(288);
    let n_elems = args.n_elems.unwrap_or(1024);
    let name = path.rsplit('/').next().unwrap_or(path)
        .replace(".hsaco", "").replace(".o", "");

    let n_bytes = n_elems * 4;
    let a_buf = device.alloc_vram(n_bytes)?;
    let b_buf = device.alloc_vram(n_bytes)?;
    let c_buf = device.alloc_vram_host(n_bytes)?;
    let a_data: Vec<f32> = (0..n_elems).map(|i| i as f32).collect();
    let b_data: Vec<f32> = (0..n_elems).map(|i| (i as f32) * 0.5).collect();
    a_buf.write(unsafe {
        std::slice::from_raw_parts(a_data.as_ptr() as *const u8, n_bytes)
    });
    b_buf.write(unsafe {
        std::slice::from_raw_parts(b_data.as_ptr() as *const u8, n_bytes)
    });

    let mut ka = vec![0u8; ka_size];
    ka[0..8].copy_from_slice(&a_buf.gpu_addr().to_le_bytes());
    ka[8..16].copy_from_slice(&b_buf.gpu_addr().to_le_bytes());
    ka[16..24].copy_from_slice(&c_buf.gpu_addr().to_le_bytes());
    ka[24..28].copy_from_slice(&(n_elems as u32).to_le_bytes());
    Ok((name, elf, g, wg, 0u32, ka, args.flops))
}

fn prepare_gemm_kernel(
    device: &std::sync::Arc<t0_gpu::kfd::KfdDevice>,
    args: &Args,
) -> Result<(String, Vec<u8>, [u32; 3], u32, u32, Vec<u8>, Option<u64>), String> {
    use t0_gpu::t0::Target;
    use t0_gpu::t0::{auto_select, compute_grid_auto};
    use t0_gpu::t0::gemm_gen::{generate, build_kernargs};

    let m = args.m.unwrap_or(4096);
    let k = args.k.unwrap_or(4096);
    let n = args.n.unwrap_or(4096);
    eprintln!("[1/3] Compiling GEMM {}x{}x{}...", m, k, n);

    let cfg = auto_select(m, k, n);
    let kernel_ir = generate(&cfg);
    let elf = kernel_ir.compile(Target::GFX1100)?;
    let grid_2d = compute_grid_auto(&cfg, m, n);
    let grid = [grid_2d.0, grid_2d.1, 1u32];

    let x_bytes = (m as usize) * (k as usize) * 2;
    let w_bytes = (n as usize) * (k as usize) * 2;
    let x_buf = device.alloc_vram(x_bytes)?;
    let w_buf = device.alloc_vram(w_bytes)?;
    let y_buf = device.alloc_vram_host((m as usize) * (n as usize) * 4)?;
    let x_data = vec![0x3F80u16; (m * k) as usize];
    x_buf.write(unsafe {
        std::slice::from_raw_parts(x_data.as_ptr() as *const u8, x_bytes)
    });
    let w_data = vec![0x3F80u16; (n * k) as usize];
    w_buf.write(unsafe {
        std::slice::from_raw_parts(w_data.as_ptr() as *const u8, w_bytes)
    });

    let ka = build_kernargs(
        x_buf.gpu_addr(), w_buf.gpu_addr(), y_buf.gpu_addr(), k, n, m, &cfg,
    );
    let ka = ka.to_vec();
    let flops = Some(2u64 * m as u64 * n as u64 * k as u64);
    Ok((
        format!("gemm_{}x{}x{}", m, k, n),
        elf, grid, cfg.wg_size, cfg.lds_total(), ka, flops,
    ))
}

fn prepare_softmax_kernel(
    device: &std::sync::Arc<t0_gpu::kfd::KfdDevice>,
    args: &Args,
) -> Result<(String, Vec<u8>, [u32; 3], u32, u32, Vec<u8>, Option<u64>), String> {
    use t0_gpu::t0::Target;
    use t0_gpu::t0::softmax_kernels::{build_softmax_forward, softmax_grid};

    let rows = args.rows.unwrap_or(1024);
    let cols = args.cols.unwrap_or(256);
    let n = rows * cols;
    eprintln!("[1/3] Compiling Softmax {}x{}...", rows, cols);

    let kb = build_softmax_forward();
    let ck = kb.compile_via_ssa(Target::GFX1100)
        .map_err(|e| format!("softmax compile: {}", e))?;
    let (gx, _) = softmax_grid(rows);
    let grid = [gx, 1, 1];
    let wg = ck.workgroup_size[0];

    let n_bytes = (n as usize) * 4;
    let input_buf = device.alloc_vram(n_bytes)?;
    let output_buf = device.alloc_vram_host(n_bytes)?;
    let input_data: Vec<f32> = (0..n).map(|i| (i as f32) * 0.001).collect();
    input_buf.write(unsafe {
        std::slice::from_raw_parts(input_data.as_ptr() as *const u8, n_bytes)
    });

    let mut ka = vec![0u8; ck.kernarg_size];
    ka[0..8].copy_from_slice(&input_buf.gpu_addr().to_le_bytes());
    ka[8..16].copy_from_slice(&output_buf.gpu_addr().to_le_bytes());
    ka[16..20].copy_from_slice(&(cols as u32).to_le_bytes());
    Ok((
        format!("softmax_{}x{}", rows, cols),
        ck.elf, grid, wg, ck.lds_size, ka, None,
    ))
}

fn prepare_rmsnorm_kernel(
    device: &std::sync::Arc<t0_gpu::kfd::KfdDevice>,
    args: &Args,
) -> Result<(String, Vec<u8>, [u32; 3], u32, u32, Vec<u8>, Option<u64>), String> {
    use t0_gpu::t0::Target;
    use t0_gpu::t0::rmsnorm_kernels::{build_rmsnorm_forward, rmsnorm_grid};

    let rows = args.rows.unwrap_or(1024);
    let cols = args.cols.unwrap_or(1024);
    let n = rows * cols;
    let eps = 1e-5f32;
    eprintln!("[1/3] Compiling RMSNorm {}x{}...", rows, cols);

    let kb = build_rmsnorm_forward();
    let ck = kb.compile_via_ssa(Target::GFX1100)
        .map_err(|e| format!("rmsnorm compile: {}", e))?;
    let (gx, _) = rmsnorm_grid(rows);
    let grid = [gx, 1, 1];
    let wg = ck.workgroup_size[0];

    let n_bytes = (n as usize) * 4;
    let input_buf = device.alloc_vram(n_bytes)?;
    let weight_buf = device.alloc_vram((cols as usize) * 4)?;
    let output_buf = device.alloc_vram_host(n_bytes)?;
    let input_data: Vec<f32> = (0..n).map(|i| (i as f32) * 0.001).collect();
    input_buf.write(unsafe {
        std::slice::from_raw_parts(input_data.as_ptr() as *const u8, n_bytes)
    });
    let weight_data: Vec<f32> = vec![1.0f32; cols as usize];
    weight_buf.write(unsafe {
        std::slice::from_raw_parts(weight_data.as_ptr() as *const u8, (cols as usize) * 4)
    });

    let mut ka = vec![0u8; ck.kernarg_size];
    ka[0..8].copy_from_slice(&input_buf.gpu_addr().to_le_bytes());
    ka[8..16].copy_from_slice(&weight_buf.gpu_addr().to_le_bytes());
    ka[16..24].copy_from_slice(&output_buf.gpu_addr().to_le_bytes());
    ka[24..28].copy_from_slice(&(cols as u32).to_le_bytes());
    ka[28..32].copy_from_slice(&eps.to_le_bytes());
    Ok((
        format!("rmsnorm_{}x{}", rows, cols),
        ck.elf, grid, wg, ck.lds_size, ka, None,
    ))
}

struct Args {
    kernel: String,
    m: Option<u32>,
    n: Option<u32>,
    k: Option<u32>,
    rows: Option<u32>,
    cols: Option<u32>,
    grid: Option<[u32; 3]>,
    wg_size: Option<u32>,
    kernarg_size: Option<usize>,
    n_elems: Option<usize>,
    flops: Option<u64>,
    format: String,
    output: Option<String>,
    cu: Option<u32>,
}

fn parse_args() -> Args {
    let mut args = Args {
        kernel: "gemm".to_string(),
        m: None, n: None, k: None,
        rows: None, cols: None,
        grid: None, wg_size: None, kernarg_size: None,
        n_elems: None, flops: None,
        format: "text".to_string(), output: None, cu: None,
    };
    let argv: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--kernel" => { i += 1; args.kernel = argv[i].clone(); }
            "--m" => { i += 1; args.m = Some(argv[i].parse().unwrap()); }
            "--n" => { i += 1; args.n = Some(argv[i].parse().unwrap()); }
            "--k" => { i += 1; args.k = Some(argv[i].parse().unwrap()); }
            "--rows" => { i += 1; args.rows = Some(argv[i].parse().unwrap()); }
            "--cols" => { i += 1; args.cols = Some(argv[i].parse().unwrap()); }
            "--grid" => { i += 1; let p: Vec<u32> = argv[i].split(',').map(|s| s.parse().unwrap()).collect(); args.grid = Some([p[0], p.get(1).copied().unwrap_or(1), p.get(2).copied().unwrap_or(1)]); }
            "--wg" => { i += 1; args.wg_size = Some(argv[i].parse().unwrap()); }
            "--kernarg-size" => { i += 1; args.kernarg_size = Some(argv[i].parse().unwrap()); }
            "--n-elems" => { i += 1; args.n_elems = Some(argv[i].parse().unwrap()); }
            "--flops" => { i += 1; args.flops = Some(argv[i].parse().unwrap()); }
            "--format" => { i += 1; args.format = argv[i].clone(); }
            "--output" => { i += 1; args.output = Some(argv[i].clone()); }
            "--cu" => { i += 1; args.cu = Some(argv[i].parse().unwrap()); }
            "--help" | "-h" => {
                eprintln!("GFX1100 Bare-Metal GPU Profiler");
                eprintln!("Profiles T0 kernels and HSACO files, collecting hardware counters.");
                eprintln!();
                eprintln!("USAGE:");
                eprintln!("  gfx_profiler --kernel <type> [options]");
                eprintln!();
                eprintln!("KERNEL TYPES:");
                eprintln!("  gemm         GEMM (bf16 WMMA)");
                eprintln!("  softmax      Softmax forward");
                eprintln!("  rmsnorm      RMSNorm forward");
                eprintln!("  elf:<path>   External HSACO file");
                eprintln!();
                eprintln!("  Use comma to profile multiple kernels:");
                eprintln!("    --kernel gemm,softmax,rmsnorm");
                eprintln!();
                eprintln!("GEMM OPTIONS:");
                eprintln!("  --m, --n, --k <N>     Dimensions [4096]");
                eprintln!();
                eprintln!("SOFTMAX / RMSNORM OPTIONS:");
                eprintln!("  --rows <N>            Number of rows [1024]");
                eprintln!("  --cols <N>            Columns per row [256/1024]");
                eprintln!();
                eprintln!("ELF OPTIONS:");
                eprintln!("  --grid <x,y,z>        Grid dimensions");
                eprintln!("  --wg <N>              Workgroup size [256]");
                eprintln!("  --kernarg-size <N>    Kernarg size in bytes [288]");
                eprintln!("  --n-elems <N>         Number of elements [1024]");
                eprintln!();
                eprintln!("OUTPUT OPTIONS:");
                eprintln!("  --format text|json    Output format [text]");
                eprintln!("  --output <path>       Output file for JSON");
                eprintln!("  --flops <N>           Total FLOPs for roofline analysis");
                eprintln!("  --cu <N>              Target CU index for SQ counters");
                eprintln!();
                eprintln!("ENVIRONMENT:");
                eprintln!("  T0_GFX_PROFILER_NO_COUNTERS=1");
                eprintln!("    Disable hardware counters (timing-only safe mode)");
                eprintln!();
                eprintln!("EXAMPLES:");
                eprintln!("  cargo run --release --features rocm,gfx-profiler --example gfx_profiler -- \\");
                eprintln!("    --kernel gemm --m 4096 --n 4096 --k 4096");
                eprintln!();
                eprintln!("  cargo run --release --features rocm,gfx-profiler --example gfx_profiler -- \\");
                eprintln!("    --kernel softmax --rows 1024 --cols 256");
                eprintln!();
                eprintln!("  cargo run --release --features rocm,gfx-profiler --example gfx_profiler -- \\");
                eprintln!("    --kernel gemm,softmax,rmsnorm --m 256 --n 256 --k 256 --rows 256 --cols 256");
                std::process::exit(0);
            }
            other => { eprintln!("Unknown: {}", other); std::process::exit(1); }
        }
        i += 1;
    }
    args
}