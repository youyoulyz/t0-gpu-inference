//! Bare-metal GPU hardware counter profiler CLI.
//!
//! # Usage
//!
//! ```bash
//! # Profile a T0 GEMM kernel
//! cargo run --release --features rocm,gfx-profiler --example gfx_profiler -- \
//!   --kernel gemm --m 4096 --n 4096 --k 4096
//!
//! # Profile an external HSACO file (e.g., HIP-compiled)
//! cargo run --release --features rocm,gfx-profiler --example gfx_profiler -- \
//!   --kernel elf:tests/hip_kernels/vector_add_gfx1100.hsaco --grid 4 --wg 256
//! ```

use t0_gpu::gfx_profiler::{GfxProfiler, OutputFormat};
use t0_gpu::kfd::{GpuKernel, KernelLoadConfig};

fn main() -> Result<(), String> {
    let args = parse_args();

    eprintln!("═══════════════════════════════════════════════════");
    eprintln!("  GFX1100 Bare-Metal GPU Profiler");
    eprintln!("═══════════════════════════════════════════════════");
    eprintln!();

    let mut profiler = GfxProfiler::new()?;
    let device = profiler.device();

    let (kernel_name, elf, grid, wg, lds, ka, flops) = match args.kernel.as_str() {
        k if k.starts_with("elf:") => {
            let path = &k[4..];
            eprintln!("[1/3] Loading HSACO: {}", path);
            let elf = std::fs::read(path).map_err(|e| format!("Failed to read {}: {}", path, e))?;
            eprintln!("  ELF: {} bytes", elf.len());
            let g = args.grid.ok_or("--grid is required for elf: kernels")?;
            let wg = args.wg_size.unwrap_or(256);
            let ka_size = args.kernarg_size.unwrap_or(288);
            let n_elems = args.n_elems.unwrap_or(1024);
            let name = path.rsplit('/').next().unwrap_or(path).replace(".hsaco", "").replace(".o", "");

            let n_bytes = n_elems * 4;
            let a_buf = device.alloc_vram(n_bytes)?;
            let b_buf = device.alloc_vram(n_bytes)?;
            let c_buf = device.alloc_vram_host(n_bytes)?;
            let a_data: Vec<f32> = (0..n_elems).map(|i| i as f32).collect();
            let b_data: Vec<f32> = (0..n_elems).map(|i| (i as f32) * 0.5).collect();
            a_buf.write(unsafe { std::slice::from_raw_parts(a_data.as_ptr() as *const u8, n_bytes) });
            b_buf.write(unsafe { std::slice::from_raw_parts(b_data.as_ptr() as *const u8, n_bytes) });

            let mut ka = vec![0u8; ka_size];
            ka[0..8].copy_from_slice(&a_buf.gpu_addr().to_le_bytes());
            ka[8..16].copy_from_slice(&b_buf.gpu_addr().to_le_bytes());
            ka[16..24].copy_from_slice(&c_buf.gpu_addr().to_le_bytes());
            ka[24..28].copy_from_slice(&(n_elems as u32).to_le_bytes());
            (name, elf, g, wg, 0u32, ka, args.flops)
        }
        "gemm" => {
            use t0_gpu::t0::{Target, auto_select, compute_grid_auto};
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
            x_buf.write(unsafe { std::slice::from_raw_parts(x_data.as_ptr() as *const u8, x_bytes) });
            let w_data = vec![0x3F80u16; (n * k) as usize];
            w_buf.write(unsafe { std::slice::from_raw_parts(w_data.as_ptr() as *const u8, w_bytes) });

            let ka = build_kernargs(x_buf.gpu_addr(), w_buf.gpu_addr(), y_buf.gpu_addr(), k, n, m, &cfg);
            let ka = ka.to_vec();
            let flops = Some(2u64 * m as u64 * n as u64 * k as u64);
            (format!("gemm_{}x{}x{}", m, k, n), elf, grid, cfg.wg_size, cfg.lds_total(), ka, flops)
        }
        other => return Err(format!("Unknown kernel: '{}'. Supported: gemm, elf:<path>", other)),
    };

    eprintln!("[2/3] Loading kernel onto GPU...");
    let gpu_kernel = GpuKernel::load(device, &elf, &KernelLoadConfig {
        workgroup_size: [wg, 1, 1], lds_size: lds,
    })?;
    eprintln!("  Grid: {}x{}x{}, wg={}", grid[0], grid[1], grid[2], wg);

    eprintln!("[3/3] Profiling '{}'...", kernel_name);
    let result = profiler.profile_t0_kernel(&gpu_kernel, grid, &ka, &kernel_name, flops, args.cu)?;
    eprintln!("  Done ({} passes, {} ns)\n", result.num_passes, result.elapsed_ns);

    let format = match args.format.as_str() { "json" => OutputFormat::Json, _ => OutputFormat::Text };
    profiler.report(&result, format);

    if args.format == "json" {
        let path = args.output.unwrap_or_else(|| format!("profile_{}.json", kernel_name));
        let json = t0_gpu::gfx_profiler::report::generate_report(
            &kernel_name, &result.metrics, &result.suggestions, OutputFormat::Json,
        );
        std::fs::write(&path, &json).map_err(|e| e.to_string())?;
        eprintln!("\nJSON written to {}", path);
    }
    Ok(())
}

struct Args {
    kernel: String, m: Option<u32>, n: Option<u32>, k: Option<u32>,
    grid: Option<[u32; 3]>, wg_size: Option<u32>, kernarg_size: Option<usize>,
    n_elems: Option<usize>, flops: Option<u64>, format: String, output: Option<String>, cu: Option<u32>,
}

fn parse_args() -> Args {
    let mut args = Args {
        kernel: "gemm".to_string(), m: None, n: None, k: None,
        grid: None, wg_size: None, kernarg_size: None, n_elems: None, flops: None,
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
            "--grid" => { i += 1; let p: Vec<u32> = argv[i].split(',').map(|s| s.parse().unwrap()).collect(); args.grid = Some([p[0], p.get(1).copied().unwrap_or(1), p.get(2).copied().unwrap_or(1)]); }
            "--wg" => { i += 1; args.wg_size = Some(argv[i].parse().unwrap()); }
            "--kernarg-size" => { i += 1; args.kernarg_size = Some(argv[i].parse().unwrap()); }
            "--n-elems" => { i += 1; args.n_elems = Some(argv[i].parse().unwrap()); }
            "--flops" => { i += 1; args.flops = Some(argv[i].parse().unwrap()); }
            "--format" => { i += 1; args.format = argv[i].clone(); }
            "--output" => { i += 1; args.output = Some(argv[i].clone()); }
            "--cu" => { i += 1; args.cu = Some(argv[i].parse().unwrap()); }
            "--help" | "-h" => {
                eprintln!("Usage: gfx_profiler [OPTIONS]");
                eprintln!("  --kernel <name>       gemm, elf:<path.hsaco>");
                eprintln!("  --m, --n, --k <N>     GEMM dimensions");
                eprintln!("  --grid <x,y,z>        Grid (for elf: kernels)");
                eprintln!("  --wg <N>              Workgroup size [256]");
                eprintln!("  --kernarg-size <N>    Kernarg size (for elf:) [288]");
                eprintln!("  --n-elems <N>         Elements [1024]");
                eprintln!("  --flops <N>           Total FLOPs (roofline)");
                eprintln!("  --format text|json    Output format");
                eprintln!("  --output <path>       Output file");
                std::process::exit(0);
            }
            other => { eprintln!("Unknown: {}", other); std::process::exit(1); }
        }
        i += 1;
    }
    args
}
