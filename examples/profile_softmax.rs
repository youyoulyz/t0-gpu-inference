//! Minimal softmax profiling example.
//!
//! Run:
//!   cargo run --release --features "rocm,profile" --example profile_softmax

use std::sync::Arc;
use t0_gpu::ignis::gpu_context::GpuRuntime;
use t0_gpu::ignis::tensor::Tensor;
use t0_gpu::ignis::ops::softmax;

fn main() {
    let runtime = GpuRuntime::new().expect("GPU runtime");

    // Test configs: (rows, cols)
    let configs: Vec<(usize, usize)> = vec![
        (4, 128),       // small: single-pass kernel
        (4, 1024),      // medium: chunked kernel
        // (32, 151936), // vocab-size: GPU hang (known large-softmax issue)
    ];

    for (rows, cols) in &configs {
        let n = rows * cols;

        // Random-ish input data
        let data: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.001).sin()).collect();
        let input = Tensor::from_f32(&runtime, &data, &[*rows, *cols], "input").expect("tensor");

        // Warmup (first dispatch compiles the kernel)
        let _ = softmax::softmax(&input, &runtime).expect("softmax warmup");

        // Reset profiler and run N iterations
        t0_gpu::profiler::reset();
        let iters = 20;
        for _ in 0..iters {
            let _out = softmax::softmax(&input, &runtime).expect("softmax");
        }

        // Sync and report
        runtime.wait_idle().expect("sync");
        eprintln!("=== Softmax {}x{} ({} iters) ===", rows, cols, iters);
        t0_gpu::profiler::report();
    }

    // Export trace
    let json = t0_gpu::profiler::to_json();
    std::fs::write("softmax_profile.json", &json).expect("write trace");
    eprintln!("Trace written to softmax_profile.json");
}
