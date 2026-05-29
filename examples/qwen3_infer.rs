//! Qwen3 bare-metal inference example.
//!
//! Runs Qwen3-0.6B or Qwen3-4B on the RX 7900 XTX via KFD.
//!
//! Usage:
//!   cargo run --release --features rocm --example qwen3_infer -- \
//!     --model-path /path/to/Qwen3-0.6B \
//!     --prompt "Hello, who are you?" \
//!     --max-tokens 128 \
//!     --temperature 0.7

use std::sync::Arc;

fn main() {
    let args = parse_args();

    eprintln!("=== Qwen3 Bare-Metal Inference ===");
    eprintln!("Model: {}", args.model_path);
    eprintln!("Prompt: {}", args.prompt);
    eprintln!("Max tokens: {}", args.max_tokens);
    eprintln!("Temperature: {}", args.temperature);
    eprintln!("Top-p: {}", args.top_p);
    eprintln!();

    // 1. Initialize GPU runtime
    eprintln!("[1/5] Initializing GPU runtime...");
    let runtime = Arc::new(
        t0_gpu::ignis::gpu_context::GpuRuntime::new()
            .expect("Failed to create GPU runtime")
    );

    // 2. Load model
    eprintln!("[2/5] Loading model from {}...", args.model_path);
    let mut model = t0_gpu::ignis::safetensors::load_qwen3_model_from_dir(
        &runtime, &args.model_path,
    ).expect("Failed to load model");
    eprintln!("  Parameters: {:.2}M", model.param_count() as f64 / 1e6);

    // 3. Load tokenizer
    eprintln!("[3/5] Loading tokenizer...");
    let tokenizer = t0_gpu::ignis::tokenizer::HfTokenizer::from_dir(&args.model_path)
        .expect("Failed to load tokenizer");
    eprintln!("  Vocab size: {}", tokenizer.vocab_size());

    // 4. Create KV cache
    eprintln!("[4/5] Allocating KV cache...");
    let kv_config = t0_gpu::ignis::kv_cache::KvCacheConfig {
        num_layers: model.config.num_layers,
        num_kv_heads: model.config.num_key_value_heads,
        head_dim: model.config.head_dim,
        max_seq_len: args.max_seq_len,
    };
    let mut kv_cache = t0_gpu::ignis::kv_cache::KvCache::new(&runtime, kv_config)
        .expect("Failed to allocate KV cache");

    // 5. Tokenize and generate
    eprintln!("[5/5] Tokenizing and generating...");
    let prompt_ids = tokenizer.encode(&args.prompt);
    eprintln!("  Prompt tokens: {} ({:?})", prompt_ids.len(), &prompt_ids[..prompt_ids.len().min(20)]);

    let eos_id = 151645u32; // Qwen3 EOS token

    let start = std::time::Instant::now();
    let generated_ids = model.generate(
        &prompt_ids,
        args.max_tokens,
        args.temperature,
        args.top_p,
        eos_id,
        &mut kv_cache,
    ).expect("Generation failed");
    let elapsed = start.elapsed();

    let output_text = tokenizer.decode(&generated_ids);

    eprintln!();
    eprintln!("=== Output ===");
    println!("{}", output_text);
    eprintln!();
    eprintln!("=== Stats ===");
    eprintln!("Generated {} tokens in {:.2}s ({:.1} tokens/s)",
        generated_ids.len(),
        elapsed.as_secs_f64(),
        generated_ids.len() as f64 / elapsed.as_secs_f64()
    );

    // Profiler report (if --profile flag was set)
    if args.profile {
        t0_gpu::profiler::report();
    }
    if args.profile_json {
        let json = t0_gpu::profiler::to_json();
        std::fs::write("profile_trace.json", &json).expect("write profile_trace.json");
        eprintln!("Chrome tracing JSON written to profile_trace.json");
    }
}

struct Args {
    model_path: String,
    prompt: String,
    max_tokens: usize,
    temperature: f32,
    top_p: f32,
    max_seq_len: usize,
    profile: bool,
    profile_json: bool,
}

fn parse_args() -> Args {
    let mut model_path = String::new();
    let mut prompt = String::from("Hello");
    let mut max_tokens = 128usize;
    let mut temperature = 0.7f32;
    let mut top_p = 0.9f32;
    let mut max_seq_len = 2048usize;
    let mut profile = false;
    let mut profile_json = false;

    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model-path" => { i += 1; model_path = args[i].clone(); }
            "--prompt" => { i += 1; prompt = args[i].clone(); }
            "--max-tokens" => { i += 1; max_tokens = args[i].parse().unwrap(); }
            "--temperature" => { i += 1; temperature = args[i].parse().unwrap(); }
            "--top-p" => { i += 1; top_p = args[i].parse().unwrap(); }
            "--max-seq-len" => { i += 1; max_seq_len = args[i].parse().unwrap(); }
            "--profile" => { profile = true; }
            "--profile-json" => { profile_json = true; }
            _ => {
                eprintln!("Unknown arg: {}", args[i]);
                std::process::exit(1);
            }
        }
        i += 1;
    }

    if model_path.is_empty() {
        eprintln!("Usage: qwen3_infer --model-path <dir> [--prompt <text>] [--max-tokens N]");
        std::process::exit(1);
    }

    Args { model_path, prompt, max_tokens, temperature, top_p, max_seq_len, profile, profile_json }
}
