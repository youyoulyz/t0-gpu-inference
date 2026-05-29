# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

t0-gpu is a pure-Rust GPU kernel compiler and bare-metal runtime targeting AMD RDNA3 (GFX1100, RX 7900 XTX). It bypasses HIP/ROCm entirely, communicating directly with the GPU via the Linux KFD (`/dev/kfd`) ioctl interface. The only external dependency is `tokenizers` (for text tokenization).

## Build & Test Commands

```bash
# Build compiler only (no GPU needed)
cargo build --release --lib

# Build with KFD runtime (requires /dev/kfd)
cargo build --release --lib --features rocm

# Lint (CI uses these)
cargo fmt --check
cargo clippy --release --lib -- -D warnings

# CPU-only unit tests (no GPU needed, safe to run anywhere)
cargo test --release --lib -- "cpu_softmax" --nocapture
cargo test --release --lib -- "cpu_ce_loss" --nocapture
cargo test --release --lib -- "cpu_rope" --nocapture
cargo test --release --lib -- "cpu_causal_mask" --nocapture
cargo test --release --lib -- "cpu_rmsnorm" --nocapture
cargo test --release --lib -- "cpu_embedding" --nocapture
cargo test --release --lib -- "cpu_adamw" --nocapture

# Kernel compile tests (T0 -> ELF, no GPU execution)
cargo test --release --lib -- "compiles" --nocapture

# GPU tests (requires --features rocm, MUST use --test-threads=1)
cargo test --release --features rocm -- test_tile_ir_correctness --nocapture --test-threads=1
cargo test --release --features rocm -- test_tune_tile_ir_4096 --nocapture --ignored --test-threads=1

# ISA assembly dump for debugging
T0_DUMP_ASM=1 cargo test --release --features rocm -- <test_name>

# Run examples
cargo run --release --features rocm --example test_gemm_correctness
cargo run --release --features rocm --example hello_gemm_gen

# Qwen3 inference
cargo run --release --features rocm --example qwen3_infer -- \
  --model-path /path/to/Qwen3-0.6B --prompt "Hello" --max-tokens 32 --temperature 0.7

# CPU reference tests for inference ops (requires --features rocm)
cargo test --release --features rocm --lib -- "cpu_qk_norm" --nocapture --test-threads=1
cargo test --release --features rocm --lib -- "cpu_attention" --nocapture --test-threads=1
cargo test --release --features rocm --lib -- "cpu_sample" --nocapture --test-threads=1
cargo test --release --features rocm --lib -- "test_rope" --nocapture --test-threads=1
```

**GPU tests must always use `--test-threads=1`** to avoid GPU resource contention.

## Qwen3 Inference

See `docs/Qwen3_推理引擎_架构与实现.md` for full documentation.

Key files: `nn/config.rs` (head_dim fix), `nn/transformer.rs` (QK-norm + RoPE + attention), `nn/model.rs` (prefill/decode/generate), `ops/rope.rs`, `ops/qk_norm.rs`, `ops/attention.rs`, `examples/qwen3_infer.rs`.

Model directory: `/mnt/public/models/huggingface/Qwen3-0.6B` (downloaded). Config: head_dim=128, 28 layers, 16 Q heads, 8 KV heads, hidden=1024, vocab=151936.

## Architecture

Four subsystems, all in a single crate (not a workspace):

### 1. T0 Compiler (`src/t0/` — 46 files)

Two independent compilation paths sharing a common backend:

**Path A — General Kernels (BlockDSL → SSA):** Triton-style DSL frontend (`block_dsl.rs`) translated to SSA IR (`ssa_ir.rs`) with explicit VCC/SCC/EXEC modeling, 6-pass optimization (`opt_passes.rs`: DCE, CSE, LICM, CopyProp, AlgSimp, Waitcnt), and linear-scan register allocation (`ssa_regalloc.rs`).

**Path B — GEMM Kernels (TileIR):** Tile-level GEMM description (`tile_ir.rs`, largest file at 5.3K lines) compiled through Tile SSA (`tile_ssa.rs`) and lowered to T0Kernel (`tile_ssa_lower.rs`). Auto-tuned via `cost_model.rs`.

**Shared backend:** `ir.rs` (~80 Op types, virtual registers), `compile.rs` (T0Kernel builder/orchestrator), `asm_emitter.rs` (ISA emission), `schedule.rs` (instruction scheduling), `isa_verifier.rs` (static hang-pattern detection).

**Built-in kernel libraries:** Hand-written ISA kernels for softmax, cross-entropy loss, RoPE, causal mask, RMSNorm, embedding, AdamW, elementwise ops, argmax, and OCPA attention (`math.rs`).

### 2. ISA Encoder & ELF Generator (`src/rdna3_asm.rs`, `src/rdna3_code_object.rs`)

Binary instruction encoder for all GFX1100 instruction classes (VOP1/VOP2/VOP3/SMEM/FLAT/WMMA/DS/MUBUF/SOPP/SOP2), verified against LLVM `llvm-mc`. Hand-crafted AMD HSA ELF code object generator — no LLVM linker dependency.

### 3. KFD Runtime (`src/kfd/mod.rs`)

Bare-metal GPU runtime via `/dev/kfd` ioctl: VRAM allocation (with small BAR fallback), AQL queue dispatch (~2μs async latency), doorbell ring + completion polling, kernel loading from ELF code objects.

### 4. Ignis Autodiff Framework (`src/ignis/`)

GPU-native automatic differentiation with reverse-mode tape (`tape.rs`), GPU-backed tensors (`tensor.rs`), differentiable ops (`ops/`), neural network layers (`nn/`), and training infrastructure (data loader, tokenizer, LR scheduler, gradient clipping, loss scaler, KV cache, safetensors loading).

## Feature Flags

- `rocm` — enables KFD runtime, Ignis autodiff framework, and GPU-executeable examples. Requires `/dev/kfd`. Without this flag, only the compiler/encoder library builds (CPU-only).

## Key Conventions

- Target GPU: AMD RX 7900 XTX (Navi 31, GFX1100, 96 CU). Cost model and auto-tuning data are specific to this hardware.
- The `docs/` directory contains 48 files including a full technical manual (`T0_技术手册.md`), architecture diagrams, SSA safety guide, and experiment logs documenting GPU hang root causes.
- `RUN_RESULTS.md` has benchmark results from this specific machine (105.2 TFLOPS achieved).
- Binary target `isa_probe` (`src/bin/isa_probe.rs`) is a standalone ISA encoding verifier.
- Rust edition 2021. CI sets `RUSTFLAGS="-D warnings"`.
