# 示例 / Examples

所有示例需要 `--features rocm` 和 `/dev/kfd` 设备。GPU 测试必须单线程运行 (`--test-threads=1`)。

All examples require `--features rocm` and `/dev/kfd`. GPU tests must use `--test-threads=1`.

---

## 入门示例 / Getting Started

### 1. hello_gemm — 最小端到端 GPU 工作流

演示完整的裸金属 GPU 工作流：T0 编译向量加法内核 → KFD 分配 VRAM → dispatch → 读回结果。

```bash
cargo run --example hello_gemm --features rocm --release
```

### 2. hello_gemm_gen — 自动选择 GEMM

演示 `auto_select(M, K, N)` 自动选择最优 GEMM 配置，编译并验证正确性。

```bash
cargo run --example hello_gemm_gen --features rocm --release
```

### 3. qwen3_infer — Qwen3 推理

完整的 Qwen3-0.6B/4B 推理流水线：tokenizer → prefill → decode → generate。

```bash
cargo run --release --features rocm --example qwen3_infer -- \
  --model-path /path/to/Qwen3-0.6B \
  --prompt "Hello, who are you?" \
  --max-tokens 128 \
  --temperature 0.7
```

### 4. train_mlp — 端到端训练

2 层 MLP 训练演示：Embedding → GEMM+ReLU → GEMM → Softmax+CE Loss → AdamW 更新。所有内核由 T0 编译，KFD 裸金属调度。

```bash
cargo run --example train_mlp --features rocm --release
```

---

## 正确性测试 / Correctness Tests

### test_gemm_correctness — GEMM 正确性验证

测试所有 T0 GEMM 配置，对比 CPU bf16 参考实现。使用随机 bf16 数据检测系统性误差。

```bash
cargo run --example test_gemm_correctness --features rocm --release
```

### test_gemm_backward — GEMM 反向传播验证

验证反向传播：`dX = dY @ W^T`（反向数据）和 `dW = dY^T @ X`（反向权重）。

```bash
cargo run --example test_gemm_backward --features rocm --release
```

---

## 性能基准 / Benchmarks

### bench_gemm — GEMM 性能基准

跨多种矩阵尺寸测试 T0 编译的 GEMM 内核，报告 TFLOPS。

```bash
cargo run --example bench_gemm --features rocm --release
```

### bench_gemm_sweep — GEMM 变体扫描

编译所有 `GemmConfig` 变体，跨多种矩阵尺寸基准测试，报告每种尺寸的最优配置。

```bash
cargo run --example bench_gemm_sweep --features rocm --release
```

### bench_tile_gemm — TileIR GEMM 基准

测试生产级 tile_ir 编译路径（SSA 优化、graduated waitcnt、双缓冲、WMMA 16x16x16）。

```bash
cargo run --example bench_tile_gemm --features rocm --release
```

### bench_tile_ir — TileIR vs gemm_gen 对比

对比 tile_ir（编译器生成）与 gemm_gen（手写）GEMM 内核的性能差异。

```bash
cargo run --example bench_tile_ir --features rocm --release
```

### bench_tile_ir_vs_gemm_gen — 编译器 vs 生成器深度对比

两条编译路径的 NT 模式 GEMM（bf16 输入，f32 输出）深度对比。

```bash
cargo run --example bench_tile_ir_vs_gemm_gen --features rocm --release
```

### bench_kfd_dispatch — KFD 调度延迟

裸金属 KFD 调度延迟基准测试。

```bash
cargo run --example bench_kfd_dispatch --features rocm --release
```

### bench_split_k — Split-K GEMM

测试 Split-K 策略在小矩阵上的并行化效果。

```bash
cargo run --example bench_split_k --features rocm --release
```

### bench_wgp_vs_cu — WGP vs CU 模式对比

对比 Workgroup Processor 模式（WG 跨 2 CU = 128KB LDS + 4 SIMD）与标准 CU 模式。

```bash
cargo run --example bench_wgp_vs_cu --features rocm --release
```

### bench_small_matrix — 小矩阵调度开销

测量小矩阵的调度开销和单 CU 效率。

```bash
cargo run --example bench_small_matrix --features rocm --release
```

### bench_thin_matrix — 窄矩阵基准

M=128/256 × K=1024 × N=4096 的 tile/WGP/split-K/grid 组合测试。

```bash
cargo run --example bench_thin_matrix --features rocm --release
```

### bench_128x4096 — 深 K 优化

128×1024×4096 矩阵的 k32/k64 tile + WGP 优化测试。

```bash
cargo run --example bench_128x4096 --features rocm --release
```

### bench_256_sweep — 256³ 全配置扫描

绕过 Ignis bf16 转换开销，直接写入 GPU 的 256³ 矩阵全配置扫描。

```bash
cargo run --example bench_256_sweep --features rocm --release
```

### bench_t0_unified — T0 统一基准

测试 T0 编译器各编译链路的 GEMM 性能。

```bash
cargo run --example bench_t0_unified --features rocm --release
```

### bench_block_dsl_gemm — BlockDSL GEMM 基准

BlockDSL `gemm_tn_naive` 跨矩阵尺寸基准测试。

```bash
cargo run --example bench_block_dsl_gemm --features rocm --release
```

### bench_auto_select — 自动选择验证

验证 `auto_select` 在不同尺寸下的配置选择安全性。

```bash
cargo run --example bench_auto_select --features rocm --release
```

---

## 调试工具 / Debug Tools

### analyze_gemm_isa — ISA 指令分析

导出 GEMM 内核汇编并统计指令类别分布。

```bash
cargo run --example analyze_gemm_isa --release
```

### dump_asm — 汇编导出

导出 T0 编译内核的 GFX1100 ISA 汇编。

```bash
cargo run --example dump_asm --release
```

### debug_tile_ir — TileIR 调试

对比 tile_ir 和 gemm_gen 对同一尺寸生成的汇编，用最小尺寸单 tile 测试 GPU 执行。

```bash
cargo run --example debug_tile_ir --features rocm --release
```

---

## 内置内核 / Kernel Sources

`kernels/` 目录包含手写 ISA 内核：

| 文件 | 说明 |
|---|---|
| `ocpa_forward_intra.rs` | OCPA chunk 内因果注意力前向 |
| `ocpa_backward_intra.rs` | OCPA chunk 内注意力反向 |
| `ocpa_state_update.rs` | OCPA 状态更新 |
| `softmax_ce_loss.rs` | 融合 Softmax + 交叉熵损失 |
