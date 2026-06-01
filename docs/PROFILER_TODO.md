# Profiler 实现 TODO

> 状态: 未开始
> 优先级: GPU Timestamp → Op Profiler → Memory Profiler → Compiler Profiler

---

## 1. GPU Kernel Timestamp Profiler

**目标**: 测量单个 GPU kernel 的执行时间（纳秒级精度）

### 实现方案

在 AQL dispatch 前后各 dispatch 一个极轻量的 timestamp kernel，读回差值。

### 需要做的事

- [ ] 创建 `src/profiler/mod.rs`，定义 `GpuProfiler` 结构体
- [ ] 创建 `src/profiler/gpu_timestamp.rs`，实现 timestamp 内核编译 + dispatch
  - 内核内容: `s_getreg_b32 s0, SHADER_CYCLES; s_store_b32 s0, desc, offset` (约 5 条指令)
  - 需要 2 个 u64 VRAM buffer (start_cycles, end_cycles)
  - `profiled_submit(kernel, grid, kernargs)` 方法: dispatch ts_start → kernel → ts_end → readback
- [ ] 在 `DispatchPool` 或 `GpuRuntime` 上加 `profiled_submit` 接口
- [ ] 频率转换: `gpu_ns = gpu_cycles * 1000 / clock_mhz`（RX 7900 XTX boost ~2500 MHz）
- [ ] 支持批量 profile: 一个推理 pass 内所有 kernel 的 timestamp 收集到 Vec，最后一次性 readback

### 关键代码位置

- AQL dispatch: `src/kfd/mod.rs` 中的 `queue.submit()`
- 内核编译: `src/t0/compile.rs` 中的 `T0Kernel::compile()`
- 时间戳寄存器: GFX1100 `s_getreg SHADER_CYCLES` (regid=3, hwreg=SHADER_CYCLES)

### 输出格式

```text
Kernel                     Cycles      Time (μs)
softmax_fwd                  1,234        0.49
gemm_128x128_k32           456,789      182.72
rmsnorm_fwd                  2,345        0.94
```

---

## 2. Op-level Profiler

**目标**: 在推理/训练层面统计每个算子的耗时

### 实现方案

在 Ignis ops 层加 profiler wrapper，自动记录每次 op dispatch 的名称、kernel 数量、CPU/GPU 耗时。

### 需要做的事

- [ ] 创建 `src/profiler/op_profiler.rs`
  - `OpProfiler` 结构体: records: Vec<OpRecord>, stack: Vec<OpFrame>
  - `OpRecord`: op_name, kernel_count, cpu_us, gpu_us
  - `begin(name)` / `end(name)` 接口
  - `report()` 打印表格 (人可读) + `to_json()` (Chrome tracing 格式)
- [ ] 在每个 Ignis op 函数入口/出口加 profiling hook
  - `ops/attention.rs:standard_attention` — 9 个 kernel dispatch
  - `ops/rmsnorm.rs:rmsnorm_forward` — 1 个 kernel dispatch
  - `ops/bf16_matmul.rs:gemm_f32_raw` — 1 个 kernel dispatch
  - `ops/softmax.rs:softmax_forward` — 1 个 kernel dispatch
  - `ops/rope.rs:rope_forward` — 1 个 kernel dispatch
  - `ops/cross_entropy.rs:cross_entropy_forward` — 2 个 kernel dispatch
  - 其余 ops 类似
- [ ] 用 feature gate 避免零开销:
  ```toml
  [features]
  profile = []
  ```
  ```rust
  #[cfg(feature = "profile")]
  { profiler.begin("attention_qk"); }
  // ... actual work ...
  #[cfg(feature = "profile")]
  { profiler.end("attention_qk"); }
  ```
- [ ] 支持嵌套: `attention` 下有 `qk_gemm`, `softmax`, `av_gemm`
- [ ] 输出 Chrome tracing JSON 格式，可用 `chrome://tracing` 或 Perfetto 可视化

### 关键代码位置

- 推理入口: `src/ignis/nn/model.rs:forward_prefill` 和 `forward_decode`
- Transformer 层: `src/ignis/nn/transformer.rs:forward_inference`
- 注意力: `src/ignis/ops/attention.rs:standard_attention`

### 输出格式

人可读:
```text
=== Decode Step Profile (layer 0) ===
Op                     CPU (μs)   GPU (μs)   Kernels
rmsnorm_qkv              12.3       8.1        1
q_proj                   45.2      38.7        1
... (see lec2 for full table)
```

Chrome tracing JSON:
```json
{"traceEvents": [
  {"name": "attention_qk", "ph": "B", "ts": 1234, "pid": 1},
  {"name": "attention_qk", "ph": "E", "ts": 1567, "pid": 1}
]}
```

---

## 3. Memory Profiler

**目标**: 追踪 VRAM 分配/释放，检测泄漏

### 需要做的事

- [ ] 创建 `src/profiler/memory.rs`
  - `MemoryProfiler` 结构体
  - `on_alloc(gpu_addr, size, caller)` / `on_free(gpu_addr)` 接口
  - `report()`: peak VRAM, current, alloc count, leaked blocks
- [ ] Hook `KfdDevice::alloc_vram()` 和 `GpuBuffer::drop()`
  - 用 `#[track_caller]` 记录调用位置
- [ ] 检测 double-free 和 use-after-free（debug 模式）
- [ ] 可选: VRAM 分配时间线可视化

### 关键代码位置

- VRAM 分配: `src/kfd/mod.rs` 中的 `alloc_vram()`
- Buffer drop: `src/kfd/mod.rs` 中 `GpuBuffer` 的 `Drop` impl
- Buffer pool: `src/ignis/buffer_pool.rs`

---

## 4. Compiler Profiler

**目标**: 分析 T0 编译器各阶段耗时

### 需要做的事

- [ ] 创建 `src/profiler/compile.rs`
  - `CompileProfiler` 结构体
  - 在 `compile_via_ssa()` 各阶段插入计时
  - `report()`: parse / SSA lift / opt / regalloc / emit 各阶段耗时
- [ ] 支持 autotuner 场景: 批量编译 13 个候选内核，报告每个的编译时间和总时间

### 关键代码位置

- 编译入口: `src/t0/compile.rs` 中的 `compile_via_ssa()`
- SSA lift: `src/t0/ssa_ir.rs:lift_to_ssa()`
- 优化: `src/t0/opt_passes.rs`
- 寄存器分配: `src/t0/ssa_regalloc.rs`
- ISA 发射: `src/t0/asm_emitter.rs`

---

## 5. CLI 集成

- [ ] `--profile` 命令行参数 (qwen3_infer 示例)
- [ ] 环境变量 `T0_PROFILE=1` (全局开关)
- [ ] `T0_PROFILE_JSON=1` 输出 Chrome tracing 格式
- [ ] `T0_PROFILE_MEMORY=1` 启用 memory profiler

---

## 6. 依赖

```
src/profiler/
├── mod.rs              # 模块入口 + Profiler trait
├── gpu_timestamp.rs    # GPU kernel 时间戳
├── op_profiler.rs      # Op 级耗时
├── memory.rs           # VRAM 追踪
└── compile.rs          # 编译器耗时
```

`Cargo.toml`:
```toml
[features]
profile = []
```

无新外部依赖，所有 profiler 用标准库 `Instant` + GFX1100 时间戳寄存器实现。
