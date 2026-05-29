# Profiler Guide

T0-GPU 内置 op 级性能分析器，用于定位推理/训练中的性能瓶颈。

## 编译

```bash
# 启用 profiler（编译时插桩，无运行时开销控制）
cargo build --release --features "rocm,profile"

# 不启用 profiler（profile_scope! 宏被编译为空操作，零开销）
cargo build --release --features rocm
```

## 使用

### 命令行参数

```bash
# 输出人可读的 profiling 表格
cargo run --release --features "rocm,profile" --example qwen3_infer -- \
  --model-path /path/to/Qwen3-0.6B \
  --prompt "Hello" \
  --max-tokens 32 \
  --profile

# 导出 Chrome tracing JSON（用 chrome://tracing 或 ui.perfetto.dev 打开）
cargo run --release --features "rocm,profile" --example qwen3_infer -- \
  --model-path /path/to/Qwen3-0.6B \
  --prompt "Hello" \
  --max-tokens 32 \
  --profile-json
```

### 输出示例

```
=== GPU Profiler Report ===
Op                             Total (ms)   Avg (μs)    Calls   Kernels
------------------------------------------------------------------------
standard_attention                  8.234    4117.0        2         2
matmul_wt_bf16                     6.156    1026.0        6         6
rmsnorm                             0.412      68.7        6         6
rope                                0.204      34.0        6         6
qk_norm                             0.312      52.0        6         6
silu_gate                           0.108      18.0        3         3
softmax                             0.056      28.0        2         2
------------------------------------------------------------------------
TOTAL                              15.482                 31        31
```

### 编程接口

```rust
use t0_gpu::profiler;

// 方式 1: begin/end 手动配对
profiler::begin("my_op");
// ... GPU dispatches ...
profiler::end("my_op");

// 方式 2: RAII guard（推荐，自动在 scope 结束时 end）
{
    let _guard = profiler::ProfileGuard::new("my_op");
    // ... GPU dispatches ...
} // 自动 end

// 方式 3: 宏（最简洁）
profiler::profile_scope!("my_op");

// 输出
profiler::report();           // 人可读表格 → stderr
let json = profiler::to_json(); // Chrome tracing JSON
profiler::reset();            // 清空数据
```

## 已插桩的 Op

| Op 名称 | 文件 | 说明 |
|---|---|---|
| `standard_attention` | `ops/attention.rs` | GQA 注意力（9 步 GPU pipeline） |
| `matmul` | `ops/bf16_matmul.rs` | bf16 WMMA 矩阵乘法（训练路径） |
| `matmul_wt_bf16` | `ops/bf16_matmul.rs` | bf16 WMMA 矩阵乘法（推理路径，预转置权重） |
| `bf16_gemm` | `ops/bf16_matmul.rs` | 底层 f32 GEMM（attention 内部调用） |
| `rmsnorm` | `ops/rmsnorm.rs` | RMS 归一化 |
| `softmax` | `ops/softmax.rs` | Softmax（小 + 大 chunked） |
| `rope` | `ops/rope.rs` | 旋转位置编码 |
| `qk_norm` | `ops/qk_norm.rs` | QK 归一化 |
| `embedding` | `ops/embedding.rs` | Token embedding gather |
| `silu_gate` | `ops/silu.rs` | SiLU gate 激活 |
| `cross_entropy` | `ops/cross_entropy.rs` | 交叉熵损失 |
| `compile_via_ssa` | `t0/block_dsl_to_ssa.rs` | T0 编译器 SSA 编译 |
| `t0_compile` | `t0/compile.rs` | T0 底层编译（ISA 发射 + ELF） |

## Chrome Tracing

`--profile-json` 输出 `profile_trace.json`，格式兼容：

- **Chrome**: 打开 `chrome://tracing`，Load 按钮加载 JSON
- **Perfetto**: 打开 `ui.perfetto.dev`，拖入 JSON 文件

时间轴上每个 op 显示为一个色块，宽度 = CPU 耗时，嵌套层级 = depth。

## 添加新的 Profiling 点

在任意函数入口加一行即可：

```rust
pub fn my_new_op(...) -> Result<..., String> {
    crate::profile_scope!("my_new_op");
    // ... your code ...
}
```

`profile_scope!` 在不启用 `profile` feature 时编译为空操作（零开销）。

## 设计决策

- **CPU 侧计时**（`Instant::now()`）而非 GPU 硬件时间戳 — 覆盖 90% 场景，实现简单，无 AQL packet 修改
- **全局 Mutex<OpProfiler>** — 单 GPU 场景够用，多 GPU 需要改为 per-device
- **Feature gate** — `--features profile` 控制编译时插桩，不用时零开销
- **嵌套支持** — `begin("A") → begin("B") → end("B") → end("A")` 正确匹配

## 未实现（Future Work）

- [ ] GPU 硬件时间戳（`s_getreg SHADER_CYCLES`，需要 AQL packet 修改）
- [ ] Memory profiler（VRAM alloc/free 追踪、泄漏检测）
- [ ] 多 GPU 支持（per-device profiler）
- [ ] 自动 bottleneck 检测（计算 vs 内存 vs 调度）
