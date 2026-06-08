# Agent Task: Fix PM4 Counter Config & Tool-ify gfx-profiler

## 背景

t0-gpu 项目 (`/mnt/luyuzhou/hpc/bare-metal-1100/t0-gpu`) 是一个纯 Rust 的 AMD RDNA3 (GFX1100, RX 7900 XTX) 裸金属 GPU 编译器和运行时，通过 `/dev/kfd` ioctl 直接与 GPU 通信。

项目已实现一个初步的 GPU profiler (`src/gfx_profiler/`)，能加载 HSACO 文件并 profile kernel，但有两个关键问题需要修复。

## 任务 1: 修复 PM4 counter config GPU hang

### 问题描述

`Pm4CounterEngine::build_counter_config_cmds()` 通过 `Pm4CmdBuilder::set_uconfig_reg()` 写 GPU 性能计数器选择寄存器时，会导致 GPU hang。当前代码已将 counter config 禁用（返回空 Vec），profiler 只能测量 timing，无法采集硬件计数器指标。

### 已知信息

1. **寄存器地址** (from `docs/KFD_WO_GFX1100_NCU_LIKE_PROFILE_GUIDE.md`):
   - `SQ_PERFCOUNTER0_SELECT = 0xD040` (per-CU, 需要 GRBM_GFX_INDEX 定向)
   - `SQ_PERFCOUNTER_CTRL = 0xD030` (bit 6 = CS enable)
   - `GRBM_PERFCOUNTER0_SELECT = 0xD000` (全局, 无需定向)
   - `TCC_PERFCOUNTER0_SELECT = 0xD200` (per-channel)
   - `GRBM_GFX_INDEX = 0x30800` (CU 实例定向)

2. **当前实现** (`src/gfx_profiler/pm4_engine.rs:build_counter_config_cmds`):
   ```rust
   // 使用 SET_UCONFIG_REG 写所有寄存器
   pm4.set_uconfig_reg(GRBM_GFX_INDEX, &[gfx_index]);
   pm4.set_uconfig_reg(SQ_PERFCOUNTER0_SELECT, &[event_id]);
   pm4.set_uconfig_reg(SQ_PERFCOUNTER_CTRL, &[1u32 << 6]);
   ```

3. **`set_uconfig_reg` 实现** (`src/kfd/mod.rs`):
   ```rust
   pub fn set_uconfig_reg(&mut self, reg_addr: u32, values: &[u32]) {
       let reg_offset = (reg_addr - UCONFIG_REG_BASE) >> 2; // UCONFIG_REG_BASE = 0xC000
       self.pkt3(PM4_SET_UCONFIG_REG, &[reg_offset, ...values]);
   }
   ```

4. **PM4 提交路径**: `Pm4CmdBuilder::finish()` → `Vec<u32>` → `AqlQueue::submit_pm4()` → VENDOR_SPECIFIC AQL packet → GPU CP 执行

5. **counter readback** (`build_counter_readback_cmds`): 使用 `RELEASE_MEM` 的 `data_sel=1/2` 读 counter0/counter1_lo 到 GTT buffer

### 可能的原因

1. **寄存器地址空间错误**: `SQ_PERFCOUNTER*_SELECT` (0xD040+) 可能不在 UCONFIG 空间 (0xC000-0xFFFF)，可能需要用 `SET_SH_REG` 或 `SET_UCONFIG_REG` 的不同 base
2. **GRBM_GFX_INDEX 地址错误**: 0x30800 可能不是正确的 MMIO 地址
3. **缺少 ACQUIRE_MEM**: 写计数器寄存器前可能需要 cache invalidation
4. **缺少 counter reset**: 计数器可能需要先 reset 再配置
5. **PM4 packet 格式错误**: count field, opcode, 或 body 格式可能有误

### 调试建议

1. **读 AMD GPU 寄存器文档**: 查找 GFX1100 的 `SQ_PERFCOUNTER*_SELECT` 寄存器的正确地址空间和访问方式
2. **参考 tinygrad/amdgpu 驱动**: 搜索 `GRBM_GFX_INDEX` 和 `SQ_PERFCOUNTER` 的 PM4 编程方式
3. **逐步测试**: 先只写 `GRBM_GFX_INDEX`，不写 counter SELECT，看是否 hang；然后加一个 counter SELECT；逐步增加
4. **尝试 SET_SH_REG**: 对 `SQ_PERFCOUNTER*_SELECT` 用 `set_sh_reg` 代替 `set_uconfig_reg`
5. **检查 PM4 dump**: 用 `T0_DUMP_ASM=1` 环境变量 dump PM4 命令，检查二进制是否正确

### 验证标准

- `build_counter_config_cmds` 返回非空 PM4 命令
- 提交 PM4 后 GPU 不 hang（后续 kernel dispatch 正常完成）
- 读回的 counter 值非零（SQ_WAVES, SQ_INSTS 等）
- `cargo run --release --features rocm,gfx-profiler --example gfx_profiler -- --kernel gemm --m 256 --n 256 --k 256` 输出非零的 occupancy/IPC/cache hit rate

## 任务 2: 工具化 — 修复 CLI 第二次 dispatch hang

### 问题描述

从 CLI binary (`examples/gfx_profiler.rs`) 调用 profiler 时，第二次 dispatch 会 hang。但从 test examples (`test_mixed`, `test_cli_mimic`, `gfx_profiler_min`) 调用同样的 profiler 代码，一切正常。

### 已知信息

1. **CLI 和 test 的代码路径完全相同**: 都调用 `GfxProfiler::new()` → `GpuKernel::load()` → `profiler.profile_t0_kernel()`
2. **小 kernel (4 blocks) 可以工作**: `--grid 4 --n-elems 1024` 成功
3. **大 kernel (4096 blocks) hang**: `--grid 4096 --n-elems 1048576` 在第二次 dispatch hang
4. **test examples 用同样的 kernel 和 grid 正常工作**
5. **`read=0, target=1`**: 第一次 dispatch 就不被 GPU 处理
6. **`read=1, target=2`**: 第一次 dispatch 完成，第二次 hang

### 调试建议

1. **对比 CLI 和 test 的二进制**: 检查链接差异、优化级别、符号表
2. **strace 对比**: 用 `strace` 跟踪 CLI 和 test 的 syscall，比较 KFD ioctl 调用
3. **检查环境变量**: CLI 可能有不同的环境变量影响 KFD 行为
4. **检查 Arg parsing**: `parse_args()` 中的 `std::env::args()` 是否有副作用
5. **最小化 CLI**: 把 CLI 精简到和 test 完全一样的代码，逐步加回差异

### 验证标准

- `cargo run --release --features rocm,gfx-profiler --example gfx_profiler -- --kernel gemm --m 256 --n 256 --k 256` 正常完成
- `--kernel elf:tests/hip_kernels/vector_add_gfx1100.hsaco --grid 4096 --n-elems 1048576` 正常完成

## 任务 3: 工具化 — 完善 CLI 功能

在任务 1 和 2 修复后，完善 CLI：

1. **支持更多 kernel 类型**: 添加 `--kernel softmax`, `--kernel rmsnorm` 等内置 kernel
2. **JSON 输出**: `--format json` 输出完整的 profiling 数据
3. **多 kernel batch**: `--kernel gemm,softmax,rmsnorm` 一次 profile 多个 kernel
4. **输出到文件**: `--output profile.json`
5. **Help 文档**: 完善 `--help` 输出

## 关键文件

| 文件 | 说明 |
|------|------|
| `src/gfx_profiler/mod.rs` | GfxProfiler 顶层, profile_t0_kernel, profile_with_events |
| `src/gfx_profiler/pm4_engine.rs` | Pm4CounterEngine, build_counter_config_cmds, build_counter_readback_cmds, execute_pass |
| `src/gfx_profiler/counter_config.rs` | CounterEvent 定义, 19 个事件, schedule_passes, 寄存器地址常量 |
| `src/gfx_profiler/metrics.rs` | RawCounters → ProfileMetrics (IPC, occupancy, hit rate, bandwidth) |
| `src/gfx_profiler/report.rs` | NCU 风格 text + JSON 报告 |
| `src/gfx_profiler/suggestions.rs` | 优化建议引擎 |
| `src/kfd/mod.rs` | Pm4CmdBuilder (set_sh_reg, set_uconfig_reg), AqlQueue (submit_pm4, dispatch_signal, submit), DispatchPool, KfdDevice |
| `examples/gfx_profiler.rs` | CLI 入口 |
| `docs/KFD_WO_GFX1100_NCU_LIKE_PROFILE_GUIDE.md` | 完整的寄存器地址、PM4 格式、counter 事件 ID、NCU metric 映射 |

## 构建和测试

```bash
# 构建
cargo build --release --features rocm,gfx-profiler

# 测试 GEMM (需要修复后)
cargo run --release --features rocm,gfx-profiler --example gfx_profiler -- --kernel gemm --m 256 --n 256 --k 256

# 测试 HIP kernel
cargo run --release --features rocm,gfx-profiler --example gfx_profiler -- \
  --kernel elf:tests/hip_kernels/vector_add_gfx1100.hsaco --grid 4 --wg 256 --n-elems 1024

# GPU 测试必须 --test-threads=1
cargo test --release --features rocm,gfx-profiler -- --test-threads=1
```

## 硬件环境

- GPU: AMD RX 7900 XTX (RDNA3, GFX1100, 96 CU)
- OS: Linux 6.14.0-37-generic
- Rust: 1.94.0
- KFD: /dev/kfd (版本 1.18)
- Resizable BAR: 已启用 (32GB)
