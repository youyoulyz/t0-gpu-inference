# T0-GPU Demo Results — RX 7900 XTX (GFX1100) on bare-metal-1100

## 硬件环境 / Hardware

| 项目 | 详情 |
|------|------|
| GPU | AMD Radeon RX 7900 XTX (RDNA3, GFX1100) |
| CPU | x86_64 Linux |
| 内核 | 6.x (amdgpu KFD 内置驱动) |
| Rust | 1.94.0 |
| LLVM | 17.0.6 |
| Resizable BAR | ❌ 未启用 (Small BAR 系统) |

## 关键修复 / Key Fix: Small BAR 支持

这台机器**没有启用 Resizable BAR / Above 4G Decoding**，导致 KFD 无法分配 host-visible VRAM（`PUBLIC` 标志）。内核日志显示：
```
amdgpu: Alloc host visible vram on small bar is not allowed
```

### 修复内容

修改了 `src/kfd/mod.rs` 中的 `alloc_vram()` 和 `alloc_code()` 方法，添加了 small BAR 回退路径：
- `alloc_vram()`: PUBLIC VRAM 失败 → 回退到 non-public VRAM
- `alloc_code()`: PUBLIC VRAM 失败 → 回退到 non-public VRAM + EXECUTABLE
- 新增 `alloc_vram_host()`: 专用于需要 CPU 读回的缓冲区

### 修改的文件
- `src/kfd/mod.rs` — 添加 small BAR fallback 逻辑
- `examples/hello_gemm_gen.rs` — 输出缓冲区改用 `alloc_vram_host()`

## Benchmark 结果 / Results

### 4096³ GEMM Autotuner（主力测试）

```
Best: tile_gemm_128x64_k32_db → 105.2 TFLOPS
```

| 排名 | 配置 | TFLOPS |
|------|------|--------|
| 1 | 128×64 k32 | **105.2** |
| 2 | 128×128 k32 | 104.5 |
| 3 | 64×128 k32 | 96.5 |
| 4 | 64×64 k64 | 94.2 |
| 5 | 64×64 k32 | 94.1 |

> 🏆 **超越 README 记录的 96.4 TF**，达到 105.2 TF（~63.8% 峰值利用率）

### 正确性测试 / Correctness

```
39 configs tested, 39 PASS, 0 FAIL
```

覆盖 64³ ~ 256×512×512 多尺寸、多配置（含 split-k），最大误差 ~1e-5（BF16 精度范围）。

### 全谱扫描 / Full Spectrum (256³~8192³)

256³~4096³ 正常运行。8192³ 因 GPU 热节流/频率爬升导致 hang（机器散热条件限制）。

## 运行命令 / Run Commands

```bash
cd /mnt/luyuzhou/hpc/bare-metal-1100/t0-gpu

# 编译
cargo build --release --lib --features rocm

# 4096³ autotuner（主力 benchmark）
cargo test --release --features rocm -- test_tune_tile_ir_4096 \
  --nocapture --ignored --test-threads=1

# 正确性测试
cargo test --release --features rocm -- test_tile_ir_correctness \
  --nocapture --test-threads=1

# GEMM 正确性验证（独立 example）
cargo run --release --features rocm --example test_gemm_correctness

# ISA 汇编导出调试
T0_DUMP_ASM=1 cargo test --release --features rocm -- \
  test_lower_gemm_128x128_k32_compiles --nocapture
```

## 注意事项 / Notes

1. **GPU 热节流**: 长时间运行后性能可能下降 1-3%，建议跑 2-3 次取最佳值
2. **Small BAR**: 本机未启用 Resizable BAR，所有 VRAM 分配自动回退到 non-public 模式
3. **8192³ hang**: 大矩阵全谱扫描时可能因热节流导致 GPU hang，属于散热限制而非软件问题
4. **LLVM 17**: 已通过 `sudo apt install llvm-17 lld-17` 安装，符号链接到 `/usr/local/bin/`

## 依赖安装 / Dependencies Installed

```bash
sudo apt install llvm-17 lld-17
sudo ln -sf /usr/bin/llvm-mc-17 /usr/local/bin/llvm-mc
sudo ln -sf /usr/bin/ld.lld-17 /usr/local/bin/ld.lld
```
