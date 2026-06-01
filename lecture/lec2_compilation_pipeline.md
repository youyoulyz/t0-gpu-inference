# Lec 2: 编译管线深度解析 — 从 BlockDSL 到 GPU 机器码

> **课程**: 人工智能硬件部署工程实践
> **目标**: 理解 T0 编译器如何把高级 DSL 变成 GPU 能执行的机器码

---

## 1. 编译管线总览

```text
BlockDSL (Rust API)
  │
  │  block_dsl_to_ssa.rs: 翻译为 SSA IR
  ▼
SSA IR (Static Single Assignment)
  │
  │  opt_passes.rs: 9-pass 优化
  │    CopyProp → CSE → LICM → ConstFold → AlgSimp
  │    → InsnCombine → LoopUnroll → Waitcnt → Schedule
  ▼
优化后的 SSA IR
  │
  │  ssa_regalloc.rs: 线性扫描寄存器分配
  ▼
Vec<Op> (带物理寄存器的机器指令)
  │
  │  asm_emitter.rs: 发射 GFX1100 二进制
  ▼
AMD HSA ELF Code Object
  │
  │  KFD runtime: 加载到 GPU
  ▼
GPU 执行
```

### 每一层的作用

| 层 | 文件 | 作用 | 类比 |
|---|---|---|---|
| DSL | `block_dsl.rs` | 用户 API | Python/Triton |
| SSA IR | `ssa_ir.rs` | 中间表示 | LLVM IR |
| 优化 | `opt_passes.rs` | 消除冗余 | LLVM -O2 |
| 寄存器分配 | `ssa_regalloc.rs` | 映射到硬件寄存器 | LLVM regalloc |
| ISA 发射 | `asm_emitter.rs` | 生成二进制 | 汇编器 |
| ELF | `rdna3_code_object.rs` | 打包为可加载格式 | 链接器 |

---

## 2. SSA IR: 编译器的通用语言

### 什么是 SSA？

SSA (Static Single Assignment) 要求：**每个变量只被赋值一次**。

```text
// 普通 IR:
x = load a[i]
x = x + 1
x = x * 2

// SSA IR:
x0 = load a[i]
x1 = x0 + 1
x2 = x1 * 2
```

### 为什么用 SSA？

1. **每个值有唯一定义** — 不需要追踪"当前值是哪个"
2. **use-def 链直接可用** — 优化 pass 不需要数据流分析
3. **寄存器分配更简单** — 每个 MVal 的生命周期是连续的

### T0 的 SSA 结构

```rust
// src/t0/ssa_ir.rs

/// SSA 值（每个定义唯一）
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MVal(pub u32);

/// SSA 指令
pub struct MachInst {
    pub op: Op,                    // 原始机器指令
    pub defs: Vec<MVal>,           // 定义的 SSA 值
    pub uses: Vec<MVal>,           // 使用的 SSA 值
    pub implicit_defs: Vec<ImplicitReg>,  // 隐式定义 (VCC/SCC)
    pub implicit_uses: Vec<ImplicitReg>,  // 隐式使用
}

/// 隐式状态寄存器（GFX1100 特有）
pub enum ImplicitReg {
    Vcc,   // 向量条件码（v_cmp_* 写入，v_cndmask 读取）
    Scc,   // 标量条件码（s_cmp_* 写入，s_cbranch 读取）
    Exec,  // 执行掩码（SIMD mask）
}
```

### 看一个实际例子

BlockDSL:
```rust
let x = kb.load(input_ptr, offset, mask);
let y = x.add(&mut kb, kb.const_f32(1.0));
let z = y.mul(&mut kb, y);
kb.store(output_ptr, offset, z, mask);
```

翻译为 SSA IR (概念):
```text
m0 = s_load_b64  input_ptr       // 加载指针
m1 = v_add_u32   pid*256, tid    // 计算偏移
m2 = v_cmp_lt_u32 m1, n          // 边界检查 → VCC
m3 = buffer_load_f32 m0, m1      // 加载数据 (masked by VCC)
    ; implicit_use: VCC
m4 = v_add_f32   m3, 1.0         // x + 1
m5 = v_mul_f32   m4, m4          // y * y
buffer_store_f32 m5, m0, m1      // 存储
    ; implicit_use: VCC
```

注意：
- `m0`, `m1`, `m2`... 每个只定义一次
- `VCC` 是隐式的 — `v_cmp` 写入，`buffer_store` 读取（作为 mask）
- SSA 编译器必须追踪这些隐式依赖，否则优化会破坏正确性

---

## 3. 优化 Pass: 消除冗余

9 个优化 pass，分 4 个阶段：

### Phase A: SSA 级优化

**Copy Propagation (复制传播)**
```text
// 优化前:
m1 = v_mov_b32 m0       // m1 = m0
m2 = v_add_f32 m1, m3   // 用到 m1

// 优化后:
m2 = v_add_f32 m0, m3   // 直接用 m0，消除 m1
```

**Common Subexpression Elimination (公共子表达式消除)**
```text
// 优化前:
m3 = v_mul_f32 m0, m1
m4 = v_mul_f32 m0, m1   // 重复计算！

// 优化后:
m3 = v_mul_f32 m0, m1
m4 = m3                  // 复用 m3
```

**LICM (循环不变量外提)**
```text
// 优化前:
loop:
    m3 = v_mul_f32 m0, m1  // m0, m1 在循环外定义，不变
    m4 = v_add_f32 m3, m2
    s_cbranch loop

// 优化后:
m3 = v_mul_f32 m0, m1      // 提到循环外
loop:
    m4 = v_add_f32 m3, m2
    s_cbranch loop
```

**Algebraic Simplification (代数化简)**
```text
m2 = v_mul_f32 m0, 1.0   →  m2 = m0      // x * 1 = x
m2 = v_add_f32 m0, 0.0   →  m2 = m0      // x + 0 = x
m2 = v_mul_f32 m0, 2.0   →  m2 = v_add_f32 m0, m0  // x * 2 = x + x (更便宜)
```

**Instruction Combine (指令合并)**
```text
// 优化前:
m3 = v_mul_f32 m0, m1
m4 = v_add_f32 m3, m2

// 优化后:
m4 = v_fma_f32 m0, m1, m2  // m4 = m0 * m1 + m2 (一条指令)
```

### Phase B: 机器指令级优化

**Loop Unroll (循环展开)**
```text
// 对于小循环（trip count 已知且小），展开为线性代码
// 消除分支开销，增加指令级并行
```

### Phase C: 迭代优化

重复执行 AlgSimp + DCE 直到不动点（没有更多优化机会）。

### Phase D: 硬件相关优化

**Waitcnt 优化 (关键！)**
```text
// GFX1100 是异步的：load 指令发出后不等结果
// 必须在使用前插入 s_wait 指令

// 优化前（保守，每条 load 后都等）:
buffer_load_f32 v1, ...
s_waitcnt vmcnt(0)         // 等 v1 就绪
v_add_f32 v2, v1, v3
buffer_load_f32 v4, ...
s_waitcnt vmcnt(0)         // 等 v4 就绪
v_mul_f32 v5, v4, v6

// 优化后（精确 waitcnt）:
buffer_load_f32 v1, ...
buffer_load_f32 v4, ...     // 两次 load 可以重叠！
s_waitcnt vmcnt(1)          // 只等 v1 就绪（v4 还在飞行）
v_add_f32 v2, v1, v3
s_waitcnt vmcnt(0)          // 现在等 v4
v_mul_f32 v5, v4, v6
```

**为什么 waitcnt 优化重要？**

GFX1100 的 load 延迟约 300 个时钟周期。如果每条 load 后都等，GPU 就变成了 CPU。通过精确计算 `vmcnt(N)`，可以让多条 load 并行执行，隐藏延迟。

---

## 4. 寄存器分配: 从无限到有限

SSA IR 假设有无限个 `MVal`，但硬件只有有限的 VGPR (1536/CU)。

### 线性扫描算法

```text
1. 计算每个 MVal 的活跃区间 (live interval):
   m3 = ...        // m3 定义
   ... m3 ...      // m3 使用
   ←── m3 活跃 ──→

2. 按起始位置排序，贪心分配:
   - 如果有空闲 VGPR → 分配
   - 如果没有 → spill 到 LDS（慢！）
```

### 对齐约束

WMMA 指令要求 8 对齐的 VGPR 组：
```text
// 正确：v16..v23 (8 对齐)
v_wmma_f32_16x16x16_bf16 v[16:23], v[0:7], v[8:15], v[16:23]

// 错误：v17..v24 (不对齐) → 编译错误
```

寄存器分配器必须保证 WMMA 片段分配到 8 对齐的 VGPR 组。

### Spill 的代价

```text
正常: v_add_f32 v0, v1, v2           // 1 个周期

Spill: s_store_b32 v0, s[sp], offset  // 写 LDS: ~20 周期
       s_load_b32 v0, s[sp], offset   // 读 LDS: ~20 周期
       v_add_f32 v0, v1, v2           // 1 个周期
       → 41x 慢！
```

所以寄存器分配的目标是：**最小化 spill**。

---

## 5. ISA 发射: 最后的翻译

优化后的 `Vec<Op>` 被翻译为 GFX1100 二进制指令：

```text
// Vec<Op>:
Op::VAddF32 { dst: VReg(2), src0: VReg(0), src1: VReg(1) }

// GFX1100 ISA (16 字节):
// VOP3 格式:
//   [31:24] 0xD1 (VOP3 opcode prefix)
//   [23:16] 0x00 (V_ADD_F32)
//   [15:8]  vsrc0, vsrc1
//   [7:0]   vdst
bytes: [0x00, 0x00, 0x02, 0xD1, 0x01, 0x00, 0x02, 0x04, ...]
```

### ISA 编码器

`rdna3_asm.rs` 包含 GFX1100 全指令集的二进制编码器：

| 指令类型 | 示例 | 用途 |
|---|---|---|
| VOP1 | `v_mov_b32`, `v_exp_f32` | 单操作数向量指令 |
| VOP2 | `v_add_f32`, `v_mul_f32` | 双操作数向量指令 |
| VOP3 | `v_fma_f32`, `v_cmp_lt` | 三操作数/比较指令 |
| SMEM | `s_load_b64`, `s_store` | 标量内存操作 |
| FLAT | `flat_load_b128` | 全局内存操作 |
| DS | `ds_read_b32`, `ds_write` | LDS 操作 |
| WMMA | `v_wmma_f32_16x16x16` | 矩阵乘法加速 |
| SOPP | `s_waitcnt`, `s_endpgm` | 流程控制 |

---

## 6. 实际观察: T0_DUMP_ASM

用 `T0_DUMP_ASM=1` 可以看到编译器生成的完整汇编：

```bash
T0_DUMP_ASM=1 cargo test --release --features rocm \
  -- test_softmax_forward --nocapture --test-threads=1
```

输出示例（简化）：
```asm
;; T0 Kernel: softmax_fwd
;; VGPRs used: 12, SGPRs used: 8

;; === Prologue ===
s_load_b64  s[0:1], s[4:5], 0x00    // 加载 input_ptr
s_load_b64  s[2:3], s[4:5], 0x08    // 加载 output_ptr
s_load_b32  s6, s[4:5], 0x10        // 加载 cols
s_waitcnt   lgkmcnt(0)              // 等待标量加载

;; === 计算偏移 ===
v_and_b32   v1, 31, v0              // tid = lane_id & 31
v_readfirstlane_b32 s7, v0          // pid = workgroup_id
v_mul_lo_u32 v2, s7, s6             // row_base = pid * cols
v_add_u32   v3, v2, v1              // offset = row_base + tid
v_cmp_lt_u32 vcc, v3, s6            // mask = offset < cols

;; === 加载输入 ===
buffer_load_b32 v4, v3, s[0:1], 0   // x = load(input, offset)
s_waitcnt   vmcnt(0)                // 等待数据

;; === Row Max ===
v_cmpx_lt_u32 exec, v3, s6         // exec = mask
v_max_f32   v5, v4, -inf            // masked max
;; ... wave reduce + LDS reduce ...

;; === Exp ===
v_sub_f32   v6, v4, v5              // shifted = x - max
v_exp_f32   v7, v6                   // exp_x = exp(shifted)

;; === Row Sum ===
;; ... wave reduce + LDS reduce ...
v_rcp_f32   v8, v9                   // inv_sum = 1/sum

;; === 输出 ===
v_mul_f32   v10, v7, v8             // result = exp_x * inv_sum
buffer_store_b32 v10, v3, s[2:3], 0 // store(output, offset)

s_endpgm                             // 结束
```

### 观察要点

1. **Prologue** — 加载 kernel arguments（SGPR 操作）
2. **偏移计算** — `pid * cols + tid`，纯 SGPU/ALU 指令
3. **VCC/EXEC** — `v_cmp` 写 VCC，`v_cndmask` 读 VCC
4. **Waitcnt** — `s_waitcnt vmcnt(0)` 确保 buffer_load 完成
5. **Wave reduce** — `v_max_f32` + `ds_write` + `ds_read` 实现 workgroup 归约
6. **s_endpgm** — 必须在内核末尾

---

## 7. ISA Verifier: 编译时安全检查

在内核编译后、发送到 GPU 前，ISA verifier 检查 8 类 hang 模式：

| 检查项 | 说明 | 为什么会导致 hang |
|---|---|---|
| VCC 残留 | 循环中 `v_cmp` 后没有 `s_and_b64 vcc` | 下次循环 VCC 状态错误 |
| EXEC 不平衡 | `s_and_saveexec` 后没有 `s_or_b64 exec` | 部分线程永久禁用 |
| 缺失 waitcnt | load 后直接使用数据 | 读到未就绪的数据 |
| 缺失 s_endpgm | 内核没有结束指令 | GPU 跑飞 |
| 自循环 | 分支跳转到自身 | 死循环 |
| VReg 超限 | VGPR > 256 | 硬件不支持 |

```rust
// src/t0/isa_verifier.rs
pub fn verify_ops(ops: &[Op]) -> Result<(), String> {
    // 检查 VCC 清理
    // 检查 EXEC 平衡
    // 检查 waitcnt
    // ...
}
```

---

## 8. Lab 2 作业

### 作业 2.1: 观察优化 Pass 的效果 (30 分)

**任务**: 用 `T0_OPT_LEVEL` 环境变量对比不同优化级别的 ISA 输出。

```bash
# 无优化
T0_OPT_LEVEL=0 T0_DUMP_ASM=1 cargo test --release --features rocm \
  -- test_softmax_forward --nocapture --test-threads=1

# 最高优化
T0_OPT_LEVEL=4 T0_DUMP_ASM=1 cargo test --release --features rocm \
  -- test_softmax_forward --nocapture --test-threads=1
```

**报告要求**:
- 统计两个版本的指令数差异
- 找出至少 2 个被消除的冗余指令，解释是哪个 pass 消除的
- 对比 waitcnt 指令数量差异

### 作业 2.2: 分析 SSA IR (40 分)

**任务**: 阅读 `src/t0/ssa_ir.rs` 中的 `lift_to_ssa()` 函数，回答以下问题：

1. 为什么 `v_cmp` 指令需要 `implicit_defs: [VCC]`？如果不建模 VCC 会怎样？

2. 为什么 `v_cndmask` 需要 `implicit_uses: [VCC]`？

3. `s_and_saveexec` 同时定义 EXEC 并使用 EXEC，编译器如何处理这种"读-改-写"模式？

4. SSA 的 Phi 节点在 T0 中是怎么实现的？（提示：看 `MachFunc` 的 block 结构）

### 作业 2.3: Waitcnt 优化分析 (30 分)

**任务**: 在下面的代码片段中，手动计算最优的 `s_waitcnt` 参数。

```text
buffer_load_f32 v1, ...    // load A
buffer_load_f32 v2, ...    // load B
buffer_load_f32 v3, ...    // load C
v_add_f32 v4, v1, v2       // 使用 A, B
buffer_load_f32 v5, ...    // load D
v_mul_f32 v6, v3, v4       // 使用 C
v_add_f32 v7, v5, v6       // 使用 D
```

**问题**:
1. 在 `v_add_f32 v4, v1, v2` 之前，`vmcnt` 应该是多少？
2. 在 `v_mul_f32 v6, v3, v4` 之前，`vmcnt` 应该是多少？
3. 在 `v_add_f32 v7, v5, v6` 之前，`vmcnt` 应该是多少？
4. 如果 GPU 一次最多能同时飞行 15 条 load（`vmcnt` 上限 = 15），上面的代码是否需要等待？

**提示**: `vmcnt(N)` 表示"还有 N 条 load 未完成"。`vmcnt(0)` 表示"所有 load 都完成了"。

---

## 9. 思考题

1. **为什么 T0 选择 SSA 而不是传统的三地址码？** 提示：考虑优化 pass 的实现复杂度。

2. **GFX1100 的 VCC 和 SCC 有什么区别？** 提示：VCC 是 per-lane 还是 per-wave？SCC 呢？

3. **如果一个内核用了 64 个 VGPR，一个 CU 最多能同时运行多少个 wave？** 提示：一个 CU 有 1536 个 VGPR。

4. **Waitcnt 优化为什么是 Phase D（最后阶段）而不是 Phase A？** 提示：考虑寄存器分配对活跃 load 数量的影响。

---

## 参考资料

- SSA IR: `src/t0/ssa_ir.rs`
- 优化 Pass: `src/t0/opt_passes.rs`
- 寄存器分配: `src/t0/ssa_regalloc.rs`
- ISA 发射: `src/t0/asm_emitter.rs`
- ISA 编码器: `src/rdna3_asm.rs`
- ISA 验证器: `src/t0/isa_verifier.rs`
- ELF 生成: `src/rdna3_code_object.rs`
