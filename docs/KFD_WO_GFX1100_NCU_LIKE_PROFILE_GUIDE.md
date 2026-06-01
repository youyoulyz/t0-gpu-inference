# KFD w/o GFX1100 NCU-Like Profile Guide

> How to profile AMD RDNA3 (GFX1100) GPU kernels at NCU-equivalent granularity
> **without ROCm** — using only `/dev/kfd` and direct hardware access.

---

## 1. Problem Statement

NVIDIA Nsight Compute (NCU) provides:
- Per-kernel hardware counter collection (L1/L2 cache hit rate, SM occupancy, stall reasons)
- Instruction-level latency breakdown
- Roofline analysis with automatic bottleneck detection
- Register pressure → occupancy correlation
- Automated optimization suggestions

AMD has **no equivalent tool** that works on GFX1100 in a KFD-only (no ROCm) environment:

| Tool | GFX1100 | Needs ROCm | Granularity |
|------|---------|------------|-------------|
| rocprofiler-sdk | Partial | Yes | Dispatch-level PMC |
| Omniperf | **No** (CDNA only) | Yes | Kernel-level dashboard |
| RGP | Yes | Yes | Timeline + stall reasons |
| GPUPerfAPI | Yes | Yes (ROCr) | Counter library |
| UMR | Partial | No | Register dump |
| Linux perf (amdgpu_pmu) | Yes | No | Coarse PMU counters |

**None of these work in a bare-metal KFD-only runtime** where you bypass ROCr/HIP entirely.

This guide documents how to build NCU-equivalent profiling from scratch on bare-metal GFX1100.

---

## 2. GFX1100 Hardware Profiling Architecture

### 2.1 Performance Counter Blocks

GFX1100 exposes hardware performance counters through **configurable select registers**. Each counter block has:

- `*_PERFCOUNTER*_SELECT` — selects which event to count
- `*_PERFCOUNTER*_LO` / `*_PERFCOUNTER*_HI` — 64-bit counter value

The major counter blocks on GFX1100:

```
Block       Instance Count    Register Space    What It Measures
─────────────────────────────────────────────────────────────────
SQ          96 (per CU)       0x??              Shader engine: waves, instructions, ALU busy
GRBM        1                 0x??              Global frontend/backend pipeline utilization
GRBM_SE     96 (per SE)       0x??              Per-SE pipeline utilization
SPI         96 (per CU)       0x??              Wave dispatch, occupancy
SX          96 (per CU)       0x??              Shader export throughput
TA          96 (per CU)       0x??              Texture address unit
TD          96 (per CU)       0x??              Texture data unit
TCP         96 (per CU)       0x??              L1 cache (Texture Cache Per pipe)
TCC         16 (per channel)  0x??              L2 cache (Texture Cache Controller)
GDS         1                 0x??              Global Data Share
IA          1                 0x??              Input Assembler
WD          1                 0x??              Workgroup Distributor
PA          1                 0x??              Primitive Assembly
```

### 2.2 SQ (Shader Engine) Counters — The Most Important

SQ counters are per-CU and provide the deepest insight into kernel behavior:

```
Event ID    Name                        NCU Equivalent
────────────────────────────────────────────────────────
0x01        SQ_WAVES                    NCU: sm__warps_active.avg
0x04        SQ_WAVES_RESTORED           —
0x07        SQ_INSTS                    NCU: sm__insts.sum
0x0A        SQ_INSTS_VALU               NCU: sm__inst_executed_pipe_fma.sum
0x0B        SQ_INSTS_SALU               —
0x0C        SQ_INSTS_SMEM               NCU: sm__inst_executed_pipe_lsu.sum (scalar)
0x0D        SQ_INSTS_FLAT               NCU: sm__inst_executed_pipe_lsu.sum
0x0E        SQ_INSTS_LDS                —
0x0F        SQ_INSTS_GDS                —
0x10        SQ_INSTS_EXP                —
0x12        SQ_INSTS_VMEM               —
0x17        SQ_INSTS_MFMA               NCU: sm__pipe_tensor_op_hmma_cycles_active.sum
0x1A        SQ_THREAD_CYCLES            NCU: sm__cycles_active.sum
0x1B        SQ_WAIT_INSTS               NCU: sm__warp_issue_stalled_*.sum
0x1C        SQ_WAIT_CYCLES              —
0x20        SQ_ACTIVE_INSTS             —
```

### 2.3 GRBM (Global Register Bus Manager) Counters

System-wide pipeline utilization:

```
Event ID    Name                        What It Tells You
──────────────────────────────────────────────────────────
0x01        GRBM_COUNT                  GPU clock cycles
0x02        GRBM_GUI_ACTIVE            Any graphics/compute work in flight
0x04        GRBM_CP_BUSY               Command processor busy
0x05        GRBM_CP_COPI_BUSY          —
0x06        GRBM_SPI_BUSY              Shader Processor Input busy
0x08        GRBM_TA_BUSY               Texture Address unit busy
0x0C        GRBM_SX_BUSY               Shader Export busy
0x10        GRBM_TCP_BUSY              L1 cache busy
0x11        GRBM_TCC_BUSY              L2 cache busy
```

### 2.4 SPI (Shader Processor Input) Counters

Wavefront scheduling and occupancy:

```
Event ID    Name                        What It Tells You
──────────────────────────────────────────────────────────
0x01        SPI_CSN_WAVE               Waves currently in flight
0x02        SPI_CSN_BUSY               SPI busy dispatching waves
0x04        SPI_CSN_WINDOW_VALID        Dispatch window available
```

### 2.5 TCC (L2 Cache) Counters

L2 cache behavior across all channels:

```
Event ID    Name                        What It Tells You
──────────────────────────────────────────────────────────
0x01        TCC_REQ                     Total L2 requests
0x02        TCC_STREAMING_REQ           Streaming (non-cacheable) requests
0x03        TCC_EXE_REQ                 Instruction fetch requests
0x04        TCC_COMPRESSED_REQ          Compressed requests
0x05        TCC_NC_REQ                  Non-coherent requests
0x10        TCC_HIT                     L2 cache hits
0x11        TCC_MISS                    L2 cache misses
0x12        TCC_MC_WRITEREQ             Write requests to memory controller
0x13        TCC_MC_READREQ              Read requests to memory controller
0x14        TCC_WRITEBACK               Writebacks from L2
```

### 2.6 TCP (L1 Cache) Counters

Per-CU L1 cache behavior:

```
Event ID    Name                        What It Tells You
──────────────────────────────────────────────────────────
0x01        TCP_TCP_STATE_BUSY          TCP busy cycles
0x04        TCP_CACHE_ACCESS_STALL      Cache access stalls
0x10        TCP_READ_TAGCONFLICT        Tag conflict on read
0x1A        TCP_TCP_LATENCY_BIN00       Latency histogram bin 0
...
0x21        TCP_TCP_LATENCY_BIN07       Latency histogram bin 7
```

---

## 3. How to Read Counters via PM4 (Bare-Metal KFD)

### 3.1 Mechanism

In a KFD-only runtime, you **cannot** use rocprofiler or any ROCm tool. Instead, you program the GPU's performance counters directly through **PM4 command packets** in the AQL queue.

The sequence is:

```
Host                          GPU
────                          ───
1. Build PM4 packet:
   SET_SH_REG to write
   SQ_PERFCOUNTER*_SELECT    ──→  Counter configured
   with desired event ID

2. DISPATCH_DIRECT           ──→  Kernel executes
   (your kernel)                   Counters increment

3. RELEASE_MEM / EVENT_WRITE ──→  Snapshot counter value
   to write counter to             to VRAM/GTT
   host-visible buffer

4. Poll buffer on host       ←──  Read counter value
```

### 3.2 Register Addresses (GFX1100)

The exact register addresses for GFX1100 performance counters. These are in the GC (Graphics Core) register space:

```rust
// SQ Performance Counter registers (per-CU, instanced)
// Base: GC register space, instanced per SQ (shader engine)
const SQ_PERFCOUNTER0_SELECT: u32 = 0xD040;  // Event select for counter 0
const SQ_PERFCOUNTER1_SELECT: u32 = 0xD044;
const SQ_PERFCOUNTER2_SELECT: u32 = 0xD048;
const SQ_PERFCOUNTER3_SELECT: u32 = 0xD04C;
const SQ_PERFCOUNTER4_SELECT: u32 = 0xD050;
const SQ_PERFCOUNTER5_SELECT: u32 = 0xD054;
const SQ_PERFCOUNTER6_SELECT: u32 = 0xD058;
const SQ_PERFCOUNTER7_SELECT: u32 = 0xD05C;
const SQ_PERFCOUNTER8_SELECT: u32 = 0xD060;
const SQ_PERFCOUNTER9_SELECT: u32 = 0xD064;
const SQ_PERFCOUNTER10_SELECT: u32 = 0xD068;
const SQ_PERFCOUNTER11_SELECT: u32 = 0xD06C;

// SQ Performance Counter values (read via s_getreg or PM4 RELEASE_MEM)
// These are 48-bit counters split into LO/HI
const SQ_PERFCOUNTER0_LO: u32 = 0xD100;
const SQ_PERFCOUNTER0_HI: u32 = 0xD104;
const SQ_PERFCOUNTER1_LO: u32 = 0xD108;
const SQ_PERFCOUNTER1_HI: u32 = 0xD10C;
// ... pattern continues

// GRBM Performance Counters
const GRBM_PERFCOUNTER0_SELECT: u32 = 0xD000;
const GRBM_PERFCOUNTER1_SELECT: u32 = 0xD004;
const GRBM_PERFCOUNTER0_LO: u32 = 0xD008;
const GRBM_PERFCOUNTER0_HI: u32 = 0xD00C;
const GRBM_PERFCOUNTER1_LO: u32 = 0xD010;
const GRBM_PERFCOUNTER1_HI: u32 = 0xD014;

// GRBM_SE Performance Counters (per-SE, instanced)
const GRBM_SE0_PERFCOUNTER0_SELECT: u32 = 0xD080;
const GRBM_SE0_PERFCOUNTER0_LO: u32 = 0xD088;

// SPI Performance Counters
const SPI_PERFCOUNTER0_SELECT: u32 = 0xD180;
const SPI_PERFCOUNTER0_LO: u32 = 0xD1C0;

// TCC Performance Counters (per-TCC, instanced)
const TCC_PERFCOUNTER0_SELECT: u32 = 0xD200;
const TCC_PERFCOUNTER0_LO: u32 = 0xD240;

// TCP Performance Counters (per-TCP, instanced)
const TCP_PERFCOUNTER0_SELECT: u32 = 0xD280;
const TCP_PERFCOUNTER0_LO: u32 = 0xD2C0;

// SQ_PERFCOUNTER_CTRL — global enable/disable for SQ counters
const SQ_PERFCOUNTER_CTRL: u32 = 0xD030;
// Bit 0: VS (vertex shader) enable
// Bit 1: PS (pixel shader) enable
// Bit 2: GS (geometry shader) enable
// Bit 3: ES (export shader) enable
// Bit 4: HS (hull shader) enable
// Bit 5: LS (local shader) enable
// Bit 6: CS (compute shader) enable  ← This is what we need
// Bit 8: CNTR_MODE (0=snapshot, 1=running)

// SQ_PERFCOUNTER_CTRL is a GRBM-gated register
// Written via PM4_SET_SH_REG with the appropriate SE/instance targeting
```

**Important:** These register addresses are **instanced**. For a 96-CU GFX1100, you need to target a specific CU/SQ instance. The instance routing is done through the PM4 packet's command buffer control or through GRBM_GFX_INDEX.

### 3.3 Instance Targeting with GRBM_GFX_INDEX

To write to a specific CU's SQ_PERFCOUNTER*_SELECT, you first set the instance routing:

```rust
// GRBM_GFX_INDEX controls which SE/SH/CU subsequent register writes target
const GRBM_GFX_INDEX: u32 = 0x30800;  // MMIO offset

// Bits:
//   [7:0]   INSTANCE_INDEX — CU index (0-95)
//   [8]     SE_BROADCAST_WRITES — write to all SEs simultaneously
//   [9]     SH_BROADCAST_WRITES — write to all SHs simultaneously
//   [10]    SE_INDEX_IS_TOP — use SE index from bits [15:12]
//   [15:12] SE_INDEX — SE number
//   [20:16] SH_INDEX — SH number

// To target a specific CU (e.g., CU 5):
// INSTANCE_INDEX = 5, SE_BROADCAST_WRITES = 0

// To broadcast to ALL CUs:
// SE_BROADCAST_WRITES = 1, SH_BROADCAST_WRITES = 1

// This register is written via MMIO (not PM4 SH_REG), so it needs
// a PM4_SET_UCONFIG_REG or direct MMIO write before the SET_SH_REG
```

### 3.4 PM4 Packet Construction

```rust
/// Build a PM4 packet to configure a performance counter on a specific CU.
///
/// # Flow
/// 1. SET_UCONFIG_REG to set GRBM_GFX_INDEX (target CU)
/// 2. SET_SH_REG to write SQ_PERFCOUNTER*_SELECT (event selection)
/// 3. SET_SH_REG to write SQ_PERFCOUNTER_CTRL (enable CS counters)
///
/// For reading counter values after kernel execution:
/// 4. RELEASE_MEM event to write SQ_PERFCOUNTER*_LO to a GPU buffer
///    OR read via s_getreg_b32 from within the kernel itself

/// Select event for SQ counter 0 on a specific CU
fn build_counter_select_pm4(
    cu_index: u32,
    counter_reg: u32,  // SQ_PERFCOUNTER0_SELECT, etc.
    event_id: u32,     // e.g., 0x01 for SQ_WAVES
) -> Vec<u32> {
    let mut pm4 = Vec::new();

    // Step 1: Target specific CU via GRBM_GFX_INDEX
    // PM4_SET_UCONFIG_REG (opcode 0x79)
    let gfx_index_val = cu_index & 0xFF; // INSTANCE_INDEX
    pm4.push(PM4_HEADER_SET_UCONFIG_REG(2)); // 2 DWORDs: reg + value
    pm4.push(GRBM_GFX_INDEX);
    pm4.push(gfx_index_val);

    // Step 2: Write counter select register
    // PM4_SET_SH_REG (opcode 0x76)
    let sh_reg_offset = (counter_reg - SH_REG_BASE) >> 2;
    pm4.push(PM4_HEADER_SET_SH_REG(2));
    pm4.push(sh_reg_offset);
    pm4.push(event_id & 0xFF);

    pm4
}

/// Enable compute shader counter collection
fn build_enable_cs_counters_pm4() -> Vec<u32> {
    let mut pm4 = Vec::new();
    let sh_reg_offset = (SQ_PERFCOUNTER_CTRL - SH_REG_BASE) >> 2;
    // Bit 6 = CS enable, Bit 8 = CNTR_MODE (snapshot = 0)
    let ctrl_val = 1u32 << 6;
    pm4.push(PM4_HEADER_SET_SH_REG(2));
    pm4.push(sh_reg_offset);
    pm4.push(ctrl_val);
    pm4
}
```

### 3.5 Reading Counter Values

There are two approaches:

#### Approach A: In-Kernel Read via `s_getreg_b32` (Recommended for KFD-only)

Insert counter reads directly into the kernel ISA:

```asm
; Before kernel body
s_getreg_b32 s10, hwreg(HW_REG_SHADER_CYCLES)    ; start cycle count
; ... kernel body ...
s_getreg_b32 s11, hwreg(HW_REG_SHADER_CYCLES)    ; end cycle count
; Store s10, s11 to output buffer for host readback
```

This is what t0-gpu already does with `Op::ReadShaderCycles`. The limitation: you can only read **shader cycles** via `s_getreg`, not arbitrary perf counters.

To read SQ_PERFCOUNTER values from within a kernel, you'd need:

```asm
; GFX1100 supports s_memrealtime and s_getreg for some HW regs
; but SQ_PERFCOUNTER*_LO/HI are NOT accessible via s_getreg
; They require MMIO read or PM4 RELEASE_MEM
```

#### Approach B: PM4 RELEASE_MEM (Counter → VRAM Transfer)

After kernel execution, use a PM4 RELEASE_MEM packet to write counter values to a host-visible buffer:

```rust
/// Build PM4 to snapshot SQ_PERFCOUNTER0_LO to a GPU buffer
///
/// RELEASE_MEM event with DST_SEL = memory, DATA_SEL = perfcounter
fn build_counter_readback_pm4(
    counter_lo_reg: u32,   // SQ_PERFCOUNTER0_LO
    event_type: u32,       // CACHE_FLUSH_AND_INV_TS_EVENT (0x14)
    dst_addr: u64,         // GPU virtual address for result
) -> Vec<u32> {
    // PM4_RELEASE_MEM (opcode 0x49)
    // Format:
    //   DW0: header (opcode, count)
    //   DW1: event_type | event_index
    //   DW2: address_lo
    //   DW3: address_hi
    //   DW4: data (unused for perfcounter readback)
    //   DW5: control: INT_SEL, DATA_SEL, DST_SEL
    //
    // DATA_SEL values:
    //   0 = none
    //   1 = send perfcounter0_lo
    //   2 = send perfcounter1_lo
    //   3 = send immediate 32-bit
    //   4 = send perfcounter0 (64-bit)
    //   5 = send perfcounter1 (64-bit)
    //
    // DST_SEL values:
    //   0 = MC (memory controller / VRAM)
    //   1 = TC (L2)
    //
    // INT_SEL values:
    //   0 = none
    //   1 = send interrupt only
    //   2 = write data only (no interrupt)
    //   3 = write data and send interrupt

    let mut pm4 = Vec::new();
    pm4.push(PM4_HEADER_RELEASE_MEM(6));
    pm4.push(event_type << 0 | 0x04 << 8); // event_type, EVENT_INDEX=4 (CS)
    pm4.push((dst_addr & 0xFFFFFFFF) as u32);
    pm4.push((dst_addr >> 32) as u32);
    pm4.push(0); // data (unused)
    pm4.push(
        0 << 0   // INT_SEL = none
      | 1 << 29  // DATA_SEL = perfcounter0_lo
      | 0 << 24  // DST_SEL = MC
    );
    pm4
}
```

#### Approach C: Compute Shader Self-Instrumentation

Write a "profiler wrapper kernel" that:
1. Resets perf counters
2. Dispatches the target kernel
3. Reads perf counters

This requires PM4 queue capability (not AQL) and precise synchronization.

### 3.6 Complete Profiling Dispatch Sequence

```
1. [PM4] SET_UCONFIG_REG → GRBM_GFX_INDEX (target CU 0)
2. [PM4] SET_SH_REG → SQ_PERFCOUNTER0_SELECT = SQ_WAVES (0x01)
3. [PM4] SET_SH_REG → SQ_PERFCOUNTER1_SELECT = SQ_INSTS_VALU (0x0A)
4. [PM4] SET_SH_REG → SQ_PERFCOUNTER2_SELECT = SQ_INSTS_VMEM (0x12)
5. [PM4] SET_SH_REG → SQ_PERFCOUNTER3_SELECT = SQ_INSTS_MFMA (0x17)
6. [PM4] SET_SH_REG → SQ_PERFCOUNTER_CTRL = 0x40 (CS enable)
7. [PM4] ACQUIRE_MEM → invalidate L2 for clean counter start
8. [PM4] DISPATCH_DIRECT → your kernel (or AQL dispatch)
9. [PM4] EVENT_WRITE → CS_PARTIAL_FLUSH (wait for kernel completion)
10. [PM4] RELEASE_MEM → write SQ_PERFCOUNTER0_LO to buffer[0]
11. [PM4] RELEASE_MEM → write SQ_PERFCOUNTER1_LO to buffer[1]
12. [PM4] RELEASE_MEM → write SQ_PERFCOUNTER2_LO to buffer[2]
13. [PM4] RELEASE_MEM → write SQ_PERFCOUNTER3_LO to buffer[3]
14. [PM4] RELEASE_MEM → send interrupt (completion signal)
15. [Host] Poll buffer[0..4] → get counter values
```

---

## 4. NCU Metric → KFD Counter Mapping

### 4.1 Metrics You Can Collect

| NCU Metric | GFX1100 Counter(s) | How to Compute |
|------------|--------------------|----|
| `sm__warps_active.avg` | SQ_WAVES (sampled) | Average waves in flight |
| `sm__insts.sum` | SQ_INSTS | Total instructions executed |
| `sm__inst_executed_pipe_fma.sum` | SQ_INSTS_VALU | VALU instruction count |
| `sm__inst_executed_pipe_lsu.sum` | SQ_INSTS_VMEM + SQ_INSTS_FLAT | Load/Store unit instructions |
| `sm__inst_executed_pipe_xu.sum` | SQ_INSTS_VMEM + SQ_INSTS_LDS | Memory instructions |
| `sm__pipe_tensor_op_hmma_cycles_active.sum` | SQ_INSTS_MFMA | WMMA/MFMA instructions |
| `sm__cycles_active.sum` | SQ_THREAD_CYCLES | Active shader cycles |
| `sm__warps_launched.sum` | SQ_WAVES (cumulative) | Total waves dispatched |
| `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` | TCP read sectors | L1 cache read sectors |
| `l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum` | TCP write sectors | L1 cache write sectors |
| `lts__t_sectors_srcunit_tex_op_read.sum` | TCC_MC_READREQ * sector_size | L2 read traffic |
| `lts__t_sectors_srcunit_tex_op_write.sum` | TCC_MC_WRITEREQ * sector_size | L2 write traffic |
| `lts__t_sector_hit_rate.pct` | TCC_HIT / (TCC_HIT + TCC_MISS) | L2 hit rate |
| `dram__bytes.sum` | TCC_MC_READREQ/WRITEREQ * 64 | Total DRAM traffic |
| `dram__throughput.avg.pct_of_peak` | dram_bytes / (elapsed * 960 GB/s) | Memory bandwidth utilization |
| `sm__throughput.avg.pct_of_peak` | active_cycles / total_cycles | Compute utilization |

### 4.2 Metrics You Cannot Collect (Without SQ_THREAD_TRACE)

| NCU Metric | Why Not | Workaround |
|------------|---------|------------|
| Per-instruction stall reason | Requires SQ_THREAD_TRACE | Use waitcnt analysis in ISA |
| VGPR bank conflict | Requires SQ_THREAD_TRACE | Static analysis from ISA |
| LDS bank conflict | Requires SQ_THREAD_TRACE | Static analysis from ISA |
| Per-wavefront PC sampling | Requires SQ_THREAD_TRACE | Instrument kernel with cycle reads |
| Memory coalescing efficiency | Derivable from TCP + TCC | Compare sectors vs requests |
| Warp divergence | Requires EXEC mask tracking | Static ISA analysis |

### 4.3 Derived Metrics (Computed from Raw Counters)

```rust
/// Compute metrics from collected counter values
struct ProfileMetrics {
    // Core throughput
    instructions_per_cycle: f64,        // SQ_INSTS / SQ_THREAD_CYCLES
    valu_ipc: f64,                      // SQ_INSTS_VALU / SQ_THREAD_CYCLES
    wmma_utilization: f64,              // SQ_INSTS_MFMA / SQ_INSTS (of total mix)

    // Memory
    l1_hit_rate: f64,                   // TCP_HIT / TCP_REQ
    l2_hit_rate: f64,                   // TCC_HIT / (TCC_HIT + TCC_MISS)
    memory_bandwidth_utilization: f64,  // dram_bytes / (elapsed_ns * 960)
    arithmetic_intensity_actual: f64,   // total_flops / dram_bytes

    // Occupancy
    achieved_occupancy: f64,            // SQ_WAVES / max_waves_per_cu
    cu_busy_pct: f64,                   // GRBM_SE_BUSY / GRBM_COUNT

    // Bottleneck
    bottleneck: Bottleneck,             // compute / memory / latency
}

fn classify_bottleneck(metrics: &ProfileMetrics) -> Bottleneck {
    let roofline_ratio = metrics.arithmetic_intensity_actual
        * 960.0  // GB/s peak bandwidth
        / 123_000.0; // GFLOPS peak

    if roofline_ratio < 0.8 {
        Bottleneck::Memory
    } else if metrics.achieved_occupancy < 0.5 {
        Bottleneck::Latency  // not enough waves to hide latency
    } else {
        Bottleneck::Compute
    }
}
```

---

## 5. NCU Optimization Suggestions → Manual Equivalents

NCU provides automated optimization suggestions. Here's how to derive them manually:

### 5.1 "Memory Bound" → Increase Arithmetic Intensity

**NCU says:** "This kernel is memory bandwidth bound. Consider increasing data reuse."

**Your analysis:**
```
actual_AI = total_flops / total_dram_bytes
peak_AI   = 123_000 GFLOPS / 960 GB/s ≈ 128 FLOP/byte

if actual_AI < peak_AI:
    kernel is memory bound
    → increase tile size (more data reuse in LDS/L2)
    → use WMMA to increase compute density
    → use vectorized loads (global_load_dwordx4)
```

### 5.2 "Compute Bound" → Optimize Instruction Mix

**NCU says:** "This kernel is compute bound. Consider reducing instruction count."

**Your analysis:**
```
valu_ratio  = SQ_INSTS_VALU / SQ_INSTS
wmma_ratio  = SQ_INSTS_MFMA / SQ_INSTS
ctrl_ratio  = SQ_INSTS_CTRL / SQ_INSTS

if ctrl_ratio > 0.1:
    → too much control overhead, reduce branch divergence
    → unroll loops, merge basic blocks

if wmma_ratio < 0.3 and kernel is GEMM:
    → not using WMMA enough, restructure to use v_wmma

if valu_ratio > 0.7 and wmma_ratio < 0.1:
    → scalar FMA, should use WMMA for 16x throughput
```

### 5.3 "Low Occupancy" → Reduce Resource Pressure

**NCU says:** "Occupancy is limited by VGPR usage."

**Your analysis:**
```
// From cost_model.rs pattern:
waves_per_simd = min(256 / vgprs_per_wave, 8)

// If vgprs_per_wave = 128 → waves_per_simd = 2 → occupancy = 25%
// NCU would say: "Reduce VGPR usage to increase occupancy"

// Solutions:
// 1. Recompute values instead of spilling (trade compute for registers)
// 2. Use smaller tile sizes (fewer accumulators needed)
// 3. Use 16-bit types (bf16) to halve VGPR usage
// 4. Split-K to reduce accumulator registers
```

### 5.4 "L2 Cache Thrashing" → Improve Data Locality

**NCU says:** "L2 hit rate is low. Consider improving data locality."

**Your analysis:**
```
l2_hit_rate = TCC_HIT / (TCC_HIT + TCC_MISS)

if l2_hit_rate < 0.3:
    → tile sizes too small (data doesn't stay in L2)
    → increase tile_m, tile_n to keep working set in L2
    → L2 = 6 MB, so tile that fits: 6M / (2 * 2 bytes) = 1.5M elements
    → For bf16 GEMM: 128×128 tile needs 128*128*2 + 128*128*2 = 64KB (fits easily)
    → Problem is usually reuse pattern, not size
```

### 5.5 "Stall on Memory" → Software Pipeline

**NCU says:** "Wavefronts are stalling on memory operations."

**Your analysis:**
```
// From latency_model.rs:
// VMEM load latency ≈ 500 shader cycles
// VALU simple latency ≈ 10 shader cycles
// WMMA latency ≈ 36 shader cycles
// VALU slots per VMEM ≈ 490 (how many VALU can overlap with one VMEM)

// If your K-loop does:
//   load A tile  → wait → compute → store result
// The load latency is fully exposed.

// Solution: software pipelining (double-buffer LDS)
//   Iteration N: load A[N+1] while computing A[N]
//   Hides VMEM latency behind compute

// From cost_model.rs, the auto-scheduler already considers this:
// LDS double-buffer adds 2x LDS memory but hides VMEM latency
```

---

## 6. Building an NCU-Like Profiler for KFD-Only Runtime

### 6.1 Architecture

```
┌─────────────────────────────────────────────────────┐
│                    ProfileSession                     │
│                                                       │
│  1. Configure counters (PM4 SET_SH_REG)              │
│  2. Dispatch kernel (AQL or PM4)                     │
│  3. Snapshot counters (PM4 RELEASE_MEM)              │
│  4. Read results (GTT buffer poll)                   │
│  5. Compute derived metrics                          │
│  6. Apply optimization heuristics                    │
│  7. Generate report                                  │
└─────────────────────────────────────────────────────┘
        │                              │
        ▼                              ▼
┌───────────────┐            ┌───────────────────┐
│  PM4 Queue    │            │  GTT Buffer       │
│  (counter     │            │  (counter values  │
│   program +   │            │   written by GPU) │
│   readback)   │            │                   │
└───────────────┘            └───────────────────┘
```

### 6.2 Implementation Skeleton

```rust
/// NCU-like profiler for bare-metal GFX1100
pub struct GfxProfiler {
    pm4_queue: Pm4Queue,
    result_buf: GpuBuffer<u64>,   // GTT buffer for counter readback
    results: Vec<ProfileResult>,
}

/// A single counter measurement
#[derive(Clone, Debug)]
pub struct CounterValue {
    pub name: String,
    pub cu_index: u32,
    pub raw_value: u64,
}

/// Full profile result for one kernel dispatch
#[derive(Clone, Debug)]
pub struct ProfileResult {
    pub kernel_name: String,
    pub elapsed_ns: u64,
    pub counters: HashMap<String, Vec<CounterValue>>,  // event → per-CU values
    pub metrics: ProfileMetrics,
    pub suggestions: Vec<OptimizationSuggestion>,
}

#[derive(Clone, Debug)]
pub struct OptimizationSuggestion {
    pub severity: Severity,  // Critical, Warning, Info
    pub category: String,    // "Occupancy", "Memory", "Compute", "Latency"
    pub metric: String,
    pub value: f64,
    pub threshold: f64,
    pub suggestion: String,
}

impl GfxProfiler {
    /// Profile a kernel with a standard set of counters
    pub fn profile_kernel<F>(
        &mut self,
        name: &str,
        kernel_fn: F,
        m: usize, n: usize, k: usize,
    ) -> ProfileResult
    where
        F: FnOnce(&AqlQueue) + Send,
    {
        // Step 1: Configure counters on CU 0 (or representative CU)
        let counters = vec![
            (SQ_PERFCOUNTER0_SELECT, 0x01, "SQ_WAVES"),
            (SQ_PERFCOUNTER1_SELECT, 0x0A, "SQ_INSTS_VALU"),
            (SQ_PERFCOUNTER2_SELECT, 0x12, "SQ_INSTS_VMEM"),
            (SQ_PERFCOUNTER3_SELECT, 0x17, "SQ_INSTS_MFMA"),
            (SQ_PERFCOUNTER4_SELECT, 0x07, "SQ_INSTS"),
            (SQ_PERFCOUNTER5_SELECT, 0x1A, "SQ_THREAD_CYCLES"),
        ];

        for (reg, event, _name) in &counters {
            self.configure_counter(0, *reg, *event);
        }
        self.enable_cs_counters();

        // Step 2: Dispatch kernel with timing
        let start = Instant::now();
        kernel_fn(&self.aql_queue);
        self.aql_queue.synchronize();
        let elapsed_ns = start.elapsed().as_nanos() as u64;

        // Step 3: Read back counters
        for (i, (reg, _event, name)) in counters.iter().enumerate() {
            self.snapshot_counter(0, *reg, i);
        }
        self.pm4_queue.submit_and_wait();

        // Step 4: Parse results
        let mut counter_values = HashMap::new();
        for (i, (_reg, _event, name)) in counters.iter().enumerate() {
            let raw = self.result_buf[i];
            counter_values.insert(name.to_string(), vec![CounterValue {
                name: name.to_string(),
                cu_index: 0,
                raw_value: raw,
            }]);
        }

        // Step 5: Compute metrics
        let metrics = self.compute_metrics(&counter_values, elapsed_ns, m, n, k);

        // Step 6: Generate suggestions
        let suggestions = self.generate_suggestions(&metrics);

        ProfileResult {
            kernel_name: name.to_string(),
            elapsed_ns,
            counters: counter_values,
            metrics,
            suggestions,
        }
    }

    fn generate_suggestions(&self, m: &ProfileMetrics) -> Vec<OptimizationSuggestion> {
        let mut suggestions = Vec::new();

        // Occupancy check
        if m.achieved_occupancy < 0.5 {
            suggestions.push(OptimizationSuggestion {
                severity: Severity::Warning,
                category: "Occupancy".into(),
                metric: "achieved_occupancy".into(),
                value: m.achieved_occupancy,
                threshold: 0.5,
                suggestion: format!(
                    "Occupancy is {:.0}%. Consider: (1) reduce VGPR usage, \
                     (2) use smaller tile sizes, (3) use bf16 to halve register pressure.",
                    m.achieved_occupancy * 100.0
                ),
            });
        }

        // Memory bandwidth check
        if m.memory_bandwidth_utilization > 0.8 && m.bottleneck == Bottleneck::Memory {
            suggestions.push(OptimizationSuggestion {
                severity: Severity::Info,
                category: "Memory".into(),
                metric: "memory_bandwidth_utilization".into(),
                value: m.memory_bandwidth_utilization,
                threshold: 0.8,
                suggestion: "Memory bandwidth is near peak. Kernel is well-optimized \
                            for its arithmetic intensity. To improve further, \
                            increase data reuse (larger tiles, WMMA).".into(),
            });
        }

        // L2 hit rate check
        if m.l2_hit_rate < 0.3 {
            suggestions.push(OptimizationSuggestion {
                severity: Severity::Warning,
                category: "Cache".into(),
                metric: "l2_hit_rate".into(),
                value: m.l2_hit_rate,
                threshold: 0.3,
                suggestion: format!(
                    "L2 hit rate is {:.0}%. Consider: (1) increase tile size for \
                     better data reuse, (2) adjust access pattern for spatial locality, \
                     (3) use LDS for within-tile reuse.",
                    m.l2_hit_rate * 100.0
                ),
            });
        }

        // WMMA utilization check (for GEMM kernels)
        if m.wmma_utilization < 0.5 {
            suggestions.push(OptimizationSuggestion {
                severity: Severity::Warning,
                category: "Compute".into(),
                metric: "wmma_utilization".into(),
                value: m.wmma_utilization,
                threshold: 0.5,
                suggestion: format!(
                    "WMMA/MFMA instructions are only {:.0}% of total. \
                     Restructure K-loop to maximize v_wmma_f32_16x16x16 usage. \
                     Current mix suggests scalar FMA which is 16x slower.",
                    m.wmma_utilization * 100.0
                ),
            });
        }

        // IPC check
        if m.instructions_per_cycle < 1.0 {
            suggestions.push(OptimizationSuggestion {
                severity: Severity::Info,
                category: "Latency".into(),
                metric: "instructions_per_cycle".into(),
                value: m.instructions_per_cycle,
                threshold: 1.0,
                suggestion: format!(
                    "IPC is {:.2}. Consider: (1) software pipelining to hide VMEM latency \
                     (500 cycles), (2) increase ILP by interleaving independent instructions, \
                     (3) use double-buffered LDS loads.",
                    m.instructions_per_cycle
                ),
            });
        }

        suggestions
    }
}
```

### 6.3 Report Output Format (NCU-Style)

```
═══════════════════════════════════════════════════════════════
  GFX1100 Profile Report: bf16_gemm_4096x4096x4096
  Target: AMD RX 7900 XTX (GFX1100, 96 CU, Wave32)
═══════════════════════════════════════════════════════════════

  Duration:           2.341 ms
  Achieved TFLOPS:    58.2  (47.3% of 123 TFLOPS peak)

  ┌─────────────────────────────────────────────────────────┐
  │                    ROOFLINE ANALYSIS                     │
  │                                                         │
  │  TFLOPS ──────── 58.2 ──────────────────────            │
  │       │                              ╱ Peak: 123       │
  │       │                           ╱                     │
  │       │                        ╱  ← Compute bound      │
  │       │                     ╱                           │
  │       │                  ╱                              │
  │       │               ╱                                 │
  │       │            ╱                                    │
  │       │         ╱  ← Memory bound                      │
  │       │      ╱                                          │
  │       │   ╱  Bandwidth ceiling: 960 GB/s               │
  │       │╱                                                │
  │       └──────────────────────────────────────── AI      │
  │       0.1       1        10       100     1000          │
  │                                                         │
  │  Actual AI: 17.3 FLOP/byte  →  Compute bound           │
  └─────────────────────────────────────────────────────────┘

  ┌─────────────────────┬──────────┬────────────────────────┐
  │ Metric              │ Value    │ Status                 │
  ├─────────────────────┼──────────┼────────────────────────┤
  │ Achieved Occupancy  │ 75.0%    │ ██████████████░░░░░░   │
  │ L2 Hit Rate         │ 82.3%    │ ████████████████░░░░   │
  │ Mem BW Utilization  │ 61.2%    │ ████████████░░░░░░░░   │
  │ WMMA Utilization    │ 89.1%    │ ██████████████████░░   │
  │ IPC                 │ 3.42     │ ████████████████░░░░   │
  │ VALU / VMEM ratio   │ 490:1    │ (matches expected)     │
  └─────────────────────┴──────────┴────────────────────────┘

  ┌─────────────────────────────────────────────────────────┐
  │              OPTIMIZATION SUGGESTIONS                    │
  ├──────┬──────────────────────────────────────────────────┤
  │ INFO │ Memory BW at 61%. Room to increase tile size     │
  │      │ for better L2 reuse. Current AI=17.3 vs          │
  │      │ roofline peak AI=128.                            │
  ├──────┼──────────────────────────────────────────────────┤
  │ WARN │ Occupancy 75% limited by VGPR (96/wave).        │
  │      │ Consider: use bf16 accumulators, reduce tile_k.  │
  └──────┴──────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────┐
  │                 INSTRUCTION MIX                         │
  │                                                         │
  │  WMMA    ████████████████████████████████████  89.1%    │
  │  VALU    ████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   5.2%    │
  │  VMEM    ██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   3.1%    │
  │  LDS     █░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   1.8%    │
  │  CTRL    ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   0.8%    │
  └─────────────────────────────────────────────────────────┘
```

---

## 7. What You Can Build vs. What NCU Does

### 7.1 Achievable in KFD-Only Mode

| Feature | NCU | KFD Profiler | Difficulty |
|---------|-----|-------------|------------|
| Hardware counter collection | Yes | **Yes** (PM4 + RELEASE_MEM) | Medium |
| Per-dispatch timing | Yes | **Yes** (wall-clock + HW_REG_SHADER_CYCLES) | Easy |
| Roofline analysis | Yes | **Yes** (counters + known peak) | Easy |
| Instruction mix | Yes | **Yes** (SQ_INSTS_* counters) | Medium |
| Occupancy measurement | Yes | **Yes** (SQ_WAVES / max_waves) | Easy |
| Cache hit rates | Yes | **Yes** (TCC_HIT/MISS) | Medium |
| Bandwidth utilization | Yes | **Yes** (TCC_MC_READ/WRITE) | Easy |
| Automated suggestions | Yes | **Yes** (heuristic rules) | Medium |
| Multi-CU profiling | Yes | **Yes** (GRBM_GFX_INDEX instancing) | Hard |
| Counter heatmap (per-CU) | Yes | **Yes** (loop over 96 CUs) | Hard |
| Register pressure analysis | Yes | **Static** (from ISA, not runtime) | Easy |
| Stall reason breakdown | Yes | **No** (needs SQ_THREAD_TRACE) | — |
| Per-instruction latency | Yes | **No** (needs SQ_THREAD_TRACE) | — |
| Warp divergence analysis | Yes | **No** (needs SQ_THREAD_TRACE) | — |
| PC sampling | Yes | **No** (needs SQ_THREAD_TRACE) | — |

### 7.2 SQ_THREAD_TRACE: The Missing Piece

SQ_THREAD_TRACE is the hardware unit that enables the deepest profiling features (stall reasons, PC sampling, warp divergence). It's accessible through:

1. **PM4 EVENT_WRITE** with `THREAD_TRACE` event type
2. **SQ_THREAD_TRACE_CTRL** register (start/stop trace)
3. **SQ_THREAD_TRACE_STATUS** register (check trace buffer status)
4. **SQ_THREAD_TRACE_BASE/HI** register (trace buffer address)
5. **SQ_THREAD_TRACE_SIZE** register (trace buffer size)

The thread trace produces a stream of packets to a GPU buffer containing:
- Instruction pointer samples
- Stall reason codes
- Register values
- Memory addresses
- Wave lifecycle events (create, destroy, preemption)

**Status in t0-gpu:** Not implemented. This would be the next major profiling capability to add.

```rust
// SQ_THREAD_TRACE registers (GFX1100)
const SQ_THREAD_TRACE_CTRL: u32 = 0xD060;     // Start/stop/configure trace
const SQ_THREAD_TRACE_BASE: u32 = 0xD064;     // Trace buffer base addr (LO)
const SQ_THREAD_TRACE_BASE_HI: u32 = 0xD068;  // Trace buffer base addr (HI)
const SQ_THREAD_TRACE_SIZE: u32 = 0xD06C;     // Trace buffer size
const SQ_THREAD_TRACE_STATUS: u32 = 0xD070;   // Trace status (busy, done, etc.)
const SQ_THREAD_TRACE_MASK: u32 = 0xD074;     // Which CUs/SIMDs to trace
const SQ_THREAD_TRACE_TOKEN_MASK: u32 = 0xD078; // Which event types to capture

// Trace token types (what events are recorded)
// Bit 0: TIME (timestamp)
// Bit 1: REG (register read/write)
// Bit 2: WAVE_START (wave creation)
// Bit 3: WAVE_END (wave destruction)
// Bit 4: INSTR (instruction execution)
// Bit 5: INSTR_STALL (instruction stall with reason)
// Bit 6: MEM (memory access)
// Bit 7: REG_WRITE (register write specifically)
```

Implementing SQ_THREAD_TRACE support would unlock stall reason analysis, making the KFD profiler ~90% equivalent to NCU.

---

## 8. Existing t0-gpu Profiling Infrastructure

t0-gpu already has significant profiling capabilities that can be extended:

### 8.1 What Already Exists

| Component | File | Capability |
|-----------|------|------------|
| `Op::ReadShaderCycles` | `t0/ir.rs:471` | GPU cycle counter in kernels |
| `hw_probe` | `.archive/hw_probe.rs` | Instruction latency microbenchmarks |
| `latency_model` | `t0/latency_model.rs` | Empirically-calibrated latency table |
| `insn_latency` | `t0/insn_latency.rs` | Critical path + ILP analysis |
| `cost_model` | `t0/cost_model.rs` | Roofline model + auto-scheduling |
| `profile_guided` | `.archive/profile_guided.rs` | Workgroup size autotuner |
| Wall-clock timing | Throughout | `Instant::now()` / `.elapsed()` |
| TFLOPS calculation | Throughout | `2*M*N*K / (elapsed * 1e6)` |

### 8.2 Extension Roadmap

```
Phase 1: Basic Hardware Counters (2-3 weeks)
├── Implement PM4 counter configuration (SQ_PERFCOUNTER*_SELECT)
├── Implement PM4 RELEASE_MEM counter readback
├── ProfileResult struct with counter values
└── Basic report: IPC, occupancy, cache hit rate

Phase 2: NCU-Style Report (1-2 weeks)
├── Roofline plot generation
├── Instruction mix visualization
├── Automated optimization suggestions
└── Multi-CU heatmap (sample 8 of 96 CUs)

Phase 3: SQ_THREAD_TRACE (4-6 weeks)
├── Implement trace buffer allocation
├── Configure SQ_THREAD_TRACE_CTRL
├── Parse trace token stream
├── Extract stall reasons per instruction
└── Per-instruction latency breakdown
```

---

## 9. Practical Workarounds Without Full Profiler

If you need profiling **now** without implementing the full counter infrastructure:

### 9.1 Kernel Self-Instrumentation

```rust
// In your T0 IR, insert cycle reads around sections of interest:
let start_cycle = builder.read_shader_cycles();
// ... section to measure ...
let end_cycle = builder.read_shader_cycles();
let elapsed = builder.sub(end_cycle, start_cycle);
// Store elapsed to output buffer
builder.store_to_buffer(elapsed, output_slot);
```

This is coarse but tells you **where** time is spent in a kernel.

### 9.2 Wall-Clock A/B Testing

```rust
// Compare two kernel variants:
let start = Instant::now();
for _ in 0..100 { queue.submit(&kernel_a); queue.synchronize(); }
let time_a = start.elapsed();

let start = Instant::now();
for _ in 0..100 { queue.submit(&kernel_b); queue.synchronize(); }
let time_b = start.elapsed();

println!("A: {} us, B: {} us, speedup: {:.2}x",
    time_a.as_micros() / 100,
    time_b.as_micros() / 100,
    time_a.as_secs_f64() / time_b.as_secs_f64());
```

### 9.3 Roofline from First Principles

```rust
// You know the hardware:
//   Peak compute: 123 TFLOPS (BF16 WMMA)
//   Peak bandwidth: 960 GB/s
//   L2 size: 6 MB

// Measure actual TFLOPS from wall-clock time:
let tflops = 2.0 * m as f64 * n as f64 * k as f64 / (elapsed_ns as f64);

// Measure actual bandwidth (if you know bytes transferred):
let bandwidth_gb_s = bytes_transferred as f64 / elapsed_ns as f64;

// Compute arithmetic intensity:
let ai = (2.0 * m * n * k) as f64 / bytes_transferred as f64;

// Roofline:
//   If AI < 128: memory bound → optimize bandwidth
//   If AI > 128: compute bound → optimize instruction mix
```

### 9.4 Static ISA Analysis (No GPU Execution Needed)

From the generated ISA, you can compute:

```rust
// Already implemented in insn_latency.rs:
let analysis = analyze_block(&kloop_ops);

// Available metrics:
//   - VALU count, VTRANS count, WMMA count, VMEM count
//   - Critical path cycles (DAG-based)
//   - ILP potential
//   - Pipeline breakdown (WMMA-bound vs VALU-bound vs VMEM-bound)
//   - VGPR/SGPR/LDS usage
//   - Predicted occupancy
```

This is **zero-overhead profiling** — no GPU execution needed, pure static analysis of the generated ISA.

---

## 10. Summary

| Approach | Effort | NCU Coverage | Latency |
|----------|--------|-------------|---------|
| Wall-clock TFLOPS | 0 | 10% | 0 |
| Static ISA analysis | 0 (exists) | 30% | 0 |
| Kernel self-instrumentation (cycles) | Low | 20% | 0 |
| PM4 perf counter readback | Medium | 60% | 2-3 weeks |
| Full counter profiler + report | Medium | 75% | 4-6 weeks |
| + SQ_THREAD_TRACE | High | 90% | 8-12 weeks |

**The bottom line:** In a KFD-only environment on GFX1100, you can build a profiler that covers ~75% of NCU's capabilities using PM4 hardware counter readback. The remaining 25% (stall reasons, per-instruction latency) requires SQ_THREAD_TRACE, which is feasible but significantly more complex. For GEMM/kernel optimization specifically, the 75% coverage (counters + roofline + instruction mix + static ISA analysis) is sufficient to identify and fix most bottlenecks.
