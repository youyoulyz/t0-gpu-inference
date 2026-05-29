//! Level 1: CPU-based TileSSA IR Interpreter.
//!
//! Executes TileSSA kernels on CPU without GPU hardware.
//! Single-thread simulation: one lane, one workgroup.
//!
//! # Use Cases
//! - Students can test kernel logic on their laptop without GPU
//! - Catches OOB access, algorithm errors, numerical issues
//! - Instant feedback (<1ms per execution)
//!
//! # Limitations
//! - Single lane (no wave/wavefront behavior)
//! - No LDS (workgroup reductions use scalar fallback)
//! - No WMMA (uses naive matmul)
//! - f32 only (no bf16 precision simulation)

use super::super::tile_ssa::*;
use std::collections::HashMap;

/// Simulated VRAM — flat f32 array with bounds checking.
pub struct SimMemory {
    /// Named buffers: name → (base_offset, len)
    buffers: HashMap<String, (usize, usize)>,
    /// Flat storage
    data: Vec<f32>,
}

impl SimMemory {
    pub fn new() -> Self {
        Self { buffers: HashMap::new(), data: Vec::new() }
    }

    /// Allocate a named buffer, return its base offset.
    pub fn alloc(&mut self, name: &str, len: usize) -> usize {
        let base = self.data.len();
        self.data.resize(base + len, 0.0);
        self.buffers.insert(name.to_string(), (base, len));
        base
    }

    /// Write data into a named buffer.
    pub fn write(&mut self, name: &str, offset: usize, data: &[f32]) {
        let (base, _) = self.buffers.get(name)
            .unwrap_or_else(|| panic!("SimMemory: buffer '{}' not allocated", name));
        for (i, &v) in data.iter().enumerate() {
            self.data[base + offset + i] = v;
        }
    }

    /// Read data from a named buffer.
    pub fn read(&self, name: &str, offset: usize, len: usize) -> Vec<f32> {
        let (base, buf_len) = self.buffers.get(name)
            .unwrap_or_else(|| panic!("SimMemory: buffer '{}' not allocated", name));
        assert!(offset + len <= *buf_len,
            "SimMemory: OOB read from '{}': offset={} + len={} > buf_len={}",
            name, offset, len, buf_len);
        self.data[base + offset..base + offset + len].to_vec()
    }

    /// Get base offset of a named buffer (for pointer simulation).
    pub fn base(&self, name: &str) -> usize {
        self.buffers.get(name)
            .unwrap_or_else(|| panic!("SimMemory: buffer '{}' not allocated", name))
            .0
    }

    /// Read a single f32 at an absolute index.
    pub fn read_abs(&self, idx: usize) -> f32 {
        if idx >= self.data.len() {
            panic!("SimMemory: OOB read at absolute index {} (len={})", idx, self.data.len());
        }
        self.data[idx]
    }

    /// Write a single f32 at an absolute index.
    pub fn write_abs(&mut self, idx: usize, val: f32) {
        if idx >= self.data.len() {
            panic!("SimMemory: OOB write at absolute index {} (len={})", idx, self.data.len());
        }
        self.data[idx] = val;
    }

    /// Total size in f32 elements.
    pub fn len(&self) -> usize { self.data.len() }
}

/// Interpreter value — either a scalar or a vector.
#[derive(Clone, Debug)]
pub enum InterpValue {
    /// Scalar f32
    F32(f32),
    /// Scalar u32 (stored as f32 for simplicity, but typed)
    U32(u32),
    /// Scalar i32
    I32(i32),
    /// Vector of f32 (simulates per-lane values)
    VecF32(Vec<f32>),
    /// Vector of u32
    VecU32(Vec<u32>),
    /// Pointer (buffer name + byte offset)
    Ptr(String, usize),
    /// Boolean vector
    VecBool(Vec<bool>),
    /// Boolean scalar
    Bool(bool),
}

impl InterpValue {
    fn as_f32(&self) -> f32 {
        match self {
            InterpValue::F32(v) => *v,
            InterpValue::U32(v) => *v as f32,
            InterpValue::I32(v) => *v as f32,
            _ => panic!("InterpValue::as_f32 on {:?}", self),
        }
    }

    fn as_u32(&self) -> u32 {
        match self {
            InterpValue::U32(v) => *v,
            InterpValue::F32(v) => *v as u32,
            InterpValue::I32(v) => *v as u32,
            _ => panic!("InterpValue::as_u32 on {:?}", self),
        }
    }

    fn as_bool(&self) -> bool {
        match self {
            InterpValue::Bool(v) => *v,
            InterpValue::U32(v) => *v != 0,
            _ => panic!("InterpValue::as_bool on {:?}", self),
        }
    }

    fn as_vec_f32(&self) -> &[f32] {
        match self {
            InterpValue::VecF32(v) => v,
            _ => panic!("InterpValue::as_vec_f32 on {:?}", self),
        }
    }

    fn as_vec_bool(&self) -> &[bool] {
        match self {
            InterpValue::VecBool(v) => v,
            _ => panic!("InterpValue::as_vec_bool on {:?}", self),
        }
    }

    fn is_scalar(&self) -> bool {
        matches!(self, InterpValue::F32(_) | InterpValue::U32(_) | InterpValue::I32(_) | InterpValue::Bool(_))
    }
}

/// Interpreter configuration.
pub struct InterpretConfig {
    /// Program ID (workgroup ID) — simulates which row this WG processes.
    pub program_id: u32,
    /// Thread ID — simulates which lane this is (0 for single-lane).
    pub thread_id: u32,
    /// Workgroup size (for reduction simulation).
    pub wg_size: u32,
    /// Maximum loop iterations (safety limit).
    pub max_loop_iters: u32,
    /// Pre-computed WgReduce results (value_id → result).
    /// When set, WgReduce operations return this value instead of the identity.
    /// Useful for single-thread simulation where cross-lane reduction is impossible.
    pub reduce_overrides: HashMap<u32, InterpValue>,
}

impl Default for InterpretConfig {
    fn default() -> Self {
        Self {
            program_id: 0,
            thread_id: 0,
            wg_size: 256,
            max_loop_iters: 100_000,
            reduce_overrides: HashMap::new(),
        }
    }
}

/// Execution error with context.
#[derive(Debug)]
pub struct InterpretError {
    pub message: String,
    pub op_index: usize,
    pub block_id: u32,
}

impl std::fmt::Display for InterpretError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "InterpretError at op[{}] in block {}: {}",
            self.op_index, self.block_id, self.message)
    }
}

/// Execute a TileSSA function on CPU.
///
/// Returns the output buffer contents after execution.
/// Panics on OOB access with a descriptive message.
///
/// # Arguments
/// * `func` — The TileSSA function to execute
/// * `mem` — Simulated VRAM (will be modified in-place)
/// * `ptr_map` — Maps pointer arg names to buffer names in SimMemory
/// * `args` — Initial values for function args (non-pointer args only; pointer args use ptr_map)
/// * `config` — Interpreter configuration
///
/// # Example
/// ```ignore
/// let func = build_softmax_large();
/// let mut mem = SimMemory::new();
/// mem.alloc("input", 514);
/// mem.write("input", 0, &test_data);
/// mem.alloc("output", 514);
/// let ptr_map = vec![("input".to_string(), "input".to_string()),
///                    ("output".to_string(), "output".to_string())];
/// let args = vec![
///     InterpValue::U32(0),   // input_ptr placeholder
///     InterpValue::U32(0),   // output_ptr placeholder
///     InterpValue::U32(257), // cols
///     InterpValue::U32(1),   // n_chunks
/// ];
/// let config = InterpretConfig { program_id: 1, ..Default::default() };
/// interpret(&func, &mut mem, &ptr_map, &args, &config).unwrap();
/// let result = mem.read("output", 257, 257);
/// ```
pub fn interpret(
    func: &TileFunc,
    mem: &mut SimMemory,
    ptr_map: &[(String, String)],
    args: &[InterpValue],
    config: &InterpretConfig,
) -> Result<(), InterpretError> {
    let all_values = func.all_values();
    let blocks = func.all_blocks();
    let ops = func.all_ops();

    // Value register file — initialize with function args
    let mut vals: HashMap<u32, InterpValue> = HashMap::new();

    // Map pointer arg names to buffer names and initialize arg values
    let mut ptr_buffers: HashMap<u32, String> = HashMap::new();
    let mut ptr_idx = 0usize;
    for (i, arg_val) in func.args.iter().enumerate() {
        let vdef = &all_values[arg_val.0 as usize];
        if vdef.ty == TileType::Ptr {
            if let Some((_, buf_name)) = ptr_map.get(ptr_idx) {
                ptr_buffers.insert(arg_val.0, buf_name.clone());
            }
            ptr_idx += 1;
        }
        // Initialize arg value from args slice
        if let Some(arg_value) = args.get(i) {
            vals.insert(arg_val.0, arg_value.clone());
        }
    }

    // Start at block 0 (entry)
    let mut current_block = 0u32;
    let mut loop_count: u32 = 0;

    // Max iterations safety
    let max_iters = config.max_loop_iters * 100; // generous limit for total ops

    for _iter in 0..max_iters {
        let block = &blocks[current_block as usize];

        // Execute all ops in this block
        for &op_idx in &block.ops {
            let op = &ops[op_idx];
            execute_op(op, &mut vals, mem, &ptr_buffers, config, op_idx, current_block)?;
        }

        // Execute terminator
        match &block.terminator {
            None => {
                return Err(InterpretError {
                    message: "Block has no terminator".to_string(),
                    op_index: 0,
                    block_id: current_block,
                });
            }
            Some(Terminator::Return) => {
                return Ok(());
            }
            Some(Terminator::Branch { target, args }) => {
                // Bind block params
                let target_block = &blocks[target.0 as usize];
                for (param_val, arg_val) in target_block.params.iter().zip(args.iter()) {
                    let v = vals.get(&arg_val.0)
                        .unwrap_or_else(|| panic!("Branch arg {:?} not defined", arg_val))
                        .clone();
                    vals.insert(param_val.0, v);
                }
                current_block = target.0;
            }
            Some(Terminator::CondBranch { cond, true_bb, true_args, false_bb, false_args }) => {
                let cond_val = vals.get(&cond.0)
                    .unwrap_or_else(|| panic!("CondBranch cond {:?} not defined", cond));

                let (target, target_args) = if cond_val.as_bool() {
                    (true_bb, true_args)
                } else {
                    (false_bb, false_args)
                };

                let target_block = &blocks[target.0 as usize];
                for (param_val, arg_val) in target_block.params.iter().zip(target_args.iter()) {
                    let v = vals.get(&arg_val.0)
                        .unwrap_or_else(|| panic!("CondBranch arg {:?} not defined", arg_val))
                        .clone();
                    vals.insert(param_val.0, v);
                }
                current_block = target.0;
            }
        }

        loop_count += 1;
        if loop_count > max_iters {
            return Err(InterpretError {
                message: format!("Exceeded maximum iterations ({}) — possible infinite loop", max_iters),
                op_index: 0,
                block_id: current_block,
            });
        }
    }

    Err(InterpretError {
        message: "Exceeded total iteration limit".to_string(),
        op_index: 0,
        block_id: current_block,
    })
}

fn execute_op(
    op: &TileOp,
    vals: &mut HashMap<u32, InterpValue>,
    mem: &mut SimMemory,
    ptr_buffers: &HashMap<u32, String>,
    config: &InterpretConfig,
    op_index: usize,
    block_id: u32,
) -> Result<(), InterpretError> {
    let err = |msg: String| InterpretError { message: msg, op_index, block_id };

    match op {
        // ── Constants ──
        TileOp::ConstU32 { result, value } => {
            vals.insert(result.0, InterpValue::U32(*value));
        }
        TileOp::ConstF32 { result, value } => {
            vals.insert(result.0, InterpValue::F32(*value));
        }

        // ── Index ──
        TileOp::ProgramId { result, axis } => {
            if *axis == 0 {
                vals.insert(result.0, InterpValue::U32(config.program_id));
            } else {
                vals.insert(result.0, InterpValue::U32(0));
            }
        }
        TileOp::ThreadIdX { result } => {
            vals.insert(result.0, InterpValue::U32(config.thread_id));
        }
        TileOp::ThreadIdX2D { result, block_x } => {
            vals.insert(result.0, InterpValue::U32(config.thread_id % block_x));
        }
        TileOp::ThreadIdY2D { result, block_x } => {
            vals.insert(result.0, InterpValue::U32(config.thread_id / block_x));
        }
        TileOp::Arange { result, start, len } => {
            let v: Vec<f32> = (0..*len).map(|i| (start + i) as f32).collect();
            vals.insert(result.0, InterpValue::VecF32(v));
        }

        // ── Shape ──
        TileOp::Splat { result, src, shape } => {
            let scalar = vals.get(&src.0)
                .ok_or_else(|| err(format!("Splat src {:?} not defined", src)))?;
            let val = scalar.as_f32();
            let total: u32 = shape.iter().product();
            vals.insert(result.0, InterpValue::VecF32(vec![val; total as usize]));
        }

        // ── Memory ──
        TileOp::Load { result, ptr, indices, mask, other, dtype } => {
            let ptr_id = ptr.0;
            let buf_name = ptr_buffers.get(&ptr_id)
                .ok_or_else(|| err(format!("Load: ptr {:?} not in ptr_map", ptr)))?;
            let buf_base = mem.base(buf_name);

            let idx_val = vals.get(&indices.0)
                .ok_or_else(|| err(format!("Load: indices {:?} not defined", indices)))?;

            let mask_val = mask.as_ref().map(|m| {
                vals.get(&m.0).unwrap_or_else(|| panic!("Load mask {:?} not defined", m))
            });

            let other_val = other.as_ref().map(|o| {
                vals.get(&o.0).unwrap_or_else(|| panic!("Load other {:?} not defined", o))
            });

            match idx_val {
                InterpValue::VecF32(indices_vec) => {
                    // Vector load
                    let mut result_vec = Vec::with_capacity(indices_vec.len());
                    for (i, &idx) in indices_vec.iter().enumerate() {
                        let abs_idx = buf_base + idx as usize;
                        let masked = if let Some(ref m) = mask_val {
                            m.as_vec_bool().get(i).copied().unwrap_or(false)
                        } else {
                            true
                        };
                        if masked && abs_idx < mem.len() {
                            result_vec.push(mem.read_abs(abs_idx));
                        } else if let Some(ref o) = other_val {
                            result_vec.push(o.as_f32());
                        } else {
                            result_vec.push(0.0);
                        }
                    }
                    vals.insert(result.0, InterpValue::VecF32(result_vec));
                }
                InterpValue::U32(idx) => {
                    // Scalar load
                    let abs_idx = buf_base + *idx as usize;
                    let masked = mask_val.map(|m| m.as_bool()).unwrap_or(true);
                    if masked && abs_idx < mem.len() {
                        vals.insert(result.0, InterpValue::F32(mem.read_abs(abs_idx)));
                    } else {
                        let default = other_val.map(|o| o.as_f32()).unwrap_or(0.0);
                        vals.insert(result.0, InterpValue::F32(default));
                    }
                }
                _ => return Err(err(format!("Load: unsupported index type {:?}", idx_val))),
            }
        }

        TileOp::Store { ptr, indices, val, mask } => {
            let ptr_id = ptr.0;
            let buf_name = ptr_buffers.get(&ptr_id)
                .ok_or_else(|| err(format!("Store: ptr {:?} not in ptr_map", ptr)))?;
            let buf_base = mem.base(buf_name);

            let idx_val = vals.get(&indices.0)
                .ok_or_else(|| err(format!("Store: indices {:?} not defined", indices)))?;
            let val_val = vals.get(&val.0)
                .ok_or_else(|| err(format!("Store: val {:?} not defined", val)))?;

            let mask_val = mask.as_ref().map(|m| {
                vals.get(&m.0).unwrap_or_else(|| panic!("Store mask {:?} not defined", m))
            });

            match idx_val {
                InterpValue::VecF32(indices_vec) => {
                    let val_vec = val_val.as_vec_f32();
                    for (i, &idx) in indices_vec.iter().enumerate() {
                        let abs_idx = buf_base + idx as usize;
                        let masked = if let Some(ref m) = mask_val {
                            m.as_vec_bool().get(i).copied().unwrap_or(false)
                        } else {
                            true
                        };
                        if masked {
                            if abs_idx >= mem.len() {
                                return Err(err(format!(
                                    "Store OOB: abs_idx={} >= mem_len={} (buf_base={}, idx={})",
                                    abs_idx, mem.len(), buf_base, idx
                                )));
                            }
                            mem.write_abs(abs_idx, val_vec[i]);
                        }
                    }
                }
                InterpValue::U32(idx) => {
                    let abs_idx = buf_base + *idx as usize;
                    let masked = mask_val.map(|m| m.as_bool()).unwrap_or(true);
                    if masked {
                        if abs_idx >= mem.len() {
                            return Err(err(format!(
                                "Store OOB: abs_idx={} >= mem_len={}", abs_idx, mem.len()
                            )));
                        }
                        mem.write_abs(abs_idx, val_val.as_f32());
                    }
                }
                _ => return Err(err(format!("Store: unsupported index type {:?}", idx_val))),
            }
        }

        // ── Binary ops ──
        TileOp::BinOp { result, op: binop, lhs, rhs } => {
            let l = vals.get(&lhs.0)
                .ok_or_else(|| err(format!("BinOp lhs {:?} not defined", lhs)))?;
            let r = vals.get(&rhs.0)
                .ok_or_else(|| err(format!("BinOp rhs {:?} not defined", rhs)))?;

            let out = match (l, r) {
                (InterpValue::VecF32(a), InterpValue::VecF32(b)) => {
                    let v: Vec<f32> = a.iter().zip(b.iter()).map(|(&a, &b)| {
                        apply_binop_f32(*binop, a, b)
                    }).collect();
                    InterpValue::VecF32(v)
                }
                (InterpValue::F32(a), InterpValue::F32(b)) => {
                    InterpValue::F32(apply_binop_f32(*binop, *a, *b))
                }
                (InterpValue::U32(a), InterpValue::U32(b)) => {
                    InterpValue::U32(apply_binop_u32(*binop, *a, *b))
                }
                (InterpValue::VecF32(a), InterpValue::F32(b)) => {
                    let v: Vec<f32> = a.iter().map(|&a| apply_binop_f32(*binop, a, *b)).collect();
                    InterpValue::VecF32(v)
                }
                (InterpValue::F32(a), InterpValue::VecF32(b)) => {
                    let v: Vec<f32> = b.iter().map(|&b| apply_binop_f32(*binop, *a, b)).collect();
                    InterpValue::VecF32(v)
                }
                _ => return Err(err(format!("BinOp: unsupported types {:?} {:?}", l, r))),
            };
            vals.insert(result.0, out);
        }

        // ── Unary ops ──
        TileOp::UnaryOp { result, op: uop, src } => {
            let s = vals.get(&src.0)
                .ok_or_else(|| err(format!("UnaryOp src {:?} not defined", src)))?;

            let out = match s {
                InterpValue::VecF32(v) => {
                    InterpValue::VecF32(v.iter().map(|&x| apply_unaryop_f32(*uop, x)).collect())
                }
                InterpValue::F32(v) => InterpValue::F32(apply_unaryop_f32(*uop, *v)),
                _ => return Err(err(format!("UnaryOp: unsupported type {:?}", s))),
            };
            vals.insert(result.0, out);
        }

        // ── FMA ──
        TileOp::Fma { result, a, b, c } => {
            let av = vals.get(&a.0).ok_or_else(|| err(format!("Fma a {:?} not defined", a)))?;
            let bv = vals.get(&b.0).ok_or_else(|| err(format!("Fma b {:?} not defined", b)))?;
            let cv = vals.get(&c.0).ok_or_else(|| err(format!("Fma c {:?} not defined", c)))?;
            match (av, bv, cv) {
                (InterpValue::F32(a), InterpValue::F32(b), InterpValue::F32(c)) => {
                    vals.insert(result.0, InterpValue::F32(a * b + c));
                }
                _ => return Err(err("Fma: only scalar f32 supported".to_string())),
            }
        }

        // ── Comparison ──
        TileOp::Cmp { result, op: cmpop, lhs, rhs } => {
            let l = vals.get(&lhs.0)
                .ok_or_else(|| err(format!("Cmp lhs {:?} not defined", lhs)))?;
            let r = vals.get(&rhs.0)
                .ok_or_else(|| err(format!("Cmp rhs {:?} not defined", rhs)))?;

            let out = match (l, r) {
                (InterpValue::VecF32(a), InterpValue::VecF32(b)) => {
                    InterpValue::VecBool(a.iter().zip(b.iter()).map(|(&a, &b)| {
                        apply_cmp_f32(*cmpop, a, b)
                    }).collect())
                }
                (InterpValue::VecF32(a), InterpValue::F32(b)) => {
                    InterpValue::VecBool(a.iter().map(|&a| apply_cmp_f32(*cmpop, a, *b)).collect())
                }
                (InterpValue::F32(a), InterpValue::F32(b)) => {
                    InterpValue::Bool(apply_cmp_f32(*cmpop, *a, *b))
                }
                (InterpValue::U32(a), InterpValue::U32(b)) => {
                    InterpValue::Bool(apply_cmp_u32(*cmpop, *a, *b))
                }
                _ => return Err(err(format!("Cmp: unsupported types {:?} {:?}", l, r))),
            };
            vals.insert(result.0, out);
        }

        // ── Select ──
        TileOp::Select { result, cond, true_val, false_val } => {
            let c = vals.get(&cond.0)
                .ok_or_else(|| err(format!("Select cond {:?} not defined", cond)))?;
            let t = vals.get(&true_val.0)
                .ok_or_else(|| err(format!("Select true_val {:?} not defined", true_val)))?;
            let f = vals.get(&false_val.0)
                .ok_or_else(|| err(format!("Select false_val {:?} not defined", false_val)))?;

            let out = match (c, t, f) {
                (InterpValue::VecBool(cv), InterpValue::VecF32(tv), InterpValue::VecF32(fv)) => {
                    InterpValue::VecF32(cv.iter().zip(tv.iter()).zip(fv.iter())
                        .map(|((c, t), f)| if *c { *t } else { *f })
                        .collect())
                }
                (InterpValue::VecBool(cv), InterpValue::VecF32(tv), InterpValue::F32(fv)) => {
                    InterpValue::VecF32(cv.iter().zip(tv.iter())
                        .map(|(c, t)| if *c { *t } else { *fv })
                        .collect())
                }
                (InterpValue::VecBool(cv), InterpValue::F32(t), InterpValue::VecF32(fv)) => {
                    InterpValue::VecF32(cv.iter().zip(fv.iter())
                        .map(|(c, f)| if *c { *t } else { *f })
                        .collect())
                }
                (InterpValue::VecBool(cv), InterpValue::F32(t), InterpValue::F32(f)) => {
                    InterpValue::VecF32(cv.iter().map(|&c| if c { *t } else { *f }).collect())
                }
                (InterpValue::Bool(c), InterpValue::F32(t), InterpValue::F32(f)) => {
                    InterpValue::F32(if *c { *t } else { *f })
                }
                _ => return Err(err(format!("Select: unsupported types c={:?} t={:?} f={:?}", c, t, f))),
            };
            vals.insert(result.0, out);
        }

        // ── Reduce ──
        TileOp::Reduce { result, src, axis: _, op: rkind } => {
            let s = vals.get(&src.0)
                .ok_or_else(|| err(format!("Reduce src {:?} not defined", src)))?;
            let v = s.as_vec_f32();
            let out = match rkind {
                ReduceKind::Sum => v.iter().sum::<f32>(),
                ReduceKind::Max => v.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
                ReduceKind::Min => v.iter().cloned().fold(f32::INFINITY, f32::min),
                ReduceKind::Prod => v.iter().product::<f32>(),
            };
            vals.insert(result.0, InterpValue::F32(out));
        }

        // ── WG Reduce (single-thread simulation) ──
        TileOp::WgReduceAdd { result, src, block_size: _ } |
        TileOp::WgReduceMax { result, src, block_size: _ } |
        TileOp::WgReduceMin { result, src, block_size: _ } => {
            // Check for pre-computed override
            if let Some(override_val) = config.reduce_overrides.get(&result.0) {
                vals.insert(result.0, override_val.clone());
            } else {
                // In single-thread mode, the result is just the input (identity).
                let s = vals.get(&src.0)
                    .ok_or_else(|| err(format!("WgReduce src {:?} not defined", src)))?;
                vals.insert(result.0, s.clone());
            }
        }

        // ── Barrier (no-op in single-thread) ──
        TileOp::Barrier => {}

        // ── LDS (simplified: use flat array) ──
        TileOp::LdsAlloc { result, size_bytes } => {
            // Allocate LDS as a flat buffer (simplified)
            vals.insert(result.0, InterpValue::U32(0)); // base offset = 0
        }
        TileOp::LdsLoad { result, base: _, offset } => {
            let _off = vals.get(&offset.0)
                .ok_or_else(|| err(format!("LdsLoad offset {:?} not defined", offset)))?;
            // Simplified: return 0 (LDS not fully simulated)
            vals.insert(result.0, InterpValue::F32(0.0));
        }
        TileOp::LdsStore { base: _, offset: _, val } => {
            // Simplified: no-op (LDS not fully simulated)
            let _ = val;
        }

        // ── Cast ──
        TileOp::Cast { result, src, to } => {
            let s = vals.get(&src.0)
                .ok_or_else(|| err(format!("Cast src {:?} not defined", src)))?;
            let out = match (s, to) {
                (InterpValue::F32(v), ScalarDType::U32) => InterpValue::U32(*v as u32),
                (InterpValue::U32(v), ScalarDType::F32) => InterpValue::F32(*v as f32),
                (InterpValue::I32(v), ScalarDType::F32) => InterpValue::F32(*v as f32),
                (InterpValue::VecF32(v), ScalarDType::U32) => {
                    InterpValue::VecU32(v.iter().map(|x| *x as u32).collect())
                }
                _ => return Err(err(format!("Cast: unsupported {:?} to {:?}", s, to))),
            };
            vals.insert(result.0, out);
        }

        // ── Dot (simplified: naive matmul) ──
        TileOp::Dot { result, a, b } => {
            let av = vals.get(&a.0)
                .ok_or_else(|| err(format!("Dot a {:?} not defined", a)))?;
            let bv = vals.get(&b.0)
                .ok_or_else(|| err(format!("Dot b {:?} not defined", b)))?;
            // Simplified: treat as element-wise multiply (not real matmul)
            match (av, bv) {
                (InterpValue::VecF32(a), InterpValue::VecF32(b)) => {
                    let v: Vec<f32> = a.iter().zip(b.iter()).map(|(&a, &b)| a * b).collect();
                    vals.insert(result.0, InterpValue::VecF32(v));
                }
                _ => return Err(err("Dot: only VecF32 supported in interpreter".to_string())),
            }
        }

        // ── WMMA (not simulated) ──
        TileOp::ZeroAcc { result } => {
            vals.insert(result.0, InterpValue::VecF32(vec![0.0; 8]));
        }
        TileOp::WmmaF32 { result, .. } => {
            vals.insert(result.0, InterpValue::VecF32(vec![0.0; 8]));
        }
        TileOp::CvtPkBf16F32 { result, .. } => {
            vals.insert(result.0, InterpValue::U32(0));
        }
        TileOp::ExtractF32 { result, src, idx } => {
            let s = vals.get(&src.0)
                .ok_or_else(|| err(format!("ExtractF32 src {:?} not defined", src)))?;
            let v = s.as_vec_f32();
            vals.insert(result.0, InterpValue::F32(v[*idx as usize]));
        }
        TileOp::SplatFragment { result, src } => {
            let s = vals.get(&src.0)
                .ok_or_else(|| err(format!("SplatFragment src {:?} not defined", src)))?;
            let val = s.as_f32();
            vals.insert(result.0, InterpValue::VecF32(vec![val; 8]));
        }

        // ── Tile 2D ops (not fully simulated) ──
        TileOp::TileLoad2D { result, .. } => {
            vals.insert(result.0, InterpValue::VecF32(vec![0.0; 16]));
        }
        TileOp::TileDot { result, acc, .. } => {
            let a = vals.get(&acc.0)
                .ok_or_else(|| err(format!("TileDot acc {:?} not defined", acc)))?;
            vals.insert(result.0, a.clone());
        }
        TileOp::TileStore2D { .. } => {}

        // ── Atomic ──
        TileOp::AtomicAddF32 { ptr: _, indices: _, val, mask: _ } => {
            // Simplified: no-op (atomic not simulated)
            let _ = val;
        }

        // ── EXEC mask (no-op in single-thread) ──
        TileOp::ExecMaskPush { .. } | TileOp::ExecMaskFlip | TileOp::ExecMaskPop => {}

        // ── Reshape/ExpandDims ──
        TileOp::Reshape { result, src, .. } | TileOp::ExpandDims { result, src, .. } => {
            let s = vals.get(&src.0)
                .ok_or_else(|| err(format!("Reshape src {:?} not defined", src)))?;
            vals.insert(result.0, s.clone());
        }
    }

    Ok(())
}

fn apply_binop_f32(op: BinOpKind, a: f32, b: f32) -> f32 {
    match op {
        BinOpKind::Add => a + b,
        BinOpKind::Sub => a - b,
        BinOpKind::Mul => a * b,
        BinOpKind::Div => a / b,
        BinOpKind::Rem => a % b,
        BinOpKind::Max => f32::max(a, b),
        BinOpKind::Min => f32::min(a, b),
        _ => panic!("apply_binop_f32: unsupported op {:?}", op),
    }
}

fn apply_binop_u32(op: BinOpKind, a: u32, b: u32) -> u32 {
    match op {
        BinOpKind::Add => a.wrapping_add(b),
        BinOpKind::Sub => a.wrapping_sub(b),
        BinOpKind::Mul => a.wrapping_mul(b),
        BinOpKind::Div => a / b,
        BinOpKind::Rem => a % b,
        BinOpKind::And => a & b,
        BinOpKind::Or => a | b,
        BinOpKind::Xor => a ^ b,
        BinOpKind::Shl => a << b,
        BinOpKind::Shr => a >> b,
        BinOpKind::Max => a.max(b),
        BinOpKind::Min => a.min(b),
    }
}

fn apply_unaryop_f32(op: UnaryOpKind, x: f32) -> f32 {
    match op {
        UnaryOpKind::Neg => -x,
        UnaryOpKind::Exp => x.exp(),
        UnaryOpKind::Log => x.ln(),
        UnaryOpKind::Sqrt => x.sqrt(),
        UnaryOpKind::Rcp => 1.0 / x,
        UnaryOpKind::Rsqrt => 1.0 / x.sqrt(),
        UnaryOpKind::Abs => x.abs(),
        UnaryOpKind::Sigmoid => 1.0 / (1.0 + (-x).exp()),
        UnaryOpKind::Relu => if x > 0.0 { x } else { 0.0 },
        UnaryOpKind::Silu => x / (1.0 + (-x).exp()),
        UnaryOpKind::Sin => x.sin(),
        UnaryOpKind::Cos => x.cos(),
        UnaryOpKind::Exp2 => x.exp2(),
        UnaryOpKind::Log2 => x.log2(),
    }
}

fn apply_cmp_f32(op: CmpOpKind, a: f32, b: f32) -> bool {
    match op {
        CmpOpKind::Eq => a == b,
        CmpOpKind::Ne => a != b,
        CmpOpKind::Lt => a < b,
        CmpOpKind::Le => a <= b,
        CmpOpKind::Gt => a > b,
        CmpOpKind::Ge => a >= b,
    }
}

fn apply_cmp_u32(op: CmpOpKind, a: u32, b: u32) -> bool {
    match op {
        CmpOpKind::Eq => a == b,
        CmpOpKind::Ne => a != b,
        CmpOpKind::Lt => a < b,
        CmpOpKind::Le => a <= b,
        CmpOpKind::Gt => a > b,
        CmpOpKind::Ge => a >= b,
    }
}

/// Convenience: interpret a softmax_large kernel and return output row.
///
/// Builds the kernel, interprets it for one workgroup (one row),
/// and returns the output values.
///
/// Since the interpreter runs a single thread, WgReduce operations need
/// pre-computed overrides. We compute the expected global_max and global_sum
/// from the input data and pass them as reduce_overrides.
#[cfg(test)]
pub fn interpret_softmax_large(
    input: &[f32],
    cols: usize,
    n_chunks: usize,
    program_id: u32,
) -> Vec<f32> {
    use crate::t0::softmax_large::build_softmax_large;

    let func = build_softmax_large();
    let rows = 1;
    let mut mem = SimMemory::new();
    mem.alloc("input", input.len());
    mem.write("input", 0, input);
    mem.alloc("output", rows * cols);

    let ptr_map = vec![
        ("input".to_string(), "input".to_string()),
        ("output".to_string(), "output".to_string()),
    ];

    let args = vec![
        InterpValue::U32(0),
        InterpValue::U32(0),
        InterpValue::U32(cols as u32),
        InterpValue::U32(n_chunks as u32),
    ];

    // Pre-compute expected WgReduce results from input data.
    // The single-thread interpreter can't simulate cross-lane reductions,
    // so we compute the correct values and pass them as overrides.

    // First pass: run with identity reductions to get per-thread partial results
    let config1 = InterpretConfig {
        program_id,
        thread_id: 0,
        wg_size: 256,
        max_loop_iters: 100_000,
        reduce_overrides: HashMap::new(),
    };

    // Run first pass to discover WgReduce result value IDs
    // We'll find them by running and collecting the reduction source values
    let global_max = input.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let log2e = 1.4426950408889634f32;
    let global_sum: f32 = input.iter()
        .map(|&x| ((x - global_max) * log2e).exp2())
        .sum();

    // Find the WgReduce result value IDs by scanning the ops
    let all_ops = func.all_ops();
    let mut reduce_max_result = None;
    let mut reduce_add_result = None;
    for op in all_ops {
        match op {
            TileOp::WgReduceMax { result, .. } => reduce_max_result = Some(result.0),
            TileOp::WgReduceAdd { result, .. } => reduce_add_result = Some(result.0),
            _ => {}
        }
    }

    let mut reduce_overrides = HashMap::new();
    if let Some(rid) = reduce_max_result {
        reduce_overrides.insert(rid, InterpValue::F32(global_max));
    }
    if let Some(rid) = reduce_add_result {
        reduce_overrides.insert(rid, InterpValue::F32(global_sum));
    }

    // Run for each lane to get the complete output
    for lane in 0..cols.min(256) {
        let config = InterpretConfig {
            program_id,
            thread_id: lane as u32,
            wg_size: 256,
            max_loop_iters: 100_000,
            reduce_overrides: reduce_overrides.clone(),
        };
        interpret(&func, &mut mem, &ptr_map, &args, &config)
            .unwrap_or_else(|e| panic!("softmax_large interpret failed at lane {}: {}", lane, e));
    }
    mem.read("output", 0, cols)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sim_memory_basic() {
        let mut mem = SimMemory::new();
        mem.alloc("buf", 10);
        mem.write("buf", 0, &[1.0, 2.0, 3.0]);
        assert_eq!(mem.read("buf", 0, 3), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    #[should_panic(expected = "OOB read")]
    fn test_sim_memory_oob() {
        let mut mem = SimMemory::new();
        mem.alloc("buf", 3);
        mem.read("buf", 0, 5); // OOB
    }

    #[test]
    fn test_interpret_simple_load_store() {
        // Kernel: output[i] = input[i] * 2
        let mut f = TileFunc::new("double");
        let x_ptr = f.arg_ptr("x");
        let out_ptr = f.arg_ptr("out");
        let _n = f.arg_u32("n");

        let pid = f.program_id(0);
        let offs = f.arange(0, 4);
        let c4 = f.const_u32(4);
        let base = f.mul(pid, c4);
        let base_v = f.splat(base, 4);
        let idx = f.add(base_v, offs);

        let data = f.load(x_ptr, idx, ScalarDType::F32);
        let two = f.const_f32(2.0);
        let two_v = f.splat(two, 4);
        let doubled = f.mul(data, two_v);
        f.store(out_ptr, idx, doubled);
        f.return_();

        let mut mem = SimMemory::new();
        mem.alloc("input", 8);
        mem.write("input", 0, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        mem.alloc("output", 8);

        let ptr_map = vec![
            ("x".to_string(), "input".to_string()),
            ("out".to_string(), "output".to_string()),
        ];

        let config = InterpretConfig { program_id: 0, ..Default::default() };
        interpret(&f, &mut mem, &ptr_map, &[], &config).unwrap();
        assert_eq!(mem.read("output", 0, 4), vec![2.0, 4.0, 6.0, 8.0]);

        let config = InterpretConfig { program_id: 1, ..Default::default() };
        // Reset output
        mem.write("output", 4, &[0.0; 4]);
        interpret(&f, &mut mem, &ptr_map, &[], &config).unwrap();
        assert_eq!(mem.read("output", 4, 4), vec![10.0, 12.0, 14.0, 16.0]);
    }

    #[test]
    fn test_interpret_softmax() {
        // Simple softmax: [1, 4] — single row
        let mut f = TileFunc::new("softmax_test");
        let x_ptr = f.arg_ptr("x");
        let out_ptr = f.arg_ptr("out");
        let _n = f.arg_u32("n");

        let offs = f.arange(0, 4);
        let x = f.load(x_ptr, offs, ScalarDType::F32);

        // max
        let row_max = f.reduce_max(x, 0);
        let max_v = f.splat(row_max, 4);

        // x - max
        let shifted = f.sub(x, max_v);

        // exp
        let exp_x = f.exp(shifted);

        // sum
        let row_sum = f.sum(exp_x, 0);
        let sum_v = f.splat(row_sum, 4);

        // normalize
        let result = f.div(exp_x, sum_v);

        f.store(out_ptr, offs, result);
        f.return_();

        let mut mem = SimMemory::new();
        mem.alloc("input", 4);
        mem.write("input", 0, &[1.0, 2.0, 3.0, 4.0]);
        mem.alloc("output", 4);

        let ptr_map = vec![
            ("x".to_string(), "input".to_string()),
            ("out".to_string(), "output".to_string()),
        ];

        let config = InterpretConfig::default();
        interpret(&f, &mut mem, &ptr_map, &[], &config).unwrap();

        let result = mem.read("output", 0, 4);
        let sum: f32 = result.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "softmax sum should be 1.0, got {}", sum);
        assert!(result[3] > result[0], "softmax should be monotonically increasing for sorted input");
    }

    #[test]
    fn test_interpret_with_loop() {
        // Kernel: sum = 0; for i in 0..10: sum += i; output[0] = sum
        let mut f = TileFunc::new("sum_loop");
        let out_ptr = f.arg_ptr("out");

        let zero_u = f.const_u32(0);
        let ten_u = f.const_u32(10);
        let zero_f = f.const_f32(0.0);

        let lp = f.for_range_with_acc_runtime(zero_u, ten_u, 1,
            zero_f, TileType::Scalar(ScalarDType::F32));

        let iv_f = f.cast(lp.iv, ScalarDType::F32);
        let new_acc = f.add(lp.acc, iv_f);
        f.end_for_acc(&lp, new_acc);

        let idx = f.const_u32(0);
        f.store(out_ptr, idx, lp.result);
        f.return_();

        let mut mem = SimMemory::new();
        mem.alloc("output", 1);

        let ptr_map = vec![("out".to_string(), "output".to_string())];
        let config = InterpretConfig::default();
        interpret(&f, &mut mem, &ptr_map, &[], &config).unwrap();

        let result = mem.read("output", 0, 1);
        assert_eq!(result[0], 45.0); // sum 0..10 = 45
    }

    #[test]
    fn test_interpret_oob_store_detects() {
        let mut f = TileFunc::new("oob_test");
        let out_ptr = f.arg_ptr("out");

        // idx = 100 (way past buffer)
        let idx = f.const_u32(100);
        let val = f.const_f32(42.0);
        f.store(out_ptr, idx, val);
        f.return_();

        let mut mem = SimMemory::new();
        mem.alloc("output", 4); // only 4 elements

        let ptr_map = vec![("out".to_string(), "output".to_string())];
        let config = InterpretConfig::default();
        let result = interpret(&f, &mut mem, &ptr_map, &[], &config);

        assert!(result.is_err(), "Should detect OOB store");
        let err = result.unwrap_err();
        assert!(err.message.contains("OOB"), "Error should mention OOB: {}", err.message);
    }

    #[test]
    fn test_interpret_masked_select() {
        // if idx < 3: output = input; else: output = 0
        let mut f = TileFunc::new("masked");
        let x_ptr = f.arg_ptr("x");
        let out_ptr = f.arg_ptr("out");

        let offs = f.arange(0, 4);
        let x = f.load(x_ptr, offs, ScalarDType::F32);

        let three = f.const_u32(3);
        let three_v = f.splat(three, 4);
        let mask = f.cmp_lt(offs, three_v);

        let zero = f.const_f32(0.0);
        let result = f.select(mask, x, zero);

        f.store(out_ptr, offs, result);
        f.return_();

        let mut mem = SimMemory::new();
        mem.alloc("input", 4);
        mem.write("input", 0, &[10.0, 20.0, 30.0, 40.0]);
        mem.alloc("output", 4);

        let ptr_map = vec![
            ("x".to_string(), "input".to_string()),
            ("out".to_string(), "output".to_string()),
        ];

        let config = InterpretConfig::default();
        interpret(&f, &mut mem, &ptr_map, &[], &config).unwrap();

        let result = mem.read("output", 0, 4);
        assert_eq!(result, vec![10.0, 20.0, 30.0, 0.0]); // last element masked
    }

    #[test]
    fn test_interpret_softmax_large_single_row() {
        let cols = 4usize;
        let input = vec![1.0f32, 2.0, 3.0, 4.0];

        let max_val = input.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_vals: Vec<f32> = input.iter().map(|x| (x - max_val).exp()).collect();
        let sum: f32 = exp_vals.iter().sum();
        let expected: Vec<f32> = exp_vals.iter().map(|x| x / sum).collect();

        let result = interpret_softmax_large(&input, cols, 1, 0);

        for (i, (got, exp)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-5,
                "softmax_large[{}]: got {}, expected {}",
                i, got, exp
            );
        }
    }

    // ── Unary operations ──

    #[test]
    fn test_interpret_unary_exp() {
        let mut f = TileFunc::new("exp_test");
        let x_ptr = f.arg_ptr("x");
        let out_ptr = f.arg_ptr("out");

        let offs = f.arange(0, 3);
        let x = f.load(x_ptr, offs, ScalarDType::F32);
        let result = f.exp(x);
        f.store(out_ptr, offs, result);
        f.return_();

        let mut mem = SimMemory::new();
        mem.alloc("input", 3);
        mem.write("input", 0, &[0.0, 1.0, 2.0]);
        mem.alloc("output", 3);

        let ptr_map = vec![
            ("x".to_string(), "input".to_string()),
            ("out".to_string(), "output".to_string()),
        ];
        let config = InterpretConfig::default();
        interpret(&f, &mut mem, &ptr_map, &[], &config).unwrap();

        let result = mem.read("output", 0, 3);
        assert!((result[0] - 1.0).abs() < 1e-5); // exp(0) = 1
        assert!((result[1] - 1.0f32.exp()).abs() < 1e-5);
        assert!((result[2] - 2.0f32.exp()).abs() < 1e-5);
    }

    #[test]
    fn test_interpret_unary_log() {
        let mut f = TileFunc::new("log_test");
        let x_ptr = f.arg_ptr("x");
        let out_ptr = f.arg_ptr("out");

        let offs = f.arange(0, 3);
        let x = f.load(x_ptr, offs, ScalarDType::F32);
        let result = f.log(x);
        f.store(out_ptr, offs, result);
        f.return_();

        let mut mem = SimMemory::new();
        mem.alloc("input", 3);
        mem.write("input", 0, &[1.0, 2.7182818, 10.0]);
        mem.alloc("output", 3);

        let ptr_map = vec![
            ("x".to_string(), "input".to_string()),
            ("out".to_string(), "output".to_string()),
        ];
        let config = InterpretConfig::default();
        interpret(&f, &mut mem, &ptr_map, &[], &config).unwrap();

        let result = mem.read("output", 0, 3);
        assert!((result[0]).abs() < 1e-5); // ln(1) = 0
        assert!((result[1] - 1.0).abs() < 1e-4); // ln(e) ≈ 1
        assert!((result[2] - 10.0f32.ln()).abs() < 1e-5);
    }

    #[test]
    fn test_interpret_unary_sqrt_rcp_abs_neg() {
        let mut f = TileFunc::new("unary_test");
        let x_ptr = f.arg_ptr("x");
        let out_sqrt = f.arg_ptr("out_sqrt");
        let out_rcp = f.arg_ptr("out_rcp");
        let out_abs = f.arg_ptr("out_abs");
        let out_neg = f.arg_ptr("out_neg");

        let offs = f.arange(0, 4);
        let x = f.load(x_ptr, offs, ScalarDType::F32);

        let sq = f.sqrt(x);
        f.store(out_sqrt, offs, sq);

        let rc = f.rcp(x);
        f.store(out_rcp, offs, rc);

        let ab = f.abs(x);
        f.store(out_abs, offs, ab);

        let ng = f.neg(x);
        f.store(out_neg, offs, ng);

        f.return_();

        let mut mem = SimMemory::new();
        mem.alloc("input", 4);
        mem.write("input", 0, &[4.0, 2.0, -3.0, 0.5]);
        mem.alloc("out_sqrt", 4);
        mem.alloc("out_rcp", 4);
        mem.alloc("out_abs", 4);
        mem.alloc("out_neg", 4);

        let ptr_map = vec![
            ("x".to_string(), "input".to_string()),
            ("out_sqrt".to_string(), "out_sqrt".to_string()),
            ("out_rcp".to_string(), "out_rcp".to_string()),
            ("out_abs".to_string(), "out_abs".to_string()),
            ("out_neg".to_string(), "out_neg".to_string()),
        ];
        let config = InterpretConfig::default();
        interpret(&f, &mut mem, &ptr_map, &[], &config).unwrap();

        let sq = mem.read("out_sqrt", 0, 4);
        let rc = mem.read("out_rcp", 0, 4);
        let ab = mem.read("out_abs", 0, 4);
        let ng = mem.read("out_neg", 0, 4);

        assert!((sq[0] - 2.0).abs() < 1e-5);
        assert!((sq[1] - 2.0f32.sqrt()).abs() < 1e-5);
        assert!((rc[0] - 0.25).abs() < 1e-5);
        assert!((rc[3] - 2.0).abs() < 1e-5);
        assert!((ab[2] - 3.0).abs() < 1e-5);
        assert!((ng[0] + 4.0).abs() < 1e-5);
        assert!((ng[2] - 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_interpret_unary_sigmoid_relu_silu() {
        let mut f = TileFunc::new("act_test");
        let x_ptr = f.arg_ptr("x");
        let out_sig = f.arg_ptr("out_sig");
        let out_relu = f.arg_ptr("out_relu");
        let out_silu = f.arg_ptr("out_silu");

        let offs = f.arange(0, 4);
        let x = f.load(x_ptr, offs, ScalarDType::F32);

        let sig = f.sigmoid(x);
        f.store(out_sig, offs, sig);

        let relu = f.relu(x);
        f.store(out_relu, offs, relu);

        let silu = f.silu(x);
        f.store(out_silu, offs, silu);

        f.return_();

        let mut mem = SimMemory::new();
        mem.alloc("input", 4);
        mem.write("input", 0, &[-1.0, 0.0, 1.0, 2.0]);
        mem.alloc("out_sig", 4);
        mem.alloc("out_relu", 4);
        mem.alloc("out_silu", 4);

        let ptr_map = vec![
            ("x".to_string(), "input".to_string()),
            ("out_sig".to_string(), "out_sig".to_string()),
            ("out_relu".to_string(), "out_relu".to_string()),
            ("out_silu".to_string(), "out_silu".to_string()),
        ];
        let config = InterpretConfig::default();
        interpret(&f, &mut mem, &ptr_map, &[], &config).unwrap();

        let sig = mem.read("out_sig", 0, 4);
        let relu = mem.read("out_relu", 0, 4);
        let silu = mem.read("out_silu", 0, 4);

        // sigmoid(0) = 0.5
        assert!((sig[1] - 0.5).abs() < 1e-5);
        // sigmoid(x) + sigmoid(-x) = 1
        assert!((sig[0] + sig[2] - 1.0).abs() < 1e-5);
        // relu: negative → 0, positive → x
        assert_eq!(relu[0], 0.0);
        assert_eq!(relu[1], 0.0);
        assert!((relu[2] - 1.0).abs() < 1e-5);
        assert!((relu[3] - 2.0).abs() < 1e-5);
        // silu(0) = 0
        assert!(silu[1].abs() < 1e-5);
        // silu(x) = x * sigmoid(x)
        assert!((silu[2] - 1.0 * sig[2]).abs() < 1e-5);
    }

    #[test]
    fn test_interpret_unary_exp2_log2() {
        let mut f = TileFunc::new("exp2_log2_test");
        let x_ptr = f.arg_ptr("x");
        let out_exp2 = f.arg_ptr("out_exp2");
        let out_log2 = f.arg_ptr("out_log2");

        let offs = f.arange(0, 4);
        let x = f.load(x_ptr, offs, ScalarDType::F32);

        let e2 = f.exp2(x);
        f.store(out_exp2, offs, e2);

        let l2 = f.log2(x);
        f.store(out_log2, offs, l2);

        f.return_();

        let mut mem = SimMemory::new();
        mem.alloc("input", 4);
        mem.write("input", 0, &[0.0, 1.0, 2.0, 8.0]);
        mem.alloc("out_exp2", 4);
        mem.alloc("out_log2", 4);

        let ptr_map = vec![
            ("x".to_string(), "input".to_string()),
            ("out_exp2".to_string(), "out_exp2".to_string()),
            ("out_log2".to_string(), "out_log2".to_string()),
        ];
        let config = InterpretConfig::default();
        interpret(&f, &mut mem, &ptr_map, &[], &config).unwrap();

        let e2 = mem.read("out_exp2", 0, 4);
        let l2 = mem.read("out_log2", 0, 4);

        assert!((e2[0] - 1.0).abs() < 1e-5); // 2^0 = 1
        assert!((e2[1] - 2.0).abs() < 1e-5); // 2^1 = 2
        assert!((e2[2] - 4.0).abs() < 1e-5); // 2^2 = 4
        assert!((e2[3] - 256.0).abs() < 1e-3); // 2^8 = 256
        assert!((l2[1]).abs() < 1e-5); // log2(1) = 0
        assert!((l2[2] - 1.0).abs() < 1e-5); // log2(2) = 1
        assert!((l2[3] - 3.0).abs() < 1e-5); // log2(8) = 3
    }

    // ── Binary operations ──

    #[test]
    fn test_interpret_binop_sub_div_max_min() {
        let mut f = TileFunc::new("binop_test");
        let a_ptr = f.arg_ptr("a");
        let b_ptr = f.arg_ptr("b");
        let out_sub = f.arg_ptr("out_sub");
        let out_div = f.arg_ptr("out_div");
        let out_max = f.arg_ptr("out_max");
        let out_min = f.arg_ptr("out_min");

        let offs = f.arange(0, 4);
        let a = f.load(a_ptr, offs, ScalarDType::F32);
        let b = f.load(b_ptr, offs, ScalarDType::F32);

        let sub = f.sub(a, b);
        f.store(out_sub, offs, sub);
        let div = f.div(a, b);
        f.store(out_div, offs, div);
        let mx = f.max(a, b);
        f.store(out_max, offs, mx);
        let mn = f.min(a, b);
        f.store(out_min, offs, mn);
        f.return_();

        let mut mem = SimMemory::new();
        mem.alloc("a", 4);
        mem.write("a", 0, &[10.0, 5.0, -3.0, 0.0]);
        mem.alloc("b", 4);
        mem.write("b", 0, &[3.0, 5.0, 2.0, -1.0]);
        mem.alloc("out_sub", 4);
        mem.alloc("out_div", 4);
        mem.alloc("out_max", 4);
        mem.alloc("out_min", 4);

        let ptr_map = vec![
            ("a".to_string(), "a".to_string()),
            ("b".to_string(), "b".to_string()),
            ("out_sub".to_string(), "out_sub".to_string()),
            ("out_div".to_string(), "out_div".to_string()),
            ("out_max".to_string(), "out_max".to_string()),
            ("out_min".to_string(), "out_min".to_string()),
        ];
        let config = InterpretConfig::default();
        interpret(&f, &mut mem, &ptr_map, &[], &config).unwrap();

        let sub = mem.read("out_sub", 0, 4);
        let div = mem.read("out_div", 0, 4);
        let mx = mem.read("out_max", 0, 4);
        let mn = mem.read("out_min", 0, 4);

        assert_eq!(sub, vec![7.0, 0.0, -5.0, 1.0]);
        assert!((div[0] - 10.0 / 3.0).abs() < 1e-5);
        assert!((div[1] - 1.0).abs() < 1e-5);
        assert_eq!(mx, vec![10.0, 5.0, 2.0, 0.0]);
        assert_eq!(mn, vec![3.0, 5.0, -3.0, -1.0]);
    }

    #[test]
    fn test_interpret_binop_scalar_vector() {
        // Test scalar + vector broadcasting
        let mut f = TileFunc::new("broadcast_test");
        let x_ptr = f.arg_ptr("x");
        let out_ptr = f.arg_ptr("out");

        let offs = f.arange(0, 4);
        let x = f.load(x_ptr, offs, ScalarDType::F32);
        let scalar = f.const_f32(100.0);
        let result = f.add(x, scalar); // scalar broadcast
        f.store(out_ptr, offs, result);
        f.return_();

        let mut mem = SimMemory::new();
        mem.alloc("input", 4);
        mem.write("input", 0, &[1.0, 2.0, 3.0, 4.0]);
        mem.alloc("output", 4);

        let ptr_map = vec![
            ("x".to_string(), "input".to_string()),
            ("out".to_string(), "output".to_string()),
        ];
        let config = InterpretConfig::default();
        interpret(&f, &mut mem, &ptr_map, &[], &config).unwrap();

        let result = mem.read("output", 0, 4);
        assert_eq!(result, vec![101.0, 102.0, 103.0, 104.0]);
    }

    // ── Comparison ──

    #[test]
    fn test_interpret_cmp_ge_with_select() {
        let mut f = TileFunc::new("cmp_test");
        let x_ptr = f.arg_ptr("x");
        let out_ge = f.arg_ptr("out_ge");

        let offs = f.arange(0, 5);
        let x = f.load(x_ptr, offs, ScalarDType::F32);
        let threshold = f.const_f32(2.5);

        let ge_mask = f.cmp_ge(x, threshold);
        let zero = f.const_f32(0.0);
        let ge_result = f.select(ge_mask, x, zero);
        f.store(out_ge, offs, ge_result);
        f.return_();

        let mut mem = SimMemory::new();
        mem.alloc("input", 5);
        mem.write("input", 0, &[1.0, 2.0, 2.5, 3.0, 4.0]);
        mem.alloc("out_ge", 5);

        let ptr_map = vec![
            ("x".to_string(), "input".to_string()),
            ("out_ge".to_string(), "out_ge".to_string()),
        ];
        let config = InterpretConfig::default();
        interpret(&f, &mut mem, &ptr_map, &[], &config).unwrap();

        let ge = mem.read("out_ge", 0, 5);
        assert_eq!(ge, vec![0.0, 0.0, 2.5, 3.0, 4.0]); // >= 2.5
    }

    // ── FMA ──

    #[test]
    fn test_interpret_fma() {
        let mut f = TileFunc::new("fma_test");
        let out_ptr = f.arg_ptr("out");

        // result = 2.0 * 3.0 + 10.0 = 16.0
        let a = f.const_f32(2.0);
        let b = f.const_f32(3.0);
        let c = f.const_f32(10.0);
        let result = f.fma(a, b, c);

        let idx = f.const_u32(0);
        f.store(out_ptr, idx, result);
        f.return_();

        let mut mem = SimMemory::new();
        mem.alloc("output", 1);

        let ptr_map = vec![("out".to_string(), "output".to_string())];
        let config = InterpretConfig::default();
        interpret(&f, &mut mem, &ptr_map, &[], &config).unwrap();

        let result = mem.read("output", 0, 1);
        assert!((result[0] - 16.0).abs() < 1e-5);
    }

    // ── Cast ──

    #[test]
    fn test_interpret_cast_u32_to_f32() {
        let mut f = TileFunc::new("cast_test");
        let out_ptr = f.arg_ptr("out");

        let val = f.const_u32(42);
        let val_f = f.cast(val, ScalarDType::F32);

        let idx = f.const_u32(0);
        f.store(out_ptr, idx, val_f);
        f.return_();

        let mut mem = SimMemory::new();
        mem.alloc("output", 1);

        let ptr_map = vec![("out".to_string(), "output".to_string())];
        let config = InterpretConfig::default();
        interpret(&f, &mut mem, &ptr_map, &[], &config).unwrap();

        let result = mem.read("output", 0, 1);
        assert!((result[0] - 42.0).abs() < 1e-5);
    }

    // ── Multi-row simulation ──

    #[test]
    fn test_interpret_multi_row() {
        // Simulate 2 workgroups processing 2 rows
        let mut f = TileFunc::new("multi_row");
        let x_ptr = f.arg_ptr("x");
        let out_ptr = f.arg_ptr("out");
        let cols = f.arg_u32("cols");

        let pid = f.program_id(0);
        let offs = f.arange(0, 4);
        let row_base = f.mul(pid, cols);
        let base_v = f.splat(row_base, 4);
        let idx = f.add(base_v, offs);

        let x = f.load(x_ptr, idx, ScalarDType::F32);
        let doubled = f.add(x, x);
        f.store(out_ptr, idx, doubled);
        f.return_();

        let mut mem = SimMemory::new();
        mem.alloc("input", 8);
        mem.write("input", 0, &[1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0]);
        mem.alloc("output", 8);

        let ptr_map = vec![
            ("x".to_string(), "input".to_string()),
            ("out".to_string(), "output".to_string()),
        ];

        let args = vec![
            InterpValue::U32(0), // x_ptr
            InterpValue::U32(0), // out_ptr
            InterpValue::U32(4), // cols
        ];

        // Row 0
        let config = InterpretConfig { program_id: 0, ..Default::default() };
        interpret(&f, &mut mem, &ptr_map, &args, &config).unwrap();

        // Row 1
        let config = InterpretConfig { program_id: 1, ..Default::default() };
        interpret(&f, &mut mem, &ptr_map, &args, &config).unwrap();

        let result = mem.read("output", 0, 8);
        assert_eq!(result, vec![2.0, 4.0, 6.0, 8.0, 20.0, 40.0, 60.0, 80.0]);
    }

    // ── Edge cases ──

    #[test]
    fn test_interpret_single_element() {
        let mut f = TileFunc::new("single");
        let x_ptr = f.arg_ptr("x");
        let out_ptr = f.arg_ptr("out");

        let idx = f.const_u32(0);
        let x = f.load(x_ptr, idx, ScalarDType::F32);
        let result = f.mul(x, x);
        f.store(out_ptr, idx, result);
        f.return_();

        let mut mem = SimMemory::new();
        mem.alloc("input", 1);
        mem.write("input", 0, &[7.0]);
        mem.alloc("output", 1);

        let ptr_map = vec![
            ("x".to_string(), "input".to_string()),
            ("out".to_string(), "output".to_string()),
        ];
        let config = InterpretConfig::default();
        interpret(&f, &mut mem, &ptr_map, &[], &config).unwrap();

        let result = mem.read("output", 0, 1);
        assert!((result[0] - 49.0).abs() < 1e-5);
    }

    #[test]
    fn test_interpret_large_vector() {
        let n = 128usize;
        let mut f = TileFunc::new("large_vec");
        let x_ptr = f.arg_ptr("x");
        let out_ptr = f.arg_ptr("out");

        let offs = f.arange(0, n as u32);
        let x = f.load(x_ptr, offs, ScalarDType::F32);
        let result = f.add(x, x);
        f.store(out_ptr, offs, result);
        f.return_();

        let mut mem = SimMemory::new();
        let input: Vec<f32> = (0..n).map(|i| i as f32).collect();
        mem.alloc("input", n);
        mem.write("input", 0, &input);
        mem.alloc("output", n);

        let ptr_map = vec![
            ("x".to_string(), "input".to_string()),
            ("out".to_string(), "output".to_string()),
        ];
        let config = InterpretConfig::default();
        interpret(&f, &mut mem, &ptr_map, &[], &config).unwrap();

        let result = mem.read("output", 0, n);
        for i in 0..n {
            assert!((result[i] - 2.0 * i as f32).abs() < 1e-5,
                "result[{}] = {}, expected {}", i, result[i], 2.0 * i as f32);
        }
    }

    #[test]
    fn test_interpret_chain_ops() {
        // Test chaining many operations: result = ((x + 1) * 2 - 3) / 4
        let mut f = TileFunc::new("chain");
        let x_ptr = f.arg_ptr("x");
        let out_ptr = f.arg_ptr("out");

        let offs = f.arange(0, 4);
        let x = f.load(x_ptr, offs, ScalarDType::F32);

        let one = f.const_f32(1.0);
        let two = f.const_f32(2.0);
        let three = f.const_f32(3.0);
        let four = f.const_f32(4.0);

        let r1 = f.add(x, one);
        let r2 = f.mul(r1, two);
        let r3 = f.sub(r2, three);
        let r = f.div(r3, four);

        f.store(out_ptr, offs, r);
        f.return_();

        let mut mem = SimMemory::new();
        mem.alloc("input", 4);
        mem.write("input", 0, &[1.0, 2.0, 3.0, 4.0]);
        mem.alloc("output", 4);

        let ptr_map = vec![
            ("x".to_string(), "input".to_string()),
            ("out".to_string(), "output".to_string()),
        ];
        let config = InterpretConfig::default();
        interpret(&f, &mut mem, &ptr_map, &[], &config).unwrap();

        let result = mem.read("output", 0, 4);
        // ((1+1)*2-3)/4 = 0.25
        // ((2+1)*2-3)/4 = 0.75
        // ((3+1)*2-3)/4 = 1.25
        // ((4+1)*2-3)/4 = 1.75
        assert!((result[0] - 0.25).abs() < 1e-5);
        assert!((result[1] - 0.75).abs() < 1e-5);
        assert!((result[2] - 1.25).abs() < 1e-5);
        assert!((result[3] - 1.75).abs() < 1e-5);
    }

    #[test]
    fn test_interpret_reduce_sum_max() {
        let mut f = TileFunc::new("reduce_test");
        let x_ptr = f.arg_ptr("x");
        let out_sum = f.arg_ptr("out_sum");
        let out_max = f.arg_ptr("out_max");

        let offs = f.arange(0, 5);
        let x = f.load(x_ptr, offs, ScalarDType::F32);

        let s = f.sum(x, 0);
        let m = f.reduce_max(x, 0);

        let idx = f.const_u32(0);
        f.store(out_sum, idx, s);
        f.store(out_max, idx, m);
        f.return_();

        let mut mem = SimMemory::new();
        mem.alloc("input", 5);
        mem.write("input", 0, &[3.0, 1.0, 4.0, 1.0, 5.0]);
        mem.alloc("out_sum", 1);
        mem.alloc("out_max", 1);

        let ptr_map = vec![
            ("x".to_string(), "input".to_string()),
            ("out_sum".to_string(), "out_sum".to_string()),
            ("out_max".to_string(), "out_max".to_string()),
        ];
        let config = InterpretConfig::default();
        interpret(&f, &mut mem, &ptr_map, &[], &config).unwrap();

        assert_eq!(mem.read("out_sum", 0, 1), vec![14.0]);
        assert_eq!(mem.read("out_max", 0, 1), vec![5.0]);
    }

    #[test]
    fn test_interpret_softmax_large_row1() {
        // Test softmax_large for row 1 (program_id=1)
        // Note: interpret_softmax_large allocates output = 1 * cols,
        // but row 1 writes to indices [cols, 2*cols). So we pass
        // the full 2-row input and let the helper handle it.
        let cols = 4usize;
        let input = vec![
            1.0f32, 2.0, 3.0, 4.0,   // Row 0
            -1.0, 0.0, 1.0, 2.0,      // Row 1
        ];

        // Expected for row 1
        let row1 = &input[4..8];
        let max_val = row1.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_vals: Vec<f32> = row1.iter().map(|x| (x - max_val).exp()).collect();
        let sum: f32 = exp_vals.iter().sum();
        let expected: Vec<f32> = exp_vals.iter().map(|x| x / sum).collect();

        // Use the helper with program_id=1 — it allocates output for 1 row
        // but the kernel writes to row_base = pid * cols = 1 * 4 = 4.
        // We need a bigger output buffer, so we build it manually here.
        use crate::t0::softmax_large::build_softmax_large;
        let func = build_softmax_large();
        let mut mem = SimMemory::new();
        mem.alloc("input", input.len());
        mem.write("input", 0, &input);
        mem.alloc("output", 8); // 2 rows worth

        let ptr_map = vec![
            ("input".to_string(), "input".to_string()),
            ("output".to_string(), "output".to_string()),
        ];

        let args = vec![
            InterpValue::U32(0),
            InterpValue::U32(0),
            InterpValue::U32(cols as u32),
            InterpValue::U32(1),
        ];

        let global_max = row1.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let log2e = 1.4426950408889634f32;
        let global_sum: f32 = row1.iter()
            .map(|&x| ((x - global_max) * log2e).exp2())
            .sum();

        let all_ops = func.all_ops();
        let mut reduce_overrides = std::collections::HashMap::new();
        for op in all_ops {
            match op {
                TileOp::WgReduceMax { result, .. } => {
                    reduce_overrides.insert(result.0, InterpValue::F32(global_max));
                }
                TileOp::WgReduceAdd { result, .. } => {
                    reduce_overrides.insert(result.0, InterpValue::F32(global_sum));
                }
                _ => {}
            }
        }

        for lane in 0..cols {
            let config = InterpretConfig {
                program_id: 1,
                thread_id: lane as u32,
                wg_size: 256,
                max_loop_iters: 100_000,
                reduce_overrides: reduce_overrides.clone(),
            };
            interpret(&func, &mut mem, &ptr_map, &args, &config).unwrap();
        }

        let result = mem.read("output", 4, cols);
        for (i, (got, exp)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-5,
                "softmax_large row1[{}]: got {}, expected {}",
                i, got, exp
            );
        }
    }

    #[test]
    fn test_interpret_softmax_sums_to_one() {
        let cols = 8usize;
        let input: Vec<f32> = (0..8).map(|i| (i as f32 * 0.7 - 2.0).sin() * 3.0).collect();

        let result = interpret_softmax_large(&input, cols, 1, 0);
        let sum: f32 = result.iter().sum();

        assert!(
            (sum - 1.0).abs() < 1e-4,
            "softmax sum should be 1.0, got {}",
            sum
        );

        // All values should be positive
        for (i, &v) in result.iter().enumerate() {
            assert!(v > 0.0, "softmax[{}] = {} should be positive", i, v);
        }
    }

    // ── SimMemory tests ──

    #[test]
    fn test_sim_memory_multiple_buffers() {
        let mut mem = SimMemory::new();
        mem.alloc("a", 3);
        mem.alloc("b", 5);
        mem.write("a", 0, &[1.0, 2.0, 3.0]);
        mem.write("b", 0, &[10.0, 20.0, 30.0, 40.0, 50.0]);

        assert_eq!(mem.read("a", 0, 3), vec![1.0, 2.0, 3.0]);
        assert_eq!(mem.read("b", 0, 5), vec![10.0, 20.0, 30.0, 40.0, 50.0]);
        assert_eq!(mem.read("b", 2, 2), vec![30.0, 40.0]); // sub-range read
    }

    #[test]
    #[should_panic(expected = "not allocated")]
    fn test_sim_memory_missing_buffer() {
        let mem = SimMemory::new();
        mem.read("nonexistent", 0, 1);
    }

    #[test]
    fn test_sim_memory_write_abs() {
        let mut mem = SimMemory::new();
        mem.alloc("buf", 10);
        mem.write_abs(5, 42.0);
        assert_eq!(mem.read_abs(5), 42.0);
    }

    // ── InterpretConfig tests ──

    #[test]
    fn test_interpret_config_default() {
        let config = InterpretConfig::default();
        assert_eq!(config.program_id, 0);
        assert_eq!(config.thread_id, 0);
        assert_eq!(config.wg_size, 256);
        assert_eq!(config.max_loop_iters, 100_000);
        assert!(config.reduce_overrides.is_empty());
    }
}
