//! Safetensors loader — parse `.safetensors` files and load weights into Ignis tensors.
//!
//! Format: [8-byte header_size][JSON header][tensor data]
//! - Header size: u64 little-endian
//! - JSON header: {"tensor_name": {"dtype": "BF16", "shape": [...], "data_offsets": [start, end]}, ...}
//! - Tensor data: raw bytes at offsets relative to end of header
//!
//! Supports multi-file models (model-00001-of-00003.safetensors, ...).

use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

/// Tensor metadata from safetensors header.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    /// Tensor name (e.g., "model.layers.0.self_attn.q_proj.weight")
    pub name: String,
    /// Data type: "BF16", "F32", "F16", "I32", etc.
    pub dtype: String,
    /// Shape dimensions
    pub shape: Vec<usize>,
    /// Byte offset range [start, end) within the file's data section
    pub data_offsets: (u64, u64),
}

/// Parsed safetensors file.
pub struct SafetensorsFile {
    /// Path to the original file
    pub path: PathBuf,
    /// Raw file contents (mmap-style: read into Vec)
    data: Vec<u8>,
    /// Tensor metadata indexed by name
    pub tensors: Vec<TensorInfo>,
}

impl SafetensorsFile {
    /// Load and parse a single `.safetensors` file.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, String> {
        let path = path.as_ref().to_path_buf();
        let mut file = File::open(&path).map_err(|e| format!("open {}: {}", path.display(), e))?;

        // Read 8-byte header size
        let mut size_buf = [0u8; 8];
        file.read_exact(&mut size_buf)
            .map_err(|e| format!("read header size from {}: {}", path.display(), e))?;
        let header_size = u64::from_le_bytes(size_buf) as usize;

        // Read JSON header
        let mut header_bytes = vec![0u8; header_size];
        file.read_exact(&mut header_bytes)
            .map_err(|e| format!("read header from {}: {}", path.display(), e))?;
        let header_str = String::from_utf8(header_bytes)
            .map_err(|e| format!("parse UTF-8 header: {}", e))?;

        // Parse JSON manually (no external deps)
        let tensors = Self::parse_header(&header_str, &path)?;

        // Read remaining data (tensor bytes)
        let mut data = Vec::new();
        file.read_to_end(&mut data)
            .map_err(|e| format!("read tensor data from {}: {}", path.display(), e))?;

        eprintln!(
            "[Safetensors] {} → {} tensors, {} MB data",
            path.file_name().unwrap_or_default().to_string_lossy(),
            tensors.len(),
            data.len() / (1024 * 1024),
        );

        Ok(Self { path, data, tensors })
    }

    /// Get tensor data bytes for a given tensor info.
    pub fn get_tensor_bytes(&self, info: &TensorInfo) -> Result<&[u8], String> {
        let (start, end) = info.data_offsets;
        let start = start as usize;
        let end = end as usize;
        if end > self.data.len() {
            return Err(format!(
                "tensor {} offsets [{}, {}) exceed data size {}",
                info.name, start, end, self.data.len()
            ));
        }
        Ok(&self.data[start..end])
    }

    /// Find a tensor by name.
    pub fn find_tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.iter().find(|t| t.name == name)
    }

    /// List all tensor names.
    pub fn names(&self) -> Vec<&str> {
        self.tensors.iter().map(|t| t.name.as_str()).collect()
    }

    /// Manual JSON parser (no serde dependency).
    fn parse_header(json: &str, path: &Path) -> Result<Vec<TensorInfo>, String> {
        let mut tensors = Vec::new();

        // Find the outer { ... } block
        let json = json.trim();
        if !json.starts_with('{') || !json.ends_with('}') {
            return Err(format!("invalid safetensors header in {}", path.display()));
        }

        // Simple state machine parser for the expected JSON structure:
        // { "name": {"dtype": "X", "shape": [...], "data_offsets": [N, M]}, ... }
        let chars: Vec<char> = json.chars().collect();
        let mut pos = 1; // skip opening {

        while pos < chars.len() {
            // Skip whitespace
            while pos < chars.len() && chars[pos].is_whitespace() { pos += 1; }
            if pos >= chars.len() || chars[pos] == '}' { break; }

            // Skip comma
            if chars[pos] == ',' { pos += 1; continue; }

            // Parse tensor name: "name"
            if chars[pos] != '"' {
                return Err(format!("expected '\"' at pos {} in {}", pos, path.display()));
            }
            pos += 1;
            let name_start = pos;
            while pos < chars.len() && chars[pos] != '"' { pos += 1; }
            let name: String = chars[name_start..pos].iter().collect();
            pos += 1; // skip closing "

            // Skip whitespace and colon
            while pos < chars.len() && (chars[pos].is_whitespace() || chars[pos] == ':') { pos += 1; }

            // Parse tensor info object: {dtype, shape, data_offsets}
            if chars[pos] != '{' {
                return Err(format!("expected '{{' for tensor {} at pos {}", name, pos));
            }
            pos += 1;

            let mut dtype = String::new();
            let mut shape = Vec::new();
            let mut offsets = (0u64, 0u64);

            while pos < chars.len() && chars[pos] != '}' {
                // Skip whitespace and commas
                while pos < chars.len() && (chars[pos].is_whitespace() || chars[pos] == ',') { pos += 1; }
                if chars[pos] == '}' { break; }

                // Parse key
                if chars[pos] != '"' {
                    return Err(format!("expected key '\"' at pos {} in tensor {}", pos, name));
                }
                pos += 1;
                let key_start = pos;
                while pos < chars.len() && chars[pos] != '"' { pos += 1; }
                let key: String = chars[key_start..pos].iter().collect();
                pos += 1; // skip "

                // Skip : and whitespace
                while pos < chars.len() && (chars[pos].is_whitespace() || chars[pos] == ':') { pos += 1; }

                match key.as_str() {
                    "dtype" => {
                        // Parse string value
                        if chars[pos] != '"' {
                            return Err(format!("expected '\"' for dtype at pos {}", pos));
                        }
                        pos += 1;
                        let val_start = pos;
                        while pos < chars.len() && chars[pos] != '"' { pos += 1; }
                        dtype = chars[val_start..pos].iter().collect();
                        pos += 1;
                    }
                    "shape" => {
                        // Parse array: [N, M, ...]
                        if chars[pos] != '[' {
                            return Err(format!("expected '[' for shape at pos {}", pos));
                        }
                        pos += 1;
                        while pos < chars.len() && chars[pos] != ']' {
                            while pos < chars.len() && (chars[pos].is_whitespace() || chars[pos] == ',') { pos += 1; }
                            if chars[pos] == ']' { break; }
                            let num_start = pos;
                            while pos < chars.len() && (chars[pos].is_ascii_digit() || chars[pos] == '-') { pos += 1; }
                            let num_str: String = chars[num_start..pos].iter().collect();
                            if let Ok(n) = num_str.parse::<usize>() {
                                shape.push(n);
                            }
                        }
                        if pos < chars.len() { pos += 1; } // skip ]
                    }
                    "data_offsets" => {
                        // Parse array: [start, end]
                        if chars[pos] != '[' {
                            return Err(format!("expected '[' for data_offsets at pos {}", pos));
                        }
                        pos += 1;
                        let mut vals = Vec::new();
                        while pos < chars.len() && chars[pos] != ']' {
                            while pos < chars.len() && (chars[pos].is_whitespace() || chars[pos] == ',') { pos += 1; }
                            if chars[pos] == ']' { break; }
                            let num_start = pos;
                            while pos < chars.len() && (chars[pos].is_ascii_digit() || chars[pos] == '-') { pos += 1; }
                            let num_str: String = chars[num_start..pos].iter().collect();
                            if let Ok(n) = num_str.parse::<u64>() {
                                vals.push(n);
                            }
                        }
                        if pos < chars.len() { pos += 1; } // skip ]
                        if vals.len() >= 2 {
                            offsets = (vals[0], vals[1]);
                        }
                    }
                    _ => {
                        // Skip unknown values (string, number, array, object)
                        Self::skip_value(&chars, &mut pos);
                    }
                }
            }
            if pos < chars.len() { pos += 1; } // skip }

            tensors.push(TensorInfo {
                name,
                dtype,
                shape,
                data_offsets: offsets,
            });
        }

        Ok(tensors)
    }

    /// Skip a JSON value (string, number, array, object, true, false, null).
    fn skip_value(chars: &[char], pos: &mut usize) {
        if *pos >= chars.len() { return; }
        match chars[*pos] {
            '"' => {
                *pos += 1;
                while *pos < chars.len() && chars[*pos] != '"' { *pos += 1; }
                if *pos < chars.len() { *pos += 1; }
            }
            '[' => {
                *pos += 1;
                let mut depth = 1;
                while *pos < chars.len() && depth > 0 {
                    if chars[*pos] == '[' { depth += 1; }
                    if chars[*pos] == ']' { depth -= 1; }
                    *pos += 1;
                }
            }
            '{' => {
                *pos += 1;
                let mut depth = 1;
                while *pos < chars.len() && depth > 0 {
                    if chars[*pos] == '{' { depth += 1; }
                    if chars[*pos] == '}' { depth -= 1; }
                    *pos += 1;
                }
            }
            _ => {
                // number, true, false, null
                while *pos < chars.len()
                    && !chars[*pos].is_whitespace()
                    && chars[*pos] != ','
                    && chars[*pos] != '}'
                    && chars[*pos] != ']'
                {
                    *pos += 1;
                }
            }
        }
    }
}

/// Load weights from safetensors file(s) into Ignis BF16 GPU tensors.
///
/// Returns a map of tensor name → (shape, Arc<GpuBuffer>) where the buffer contains
/// the raw bf16 data ready for GEMM and other operations.
///
/// Note: Only BF16 tensors are loaded. F32 tensors are skipped (use get_or_create_bf16
/// for f32→bf16 conversion if needed).
#[cfg(feature = "rocm")]
pub fn load_safetensors_bf16(
    runtime: &std::sync::Arc<crate::ignis::gpu_context::GpuRuntime>,
    path: impl AsRef<std::path::Path>,
) -> Result<std::collections::HashMap<String, (Vec<usize>, std::sync::Arc<crate::kfd::GpuBuffer>)>, String> {
    use std::sync::Arc;
    use crate::kfd::GpuBuffer;

    let st = SafetensorsFile::load(path)?;
    let mut result = std::collections::HashMap::new();

    for tensor in &st.tensors {
        let bytes = st.get_tensor_bytes(tensor)?;

        // Only load BF16 tensors directly
        if tensor.dtype != "BF16" {
            continue;
        }

        // Direct upload: bf16 data is already in the right format
        let buf = runtime.device.alloc_vram(bytes.len())?;
        buf.write_bytes(0, bytes);
        result.insert(tensor.name.clone(), (tensor.shape.clone(), Arc::new(buf)));
    }

    eprintln!("[Safetensors] Loaded {} BF16 tensors into VRAM", result.len());
    Ok(result)
}

/// Load safetensors weights directly into Ignis Tensor objects.
///
/// Returns a map of tensor name → Tensor where the Tensor wraps the GPU buffer
/// containing the raw bf16 data. This is the bridge between raw safetensors files
/// and the Ignis autodiff framework.
///
/// # Example
/// ```ignore
/// let tensors = load_safetensors_tensors(&runtime, "model-00001-of-00003.safetensors")?;
/// let embed_weight = tensors.get("model.embed_tokens.weight").unwrap();
/// ```
#[cfg(feature = "rocm")]
pub fn load_safetensors_tensors(
    runtime: &std::sync::Arc<crate::ignis::gpu_context::GpuRuntime>,
    path: impl AsRef<std::path::Path>,
) -> Result<std::collections::HashMap<String, crate::ignis::tensor::Tensor>, String> {
    use std::sync::Arc;
    use crate::ignis::tensor::{Tensor, DType};

    let raw_map = load_safetensors_bf16(runtime, path)?;
    let mut result = std::collections::HashMap::new();

    for (name, (shape, buf)) in raw_map {
        let tensor = crate::ignis::tensor::Tensor::from_buffer(buf, runtime, &shape, crate::ignis::tensor::DType::BF16, &name);
        result.insert(name, tensor);
    }

    Ok(result)
}

/// Load safetensors weights into a complete Ignis LanguageModel.
///
/// This function:
/// 1. Parses the safetensors file(s)
/// 2. Maps Qwen3 weight names to Ignis layer fields
/// 3. Creates Tensor objects from the raw GPU buffers
/// 4. Assigns them to the model's layers
///
/// # Arguments
/// * `runtime` - GPU runtime for VRAM allocation
/// * `paths` - List of safetensors file paths (for sharded models)
/// * `model` - Target LanguageModel to load weights into
///
/// # Note
/// The model must already be constructed with the correct architecture
/// (dim, n_layers, vocab_size, etc.) matching the safetensors file.
#[cfg(feature = "rocm")]
pub fn load_qwen3_into_model(
    runtime: &std::sync::Arc<crate::ignis::gpu_context::GpuRuntime>,
    paths: &[impl AsRef<std::path::Path>],
    model: &mut crate::ignis::nn::model::LanguageModel,
) -> Result<(), String> {
    use crate::ignis::tensor::{Tensor, DType};
    use std::sync::Arc;

    // Load all tensors from all files
    let mut all_tensors = std::collections::HashMap::new();
    for path in paths {
        let raw_map = load_safetensors_bf16(runtime, path)?;
        for (name, (shape, buf)) in raw_map {
            let tensor = crate::ignis::tensor::Tensor::from_buffer(buf, runtime, &shape, crate::ignis::tensor::DType::BF16, &name);
            all_tensors.insert(name, tensor);
        }
    }

    eprintln!("[Safetensors] Loaded {} tensors, assigning to model...", all_tensors.len());

    // Helper: transpose a 2D bf16 tensor [M, N] → [N, M] as f32
    // Reads bf16 bytes, converts to f32 during transpose, uploads as f32.
    fn transpose_bf16(t: &crate::ignis::tensor::Tensor, runtime: &std::sync::Arc<crate::ignis::gpu_context::GpuRuntime>) -> crate::ignis::tensor::Tensor {
        let shape = t.shape();
        assert_eq!(shape.len(), 2, "transpose_bf16: expected 2D");
        let m = shape[0];
        let n = shape[1];

        // Read raw bf16 bytes from GPU
        let _ = runtime.wait_idle();
        let bf16_bytes = m * n * 2;
        let mut raw = vec![0u8; bf16_bytes];
        t.buffer().read(&mut raw);

        // Convert bf16→f32 while transposing: raw[i*n+j] → out[j*m+i]
        let mut out = vec![0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let idx = i * n + j;
                let bits = u16::from_le_bytes([raw[idx * 2], raw[idx * 2 + 1]]);
                let val = f32::from_bits((bits as u32) << 16);
                out[j * m + i] = val;
            }
        }

        // Upload transposed f32 data
        let buf = runtime.upload_f32(&out).expect("transpose upload");
        crate::ignis::tensor::Tensor::from_buffer(
            std::sync::Arc::new(buf), runtime, &[n, m],
            crate::ignis::tensor::DType::F32, "transposed"
        )
    }

    // Assign embedding weights
    if let Some(embed_weight) = all_tensors.get("model.embed_tokens.weight") {
        model.embedding.weight = embed_weight.clone();
        eprintln!("[Safetensors] Assigned model.embed_tokens.weight");
    } else {
        return Err("missing model.embed_tokens.weight".to_string());
    }

    // Assign layer weights — fast path: keep bf16, pad directly, skip f32 intermediate
    for (layer_idx, layer) in model.layers.iter_mut().enumerate() {
        let prefix = format!("model.layers.{}", layer_idx);

        // Helper: load bf16 weight, pre-compute padded bf16, set f32 weight + cache
        fn assign_weight_fast(
            layer_linear: &mut crate::ignis::nn::linear::Linear,
            bf16_tensor: &crate::ignis::tensor::Tensor,
            runtime: &std::sync::Arc<crate::ignis::gpu_context::GpuRuntime>,
        ) -> Result<(), String> {
            // bf16_tensor is [N, K] bf16 from safetensors (HF format: [out, in])
            // We need: f32 weight [K, N] for backward/other uses
            //          bf16 padded [N_pad, K_pad] for GEMM
            let shape = bf16_tensor.shape();
            let n = shape[0]; // out_features
            let k = shape[1]; // in_features

            // Convert bf16→f32 for the weight tensor (needed for non-GEMM paths)
            layer_linear.weight = transpose_bf16(bf16_tensor, runtime);

            // Note: bf16 weight cache is computed lazily in Linear::forward()
            // via precompute_wt_bf16 (which correctly transposes [K,N]→[N,K]).
            // precompute_wt_bf16_from_raw was removed because it doesn't transpose,
            // causing incorrect GEMM results.
            Ok(())
        }

        // Attention projections
        if let Some(w) = all_tensors.get(&format!("{}.self_attn.q_proj.weight", prefix)) {
            assign_weight_fast(&mut layer.wq, w, runtime)?;
        }
        if let Some(w) = all_tensors.get(&format!("{}.self_attn.k_proj.weight", prefix)) {
            assign_weight_fast(&mut layer.wk, w, runtime)?;
        }
        if let Some(w) = all_tensors.get(&format!("{}.self_attn.v_proj.weight", prefix)) {
            assign_weight_fast(&mut layer.wv, w, runtime)?;
        }
        if let Some(w) = all_tensors.get(&format!("{}.self_attn.o_proj.weight", prefix)) {
            assign_weight_fast(&mut layer.wo, w, runtime)?;
        }

        // FFN projections
        if let Some(w) = all_tensors.get(&format!("{}.mlp.gate_proj.weight", prefix)) {
            assign_weight_fast(&mut layer.w_gate, w, runtime)?;
        }
        if let Some(w) = all_tensors.get(&format!("{}.mlp.up_proj.weight", prefix)) {
            assign_weight_fast(&mut layer.w_up, w, runtime)?;
        }
        if let Some(w) = all_tensors.get(&format!("{}.mlp.down_proj.weight", prefix)) {
            assign_weight_fast(&mut layer.w_down, w, runtime)?;
        }

        // RMSNorm gammas
        if let Some(w) = all_tensors.get(&format!("{}.input_layernorm.weight", prefix)) {
            layer.attn_norm_gamma = w.clone();
        }
        if let Some(w) = all_tensors.get(&format!("{}.post_attention_layernorm.weight", prefix)) {
            layer.ffn_norm_gamma = w.clone();
        }

        // QK-norm weights (Qwen3-specific)
        if let Some(w) = all_tensors.get(&format!("{}.self_attn.q_norm.weight", prefix)) {
            layer.q_norm_gamma = w.clone();
        }
        if let Some(w) = all_tensors.get(&format!("{}.self_attn.k_norm.weight", prefix)) {
            layer.k_norm_gamma = w.clone();
        }

        eprintln!("[Safetensors] Assigned layer {}", layer_idx);
    }

    // Final norm
    if let Some(w) = all_tensors.get("model.norm.weight") {
        model.final_norm_gamma = w.clone();
        eprintln!("[Safetensors] Assigned model.norm.weight");
    }

    // LM head — either weight-tied or transposed
    if model.config.tie_word_embeddings {
        // Weight tying: lm_head shares embedding weight (HF stores as [vocab, hidden],
        // same as embedding — no transpose needed, matmul handles it)
        model.tie_lm_head();
        // Pre-compute bf16 padded weight for lm_head (same as embedding.weight)
        // Without this, lm_head.forward() triggers a GPU f32→bf16 conversion that hangs.
        if let Some(embed_tensor) = all_tensors.get("model.embed_tokens.weight") {
            let n = model.config.vocab_size;
            let k = model.config.hidden_size;
            let wt_bf16 = crate::ignis::ops::bf16_matmul::precompute_wt_bf16_from_raw(
                runtime, embed_tensor.buffer(), n, k,
            )?;
            model.lm_head.set_cached_wt_bf16(wt_bf16);
        }
        eprintln!("[Safetensors] Tied lm_head.weight → embedding.weight (bf16 cached)");
    } else if let Some(w) = all_tensors.get("lm_head.weight") {
        // No tying: transpose from HF [vocab, hidden] to our [hidden, vocab]
        model.lm_head.weight = transpose_bf16(w, runtime);
        eprintln!("[Safetensors] Assigned lm_head.weight (transposed)");
    }

    eprintln!("[Safetensors] Model weight loading complete!");
    Ok(())
}

/// Discover safetensors files in a model directory.
///
/// Returns paths sorted by filename (handles sharded models like
/// `model-00001-of-00003.safetensors`).
pub fn discover_safetensors_files(dir: &str) -> Result<Vec<std::path::PathBuf>, String> {
    let dir_path = std::path::Path::new(dir);
    if !dir_path.is_dir() {
        return Err(format!("{} is not a directory", dir));
    }

    let mut files: Vec<std::path::PathBuf> = Vec::new();
    let entries = std::fs::read_dir(dir_path)
        .map_err(|e| format!("read_dir {}: {}", dir, e))?;

    for entry in entries {
        let entry = entry.map_err(|e| format!("read_dir entry: {}", e))?;
        let path = entry.path();
        if path.extension().map_or(false, |ext| ext == "safetensors") {
            files.push(path);
        }
    }

    files.sort();
    if files.is_empty() {
        return Err(format!("no .safetensors files found in {}", dir));
    }

    eprintln!("[Safetensors] Found {} safetensors file(s) in {}", files.len(), dir);
    Ok(files)
}

/// Load Qwen3 model from a directory path.
///
/// 1. Reads config.json from the directory
/// 2. Discovers all .safetensors files
/// 3. Constructs LanguageModel from config
/// 4. Loads weights into the model
pub fn load_qwen3_model_from_dir(
    runtime: &std::sync::Arc<crate::ignis::gpu_context::GpuRuntime>,
    dir: &str,
) -> Result<crate::ignis::nn::model::LanguageModel, String> {
    use crate::ignis::nn::config::Qwen3Config;

    let config = Qwen3Config::from_file(dir)?;
    eprintln!("[Safetensors] Model config: {} layers, {} hidden, {} heads ({} kv), vocab {}",
        config.num_layers, config.hidden_size, config.num_attention_heads,
        config.num_key_value_heads, config.vocab_size);

    let mut model = crate::ignis::nn::model::LanguageModel::from_config(runtime, &config)?;
    let paths = discover_safetensors_files(dir)?;
    load_qwen3_into_model(runtime, &paths, &mut model)?;

    // Note: tie_word_embeddings is handled inside load_qwen3_into_model
    // (lm_head.weight is loaded from file and transposed, or tied to embedding)

    Ok(model)
}

/// Qwen3 weight name mapping: safetensors name → Ignis layer path.
///
/// Qwen3 naming convention:
/// - model.embed_tokens.weight → embedding.weight
/// - model.layers.{N}.input_layernorm.weight → layers.{N}.attn_norm.weight
/// - model.layers.{N}.self_attn.q_proj.weight → layers.{N}.attn.q_proj.weight
/// - model.layers.{N}.self_attn.k_proj.weight → layers.{N}.attn.k_proj.weight
/// - model.layers.{N}.self_attn.v_proj.weight → layers.{N}.attn.v_proj.weight
/// - model.layers.{N}.self_attn.o_proj.weight → layers.{N}.attn.o_proj.weight
/// - model.layers.{N}.post_attention_layernorm.weight → layers.{N}.ffn_norm.weight
/// - model.layers.{N}.mlp.gate_proj.weight → layers.{N}.ffn.gate_proj.weight
/// - model.layers.{N}.mlp.up_proj.weight → layers.{N}.ffn.up_proj.weight
/// - model.layers.{N}.mlp.down_proj.weight → layers.{N}.ffn.down_proj.weight
/// - model.norm.weight → final_norm.weight
/// - lm_head.weight → lm_head.weight
pub fn qwen3_weight_map(safetensors_name: &str) -> Option<String> {
    // Exact mappings
    if safetensors_name == "model.embed_tokens.weight" {
        return Some("embedding.weight".to_string());
    }
    if safetensors_name == "model.norm.weight" {
        return Some("final_norm.weight".to_string());
    }
    if safetensors_name == "lm_head.weight" {
        return Some("lm_head.weight".to_string());
    }

    // Layer-level patterns: model.layers.{N}.XXX
    let prefix = "model.layers.";
    if let Some(rest) = safetensors_name.strip_prefix(prefix) {
        // Parse layer index
        if let Some(dot_pos) = rest.find('.') {
            if let Ok(layer_idx) = rest[..dot_pos].parse::<usize>() {
                let sub = &rest[dot_pos + 1..];

                let mapped = match sub {
                    "input_layernorm.weight" => format!("layers.{}.attn_norm.weight", layer_idx),
                    "post_attention_layernorm.weight" => format!("layers.{}.ffn_norm.weight", layer_idx),
                    "self_attn.q_proj.weight" => format!("layers.{}.attn.q_proj.weight", layer_idx),
                    "self_attn.k_proj.weight" => format!("layers.{}.attn.k_proj.weight", layer_idx),
                    "self_attn.v_proj.weight" => format!("layers.{}.attn.v_proj.weight", layer_idx),
                    "self_attn.o_proj.weight" => format!("layers.{}.attn.o_proj.weight", layer_idx),
                    "self_attn.q_norm.weight" => format!("layers.{}.attn.q_norm.weight", layer_idx),
                    "self_attn.k_norm.weight" => format!("layers.{}.attn.k_norm.weight", layer_idx),
                    "mlp.gate_proj.weight" => format!("layers.{}.ffn.gate_proj.weight", layer_idx),
                    "mlp.up_proj.weight" => format!("layers.{}.ffn.up_proj.weight", layer_idx),
                    "mlp.down_proj.weight" => format!("layers.{}.ffn.down_proj.weight", layer_idx),
                    _ => return None,
                };
                return Some(mapped);
            }
        }
    }

    None
}

/// Validate that a safetensors file contains the expected Qwen3 structure.
pub fn validate_qwen3_structure(st: &SafetensorsFile) -> Result<(), String> {
    let names: std::collections::HashSet<&str> = st.tensors.iter().map(|t| t.name.as_str()).collect();

    // Check embedding
    if !names.contains("model.embed_tokens.weight") {
        return Err("missing model.embed_tokens.weight".to_string());
    }

    // Check at least one layer
    let has_layer = names.iter().any(|n| n.starts_with("model.layers.0."));
    if !has_layer {
        return Err("missing model.layers.0.* tensors".to_string());
    }

    // Check final norm
    if !names.contains("model.norm.weight") {
        return Err("missing model.norm.weight".to_string());
    }

    // Count layers
    let mut max_layer = 0;
    for name in &names {
        if let Some(rest) = name.strip_prefix("model.layers.") {
            if let Some(dot) = rest.find('.') {
                if let Ok(idx) = rest[..dot].parse::<usize>() {
                    if idx > max_layer {
                        max_layer = idx;
                    }
                }
            }
        }
    }

    eprintln!("[Safetensors] Validated Qwen3 structure: {} layers found", max_layer + 1);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_header() {
        let json = r#"{"tensor1": {"dtype": "BF16", "shape": [2, 3], "data_offsets": [0, 12]}, "tensor2": {"dtype": "F32", "shape": [4], "data_offsets": [12, 28]}}"#;
        let tensors = SafetensorsFile::parse_header(json, Path::new("test.safetensors")).unwrap();
        assert_eq!(tensors.len(), 2);
        assert_eq!(tensors[0].name, "tensor1");
        assert_eq!(tensors[0].dtype, "BF16");
        assert_eq!(tensors[0].shape, vec![2, 3]);
        assert_eq!(tensors[0].data_offsets, (0, 12));
        assert_eq!(tensors[1].name, "tensor2");
        assert_eq!(tensors[1].dtype, "F32");
        assert_eq!(tensors[1].shape, vec![4]);
    }

    #[test]
    fn test_qwen3_weight_map() {
        assert_eq!(
            qwen3_weight_map("model.embed_tokens.weight"),
            Some("embedding.weight".to_string())
        );
        assert_eq!(
            qwen3_weight_map("model.layers.0.self_attn.q_proj.weight"),
            Some("layers.0.attn.q_proj.weight".to_string())
        );
        assert_eq!(
            qwen3_weight_map("model.layers.5.mlp.gate_proj.weight"),
            Some("layers.5.ffn.gate_proj.weight".to_string())
        );
        assert_eq!(
            qwen3_weight_map("model.norm.weight"),
            Some("final_norm.weight".to_string())
        );
        assert_eq!(qwen3_weight_map("unknown.tensor"), None);
    }

    /// Creates a synthetic safetensors file for testing.
    fn create_test_safetensors(path: &Path, tensors: &[(&str, &str, &[usize], &[u8])]) {
        use std::io::Write;

        // Build JSON header and collect tensor data
        let mut json_parts = Vec::new();
        let mut data = Vec::new();
        let mut offset = 0u64;

        for (name, dtype, shape, tensor_data) in tensors {
            let shape_str = shape.iter().map(|n| n.to_string()).collect::<Vec<_>>().join(", ");
            let end_offset = offset + tensor_data.len() as u64;
            json_parts.push(format!(
                r#""{}": {{"dtype": "{}", "shape": [{}], "data_offsets": [{}, {}]}}"#,
                name, dtype, shape_str, offset, end_offset
            ));
            data.extend_from_slice(tensor_data);
            offset = end_offset;
        }

        let json = format!("{{{}}}", json_parts.join(", "));
        let header_bytes = json.as_bytes();
        let header_size = header_bytes.len() as u64;

        // Write file: [8-byte header_size][JSON header][tensor data]
        let mut file = File::create(path).expect("create test safetensors file");
        file.write_all(&header_size.to_le_bytes()).expect("write header size");
        file.write_all(header_bytes).expect("write header");
        file.write_all(&data).expect("write tensor data");
        file.flush().expect("flush test file");
    }

    #[test]
    fn test_load_synthetic_safetensors() {
        use std::fs;

        let tmp_dir = std::env::temp_dir();
        let path = tmp_dir.join("test_model.safetensors");

        // Create a mini Qwen3-like safetensors file with BF16 data
        // BF16 is 2 bytes per element
        let embed_data: Vec<u8> = (0..16).map(|i| i as u8).collect(); // 8 elements × 2 bytes
        let q_proj_data: Vec<u8> = (0..32).map(|i| i as u8).collect(); // 16 elements × 2 bytes
        let k_proj_data: Vec<u8> = (0..32).map(|i| i as u8).collect();
        let v_proj_data: Vec<u8> = (0..32).map(|i| i as u8).collect();
        let norm_data: Vec<u8> = (0..4).map(|i| i as u8).collect(); // 2 elements × 2 bytes

        create_test_safetensors(&path, &[
            ("model.embed_tokens.weight", "BF16", &[2, 4], &embed_data),
            ("model.layers.0.self_attn.q_proj.weight", "BF16", &[4, 4], &q_proj_data),
            ("model.layers.0.self_attn.k_proj.weight", "BF16", &[4, 4], &k_proj_data),
            ("model.layers.0.self_attn.v_proj.weight", "BF16", &[4, 4], &v_proj_data),
            ("model.norm.weight", "BF16", &[2], &norm_data),
        ]);

        // Load and verify
        let st = SafetensorsFile::load(&path).expect("load test safetensors");
        assert_eq!(st.tensors.len(), 5);

        // Verify embed_tokens
        let embed = st.find_tensor("model.embed_tokens.weight").expect("find embed");
        assert_eq!(embed.dtype, "BF16");
        assert_eq!(embed.shape, vec![2, 4]);
        let embed_bytes = st.get_tensor_bytes(embed).expect("get embed bytes");
        assert_eq!(embed_bytes, &embed_data);

        // Verify q_proj
        let q_proj = st.find_tensor("model.layers.0.self_attn.q_proj.weight").expect("find q_proj");
        assert_eq!(q_proj.shape, vec![4, 4]);
        let q_bytes = st.get_tensor_bytes(q_proj).expect("get q_proj bytes");
        assert_eq!(q_bytes, &q_proj_data);

        // Verify weight mapping
        assert_eq!(
            qwen3_weight_map("model.embed_tokens.weight"),
            Some("embedding.weight".to_string())
        );
        assert_eq!(
            qwen3_weight_map("model.layers.0.self_attn.q_proj.weight"),
            Some("layers.0.attn.q_proj.weight".to_string())
        );
        assert_eq!(
            qwen3_weight_map("model.norm.weight"),
            Some("final_norm.weight".to_string())
        );

        // Validate Qwen3 structure
        validate_qwen3_structure(&st).expect("valid Qwen3 structure");

        // Cleanup
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn test_validate_qwen3_missing_embed() {
        let tmp_dir = std::env::temp_dir();
        let path = tmp_dir.join("test_incomplete.safetensors");

        // Missing embed_tokens.weight
        let weight_data: Vec<u8> = (0..16).map(|i| i as u8).collect();
        create_test_safetensors(&path, &[
            ("model.layers.0.self_attn.q_proj.weight", "BF16", &[4, 4], &weight_data),
            ("model.norm.weight", "BF16", &[2], &weight_data[..4]),
        ]);

        let st = SafetensorsFile::load(&path).expect("load test file");
        let result = validate_qwen3_structure(&st);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("embed_tokens"));

        // Cleanup
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_qwen3_weight_map_all_keys() {
        // Verify all expected Qwen3 weight names map correctly
        let expected = vec![
            ("model.embed_tokens.weight", "embedding.weight"),
            ("model.layers.0.input_layernorm.weight", "layers.0.attn_norm.weight"),
            ("model.layers.0.post_attention_layernorm.weight", "layers.0.ffn_norm.weight"),
            ("model.layers.0.self_attn.q_proj.weight", "layers.0.attn.q_proj.weight"),
            ("model.layers.0.self_attn.k_proj.weight", "layers.0.attn.k_proj.weight"),
            ("model.layers.0.self_attn.v_proj.weight", "layers.0.attn.v_proj.weight"),
            ("model.layers.0.self_attn.o_proj.weight", "layers.0.attn.o_proj.weight"),
            ("model.layers.0.mlp.gate_proj.weight", "layers.0.ffn.gate_proj.weight"),
            ("model.layers.0.mlp.up_proj.weight", "layers.0.ffn.up_proj.weight"),
            ("model.layers.0.mlp.down_proj.weight", "layers.0.ffn.down_proj.weight"),
            ("model.norm.weight", "final_norm.weight"),
            ("lm_head.weight", "lm_head.weight"),
        ];
        for (st_name, expected_path) in expected {
            assert_eq!(
                qwen3_weight_map(st_name),
                Some(expected_path.to_string()),
                "mapping failed for {}",
                st_name
            );
        }
    }
}
