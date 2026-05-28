//! Qwen3 model configuration.

/// Qwen3 model configuration.
///
/// Preset constructors provided for common model sizes.
/// Can also be loaded from a HuggingFace `config.json` file.
#[derive(Debug, Clone)]
pub struct Qwen3Config {
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    /// Explicit head dimension (Qwen3 uses 128, independent of hidden_size / num_heads).
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub tie_word_embeddings: bool,
}

impl Qwen3Config {
    /// Qwen3-0.6B configuration.
    pub fn qwen3_0_6b() -> Self {
        Self {
            hidden_size: 1024,
            num_layers: 28,
            num_attention_heads: 16,
            num_key_value_heads: 8,
            head_dim: 128,
            intermediate_size: 3072,
            vocab_size: 151936,
            max_position_embeddings: 40960,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            tie_word_embeddings: true,
        }
    }

    /// Qwen3-4B configuration.
    pub fn qwen3_4b() -> Self {
        Self {
            hidden_size: 2560,
            num_layers: 36,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            head_dim: 128,
            intermediate_size: 12288,
            vocab_size: 151936,
            max_position_embeddings: 40960,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            tie_word_embeddings: true,
        }
    }

    /// Head dimension (explicit from config, not derived).
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// K/V projection dimension: num_key_value_heads * head_dim.
    pub fn kv_dim(&self) -> usize {
        self.num_key_value_heads * self.head_dim
    }

    /// Q projection output dimension: num_attention_heads * head_dim.
    pub fn q_dim(&self) -> usize {
        self.num_attention_heads * self.head_dim
    }

    /// Load from a HuggingFace config.json file.
    ///
    /// Supports both full paths and directory paths (will look for config.json).
    pub fn from_file(path: &str) -> Result<Self, String> {
        let config_path = if path.ends_with(".json") {
            std::path::PathBuf::from(path)
        } else {
            std::path::PathBuf::from(path).join("config.json")
        };

        let text = std::fs::read_to_string(&config_path)
            .map_err(|e| format!("Failed to read {}: {}", config_path.display(), e))?;

        Self::from_json(&text)
    }

    /// Parse from JSON string (HuggingFace config.json format).
    pub fn from_json(json: &str) -> Result<Self, String> {
        // Manual JSON parsing (zero external deps)
        let get_usize = |key: &str| -> Result<usize, String> {
            find_json_number(json, key)?
                .parse::<usize>()
                .map_err(|e| format!("{}: not a usize: {}", key, e))
        };
        let get_f32 = |key: &str| -> Result<f32, String> {
            find_json_number(json, key)?
                .parse::<f32>()
                .map_err(|e| format!("{}: not a f32: {}", key, e))
        };
        let get_bool = |key: &str| -> Result<bool, String> {
            if json.contains(&format!("\"{}\":true", key)) || json.contains(&format!("\"{}\": true", key)) {
                Ok(true)
            } else if json.contains(&format!("\"{}\":false", key)) || json.contains(&format!("\"{}\": false", key)) {
                Ok(false)
            } else {
                Ok(false) // default
            }
        };

        let num_heads = get_usize("num_attention_heads")?;
        let hidden = get_usize("hidden_size")?;
        let head_dim = get_usize("head_dim").unwrap_or(hidden / num_heads);

        Ok(Self {
            hidden_size: hidden,
            num_layers: get_usize("num_hidden_layers")?,
            num_attention_heads: num_heads,
            num_key_value_heads: get_usize("num_key_value_heads").unwrap_or(num_heads),
            head_dim,
            intermediate_size: get_usize("intermediate_size")?,
            vocab_size: get_usize("vocab_size")?,
            max_position_embeddings: get_usize("max_position_embeddings").unwrap_or(40960),
            rms_norm_eps: get_f32("rms_norm_eps").unwrap_or(1e-6),
            rope_theta: get_f32("rope_theta").unwrap_or(1_000_000.0),
            tie_word_embeddings: get_bool("tie_word_embeddings").unwrap_or(false),
        })
    }

    /// Validate internal consistency.
    pub fn validate(&self) -> Result<(), String> {
        if self.num_attention_heads % self.num_key_value_heads != 0 {
            return Err(format!(
                "num_attention_heads ({}) must be divisible by num_key_value_heads ({})",
                self.num_attention_heads, self.num_key_value_heads
            ));
        }
        if self.head_dim % 2 != 0 {
            return Err(format!("head_dim ({}) must be even (for RoPE)", self.head_dim));
        }
        if self.head_dim > 256 {
            return Err(format!(
                "head_dim ({}) exceeds RoPE/RMSNorm kernel limit (256)",
                self.head_dim
            ));
        }
        Ok(())
    }
}

/// Find a JSON number value by key (handles both `"key": 123` and `"key":123`).
fn find_json_number(json: &str, key: &str) -> Result<String, String> {
    let patterns = [format!("\"{}\":", key)];
    for pat in &patterns {
        if let Some(pos) = json.find(pat.as_str()) {
            let after = &json[pos + pat.len()..];
            let after = after.trim_start();
            // Read until comma, closing brace, or newline
            let end = after.find(|c: char| c == ',' || c == '}' || c == '\n' || c == '\r')
                .unwrap_or(after.len());
            let val = after[..end].trim().trim_matches('"');
            return Ok(val.to_string());
        }
    }
    Err(format!("key '{}' not found in JSON", key))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qwen3_0_6b_preset() {
        let cfg = Qwen3Config::qwen3_0_6b();
        assert_eq!(cfg.hidden_size, 1024);
        assert_eq!(cfg.num_layers, 28);
        assert_eq!(cfg.num_attention_heads, 16);
        assert_eq!(cfg.num_key_value_heads, 8);
        assert_eq!(cfg.head_dim(), 128);
        assert_eq!(cfg.intermediate_size, 3072);
        assert_eq!(cfg.vocab_size, 151936);
        assert_eq!(cfg.q_dim(), 2048);
        assert_eq!(cfg.kv_dim(), 1024);
        cfg.validate().unwrap();
    }

    #[test]
    fn test_qwen3_4b_preset() {
        let cfg = Qwen3Config::qwen3_4b();
        assert_eq!(cfg.hidden_size, 2560);
        assert_eq!(cfg.num_layers, 36);
        assert_eq!(cfg.num_attention_heads, 32);
        assert_eq!(cfg.num_key_value_heads, 8);
        assert_eq!(cfg.head_dim(), 128);
        assert_eq!(cfg.intermediate_size, 12288);
        assert_eq!(cfg.q_dim(), 4096);
        assert_eq!(cfg.kv_dim(), 1024);
        cfg.validate().unwrap();
    }

    #[test]
    fn test_from_json() {
        let json = r#"{
            "hidden_size": 1024,
            "num_hidden_layers": 28,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "intermediate_size": 3072,
            "vocab_size": 151936,
            "max_position_embeddings": 40960,
            "rms_norm_eps": 1e-06,
            "rope_theta": 1000000.0,
            "tie_word_embeddings": true
        }"#;
        let cfg = Qwen3Config::from_json(json).unwrap();
        assert_eq!(cfg.hidden_size, 1024);
        assert_eq!(cfg.num_layers, 28);
        assert_eq!(cfg.num_key_value_heads, 8);
        assert_eq!(cfg.head_dim(), 128);
        assert_eq!(cfg.q_dim(), 2048);
        assert_eq!(cfg.kv_dim(), 1024);
        assert_eq!(cfg.vocab_size, 151936);
        assert!(cfg.tie_word_embeddings);
        cfg.validate().unwrap();
    }

    #[test]
    fn test_from_json_no_kv_heads() {
        // Some models don't have num_key_value_heads — defaults to num_attention_heads
        let json = r#"{
            "hidden_size": 1024,
            "num_hidden_layers": 28,
            "num_attention_heads": 16,
            "intermediate_size": 3072,
            "vocab_size": 151936
        }"#;
        let cfg = Qwen3Config::from_json(json).unwrap();
        assert_eq!(cfg.num_key_value_heads, 16); // defaults to num_attention_heads
    }

    #[test]
    fn test_validate_fails() {
        let mut cfg = Qwen3Config::qwen3_0_6b();
        cfg.num_attention_heads = 7; // not divisible into num_key_value_heads
        assert!(cfg.validate().is_err());
    }
}
