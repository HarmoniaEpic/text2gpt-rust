use candle_core::{Device, Result as CandleResult};
use serde::{Deserialize, Serialize};
use std::str::FromStr;

/// Model size configurations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelSize {
    #[serde(rename = "12M")]
    Small,  // 12M parameters
    #[serde(rename = "33M")]
    Medium, // 33M parameters
    #[serde(rename = "117M")]
    Large,  // 117M parameters
}

impl FromStr for ModelSize {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "12M" | "small" => Ok(ModelSize::Small),
            "33M" | "medium" => Ok(ModelSize::Medium),
            "117M" | "large" => Ok(ModelSize::Large),
            _ => Err(format!("Invalid model size: {}. Use 12M, 33M, or 117M", s)),
        }
    }
}

impl std::fmt::Display for ModelSize {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelSize::Small => write!(f, "12M"),
            ModelSize::Medium => write!(f, "33M"),
            ModelSize::Large => write!(f, "117M"),
        }
    }
}

/// Main configuration struct for the model and training
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    // Model architecture
    pub vocab_size: usize,
    pub n_embd: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub n_positions: usize,
    pub model_size: ModelSize,
    
    // Training settings
    pub batch_size: usize,
    pub learning_rate: f64,
    pub num_epochs: usize,
    pub dropout: f64,
    
    // Data settings
    pub initial_tokens: usize,
    pub final_tokens: usize,
    pub max_length: usize,
    
    // Other settings
    pub seed: u64,
    
    // Device configuration (not serialized)
    #[serde(skip)]
    pub device: Option<Device>,
}

impl Config {
    /// Create a new configuration with the specified model size
    pub fn new(model_size: ModelSize, seed: u64) -> Self {
        let (n_embd, n_layer, n_head) = match model_size {
            ModelSize::Small => (384, 6, 6),    // 12M params
            ModelSize::Medium => (512, 8, 8),   // 33M params
            ModelSize::Large => (768, 12, 12),  // 117M params
        };
        
        Config {
            // Model architecture
            vocab_size: 50257,  // GPT-2 tokenizer vocab size
            n_embd,
            n_layer,
            n_head,
            n_positions: 512,
            model_size,
            
            // Training settings
            batch_size: 2,
            learning_rate: 3e-4,
            num_epochs: 20,
            dropout: 0.1,
            
            // Data settings
            initial_tokens: 2000,
            final_tokens: 1000,
            max_length: 128,
            
            // Other settings
            seed,
            device: None,
        }
    }
    
    /// Get the device for computation
    pub fn device(&self) -> &Device {
        self.device.as_ref().expect("Device not initialized")
    }
    
    /// Initialize and get mutable device reference
    pub fn init_device(&mut self) -> CandleResult<&Device> {
        if self.device.is_none() {
            self.device = Some(Device::cuda_if_available(0)?);
        }
        Ok(self.device.as_ref().unwrap())
    }
    
    /// Get parameter count in millions
    pub fn param_count_millions(&self) -> f32 {
        let embeddings = self.vocab_size * self.n_embd + self.n_positions * self.n_embd;
        let attention = self.n_layer * (3 * self.n_embd * self.n_embd + self.n_embd * self.n_embd);
        let mlp = self.n_layer * (2 * self.n_embd * 4 * self.n_embd);
        let layer_norm = self.n_layer * 2 * self.n_embd + self.n_embd;
        // Note: lm_head is not counted as we use weight sharing with embeddings
        
        let total = embeddings + attention + mlp + layer_norm;
        total as f32 / 1_000_000.0
    }
}

impl Default for Config {
    fn default() -> Self {
        Config::new(ModelSize::Small, 42)
    }
}

/// Category information for data generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CategoryInfo {
    pub key: String,
    pub name: String,
    pub name_ja: String,
    pub description: String,
    pub icon: String,
}

/// Preset information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PresetInfo {
    pub title: String,
    pub title_en: String,
    pub prompt: String,
    pub description: String,
}

/// Training result information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingResult {
    pub model_path: std::path::PathBuf,
    pub domain: String,
    pub final_loss: f32,
    pub training_time_seconds: f64,
}

/// Generation information saved with the model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationInfo {
    pub prompt: String,
    pub category: String,
    pub preset: Option<String>,
    pub model_size: String,
    pub training_params: TrainingParams,
    pub data_stats: DataStats,
    pub dataset_files: DatasetFiles,
    pub creation_date: String,
    pub generation_method: String,
    pub generation_model: Option<String>,
    pub refinement_model: Option<String>,
    pub ollama_used: bool,
    pub generation_time: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingParams {
    pub epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataStats {
    pub samples_generated: usize,
    pub samples_refined: usize,
    pub initial_tokens: usize,
    pub final_tokens: usize,
    pub generation_method: String,
    pub generation_model: Option<String>,
    pub refinement_model: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetFiles {
    pub json: String,
    pub text: String,
}

/// Ollama model information
pub struct OllamaModelInfo {
    pub name: &'static str,
    pub size: &'static str,
    pub description: &'static str,
    pub best_for: Vec<&'static str>,
}

/// Get Ollama model information
pub fn get_ollama_model_info(model: &str) -> Option<OllamaModelInfo> {
    match model {
        "llama3" => Some(OllamaModelInfo {
            name: "Llama 3",
            size: "8B",
            description: "Meta's latest model - Well-balanced general model",
            best_for: vec!["general", "cooking", "poetry"],
        }),
        "llama3:70b" => Some(OllamaModelInfo {
            name: "Llama 3 Large",
            size: "70B",
            description: "Highest quality generation (requires high-end GPU)",
            best_for: vec!["technical", "poetry"],
        }),
        "mistral" => Some(OllamaModelInfo {
            name: "Mistral",
            size: "7B",
            description: "Fast and efficient generation",
            best_for: vec!["general", "technical"],
        }),
        "gemma" => Some(OllamaModelInfo {
            name: "Gemma",
            size: "2B",
            description: "Google's lightweight model - Fast processing",
            best_for: vec!["general", "cooking"],
        }),
        "gemma:7b" => Some(OllamaModelInfo {
            name: "Gemma Large",
            size: "7B",
            description: "Larger Gemma - Higher quality",
            best_for: vec!["general", "technical"],
        }),
        "codellama" => Some(OllamaModelInfo {
            name: "Code Llama",
            size: "7B",
            description: "Specialized for programming and technical docs",
            best_for: vec!["technical"],
        }),
        "phi" => Some(OllamaModelInfo {
            name: "Phi-2",
            size: "2.7B",
            description: "Microsoft's small high-performance model",
            best_for: vec!["general", "technical"],
        }),
        _ => None,
    }
}