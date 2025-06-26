use candle_core::{Device, Result as CandleResult};
use serde::{Deserialize, Serialize};
use std::str::FromStr;
use std::time::Duration;

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

/// Ollama timeout preset
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OllamaTimeoutPreset {
    Auto,
    Gpu,
    Cpu,
}

impl FromStr for OllamaTimeoutPreset {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "auto" => Ok(OllamaTimeoutPreset::Auto),
            "gpu" => Ok(OllamaTimeoutPreset::Gpu),
            "cpu" => Ok(OllamaTimeoutPreset::Cpu),
            _ => Err(format!("Invalid timeout preset: {}. Use auto, gpu, or cpu", s)),
        }
    }
}

impl std::fmt::Display for OllamaTimeoutPreset {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OllamaTimeoutPreset::Auto => write!(f, "auto"),
            OllamaTimeoutPreset::Gpu => write!(f, "gpu"),
            OllamaTimeoutPreset::Cpu => write!(f, "cpu"),
        }
    }
}

/// Ollama timeout settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaTimeouts {
    pub connection_check: Duration,
    pub generation: Duration,
    pub evaluation: Duration,
    pub request_interval: Duration,
}

impl OllamaTimeouts {
    /// Create GPU-optimized timeouts
    pub fn gpu_preset() -> Self {
        Self {
            connection_check: Duration::from_secs(5),
            generation: Duration::from_secs(30),
            evaluation: Duration::from_secs(10),
            request_interval: Duration::from_millis(500),
        }
    }
    
    /// Create CPU-optimized timeouts
    pub fn cpu_preset() -> Self {
        Self {
            connection_check: Duration::from_secs(5),
            generation: Duration::from_secs(300),  // 5 minutes
            evaluation: Duration::from_secs(60),   // 1 minute
            request_interval: Duration::from_millis(1000),
        }
    }
    
    /// Create timeouts based on preset and device
    pub fn from_preset(preset: OllamaTimeoutPreset, device: Option<&Device>) -> Self {
        match preset {
            OllamaTimeoutPreset::Auto => {
                // Check if CUDA is available
                let is_gpu = device.map_or(false, |d| matches!(d, Device::Cuda(_)));
                if is_gpu {
                    Self::gpu_preset()
                } else {
                    Self::cpu_preset()
                }
            }
            OllamaTimeoutPreset::Gpu => Self::gpu_preset(),
            OllamaTimeoutPreset::Cpu => Self::cpu_preset(),
        }
    }
    
    /// Apply overrides from command line arguments or environment variables
    pub fn with_overrides(
        mut self,
        connection: Option<u64>,
        generation: Option<u64>,
        evaluation: Option<u64>,
        interval: Option<u64>,
    ) -> Self {
        if let Some(secs) = connection {
            self.connection_check = Duration::from_secs(secs);
        }
        if let Some(secs) = generation {
            self.generation = Duration::from_secs(secs);
        }
        if let Some(secs) = evaluation {
            self.evaluation = Duration::from_secs(secs);
        }
        if let Some(millis) = interval {
            self.request_interval = Duration::from_millis(millis);
        }
        self
    }
    
    /// Load from environment variables
    pub fn from_env_overrides(self) -> Self {
        let connection = std::env::var("TEXT2GPT1_OLLAMA_TIMEOUT_CONNECTION")
            .ok()
            .and_then(|s| s.parse::<u64>().ok());
        let generation = std::env::var("TEXT2GPT1_OLLAMA_TIMEOUT_GENERATION")
            .ok()
            .and_then(|s| s.parse::<u64>().ok());
        let evaluation = std::env::var("TEXT2GPT1_OLLAMA_TIMEOUT_EVALUATION")
            .ok()
            .and_then(|s| s.parse::<u64>().ok());
        let interval = std::env::var("TEXT2GPT1_OLLAMA_REQUEST_INTERVAL")
            .ok()
            .and_then(|s| s.parse::<u64>().ok());
        
        self.with_overrides(connection, generation, evaluation, interval)
    }
    
    /// Log current timeout settings
    pub fn log_settings(&self, preset_used: OllamaTimeoutPreset) {
        log::info!("Ollama timeout settings (preset: {}):", preset_used);
        log::info!("  Connection check: {:?}", self.connection_check);
        log::info!("  Generation: {:?}", self.generation);
        log::info!("  Evaluation: {:?}", self.evaluation);
        log::info!("  Request interval: {:?}", self.request_interval);
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
    pub learning_rate: f32,  // Changed from f64 to f32
    pub num_epochs: usize,
    pub dropout: f32,        // Changed from f64 to f32
    
    // Data settings
    pub initial_tokens: usize,
    pub final_tokens: usize,
    pub max_length: usize,
    
    // Other settings
    pub seed: u64,
    
    // Device configuration (not serialized)
    #[serde(skip)]
    pub device: Option<Device>,
    
    // Ollama timeout settings (not serialized)
    #[serde(skip)]
    pub ollama_timeouts: Option<OllamaTimeouts>,
}

impl Config {
    /// Create a new configuration with the specified model size
    pub fn new(model_size: ModelSize, seed: u64) -> Self {
        let (n_embd, n_layer, n_head) = match model_size {
            ModelSize::Small => (384, 6, 6),    // 12M params
            ModelSize::Medium => (512, 8, 8),   // 33M params
            ModelSize::Large => (768, 12, 12),  // 117M params
        };
        
        // Model size-aware default token counts
        let (initial_tokens, final_tokens) = match model_size {
            ModelSize::Small => (30_000, 15_000),   // 12M: smaller dataset
            ModelSize::Medium => (50_000, 25_000),  // 33M: balanced dataset
            ModelSize::Large => (100_000, 50_000),  // 117M: larger dataset
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
            learning_rate: 3e-4_f32,  // Changed to f32
            num_epochs: 20,
            dropout: 0.1_f32,         // Changed to f32
            
            // Data settings
            initial_tokens,
            final_tokens,
            max_length: 128,
            
            // Other settings
            seed,
            device: None,
            ollama_timeouts: None,
        }
    }
    
    /// Get recommended token counts for a model size
    pub fn get_recommended_tokens(model_size: ModelSize) -> (usize, usize) {
        match model_size {
            ModelSize::Small => (30_000, 15_000),
            ModelSize::Medium => (50_000, 25_000),
            ModelSize::Large => (100_000, 50_000),
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
    
    /// Get Ollama timeouts
    pub fn ollama_timeouts(&self) -> &OllamaTimeouts {
        self.ollama_timeouts.as_ref().expect("Ollama timeouts not initialized")
    }
    
    /// Initialize Ollama timeouts
    pub fn init_ollama_timeouts(&mut self, preset: OllamaTimeoutPreset) -> &OllamaTimeouts {
        if self.ollama_timeouts.is_none() {
            let timeouts = OllamaTimeouts::from_preset(preset, self.device.as_ref())
                .from_env_overrides();
            self.ollama_timeouts = Some(timeouts);
        }
        self.ollama_timeouts.as_ref().unwrap()
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
    pub learning_rate: f32,  // Changed from f64 to f32
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
        // Llama 3.1 series - Latest Meta models
        "llama3.1:8b" => Some(OllamaModelInfo {
            name: "Llama 3.1",
            size: "8B",
            description: "Latest Llama model - Excellent balance of quality and speed",
            best_for: vec!["general", "cooking", "poetry", "conversation"],
        }),
        "llama3.1:70b" => Some(OllamaModelInfo {
            name: "Llama 3.1 Large",
            size: "70B",
            description: "Highest quality generation (requires 40GB+ VRAM)",
            best_for: vec!["technical", "poetry", "creative"],
        }),
        
        // Qwen 2.5 series - Excellent multilingual support
        "qwen2.5:3b" => Some(OllamaModelInfo {
            name: "Qwen 2.5 Tiny",
            size: "3B",
            description: "Lightweight model with good multilingual support",
            best_for: vec!["general", "conversation"],
        }),
        "qwen2.5:7b" => Some(OllamaModelInfo {
            name: "Qwen 2.5",
            size: "7B",
            description: "Excellent multilingual model - Great for Japanese",
            best_for: vec!["general", "cooking", "poetry", "technical"],
        }),
        "qwen2.5:14b" => Some(OllamaModelInfo {
            name: "Qwen 2.5 Medium",
            size: "14B",
            description: "High-quality multilingual generation",
            best_for: vec!["technical", "creative", "poetry"],
        }),
        "qwen2.5:72b" => Some(OllamaModelInfo {
            name: "Qwen 2.5 Large",
            size: "72B",
            description: "Top-tier multilingual model (requires 40GB+ VRAM)",
            best_for: vec!["technical", "creative", "professional"],
        }),
        
        // Mistral series
        "mistral:7b-v0.3" => Some(OllamaModelInfo {
            name: "Mistral v0.3",
            size: "7B",
            description: "Fast and efficient - Good general purpose",
            best_for: vec!["general", "technical", "conversation"],
        }),
        "mixtral:8x7b" => Some(OllamaModelInfo {
            name: "Mixtral MoE",
            size: "8x7B",
            description: "Mixture of Experts - High quality with efficiency",
            best_for: vec!["technical", "creative", "professional"],
        }),
        
        // Google Gemma series
        "gemma2:2b" => Some(OllamaModelInfo {
            name: "Gemma 2",
            size: "2B",
            description: "Google's efficient small model",
            best_for: vec!["general", "cooking", "conversation"],
        }),
        "gemma2:9b" => Some(OllamaModelInfo {
            name: "Gemma 2 Medium",
            size: "9B",
            description: "Balanced Google model with good performance",
            best_for: vec!["general", "technical", "creative"],
        }),
        
        // Microsoft Phi series
        "phi3:mini" => Some(OllamaModelInfo {
            name: "Phi-3 Mini",
            size: "3.8B",
            description: "Microsoft's efficient small model",
            best_for: vec!["general", "technical", "conversation"],
        }),
        "phi3:medium" => Some(OllamaModelInfo {
            name: "Phi-3 Medium",
            size: "14B",
            description: "Microsoft's balanced model",
            best_for: vec!["technical", "creative"],
        }),
        
        // Code-specific models
        "codellama:7b" => Some(OllamaModelInfo {
            name: "Code Llama",
            size: "7B",
            description: "Specialized for code and technical documentation",
            best_for: vec!["technical", "code"],
        }),
        "deepseek-coder:6.7b" => Some(OllamaModelInfo {
            name: "DeepSeek Coder",
            size: "6.7B",
            description: "Excellent code generation model",
            best_for: vec!["technical", "code"],
        }),
        
        // Ultra-lightweight models
        "tinyllama:1.1b" => Some(OllamaModelInfo {
            name: "TinyLlama",
            size: "1.1B",
            description: "Ultra-lightweight - Perfect for CPU or low VRAM",
            best_for: vec!["general", "conversation"],
        }),
        "orca-mini:3b" => Some(OllamaModelInfo {
            name: "Orca Mini",
            size: "3B",
            description: "CPU-optimized lightweight model",
            best_for: vec!["general", "conversation"],
        }),
        
        _ => None,
    }
}
