use anyhow::{anyhow, Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use safetensors::{serialize, SafeTensors};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

use crate::config::Config;
use crate::model::GPT;
use crate::tokenizer::GPT2Tokenizer;

/// Save model in safetensors format
pub fn save_model<P: AsRef<Path>>(
    _model: &GPT,
    config: &Config,
    path: P,
    metadata: HashMap<String, String>,
) -> Result<()> {
    let path = path.as_ref();
    
    // Note: In a real implementation, we would need a way to extract tensors from the model
    // For now, this is a placeholder that demonstrates the structure
    // In practice, you might need to modify the GPT struct to store references to its tensors
    
    let tensors: HashMap<String, Tensor> = HashMap::new();
    // TODO: Implement tensor extraction from model
    
    // Convert to safetensors format
    let data = serialize(tensors, &Some(metadata))
        .context("Failed to serialize model tensors to safetensors format")?;
    
    // Save to file
    fs::write(path, data)
        .with_context(|| format!("Failed to write model file: {}", path.display()))?;
    
    log::info!("Model saved to: {}", path.display());
    Ok(())
}

/// Load model from safetensors format
pub fn load_model<P: AsRef<Path>>(
    path: P,
    device: &Device,
) -> Result<(GPT, Config)> {
    let path = path.as_ref();
    
    // Read the file
    let data = fs::read(path)
        .with_context(|| format!("Failed to read model file: {}", path.display()))?;
    let tensors = SafeTensors::deserialize(&data)
        .context("Failed to deserialize safetensors data")?;
    
    // Extract config from metadata
    let metadata_map = tensors.metadata()
        .map_err(|e| anyhow!("Failed to get metadata: {}", e))?;
    let config = config_from_metadata(&metadata_map)?;
    
    // Convert tensors to HashMap
    let mut tensor_map = HashMap::new();
    for (name, view) in tensors.tensors() {
        let shape = view.shape();
        let dtype = match view.dtype() {
            safetensors::Dtype::F32 => DType::F32,
            safetensors::Dtype::F16 => DType::F16,
            safetensors::Dtype::BF16 => DType::BF16,
            _ => return Err(anyhow!("Unsupported dtype: {:?}", view.dtype())),
        };
        
        // Create tensor from raw data
        let data = view.data();
        let tensor = Tensor::from_raw_buffer(data, dtype, shape, device)?;
        tensor_map.insert(name.to_string(), tensor);
    }
    
    // Create VarBuilder with the tensors
    let vb = VarBuilder::from_tensors(tensor_map, DType::F32, device);
    
    // Create and load the model
    let model = GPT::new(&config, vb)?;
    
    log::info!("Model loaded from: {}", path.display());
    Ok((model, config))
}

/// Save complete model folder
pub fn save_model_folder<P: AsRef<Path>>(
    model: &GPT,
    tokenizer: &GPT2Tokenizer,
    config: &Config,
    folder_path: P,
    metadata: HashMap<String, String>,
) -> Result<()> {
    let folder_path = folder_path.as_ref();
    
    // Create folder if it doesn't exist
    fs::create_dir_all(folder_path)
        .with_context(|| format!("Failed to create model directory: {}", folder_path.display()))?;
    
    // Save model
    let model_path = folder_path.join("model.safetensors");
    save_model(model, config, model_path, metadata)?;
    
    // Save tokenizer
    tokenizer.save_pretrained(folder_path)
        .context("Failed to save tokenizer")?;
    
    // Save config
    let config_path = folder_path.join("config.json");
    let config_json = serde_json::to_string_pretty(config)
        .context("Failed to serialize config to JSON")?;
    fs::write(&config_path, config_json)
        .with_context(|| format!("Failed to write config file: {}", config_path.display()))?;
    
    log::info!("Model folder saved to: {}", folder_path.display());
    Ok(())
}

/// Load complete model folder
pub fn load_model_folder<P: AsRef<Path>>(
    folder_path: P,
    device: &Device,
) -> Result<(GPT, GPT2Tokenizer, Config)> {
    let folder_path = folder_path.as_ref();
    
    // Load config
    let config_path = folder_path.join("config.json");
    let config_json = fs::read_to_string(&config_path)
        .with_context(|| format!("Failed to read config file: {}", config_path.display()))?;
    let mut config: Config = serde_json::from_str(&config_json)
        .context("Failed to parse config JSON")?;
    config.device = Some(device.clone());
    
    // Load model
    let model_path = folder_path.join("model.safetensors");
    let (model, _) = load_model(model_path, device)?;
    
    // Load tokenizer
    let tokenizer = GPT2Tokenizer::from_pretrained(folder_path)
        .context("Failed to load tokenizer")?;
    
    log::info!("Model folder loaded from: {}", folder_path.display());
    Ok((model, tokenizer, config))
}

/// Create config from metadata
fn config_from_metadata(metadata: &HashMap<String, String>) -> Result<Config> {
    let vocab_size = metadata.get("vocab_size")
        .ok_or_else(|| anyhow!("Missing vocab_size in metadata"))?
        .parse::<usize>()
        .context("Failed to parse vocab_size as usize")?;
    
    let n_embd = metadata.get("n_embd")
        .ok_or_else(|| anyhow!("Missing n_embd in metadata"))?
        .parse::<usize>()
        .context("Failed to parse n_embd as usize")?;
    
    let n_layer = metadata.get("n_layer")
        .ok_or_else(|| anyhow!("Missing n_layer in metadata"))?
        .parse::<usize>()
        .context("Failed to parse n_layer as usize")?;
    
    let n_head = metadata.get("n_head")
        .ok_or_else(|| anyhow!("Missing n_head in metadata"))?
        .parse::<usize>()
        .context("Failed to parse n_head as usize")?;
    
    let n_positions = metadata.get("n_positions")
        .ok_or_else(|| anyhow!("Missing n_positions in metadata"))?
        .parse::<usize>()
        .context("Failed to parse n_positions as usize")?;
    
    let model_size = if n_embd == 384 && n_layer == 6 {
        crate::config::ModelSize::Small
    } else if n_embd == 512 && n_layer == 8 {
        crate::config::ModelSize::Medium
    } else if n_embd == 768 && n_layer == 12 {
        crate::config::ModelSize::Large
    } else {
        return Err(anyhow!("Unknown model configuration"));
    };
    
    let mut config = Config::new(model_size, 42);
    config.vocab_size = vocab_size;
    config.n_embd = n_embd;
    config.n_layer = n_layer;
    config.n_head = n_head;
    config.n_positions = n_positions;
    
    Ok(config)
}
