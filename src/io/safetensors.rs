use anyhow::{anyhow, Context, Result};
use candle_core::{DType, Device};
use candle_nn::{VarBuilder, VarMap};
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::Arc;

use crate::config::Config;
use crate::model::GPT;
use crate::tokenizer::GPT2Tokenizer;

/// Save model in safetensors format using VarMap
pub fn save_model<P: AsRef<Path>>(
    varmap: &VarMap,
    path: P,
) -> Result<()> {
    let path = path.as_ref();
    
    // VarMap provides a direct save method that handles safetensors format
    varmap.save(path)
        .context("Failed to save model using VarMap")?;
    
    log::info!("Model saved to: {}", path.display());
    Ok(())
}

/// Load model from safetensors format
pub fn load_model<P: AsRef<Path>>(
    path: P,
    device: &Device,
) -> Result<(GPT, Config, Arc<VarMap>)> {
    let path = path.as_ref();
    
    // First, we need to load the config to know the model architecture
    // Assuming config is saved alongside the model
    let config_path = path.with_extension("json");
    let config = if config_path.exists() {
        let config_json = fs::read_to_string(&config_path)
            .context("Failed to read config file")?;
        serde_json::from_str(&config_json)
            .context("Failed to parse config JSON")?
    } else {
        // Try to infer config from the safetensors metadata
        load_config_from_safetensors(path)?
    };
    
    // Create VarMap and load weights
    let varmap = Arc::new(VarMap::new());
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
    
    // Create model structure
    let model = GPT::new(&config, vb)?;
    
    // Load weights into VarMap
    varmap.load(path)
        .context("Failed to load weights into VarMap")?;
    
    log::info!("Model loaded from: {}", path.display());
    Ok((model, config, varmap))
}

/// Load model using memory-mapped safetensors (most efficient for large models)
pub fn load_model_mmaped<P: AsRef<Path>>(
    path: P,
    device: &Device,
) -> Result<(GPT, Config)> {
    let path = path.as_ref();
    
    // Load config
    let config_path = path.with_extension("json");
    let config: Config = if config_path.exists() {
        let config_json = fs::read_to_string(&config_path)?;
        serde_json::from_str(&config_json)?
    } else {
        load_config_from_safetensors(path)?
    };
    
    // Use memory-mapped loading for efficiency
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[path], DType::F32, device)?
    };
    
    // Create model
    let model = GPT::new(&config, vb)?;
    
    log::info!("Model loaded (mmaped) from: {}", path.display());
    Ok((model, config))
}

/// Save complete model folder
pub fn save_model_folder<P: AsRef<Path>>(
    varmap: &VarMap,
    tokenizer: &GPT2Tokenizer,
    config: &Config,
    folder_path: P,
) -> Result<()> {
    let folder_path = folder_path.as_ref();
    
    // Create folder if it doesn't exist
    fs::create_dir_all(folder_path)
        .with_context(|| format!("Failed to create model directory: {}", folder_path.display()))?;
    
    // Save model weights using VarMap
    let model_path = folder_path.join("model.safetensors");
    save_model(varmap, &model_path)?;
    
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
) -> Result<(GPT, GPT2Tokenizer, Config, Arc<VarMap>)> {
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
    let (model, _, varmap) = load_model(&model_path, device)?;
    
    // Load tokenizer
    let tokenizer = GPT2Tokenizer::from_pretrained(folder_path)
        .context("Failed to load tokenizer")?;
    
    log::info!("Model folder loaded from: {}", folder_path.display());
    Ok((model, tokenizer, config, varmap))
}

/// Load model from multiple safetensors files (for large models)
pub fn load_model_sharded<P: AsRef<Path>>(
    model_files: &[P],
    config_path: P,
    device: &Device,
) -> Result<(GPT, Config)> {
    // Load config
    let config_json = fs::read_to_string(config_path.as_ref())?;
    let config: Config = serde_json::from_str(&config_json)?;
    
    // Convert paths to string slices for VarBuilder
    let file_paths: Vec<&Path> = model_files.iter()
        .map(|p| p.as_ref())
        .collect();
    
    // Use memory-mapped loading for multiple files
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&file_paths, DType::F32, device)?
    };
    
    // Create model
    let model = GPT::new(&config, vb)?;
    
    log::info!("Sharded model loaded from {} files", model_files.len());
    Ok((model, config))
}

/// Try to load config from safetensors metadata
fn load_config_from_safetensors<P: AsRef<Path>>(path: P) -> Result<Config> {
    use safetensors::SafeTensors;
    
    let data = fs::read(path.as_ref())?;
    let tensors = SafeTensors::deserialize(&data)?;
    
    // Access metadata
    let metadata = tensors.metadata()
        .map_err(|e| anyhow!("Failed to get metadata: {}", e))?;
    
    // Extract config from metadata
    let vocab_size = metadata.get("vocab_size")
        .ok_or_else(|| anyhow!("Missing vocab_size in metadata"))?
        .parse::<usize>()?;
    
    let n_embd = metadata.get("n_embd")
        .ok_or_else(|| anyhow!("Missing n_embd in metadata"))?
        .parse::<usize>()?;
    
    let n_layer = metadata.get("n_layer")
        .ok_or_else(|| anyhow!("Missing n_layer in metadata"))?
        .parse::<usize>()?;
    
    let n_head = metadata.get("n_head")
        .ok_or_else(|| anyhow!("Missing n_head in metadata"))?
        .parse::<usize>()?;
    
    let n_positions = metadata.get("n_positions")
        .unwrap_or(&"512".to_string())
        .parse::<usize>()?;
    
    // Determine model size based on architecture
    let model_size = match (n_embd, n_layer) {
        (384, 6) => crate::config::ModelSize::Small,
        (512, 8) => crate::config::ModelSize::Medium,
        (768, 12) => crate::config::ModelSize::Large,
        _ => return Err(anyhow!("Unknown model configuration: n_embd={}, n_layer={}", n_embd, n_layer)),
    };
    
    let mut config = Config::new(model_size, 42);
    config.vocab_size = vocab_size;
    config.n_embd = n_embd;
    config.n_layer = n_layer;
    config.n_head = n_head;
    config.n_positions = n_positions;
    
    Ok(config)
}

/// Save model with metadata
pub fn save_model_with_metadata<P: AsRef<Path>>(
    varmap: &VarMap,
    path: P,
    config: &Config,
    additional_metadata: HashMap<String, String>,
) -> Result<()> {
    // First save the model normally
    varmap.save(path.as_ref())?;
    
    // Now we need to reload and add metadata
    // This is a limitation of the current VarMap API
    // In production, you might want to extend VarMap to support metadata directly
    
    // For now, we save metadata separately
    let metadata_path = path.as_ref().with_extension("metadata.json");
    let mut metadata = additional_metadata;
    
    // Add config information to metadata
    metadata.insert("vocab_size".to_string(), config.vocab_size.to_string());
    metadata.insert("n_embd".to_string(), config.n_embd.to_string());
    metadata.insert("n_layer".to_string(), config.n_layer.to_string());
    metadata.insert("n_head".to_string(), config.n_head.to_string());
    metadata.insert("n_positions".to_string(), config.n_positions.to_string());
    metadata.insert("model_size".to_string(), config.model_size.to_string());
    
    let metadata_json = serde_json::to_string_pretty(&metadata)?;
    fs::write(&metadata_path, metadata_json)?;
    
    log::info!("Model with metadata saved to: {}", path.as_ref().display());
    Ok(())
}

/// Utility function to inspect a safetensors file
pub fn inspect_safetensors<P: AsRef<Path>>(path: P) -> Result<()> {
    use safetensors::SafeTensors;
    
    let data = fs::read(path.as_ref())?;
    let tensors = SafeTensors::deserialize(&data)?;
    
    println!("=== Safetensors File Inspection ===");
    println!("File: {}", path.as_ref().display());
    
    // Print metadata
    if let Ok(metadata) = tensors.metadata() {
        println!("\nMetadata:");
        for (key, value) in metadata {
            println!("  {}: {}", key, value);
        }
    }
    
    // Print tensor information
    println!("\nTensors:");
    for (name, tensor_view) in tensors.tensors() {
        let shape = tensor_view.shape();
        let dtype = tensor_view.dtype();
        let size = shape.iter().product::<usize>();
        println!("  {} - Shape: {:?}, Dtype: {:?}, Size: {}", name, shape, dtype, size);
    }
    
    Ok(())
}
