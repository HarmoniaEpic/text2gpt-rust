use anyhow::Result;
use chrono::Local;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

use crate::config::{DataStats, DatasetFiles, GenerationInfo, TrainingParams};

/// Dataset information saved as JSON
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetInfo {
    pub metadata: DatasetMetadata,
    pub samples: Vec<DatasetSample>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetMetadata {
    pub total_samples: usize,
    pub total_tokens: usize,
    pub average_tokens_per_sample: f64,
    pub creation_date: String,
    pub tokenizer: String,
    pub prompt: String,
    pub domain: String,
    pub generation_method: String,
    pub generation_model: Option<String>,
    pub refinement_model: Option<String>,
    pub ollama_used: bool,
    pub quality_threshold: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetSample {
    pub text: String,
    pub tokens: usize,
}

/// Model information for listing
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub path: PathBuf,
    pub folder_name: String,
    pub info: GenerationInfo,
}

/// Save dataset as JSON
pub fn save_dataset<P: AsRef<Path>>(
    path: P,
    samples: &[String],
    tokenizer: &crate::tokenizer::GPT2Tokenizer,
    metadata: DatasetMetadata,
) -> Result<()> {
    let path = path.as_ref();
    
    // Create parent directory if needed
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    
    // Convert samples to dataset format
    let mut dataset_samples = Vec::new();
    for sample in samples {
        let tokens = tokenizer.encode(sample)?;
        dataset_samples.push(DatasetSample {
            text: sample.clone(),
            tokens: tokens.len(),
        });
    }
    
    let dataset_info = DatasetInfo {
        metadata,
        samples: dataset_samples,
    };
    
    // Save as JSON
    let json = serde_json::to_string_pretty(&dataset_info)?;
    fs::write(path, json)?;
    
    log::info!("Dataset saved to: {}", path.display());
    Ok(())
}

/// Save dataset as plain text
pub fn save_dataset_text<P: AsRef<Path>>(
    path: P,
    samples: &[String],
) -> Result<()> {
    let path = path.as_ref();
    
    // Create parent directory if needed
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    
    let mut content = String::new();
    for (i, sample) in samples.iter().enumerate() {
        content.push_str(&format!("=== Sample {} ===\n", i + 1));
        content.push_str(sample);
        content.push_str("\n\n");
    }
    
    fs::write(path, content)?;
    
    log::info!("Text dataset saved to: {}", path.display());
    Ok(())
}

/// Save generation info
pub fn save_generation_info<P: AsRef<Path>>(
    path: P,
    prompt: &str,
    category: &str,
    preset: Option<&str>,
    model_size: &str,
    config: &crate::config::Config,
    data_stats: DataStats,
    generation_method: &str,
    generation_model: Option<&str>,
    refinement_model: Option<&str>,
    ollama_used: bool,
    generation_time: f64,
) -> Result<()> {
    let path = path.as_ref();
    
    // Create parent directory if needed
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    
    let info = GenerationInfo {
        prompt: prompt.to_string(),
        category: category.to_string(),
        preset: preset.map(|s| s.to_string()),
        model_size: model_size.to_string(),
        training_params: TrainingParams {
            epochs: config.num_epochs,
            batch_size: config.batch_size,
            learning_rate: config.learning_rate,
        },
        data_stats,
        dataset_files: DatasetFiles {
            json: "dataset.json".to_string(),
            text: "dataset.txt".to_string(),
        },
        creation_date: Local::now().format("%Y-%m-%d %H:%M:%S").to_string(),
        generation_method: generation_method.to_string(),
        generation_model: generation_model.map(|s| s.to_string()),
        refinement_model: refinement_model.map(|s| s.to_string()),
        ollama_used,
        generation_time: format!("{:.1} seconds", generation_time),
    };
    
    let json = serde_json::to_string_pretty(&info)?;
    fs::write(path, json)?;
    
    log::info!("Generation info saved to: {}", path.display());
    Ok(())
}

/// List saved models
pub fn list_saved_models<P: AsRef<Path>>(models_dir: P) -> Result<Vec<ModelInfo>> {
    let models_dir = models_dir.as_ref();
    let mut models = Vec::new();
    
    if !models_dir.exists() {
        return Ok(models);
    }
    
    for entry in fs::read_dir(models_dir)? {
        let entry = entry?;
        let path = entry.path();
        
        if path.is_dir() {
            let info_path = path.join("generation_info.json");
            if info_path.exists() {
                match fs::read_to_string(&info_path) {
                    Ok(content) => {
                        match serde_json::from_str::<GenerationInfo>(&content) {
                            Ok(info) => {
                                models.push(ModelInfo {
                                    folder_name: path.file_name()
                                        .unwrap_or_default()
                                        .to_string_lossy()
                                        .to_string(),
                                    path: path.clone(),
                                    info,
                                });
                            }
                            Err(e) => {
                                log::warn!("Failed to parse generation info at {}: {}", info_path.display(), e);
                            }
                        }
                    }
                    Err(e) => {
                        log::warn!("Failed to read generation info at {}: {}", info_path.display(), e);
                    }
                }
            }
        }
    }
    
    // Sort by creation date (newest first)
    models.sort_by(|a, b| b.info.creation_date.cmp(&a.info.creation_date));
    
    Ok(models)
}