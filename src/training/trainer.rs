use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use chrono::Local;
use colored::*;
use indicatif::{ProgressBar, ProgressStyle};
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use crate::config::{Config, DataStats, TrainingResult};
use crate::data::{DataGenerationMethod, DataGenerator, DataRefiner, TextDataset};
use crate::io::json;
use crate::model::GPT;
use crate::tokenizer::GPT2Tokenizer;

/// Run the full training pipeline
pub async fn run_full_pipeline(
    prompt: &str,
    config: &Config,
    output_dir: &Path,
    generation_method: DataGenerationMethod,
) -> Result<TrainingResult> {
    let start_time = Instant::now();
    
    println!("\n{}", "Starting Text2GPT1 pipeline...".bright_cyan());
    
    // Initialize tokenizer
    println!("{}", "Loading tokenizer...".bright_yellow());
    let tokenizer = GPT2Tokenizer::new()
        .context("Failed to initialize GPT2 tokenizer - ensure you have internet access for first-time download")?;
    
    // Get timeouts from config
    let timeouts = config.ollama_timeouts()
        .clone();
    
    // Data generation
    println!("\n{}", format!("[Step 1/7] Generating {} initial tokens", config.initial_tokens).bright_green());
    
    let mut generator = DataGenerator::new(prompt, tokenizer.clone(), "http://localhost:11434", timeouts.clone())?;
    
    // Check Ollama if using it
    if let DataGenerationMethod::Ollama { ref gen_model, .. } = generation_method {
        generator.check_ollama(gen_model).await
            .context("Failed to check Ollama availability")?;
    }
    
    let raw_samples = generator.generate_samples(config.initial_tokens, &generation_method).await
        .context("Failed to generate initial data samples")?;
    let domain = generator.get_domain().to_string();
    
    // Data refinement
    println!("\n{}", format!("[Step 2/7] Refining to {} tokens", config.final_tokens).bright_green());
    
    let refiner = DataRefiner::new(prompt, tokenizer.clone(), "http://localhost:11434", timeouts);
    
    let refined_samples = match &generation_method {
        DataGenerationMethod::Ollama { refine_model, .. } => {
            refiner.refine_samples(raw_samples.clone(), config.final_tokens, true, refine_model).await
                .context("Failed to refine data samples with Ollama")?
        }
        DataGenerationMethod::Template => {
            refiner.refine_samples(raw_samples.clone(), config.final_tokens, false, "").await
                .context("Failed to refine data samples")?
        }
    };
    
    // Create dataset
    println!("\n{}", "[Step 3/7] Creating dataset".bright_green());
    let mut config_mut = config.clone();
    let device = config_mut.init_device()
        .context("Failed to initialize compute device")?
        .clone();
    let dataset = TextDataset::new(&refined_samples, &tokenizer, config.max_length, &device)
        .context("Failed to create text dataset")?;
    
    println!("Dataset created with {} samples", dataset.len());
    
    // Initialize model with VarMap
    println!("\n{}", "[Step 4/7] Initializing model".bright_green());
    let varmap = Arc::new(VarMap::new());
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = GPT::new(config, vb)
        .context("Failed to initialize GPT model")?;
    
    // Initialize weights properly
    initialize_model_weights(&varmap)?;
    
    println!("Model initialized with {:.2}M parameters", config.param_count_millions());
    
    // Train model
    println!("\n{}", "[Step 5/7] Training model".bright_green());
    let final_loss = train_model(&model, &dataset, config, &varmap)
        .context("Failed during model training")?;
    
    // Test generation
    println!("\n{}", "[Step 6/7] Testing generation".bright_green());
    test_generation(&model, &tokenizer, &device)
        .context("Failed during generation test")?;
    
    // Save everything
    println!("\n{}", "[Step 7/7] Saving model and dataset".bright_green());
    
    // Create folder name
    let timestamp = Local::now().format("%Y%m%d_%H%M%S");
    let folder_name = format!("{}_{}", 
        prompt.chars().take(30).collect::<String>().replace(" ", "_").to_lowercase(),
        timestamp
    );
    let folder_path = output_dir.join(&folder_name);
    
    // Create directory
    std::fs::create_dir_all(&folder_path)
        .context("Failed to create output directory")?;
    
    // Save dataset
    let dataset_metadata = json::DatasetMetadata {
        total_samples: refined_samples.len(),
        total_tokens: refined_samples.iter()
            .map(|s| tokenizer.encode(s).unwrap_or_default().len())
            .sum(),
        average_tokens_per_sample: refined_samples.iter()
            .map(|s| tokenizer.encode(s).unwrap_or_default().len())
            .sum::<usize>() as f64 / refined_samples.len() as f64,
        creation_date: Local::now().format("%Y-%m-%d %H:%M:%S").to_string(),
        tokenizer: "gpt2".to_string(),
        prompt: prompt.to_string(),
        domain: domain.clone(),
        generation_method: match &generation_method {
            DataGenerationMethod::Template => "template".to_string(),
            DataGenerationMethod::Ollama { .. } => "ollama".to_string(),
        },
        generation_model: match &generation_method {
            DataGenerationMethod::Ollama { gen_model, .. } => Some(gen_model.clone()),
            _ => None,
        },
        refinement_model: match &generation_method {
            DataGenerationMethod::Ollama { refine_model, .. } => Some(refine_model.clone()),
            _ => None,
        },
        ollama_used: matches!(generation_method, DataGenerationMethod::Ollama { .. }),
        quality_threshold: "refined".to_string(),
    };
    
    json::save_dataset(folder_path.join("dataset.json"), &refined_samples, &tokenizer, dataset_metadata)
        .context("Failed to save dataset as JSON")?;
    json::save_dataset_text(folder_path.join("dataset.txt"), &refined_samples)
        .context("Failed to save dataset as text")?;
    
    // Save model using VarMap's save method
    let model_path = folder_path.join("model.safetensors");
    varmap.save(&model_path)
        .context("Failed to save model weights")?;
    
    // Save tokenizer
    tokenizer.save_pretrained(&folder_path)
        .context("Failed to save tokenizer")?;
    
    // Save config
    let config_path = folder_path.join("config.json");
    let config_json = serde_json::to_string_pretty(config)
        .context("Failed to serialize config to JSON")?;
    std::fs::write(&config_path, config_json)
        .context("Failed to write config file")?;
    
    // Save generation info
    let data_stats = DataStats {
        samples_generated: raw_samples.len(),
        samples_refined: refined_samples.len(),
        initial_tokens: config.initial_tokens,
        final_tokens: config.final_tokens,
        generation_method: match &generation_method {
            DataGenerationMethod::Template => "template".to_string(),
            DataGenerationMethod::Ollama { .. } => "ollama".to_string(),
        },
        generation_model: match &generation_method {
            DataGenerationMethod::Ollama { gen_model, .. } => Some(gen_model.clone()),
            _ => None,
        },
        refinement_model: match &generation_method {
            DataGenerationMethod::Ollama { refine_model, .. } => Some(refine_model.clone()),
            _ => None,
        },
    };
    
    let (gen_method, gen_model, ref_model) = match &generation_method {
        DataGenerationMethod::Template => ("template", None, None),
        DataGenerationMethod::Ollama { gen_model, refine_model } => 
            ("ollama", Some(gen_model.as_str()), Some(refine_model.as_str())),
    };
    
    json::save_generation_info(
        folder_path.join("generation_info.json"),
        prompt,
        &domain,
        None,
        &config.model_size.to_string(),
        config,
        data_stats,
        gen_method,
        gen_model,
        ref_model,
        matches!(generation_method, DataGenerationMethod::Ollama { .. }),
        start_time.elapsed().as_secs_f64(),
    ).context("Failed to save generation info")?;
    
    let result = TrainingResult {
        model_path: folder_path,
        domain,
        final_loss,
        training_time_seconds: start_time.elapsed().as_secs_f64(),
    };
    
    Ok(result)
}

/// Initialize model weights with proper values
fn initialize_model_weights(_varmap: &VarMap) -> Result<()> {
    // Note: In the current version of Candle, weights are automatically initialized
    // when creating layers with VarBuilder. The initialization strategy depends on
    // the layer type (e.g., Xavier/Kaiming for linear layers).
    // 
    // If custom initialization is needed, it should be done at layer creation time
    // using VarBuilder's get_with_hints method.
    
    Ok(())
}

/// Train the model using Candle's training pattern
pub fn train_model(
    model: &GPT,
    dataset: &TextDataset,
    config: &Config,
    varmap: &Arc<VarMap>,
) -> Result<f32> {
    // Initialize AdamW optimizer with all model parameters
    let params = ParamsAdamW {
        lr: config.learning_rate as f64,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
        weight_decay: 0.01,
    };
    
    let mut optimizer = AdamW::new(varmap.all_vars(), params)
        .context("Failed to create AdamW optimizer")?;
    
    let mut dataloader = dataset.dataloader(config.batch_size, true);
    
    let total_steps = config.num_epochs * dataloader.len();
    let pb = ProgressBar::new(total_steps as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} steps | Loss: {msg} | ETA: {eta}")
            .unwrap()
            .progress_chars("#>-"),
    );
    
    let mut step = 0;
    let mut recent_losses = Vec::new();
    let mut final_loss = 0.0;
    
    for epoch in 0..config.num_epochs {
        let mut epoch_loss = 0.0;
        let mut num_batches = 0;
        
        dataloader.reset(true);
        
        for batch_result in &mut dataloader {
            let (x, y) = batch_result?;
            
            // Forward pass
            let (_logits, loss) = model.forward(&x, Some(&y), true)?;
            
            if let Some(loss_tensor) = loss {
                // Backward pass and parameter update - key Candle pattern
                optimizer.backward_step(&loss_tensor)?;
                
                // Track loss
                let loss_val = loss_tensor.to_scalar::<f32>()?;
                epoch_loss += loss_val;
                num_batches += 1;
                
                // Keep track of recent losses for smoothed display
                recent_losses.push(loss_val);
                if recent_losses.len() > 10 {
                    recent_losses.remove(0);
                }
                let smoothed_loss = recent_losses.iter().sum::<f32>() / recent_losses.len() as f32;
                
                pb.set_position(step as u64);
                pb.set_message(format!("{:.4}", smoothed_loss));
                
                final_loss = smoothed_loss;
            }
            
            step += 1;
        }
        
        let avg_loss = epoch_loss / num_batches as f32;
        
        // Display epoch summary every 5 epochs or at start/end
        if (epoch + 1) % 5 == 0 || epoch == 0 || epoch == config.num_epochs - 1 {
            pb.println(format!(
                "Epoch {}/{} - Average loss: {:.4} - Learning rate: {:.2e}",
                epoch + 1,
                config.num_epochs,
                avg_loss,
                config.learning_rate
            ));
        }
        
        // Optional: Learning rate scheduling could be added here
        // For example: optimizer.set_learning_rate(new_lr);
    }
    
    pb.finish_with_message(format!("Training complete - Final loss: {:.4}", final_loss));
    
    Ok(final_loss)
}

/// Test the model with some generations
fn test_generation(
    model: &GPT,
    tokenizer: &GPT2Tokenizer,
    device: &Device,
) -> Result<()> {
    let test_prompts = vec![
        "",
        "The",
        "Today",
    ];
    
    println!("\nTest generations:");
    for (i, prompt) in test_prompts.iter().enumerate() {
        let start_ids = if prompt.is_empty() {
            vec![tokenizer.bos_token_id()]
        } else {
            tokenizer.encode(prompt)?
        };
        
        // Convert to i64 to match the generate function's expectations
        let start_ids_i64: Vec<i64> = start_ids.iter().map(|&x| x as i64).collect();
        
        let input = Tensor::new(start_ids_i64.as_slice(), device)?.unsqueeze(0)?;
        let generated = model.generate(&input, 30, 0.8, Some(40))?;
        
        let generated_ids: Vec<u32> = generated.squeeze(0)?.to_vec1::<i64>()?
            .into_iter()
            .map(|x| x as u32)
            .collect();
        
        let text = tokenizer.decode(&generated_ids, true)?;
        println!("Test {}: {}", i + 1, text);
    }
    
    Ok(())
}
