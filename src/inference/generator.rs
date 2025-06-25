use anyhow::Result;
use candle_core::{Device, IndexOp, Tensor};
use std::path::Path;

use crate::config::Config;
use crate::io::safetensors;
use crate::model::GPT;
use crate::tokenizer::GPT2Tokenizer;

/// Text generator for inference
pub struct TextGenerator {
    model: GPT,
    tokenizer: GPT2Tokenizer,
    config: Config,
    device: Device,
}

impl TextGenerator {
    /// Load a text generator from a model folder
    pub fn load<P: AsRef<Path>>(model_path: P) -> Result<Self> {
        let device = Device::cuda_if_available(0)?;
        
        // Use the improved loading function
        let (model, tokenizer, config, _varmap) = safetensors::load_model_folder(model_path, &device)?;
        
        Ok(Self {
            model,
            tokenizer,
            config,
            device,
        })
    }
    
    /// Load using memory-mapped files for better performance
    pub fn load_mmaped<P: AsRef<Path>>(model_path: P) -> Result<Self> {
        let device = Device::cuda_if_available(0)?;
        let model_path = model_path.as_ref();
        
        // Load tokenizer
        let tokenizer = GPT2Tokenizer::from_pretrained(model_path)?;
        
        // Load config
        let config_path = model_path.join("config.json");
        let config_json = std::fs::read_to_string(&config_path)?;
        let mut config: Config = serde_json::from_str(&config_json)?;
        config.device = Some(device.clone());
        
        // Load model using memory mapping
        let model_file = model_path.join("model.safetensors");
        let (model, _) = safetensors::load_model_mmaped(&model_file, &device)?;
        
        Ok(Self {
            model,
            tokenizer,
            config,
            device,
        })
    }
    
    /// Generate text from a prompt
    pub fn generate(
        &self,
        prompt: &str,
        max_length: usize,
        temperature: f64,
        top_k: usize,
    ) -> Result<String> {
        // Encode the prompt
        let input_ids = if prompt.is_empty() {
            vec![self.tokenizer.bos_token_id()]
        } else {
            self.tokenizer.encode(prompt)?
        };
        
        // Convert to i64 for consistency with model's generate method
        let input_ids_i64: Vec<i64> = input_ids.iter().map(|&x| x as i64).collect();
        
        // Convert to tensor
        let input = Tensor::new(input_ids_i64.as_slice(), &self.device)?.unsqueeze(0)?;
        
        // Generate
        let generated = self.model.generate(&input, max_length, temperature, Some(top_k))?;
        
        // Decode
        let generated_ids: Vec<u32> = generated.squeeze(0)?.to_vec1::<i64>()?
            .into_iter()
            .map(|x| x as u32)
            .collect();
        
        let text = self.tokenizer.decode(&generated_ids, true)?;
        
        Ok(text)
    }
    
    /// Generate multiple samples
    pub fn generate_batch(
        &self,
        prompts: &[String],
        max_length: usize,
        temperature: f64,
        top_k: usize,
    ) -> Result<Vec<String>> {
        let mut results = Vec::new();
        
        for prompt in prompts {
            let generated = self.generate(prompt, max_length, temperature, top_k)?;
            results.push(generated);
        }
        
        Ok(results)
    }
    
    /// Generate with custom parameters
    pub fn generate_custom(
        &self,
        prompt: &str,
        max_length: usize,
        temperature: f64,
        top_k: Option<usize>,
        top_p: Option<f32>,
        repetition_penalty: Option<f32>,
    ) -> Result<String> {
        // For now, we only support top_k
        // top_p and repetition_penalty could be added as extensions
        if top_p.is_some() || repetition_penalty.is_some() {
            log::warn!("top_p and repetition_penalty are not yet implemented, using top_k only");
        }
        
        let top_k = top_k.unwrap_or(40);
        self.generate(prompt, max_length, temperature, top_k)
    }
    
    /// Stream generation token by token
    pub fn generate_stream<F>(
        &self,
        prompt: &str,
        max_length: usize,
        temperature: f64,
        top_k: usize,
        mut callback: F,
    ) -> Result<String>
    where
        F: FnMut(&str) -> Result<()>,
    {
        // Encode the prompt
        let input_ids = if prompt.is_empty() {
            vec![self.tokenizer.bos_token_id()]
        } else {
            self.tokenizer.encode(prompt)?
        };
        
        // Convert to i64 for consistency
        let input_ids_i64: Vec<i64> = input_ids.iter().map(|&x| x as i64).collect();
        
        let mut idx = Tensor::new(input_ids_i64.as_slice(), &self.device)?.unsqueeze(0)?;
        let mut generated_tokens = Vec::new();
        
        for _ in 0..max_length {
            // Generate one token at a time
            let next_token_tensor = self.model.generate(&idx, 1, temperature, Some(top_k))?;
            
            // Extract the new token
            let new_token_id = next_token_tensor
                .i((.., next_token_tensor.dims()[1] - 1))?
                .squeeze(0)?
                .to_scalar::<i64>()? as u32;
            
            generated_tokens.push(new_token_id);
            
            // Decode just the new token
            let token_text = self.tokenizer.decode(&[new_token_id], false)?;
            callback(&token_text)?;
            
            // Update idx for next iteration
            idx = next_token_tensor;
            
            // Check for EOS token
            if new_token_id == self.tokenizer.eos_token_id() {
                break;
            }
        }
        
        // Return the full generated text
        // Convert original input_ids to u32 for concatenation
        let input_ids_u32: Vec<u32> = input_ids_i64.into_iter()
            .map(|x| x as u32)
            .collect();
        
        let full_ids: Vec<u32> = input_ids_u32.into_iter()
            .chain(generated_tokens.into_iter())
            .collect();
        
        self.tokenizer.decode(&full_ids, true)
    }
    
    /// Get model information
    pub fn model_info(&self) -> String {
        format!(
            "Model size: {}\nVocab size: {}\nEmbedding dim: {}\nLayers: {}\nHeads: {}\nParameters: {:.2}M\nDevice: {:?}",
            self.config.model_size,
            self.config.vocab_size,
            self.config.n_embd,
            self.config.n_layer,
            self.config.n_head,
            self.config.param_count_millions(),
            self.device
        )
    }
    
    /// Get the device being used
    pub fn device(&self) -> &Device {
        &self.device
    }
    
    /// Get the tokenizer
    pub fn tokenizer(&self) -> &GPT2Tokenizer {
        &self.tokenizer
    }
}
