use anyhow::Result;
use candle_core::{Device, Tensor};
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
        let (model, tokenizer, config) = safetensors::load_model_folder(model_path, &device)?;
        
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
        
        // Convert to tensor
        let input = Tensor::new(input_ids.as_slice(), &self.device)?.unsqueeze(0)?;
        
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
    
    /// Get model information
    pub fn model_info(&self) -> String {
        format!(
            "Model size: {}\nVocab size: {}\nEmbedding dim: {}\nLayers: {}\nHeads: {}\nParameters: {:.2}M",
            self.config.model_size,
            self.config.vocab_size,
            self.config.n_embd,
            self.config.n_layer,
            self.config.n_head,
            self.config.param_count_millions()
        )
    }
}