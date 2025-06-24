use anyhow::Result;
use indicatif::{ProgressBar, ProgressStyle};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::time::Duration;

use crate::tokenizer::GPT2Tokenizer;

/// Data refiner that evaluates and selects high-quality samples
pub struct DataRefiner {
    prompt: String,
    tokenizer: GPT2Tokenizer,
    ollama_host: String,
}

impl DataRefiner {
    pub fn new(prompt: &str, tokenizer: GPT2Tokenizer, ollama_host: &str) -> Self {
        Self {
            prompt: prompt.to_string(),
            tokenizer,
            ollama_host: ollama_host.to_string(),
        }
    }
    
    /// Refine samples to target token count
    pub async fn refine_samples(
        &self,
        samples: Vec<String>,
        target_tokens: usize,
        use_ollama: bool,
        model: &str,
    ) -> Result<Vec<String>> {
        log::info!("Refining {} samples to approximately {} tokens", samples.len(), target_tokens);
        
        let pb = ProgressBar::new(samples.len() as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} samples evaluated")
                .unwrap()
                .progress_chars("#>-"),
        );
        
        // Score all samples
        let mut scored_samples = Vec::new();
        
        for (i, sample) in samples.iter().enumerate() {
            let score = if use_ollama {
                self.evaluate_with_ollama(sample, model).await.unwrap_or_else(|_| {
                    self.calculate_template_score(sample)
                })
            } else {
                self.calculate_template_score(sample)
            };
            
            scored_samples.push((score, sample.clone()));
            pb.set_position((i + 1) as u64);
        }
        
        pb.finish_with_message("Evaluation complete");
        
        // Sort by score (highest first)
        scored_samples.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        
        // Select top samples up to target token count
        let mut refined_samples = Vec::new();
        let mut current_tokens = 0;
        
        // Use reference to avoid move
        for (_score, sample) in &scored_samples {
            let sample_tokens = self.tokenizer.encode(sample)?.len();
            
            if current_tokens + sample_tokens <= target_tokens {
                refined_samples.push(sample.clone());
                current_tokens += sample_tokens;
            }
            
            if current_tokens >= (target_tokens as f64 * 0.95) as usize {
                break;
            }
        }
        
        let avg_score = if refined_samples.is_empty() {
            0.0
        } else {
            scored_samples.iter()
                .take(refined_samples.len())
                .map(|(score, _)| score)
                .sum::<f32>() / refined_samples.len() as f32
        };
        
        log::info!(
            "Refined {} samples -> {} samples, {} tokens, avg quality: {:.2}",
            samples.len(),
            refined_samples.len(),
            current_tokens,
            avg_score
        );
        
        Ok(refined_samples)
    }
    
    /// Evaluate sample quality using Ollama
    async fn evaluate_with_ollama(&self, sample: &str, model: &str) -> Result<f32> {
        let evaluation_prompt = format!(
            "Rate how well this text matches the goal '{}' from 0-10 (0=worst, 10=best). \
             Reply with only a number:\n\nText:\n{}\n\nRating:",
            self.prompt, sample
        );
        
        #[derive(Serialize)]
        struct GenerateRequest {
            model: String,
            prompt: String,
            stream: bool,
            options: GenerateOptions,
        }
        
        #[derive(Serialize)]
        struct GenerateOptions {
            num_predict: u32,
            temperature: f64,
        }
        
        #[derive(Deserialize)]
        struct GenerateResponse {
            response: String,
        }
        
        let request = GenerateRequest {
            model: model.to_string(),
            prompt: evaluation_prompt,
            stream: false,
            options: GenerateOptions {
                num_predict: 10,
                temperature: 0.1,
            },
        };
        
        let client = reqwest::Client::new();
        let url = format!("{}/api/generate", self.ollama_host);
        
        match client
            .post(&url)
            .json(&request)
            .timeout(Duration::from_secs(10))
            .send()
            .await
        {
            Ok(response) if response.status().is_success() => {
                match response.json::<GenerateResponse>().await {
                    Ok(gen_response) => {
                        let score_text = gen_response.response.trim();
                        if let Ok(score) = score_text.split_whitespace().next().unwrap_or("5").parse::<f32>() {
                            return Ok((score / 10.0).clamp(0.0, 1.0));
                        }
                    }
                    Err(e) => {
                        log::debug!("Failed to parse Ollama evaluation response: {}", e);
                    }
                }
            }
            Err(e) => {
                log::debug!("Ollama evaluation request failed: {}. Using template scoring.", e);
            }
            _ => {
                log::debug!("Ollama returned non-success status. Using template scoring.");
            }
        }
        
        // Fall back to template scoring
        Ok(self.calculate_template_score(sample))
    }
    
    /// Calculate quality score using templates
    fn calculate_template_score(&self, sample: &str) -> f32 {
        let mut score = 0.0;
        
        // Length score
        let length = sample.len();
        if (20..=100).contains(&length) {
            score += 1.0;
        } else if (10..=150).contains(&length) {
            score += 0.5;
        }
        
        // Punctuation score
        if sample.matches('.').count() >= 1 {
            score += 0.5;
        }
        if sample.matches(',').count() >= 1 {
            score += 0.3;
        }
        
        // Relevance score (simple keyword matching)
        let prompt_words: HashSet<&str> = self.prompt.split_whitespace().collect();
        let sample_words: HashSet<&str> = sample.split_whitespace().collect();
        let overlap = prompt_words.intersection(&sample_words).count();
        score += overlap as f32 * 0.2;
        
        // Add small random variation for diversity
        let mut rng = rand::thread_rng();
        score += rng.gen_range(0.0..0.3);
        
        score.min(1.0)
    }
}
