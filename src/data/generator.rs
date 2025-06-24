use anyhow::{anyhow, Context, Result};
use indicatif::{ProgressBar, ProgressStyle};
use rand::seq::SliceRandom;
use rand::thread_rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

use crate::tokenizer::GPT2Tokenizer;

/// Data generation method
#[derive(Debug, Clone)]
pub enum DataGenerationMethod {
    Template,
    Ollama {
        gen_model: String,
        refine_model: String,
    },
}

/// Data generator that creates training samples
pub struct DataGenerator {
    prompt: String,
    tokenizer: GPT2Tokenizer,
    domain: String,
    ollama_host: String,
    ollama_available: bool,
}

impl DataGenerator {
    pub fn new(prompt: &str, tokenizer: GPT2Tokenizer, ollama_host: &str) -> Result<Self> {
        let domain = extract_domain(prompt);
        
        Ok(Self {
            prompt: prompt.to_string(),
            tokenizer,
            domain,
            ollama_host: ollama_host.to_string(),
            ollama_available: false,
        })
    }
    
    /// Check Ollama connection and model availability
    pub async fn check_ollama(&mut self, model: &str) -> Result<bool> {
        #[derive(Deserialize)]
        struct OllamaModels {
            models: Vec<OllamaModel>,
        }
        
        #[derive(Deserialize)]
        struct OllamaModel {
            name: String,
        }
        
        let client = reqwest::Client::new();
        let url = format!("{}/api/tags", self.ollama_host);
        
        match client.get(&url).timeout(Duration::from_secs(5)).send().await {
            Ok(response) if response.status().is_success() => {
                let models: OllamaModels = response.json().await
                    .context("Failed to parse Ollama models response")?;
                let model_names: Vec<String> = models.models.iter().map(|m| m.name.clone()).collect();
                
                self.ollama_available = model_names.contains(&model.to_string());
                
                if !self.ollama_available {
                    log::warn!("Model {} not found in Ollama. Available models: {:?}", model, model_names);
                }
                
                Ok(self.ollama_available)
            }
            _ => {
                log::warn!("Failed to connect to Ollama at {}", self.ollama_host);
                self.ollama_available = false;
                Ok(false)
            }
        }
    }
    
    /// Generate samples using the specified method
    pub async fn generate_samples(
        &self,
        num_tokens: usize,
        method: &DataGenerationMethod,
    ) -> Result<Vec<String>> {
        match method {
            DataGenerationMethod::Template => {
                log::info!("Using template-based generation for domain: {}", self.domain);
                self.generate_template_samples(num_tokens)
            }
            DataGenerationMethod::Ollama { gen_model, .. } => {
                if self.ollama_available {
                    log::info!("Using Ollama model {} for generation", gen_model);
                    self.generate_ollama_samples(num_tokens, gen_model).await
                } else {
                    log::warn!("Ollama not available, falling back to template generation");
                    self.generate_template_samples(num_tokens)
                }
            }
        }
    }
    
    /// Generate samples using templates
    fn generate_template_samples(&self, num_tokens: usize) -> Result<Vec<String>> {
        let mut samples = Vec::new();
        let mut current_tokens = 0;
        let mut rng = thread_rng();
        
        let pb = ProgressBar::new(num_tokens as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} tokens")
                .unwrap()
                .progress_chars("#>-"),
        );
        
        while current_tokens < num_tokens {
            let sample = match self.domain.as_str() {
                "cooking" => generate_cooking_sample(&mut rng),
                "poetry" => generate_poetry_sample(&mut rng),
                "technical" => generate_technical_sample(&mut rng),
                _ => generate_general_sample(&mut rng),
            };
            
            let tokens = self.tokenizer.encode(&sample)
                .context("Failed to encode template-generated sample")?;
            current_tokens += tokens.len();
            samples.push(sample);
            
            pb.set_position(current_tokens.min(num_tokens) as u64);
        }
        
        pb.finish_with_message("Template generation complete");
        
        log::info!("Generated {} samples with {} tokens", samples.len(), current_tokens);
        Ok(samples)
    }
    
    /// Generate samples using Ollama
    async fn generate_ollama_samples(&self, num_tokens: usize, model: &str) -> Result<Vec<String>> {
        let mut samples = Vec::new();
        let mut current_tokens = 0;
        
        let pb = ProgressBar::new(num_tokens as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} tokens")
                .unwrap()
                .progress_chars("#>-"),
        );
        
        let client = reqwest::Client::new();
        let prompts = create_generation_prompts(&self.prompt, &self.domain);
        let mut prompt_idx = 0;
        
        while current_tokens < num_tokens {
            let current_prompt = &prompts[prompt_idx % prompts.len()];
            
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
                top_p: f64,
            }
            
            #[derive(Deserialize)]
            struct GenerateResponse {
                response: String,
            }
            
            let request = GenerateRequest {
                model: model.to_string(),
                prompt: current_prompt.clone(),
                stream: false,
                options: GenerateOptions {
                    num_predict: 150,
                    temperature: 0.8,
                    top_p: 0.9,
                },
            };
            
            let url = format!("{}/api/generate", self.ollama_host);
            
            match client
                .post(&url)
                .json(&request)
                .timeout(Duration::from_secs(30))
                .send()
                .await
            {
                Ok(response) if response.status().is_success() => {
                    if let Ok(gen_response) = response.json::<GenerateResponse>().await {
                        let sample = gen_response.response.trim().to_string();
                        if sample.len() > 20 {
                            let tokens = self.tokenizer.encode(&sample)
                                .context("Failed to encode generated sample")?;
                            current_tokens += tokens.len();
                            samples.push(sample);
                            pb.set_position(current_tokens.min(num_tokens) as u64);
                        }
                    }
                }
                Err(e) => {
                    log::warn!("Ollama request failed: {}. Falling back to template generation.", e);
                    // Fall back to template generation for this sample
                    let sample = match self.domain.as_str() {
                        "cooking" => generate_cooking_sample(&mut thread_rng()),
                        "poetry" => generate_poetry_sample(&mut thread_rng()),
                        "technical" => generate_technical_sample(&mut thread_rng()),
                        _ => generate_general_sample(&mut thread_rng()),
                    };
                    
                    let tokens = self.tokenizer.encode(&sample)
                        .context("Failed to encode fallback sample")?;
                    current_tokens += tokens.len();
                    samples.push(sample);
                    pb.set_position(current_tokens.min(num_tokens) as u64);
                }
                _ => {
                    // Fall back to template generation for this sample
                    let sample = match self.domain.as_str() {
                        "cooking" => generate_cooking_sample(&mut thread_rng()),
                        "poetry" => generate_poetry_sample(&mut thread_rng()),
                        "technical" => generate_technical_sample(&mut thread_rng()),
                        _ => generate_general_sample(&mut thread_rng()),
                    };
                    
                    let tokens = self.tokenizer.encode(&sample)
                        .context("Failed to encode template fallback sample")?;
                    current_tokens += tokens.len();
                    samples.push(sample);
                    pb.set_position(current_tokens.min(num_tokens) as u64);
                }
            }
            
            prompt_idx += 1;
            
            // Add small delay to avoid overwhelming the API
            tokio::time::sleep(Duration::from_millis(500)).await;
        }
        
        pb.finish_with_message("Ollama generation complete");
        
        log::info!("Generated {} samples with {} tokens using Ollama", samples.len(), current_tokens);
        Ok(samples)
    }
    
    pub fn get_domain(&self) -> &str {
        &self.domain
    }
}

/// Extract domain from prompt
fn extract_domain(prompt: &str) -> String {
    let prompt_lower = prompt.to_lowercase();
    
    if prompt_lower.contains("cook") || prompt_lower.contains("recipe") || prompt_lower.contains("food") {
        "cooking".to_string()
    } else if prompt_lower.contains("poet") || prompt_lower.contains("creative") || prompt_lower.contains("story") {
        "poetry".to_string()
    } else if prompt_lower.contains("tech") || prompt_lower.contains("code") || prompt_lower.contains("program") {
        "technical".to_string()
    } else {
        "general".to_string()
    }
}

/// Create generation prompts based on domain
fn create_generation_prompts(base_prompt: &str, domain: &str) -> Vec<String> {
    let base_instruction = format!(
        "Generate text following this instruction: {}. Write a short, natural text:\n\n",
        base_prompt
    );
    
    match domain {
        "cooking" => vec![
            format!("{}Write a simple cooking recipe with ingredients and steps.", base_instruction),
            format!("{}Describe a seasonal dish using fresh ingredients.", base_instruction),
            format!("{}Share a cooking tip for beginners.", base_instruction),
            format!("{}Explain a professional cooking technique.", base_instruction),
            format!("{}Describe a healthy cooking approach.", base_instruction),
        ],
        "poetry" => vec![
            format!("{}Write a short poem about nature.", base_instruction),
            format!("{}Express an emotion through poetic language.", base_instruction),
            format!("{}Describe the changing seasons poetically.", base_instruction),
            format!("{}Write a reflective piece about life.", base_instruction),
            format!("{}Create a poem about love or friendship.", base_instruction),
        ],
        "technical" => vec![
            format!("{}Explain a basic programming concept.", base_instruction),
            format!("{}Describe a technical problem-solving approach.", base_instruction),
            format!("{}Share a software development best practice.", base_instruction),
            format!("{}Explain an important system design principle.", base_instruction),
            format!("{}Discuss a current technology trend.", base_instruction),
        ],
        _ => vec![
            format!("{}Write about a daily topic naturally.", base_instruction),
            format!("{}Share an interesting fact or knowledge.", base_instruction),
            format!("{}Describe a common situation or event.", base_instruction),
            format!("{}Express an opinion on a social topic.", base_instruction),
            format!("{}Give a piece of life advice.", base_instruction),
        ],
    }
}

// Template generation functions
fn generate_cooking_sample(rng: &mut impl rand::Rng) -> String {
    let dishes = ["pasta", "curry", "salad", "soup", "cake", "sushi", "tempura", "ramen"];
    let ingredients = ["tomato", "onion", "carrot", "potato", "meat", "fish", "egg", "cheese"];
    
    let dish = dishes.choose(rng).unwrap();
    let ing1 = ingredients.choose(rng).unwrap();
    let ing2 = ingredients.choose(rng).unwrap();
    
    let templates = [
        format!("How to make {}: First, prepare the {}. Then add {} and cook until done.", dish, ing1, ing2),
        format!("Simple {} recipe: Combine {} and {} for a delicious meal.", dish, ing1, ing2),
        format!("Today's dish is {}. Main ingredients are {} and {}.", dish, ing1, ing2),
        format!("Professional {} tip: The key is properly preparing the {}.", dish, ing1),
    ];
    
    templates.choose(rng).unwrap().clone()
}

fn generate_poetry_sample(rng: &mut impl rand::Rng) -> String {
    let themes = ["spring", "summer", "autumn", "winter", "love", "hope", "dreams", "time", "memory"];
    let emotions = ["joy", "sadness", "nostalgia", "anticipation", "peace", "longing", "wonder"];
    
    let theme = themes.choose(rng).unwrap();
    let emotion = emotions.choose(rng).unwrap();
    
    let templates = [
        format!("In {} I feel {}, walking quietly through moments.", theme, emotion),
        format!("Days of {} wrapped in {}, echoing in the heart.", theme, emotion),
        format!("{}, eternal companion. Living with {} beside me.", theme, emotion),
        format!("Unspeakable {} trembles in the midst of {}.", emotion, theme),
    ];
    
    templates.choose(rng).unwrap().clone()
}

fn generate_technical_sample(rng: &mut impl rand::Rng) -> String {
    let topics = ["Python", "machine learning", "API", "database", "algorithm", "security", "cloud"];
    let actions = ["implement", "optimize", "design", "analyze", "build", "debug", "test"];
    
    let topic = topics.choose(rng).unwrap();
    let action = actions.choose(rng).unwrap();
    
    let templates = [
        format!("To {} with {}, start with understanding the basics.", action, topic),
        format!("Best practices for {} in {}: focus on efficiency and clarity.", action, topic),
        format!("Effective {} {} methods: performance optimization tips.", topic, action),
        format!("Understanding {} for {}: importance and practical steps.", action, topic),
    ];
    
    templates.choose(rng).unwrap().clone()
}

fn generate_general_sample(rng: &mut impl rand::Rng) -> String {
    let subjects = ["today", "tomorrow", "life", "world", "we", "society", "future"];
    let adjectives = ["wonderful", "new", "important", "interesting", "exciting", "challenging"];
    
    let subject = subjects.choose(rng).unwrap();
    let adjective = adjectives.choose(rng).unwrap();
    
    let templates = [
        format!("{} is {}. That's my perspective.", subject, adjective),
        format!("Let's think about {} {}.", adjective, subject),
        format!("What can we do to make {} more {}?", subject, adjective),
    ];
    
    templates.choose(rng).unwrap().clone()
}