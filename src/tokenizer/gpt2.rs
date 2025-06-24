use anyhow::{anyhow, Result};
use std::path::{Path, PathBuf};
use tokenizers::tokenizer::Tokenizer;

/// GPT2 Tokenizer wrapper
#[derive(Clone)]
pub struct GPT2Tokenizer {
    tokenizer: Tokenizer,
    vocab_size: usize,
}

impl GPT2Tokenizer {
    /// Create a new GPT2 tokenizer
    pub fn new() -> Result<Self> {
        // Try to load from HuggingFace
        let tokenizer = match Tokenizer::from_pretrained("gpt2", None) {
            Ok(t) => t,
            Err(_) => {
                // Try to load from local cache
                let cache_dir = dirs::cache_dir()
                    .ok_or_else(|| anyhow!("Could not find cache directory"))?;
                let tokenizer_path = cache_dir.join("huggingface").join("tokenizers").join("gpt2.json");
                
                if tokenizer_path.exists() {
                    Tokenizer::from_file(&tokenizer_path)
                        .map_err(|e| anyhow!("Failed to load tokenizer from cache: {}", e))?
                } else {
                    return Err(anyhow!(
                        "Could not load GPT2 tokenizer. Please ensure you have internet connection \
                         for first-time download, or place the tokenizer file at: {}",
                        tokenizer_path.display()
                    ));
                }
            }
        };
        
        let vocab_size = tokenizer.get_vocab_size(true);
        
        Ok(Self {
            tokenizer,
            vocab_size,
        })
    }
    
    /// Load tokenizer from a directory
    pub fn from_pretrained<P: AsRef<Path>>(path: P) -> Result<Self> {
        let tokenizer_path = path.as_ref().join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow!("Failed to load tokenizer from {}: {}", tokenizer_path.display(), e))?;
        
        let vocab_size = tokenizer.get_vocab_size(true);
        
        Ok(Self {
            tokenizer,
            vocab_size,
        })
    }
    
    /// Save tokenizer to a directory
    pub fn save_pretrained<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let save_path = path.as_ref().join("tokenizer.json");
        self.tokenizer.save(&save_path, false)
            .map_err(|e| anyhow!("Failed to save tokenizer to {}: {}", save_path.display(), e))?;
        Ok(())
    }
    
    /// Encode text to token ids
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let encoding = self.tokenizer.encode(text, false)
            .map_err(|e| anyhow!("Failed to encode text: {}", e))?;
        Ok(encoding.get_ids().to_vec())
    }
    
    /// Decode token ids to text
    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        self.tokenizer.decode(ids, skip_special_tokens)
            .map_err(|e| anyhow!("Failed to decode tokens: {}", e))
    }
    
    /// Get vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }
    
    /// Get the pad token id
    pub fn pad_token_id(&self) -> u32 {
        // GPT2 uses EOS token as pad token
        50256
    }
    
    /// Get the BOS token id
    pub fn bos_token_id(&self) -> u32 {
        50256
    }
    
    /// Get the EOS token id
    pub fn eos_token_id(&self) -> u32 {
        50256
    }
}

impl Default for GPT2Tokenizer {
    fn default() -> Self {
        Self::new().expect("Failed to create default GPT2 tokenizer")
    }
}