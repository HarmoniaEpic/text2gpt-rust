use candle_core::{IndexOp, Result, Tensor};
use candle_nn::{embedding, layer_norm, ops, Embedding, LayerNorm, Module, VarBuilder};
use rand::{thread_rng, Rng};

use super::block::Block;
use crate::config::Config;

/// GPT Language Model
pub struct GPT {
    wte: Embedding,       // token embeddings
    wpe: Embedding,       // position embeddings
    drop: f32,           // dropout probability
    h: Vec<Block>,       // transformer blocks
    ln_f: LayerNorm,     // final layer norm
    n_positions: usize,
    vocab_size: usize,
    n_embd: usize,
}

impl GPT {
    /// Create a new GPT model
    pub fn new(config: &Config, vb: VarBuilder) -> Result<Self> {
        let wte = embedding(config.vocab_size, config.n_embd, vb.pp("wte"))?;
        let wpe = embedding(config.n_positions, config.n_embd, vb.pp("wpe"))?;
        
        let mut h = Vec::new();
        for i in 0..config.n_layer {
            h.push(Block::new(
                config.n_embd,
                config.n_head,
                config.n_positions,
                vb.pp(&format!("h.{}", i)),
            )?);
        }
        
        let ln_f = layer_norm(config.n_embd, 1e-5, vb.pp("ln_f"))?;
        
        // Note: lm_head is not created separately as we use weight sharing with wte
        
        Ok(Self {
            wte,
            wpe,
            drop: config.dropout,
            h,
            ln_f,
            n_positions: config.n_positions,
            vocab_size: config.vocab_size,
            n_embd: config.n_embd,
        })
    }
    
    /// Forward pass
    pub fn forward(&self, idx: &Tensor, targets: Option<&Tensor>, training: bool) -> Result<(Tensor, Option<Tensor>)> {
        let device = idx.device();
        let (b, t) = idx.dims2()?;
        assert!(t <= self.n_positions, "Sequence length {} exceeds maximum {}", t, self.n_positions);
        
        // Token embeddings
        let tok_emb = self.wte.forward(idx)?;
        
        // Position embeddings
        let pos = Tensor::arange(0, t as i64, device)?.unsqueeze(0)?;
        let pos_emb = self.wpe.forward(&pos)?;
        
        // Combine embeddings
        let mut x = (tok_emb + pos_emb)?;
        
        // Dropout (if training)
        if training && self.drop > 0.0 {
            x = ops::dropout(&x, self.drop)?;
        }
        
        // Pass through transformer blocks
        for block in &self.h {
            x = block.forward(&x, training)?;
        }
        
        // Final layer norm
        x = self.ln_f.forward(&x)?;
        
        // Language modeling head with weight sharing
        // GPT-1/GPT-2 architecture uses weight sharing between input embeddings (wte) 
        // and output projection layer. This reduces parameters by vocab_size * n_embd.
        
        // Get the embedding weight tensor and compute logits
        let wte_weight = self.wte.embeddings();
        let logits = x.matmul(&wte_weight.t()?)?;
        
        // Calculate loss if targets provided
        let loss = if let Some(targets) = targets {
            let logits_view = logits.reshape((b * t, self.vocab_size))?;
            let targets_view = targets.reshape((b * t,))?;
            
            let loss = candle_nn::loss::cross_entropy(&logits_view, &targets_view)?;
            Some(loss)
        } else {
            None
        };
        
        Ok((logits, loss))
    }
    
    /// Generate text autoregressively
    pub fn generate(
        &self,
        idx: &Tensor,
        max_new_tokens: usize,
        temperature: f64,
        top_k: Option<usize>,
    ) -> Result<Tensor> {
        let mut idx = idx.clone();
        let mut rng = thread_rng();
        
        for _ in 0..max_new_tokens {
            // Crop idx to last n_positions tokens if necessary
            let seq_len = idx.dims()[1];
            let idx_cond = if seq_len <= self.n_positions {
                idx.clone()
            } else {
                idx.i((.., seq_len - self.n_positions..))?
            };
            
            // Forward pass (no training mode for generation)
            let (logits, _) = self.forward(&idx_cond, None, false)?;
            
            // Get logits for last position
            let logits = logits.i((.., logits.dims()[1] - 1, ..))?;
            
            // Apply temperature
            let logits = if (temperature - 1.0).abs() > 1e-6 {
                (logits / temperature)?
            } else {
                logits
            };
            
            // Apply top-k filtering if specified
            let logits = if let Some(k) = top_k {
                let device = idx.device();
                apply_top_k(&logits, k, device)?
            } else {
                logits
            };
            
            // Convert to probabilities
            let probs = ops::softmax_last_dim(&logits)?;
            
            // Sample from the distribution
            let idx_next = sample_from_probs(&probs, &mut rng)?;
            
            // Append sampled index to the running sequence
            idx = Tensor::cat(&[&idx, &idx_next.unsqueeze(0)?], 1)?;
        }
        
        Ok(idx)
    }
}

/// Apply top-k filtering to logits
fn apply_top_k(logits: &Tensor, k: usize, device: &candle_core::Device) -> Result<Tensor> {
    let vocab_size = logits.dims()[logits.dims().len() - 1];
    if k >= vocab_size {
        return Ok(logits.clone());
    }
    
    // Get logits as vector
    let logits_vec: Vec<f32> = logits.squeeze(0)?.to_vec1()?;
    
    // Create indexed pairs and sort
    let mut indexed: Vec<(usize, f32)> = logits_vec.iter()
        .enumerate()
        .map(|(i, &v)| (i, v))
        .collect();
    
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    
    // Get threshold (k-th largest value)
    let threshold = indexed.get(k.saturating_sub(1))
        .map(|(_, v)| *v)
        .unwrap_or(f32::NEG_INFINITY);
    
    // Create filtered logits
    let result: Vec<f32> = logits_vec.iter()
        .map(|&val| if val >= threshold { val } else { f32::NEG_INFINITY })
        .collect();
    
    // Convert back to tensor
    Tensor::from_vec(result, logits.shape(), device)
}

/// Sample an index from probability distribution
fn sample_from_probs(probs: &Tensor, rng: &mut impl Rng) -> Result<Tensor> {
    let device = probs.device();
    
    // Convert to vector for sampling
    let probs_vec: Vec<f32> = probs.squeeze(0)?.to_vec1()?;
    
    // Generate random value
    let sample: f32 = rng.gen::<f32>();
    
    // Build cumulative distribution and sample
    let mut cumsum = 0.0;
    let mut idx = 0;
    
    for (i, &p) in probs_vec.iter().enumerate() {
        cumsum += p;
        if sample <= cumsum {
            idx = i;
            break;
        }
    }
    
    Tensor::new(&[idx as i64], device)
}
