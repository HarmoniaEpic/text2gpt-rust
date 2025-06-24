use candle_core::{DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::{embedding, layer_norm, ops, Embedding, LayerNorm, Module, VarBuilder, VarMap};
use rand::{thread_rng, Rng};
use std::sync::Arc;

use super::block::Block;
use crate::config::Config;

/// GPT Language Model
pub struct GPT {
    wte: Embedding,       // token embeddings
    wpe: Embedding,       // position embeddings
    drop: f64,           // dropout probability
    h: Vec<Block>,       // transformer blocks
    ln_f: LayerNorm,     // final layer norm
    n_positions: usize,
    device: Device,
    varmap: Arc<VarMap>, // Reference to VarMap for weight access
}

impl GPT {
    /// Create a new GPT model
    pub fn new(config: &Config, vb: VarBuilder) -> Result<Self> {
        // Get the VarMap reference
        let varmap = vb.varmap().clone();
        
        let wte = embedding(config.vocab_size, config.n_embd, vb.pp("wte"))?;
        let wpe = embedding(config.n_positions, config.n_embd, vb.pp("wpe"))?;
        
        let mut h = Vec::new();
        for i in 0..config.n_layer {
            h.push(Block::new(
                config.n_embd,
                config.n_head,
                config.n_positions,
                vb.pp(format!("h.{}", i)),
            )?);
        }
        
        let ln_f = layer_norm(config.n_embd, 1e-5, vb.pp("ln_f"))?;
        
        // Create lm_head with weight sharing
        // In GPT-1/2, lm_head shares weights with wte (transposed)
        let lm_head = linear(config.n_embd, config.vocab_size, vb.pp("lm_head"))?;
        
        Ok(Self {
            wte,
            wpe,
            drop: config.dropout,
            h,
            ln_f,
            lm_head,
            n_positions: config.n_positions,
            device: vb.device().clone(),
            varmap: Arc::new(varmap),
        })
    }

    
    /// Get reference to VarMap
    pub fn varmap(&self) -> &VarMap {
        &self.varmap
    }
    
    /// Initialize model weights
    pub fn init_weights(varmap: &VarMap) {
        for (name, var) in varmap.all_vars().iter() {
            let tensor = var.as_tensor();
            
            // Initialize weights based on layer type
            if name.contains("c_attn") || name.contains("c_proj") || name.contains("c_fc") {
                // Linear layers: normal initialization
                let _ = tensor.init(candle_nn::init::Init::Randn { 
                    mean: 0.0, 
                    stdev: 0.02 
                });
            } else if name.contains("ln") {
                // LayerNorm: ones for weight, zeros for bias
                if name.ends_with(".weight") {
                    let _ = tensor.init(candle_nn::init::Init::Const(1.0));
                } else if name.ends_with(".bias") {
                    let _ = tensor.init(candle_nn::init::Init::Const(0.0));
                }
            } else if name.contains("wte") || name.contains("wpe") {
                // Embeddings: normal initialization
                let _ = tensor.init(candle_nn::init::Init::Randn { 
                    mean: 0.0, 
                    stdev: 0.02 
                });
            }
            // Note: lm_head initialization removed as we use weight sharing with wte
        }
    }
    
    /// Forward pass
    pub fn forward(&self, idx: &Tensor, targets: Option<&Tensor>, training: bool) -> Result<(Tensor, Option<Tensor>)> {
        let (b, t) = idx.dims2()?;
        assert!(t <= self.n_positions, "Sequence length {} exceeds maximum {}", t, self.n_positions);
        
        // Token embeddings
        let tok_emb = self.wte.forward(idx)?;
        
        // Position embeddings
        let pos = Tensor::arange(0, t as i64, &self.device)?.unsqueeze(0)?;
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
        
        // Language modeling head
        // GPT-1/GPT-2 architecture uses weight sharing between input embeddings (wte) 
        // and output projection layer. This reduces parameters by vocab_size * n_embd.
        // Instead of having a separate lm_head layer, we use the embedding matrix directly.
        
        // Get the embedding weight tensor
        let embed_weight = self.varmap.get("wte.weight")
            .ok_or_else(|| candle_core::Error::Msg("wte.weight not found in varmap".to_string()))?
            .as_tensor();
        
        // Compute logits: x @ embed_weight.T
        // x shape: (batch, seq_len, n_embd)
        // embed_weight shape: (vocab_size, n_embd)
        // result shape: (batch, seq_len, vocab_size)
        let logits = x.matmul(&embed_weight.t()?)?;
        
        // Calculate loss if targets provided
        let loss = if let Some(targets) = targets {
            let logits_view = logits.reshape((b * t, logits.dims()[2]))?;
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
            
            // Forward pass
            let (logits, _) = self.forward(&idx_cond, None, false)?;
            
            // Get logits for last position
            let logits = logits.i((.., logits.dims()[1] - 1, ..))?;
            
            // Apply temperature
            let logits = if temperature != 1.0 {
                (logits / temperature)?
            } else {
                logits
            };
            
            // Apply top-k filtering if specified
            let logits = if let Some(k) = top_k {
                apply_top_k(&logits, k)?
            } else {
                logits
            };
            
            // Convert to probabilities
            let probs = ops::softmax_last_dim(&logits)?;
            
            // Sample from the distribution
            let idx_next = sample_from_probs(&probs, &mut rng)?;
            
            // Append sampled index to the running sequence
            idx = Tensor::cat(&[idx, idx_next.unsqueeze(0)?], 1)?;
        }
        
        Ok(idx)
    }
}

/// Apply top-k filtering to logits
fn apply_top_k(logits: &Tensor, k: usize) -> Result<Tensor> {
    let vocab_size = logits.dims()[logits.dims().len() - 1];
    if k >= vocab_size {
        return Ok(logits.clone());
    }
    
    // Get top-k values and indices
    let (topk_values, _topk_indices) = logits.topk(k, D::Minus1, true, true)?;
    
    // Get the k-th largest value (threshold)
    let threshold = topk_values.i((.., k - 1))?.unsqueeze(D::Minus1)?;
    
    // Create mask for values below threshold
    let mask = logits.ge(&threshold)?;
    
    // Apply mask (set values below threshold to -inf)
    let filtered = logits.where_cond(
        &mask,
        &logits,
        &Tensor::new(f32::NEG_INFINITY, logits.device())?.broadcast_as(logits.shape())?,
    )?;
    
    Ok(filtered)
}

/// Sample an index from probability distribution
fn sample_from_probs(probs: &Tensor, rng: &mut impl Rng) -> Result<Tensor> {
    // For now, we'll use a simple implementation
    // In practice, you might want to use a more efficient method
    
    let probs_vec: Vec<f32> = probs.squeeze(0)?.to_vec1()?;
    let mut cumsum = vec![0.0];
    let mut sum = 0.0;
    
    for &p in &probs_vec {
        sum += p;
        cumsum.push(sum);
    }
    
    let sample: f32 = rng.gen();
    let mut idx = 0;
    
    for i in 0..probs_vec.len() {
        if sample <= cumsum[i + 1] {
            idx = i;
            break;
        }
    }
    
    Tensor::new(&[idx as i64], probs.device())
}