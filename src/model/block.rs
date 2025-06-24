use candle_core::{Result, Tensor};
use candle_nn::{layer_norm, linear, ops, LayerNorm, Linear, Module, VarBuilder};

use super::attention::CausalSelfAttention;

/// MLP (Multi-Layer Perceptron) module
pub struct MLP {
    c_fc: Linear,
    c_proj: Linear,
    dropout_p: f32,  // Changed from f64 to f32
}

impl MLP {
    pub fn new(n_embd: usize, vb: VarBuilder) -> Result<Self> {
        let c_fc = linear(n_embd, 4 * n_embd, vb.pp("c_fc"))?;
        let c_proj = linear(4 * n_embd, n_embd, vb.pp("c_proj"))?;
        
        Ok(Self {
            c_fc,
            c_proj,
            dropout_p: 0.1_f32,  // Changed to f32
        })
    }
    
    pub fn forward(&self, x: &Tensor, training: bool) -> Result<Tensor> {
        let mut x = self.c_fc.forward(x)?;
        x = x.gelu()?;
        x = self.c_proj.forward(&x)?;
        
        if training && self.dropout_p > 0.0 {
            x = ops::dropout(&x, self.dropout_p)?;
        }
        
        Ok(x)
    }
}

/// Transformer block
pub struct Block {
    ln_1: LayerNorm,
    attn: CausalSelfAttention,
    ln_2: LayerNorm,
    mlp: MLP,
}

impl Block {
    pub fn new(
        n_embd: usize,
        n_head: usize,
        n_positions: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let ln_1 = layer_norm(n_embd, 1e-5, vb.pp("ln_1"))?;
        let attn = CausalSelfAttention::new(n_embd, n_head, n_positions, vb.pp("attn"))?;
        let ln_2 = layer_norm(n_embd, 1e-5, vb.pp("ln_2"))?;
        let mlp = MLP::new(n_embd, vb.pp("mlp"))?;
        
        Ok(Self {
            ln_1,
            attn,
            ln_2,
            mlp,
        })
    }
    
    pub fn forward(&self, x: &Tensor, training: bool) -> Result<Tensor> {
        // Attention block with residual connection
        let attn_out = self.attn.forward(&self.ln_1.forward(x)?, training)?;
        let x = (x + attn_out)?;
        
        // MLP block with residual connection
        let mlp_out = self.mlp.forward(&self.ln_2.forward(&x)?, training)?;
        let x = (x + mlp_out)?;
        
        Ok(x)
    }
}
