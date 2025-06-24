use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::{linear, ops, Linear, Module, VarBuilder};

/// Causal self-attention layer
pub struct CausalSelfAttention {
    c_attn: Linear,
    c_proj: Linear,
    n_head: usize,
    n_embd: usize,
    bias: Tensor,
    attn_dropout_p: f64,
    resid_dropout_p: f64,
}

impl CausalSelfAttention {
    pub fn new(
        n_embd: usize,
        n_head: usize,
        n_positions: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        assert_eq!(n_embd % n_head, 0, "n_embd must be divisible by n_head");
        
        let c_attn = linear(n_embd, 3 * n_embd, vb.pp("c_attn"))?;
        let c_proj = linear(n_embd, n_embd, vb.pp("c_proj"))?;
        
        // Create causal mask
        let mut mask_data = vec![0f32; n_positions * n_positions];
        for i in 0..n_positions {
            for j in 0..=i {
                mask_data[i * n_positions + j] = 1.0;
            }
        }
        
        let bias = Tensor::from_vec(
            mask_data,
            (1, 1, n_positions, n_positions),
            vb.device(),
        )?;
        
        Ok(Self {
            c_attn,
            c_proj,
            n_head,
            n_embd,
            bias,
            attn_dropout_p: 0.1,
            resid_dropout_p: 0.1,
        })
    }
    
    pub fn forward(&self, x: &Tensor, training: bool) -> Result<Tensor> {
        let (b, t, c) = x.dims3()?;
        
        // Calculate query, key, values for all heads in batch
        let qkv = self.c_attn.forward(x)?;
        let qkv = qkv.reshape((b, t, 3, self.n_head, c / self.n_head))?;
        let qkv = qkv.transpose(1, 2)?; // (B, 3, num_heads, T, head_dim)
        
        let q = qkv.narrow(1, 0, 1)?.squeeze(1)?;
        let k = qkv.narrow(1, 1, 1)?.squeeze(1)?;
        let v = qkv.narrow(1, 2, 1)?.squeeze(1)?;
        
        // Attention scores
        let scale = 1.0 / (k.dims()[D::Minus1] as f64).sqrt();
        let att = (q.matmul(&k.transpose(D::Minus2, D::Minus1)?)? * scale)?;
        
        // Apply causal mask
        let mask = &self.bias.narrow(2, 0, t)?.narrow(3, 0, t)?;
        let mask = mask.broadcast_as(att.shape())?;
        let att = att.where_cond(
            &mask.eq(&Tensor::zeros_like(&mask)?),
            &Tensor::new(f32::NEG_INFINITY, att.device())?.broadcast_as(att.shape())?,
            &att,
        )?;
        
        // Softmax
        let att = ops::softmax_last_dim(&att)?;
        
        // Dropout (if training)
        let att = if training && self.attn_dropout_p > 0.0 {
            ops::dropout(&att, self.attn_dropout_p)?
        } else {
            att
        };
        
        // Attention output
        let y = att.matmul(&v)?;
        let y = y.transpose(1, 2)?.contiguous()?.reshape((b, t, c))?;
        
        // Output projection
        let y = self.c_proj.forward(&y)?;
        
        // Residual dropout (if training)
        let y = if training && self.resid_dropout_p > 0.0 {
            ops::dropout(&y, self.resid_dropout_p)?
        } else {
            y
        };
        
        Ok(y)
    }
}