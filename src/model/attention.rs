use candle_core::{Result, Tensor, D};
use candle_nn::{linear, ops, Linear, Module, VarBuilder};

/// Causal self-attention layer
pub struct CausalSelfAttention {
    c_attn: Linear,
    c_proj: Linear,
    n_head: usize,
    n_embd: usize,
    bias: Tensor,
    attn_dropout_p: f32,
    resid_dropout_p: f32,
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
            attn_dropout_p: 0.1_f32,
            resid_dropout_p: 0.1_f32,
        })
    }
    
    pub fn forward(&self, x: &Tensor, training: bool) -> Result<Tensor> {
        let (b, t, c) = x.dims3()?;
        
        // Calculate query, key, values for all heads in batch
        let qkv = self.c_attn.forward(x)?;
        
        // Reshape to separate q, k, v
        // From [b, t, 3 * n_embd] to [b, t, 3, n_head, head_dim]
        let qkv = qkv.reshape((b, t, 3, self.n_head, c / self.n_head))?;
        
        // Permute to [b, 3, n_head, t, head_dim]
        // First transpose to [b, 3, t, n_head, head_dim]
        let qkv = qkv.transpose(1, 2)?;
        // Then transpose to [b, 3, n_head, t, head_dim]
        let qkv = qkv.transpose(2, 3)?;
        
        // Make contiguous after transposes
        let qkv = qkv.contiguous()?;
        
        // Extract q, k, v and make them contiguous
        let q = qkv.narrow(1, 0, 1)?.squeeze(1)?.contiguous()?; // [b, n_head, t, head_dim]
        let k = qkv.narrow(1, 1, 1)?.squeeze(1)?.contiguous()?; // [b, n_head, t, head_dim]
        let v = qkv.narrow(1, 2, 1)?.squeeze(1)?.contiguous()?; // [b, n_head, t, head_dim]
        
        // Attention scores
        let head_dim = c / self.n_head;
        let scale = 1.0 / (head_dim as f64).sqrt();
        
        // Use explicit broadcast for scaling
        let scale_tensor = Tensor::new(scale as f32, x.device())?;
        let k_t = k.transpose(D::Minus2, D::Minus1)?.contiguous()?;
        let att = q.matmul(&k_t)?
            .broadcast_mul(&scale_tensor)?;
        
        // Apply causal mask
        // Extract the relevant portion of the mask
        let mask = self.bias.narrow(2, 0, t)?.narrow(3, 0, t)?;
        
        // Expand mask to match attention shape [b, n_head, t, t]
        let mask = mask.expand(&[b, self.n_head, t, t])?;
        
        // Create attention mask: where mask is 0, use large negative value
        // This is the standard transformer approach
        let mask_value = -1e10_f32;
        
        // Apply: att = att + mask_value * (1 - mask)
        // Where mask is 1 (allowed), add 0
        // Where mask is 0 (masked), add mask_value
        let ones = Tensor::ones(mask.shape(), mask.dtype(), mask.device())?;
        let inv_mask = ones.broadcast_sub(&mask)?;
        let mask_add = inv_mask.broadcast_mul(&Tensor::new(mask_value, att.device())?)?;
        let att = att.broadcast_add(&mask_add)?;
        
        // Softmax
        let att = ops::softmax_last_dim(&att)?;
        
        // Dropout (if training)
        let att = if training && self.attn_dropout_p > 0.0 {
            ops::dropout(&att, self.attn_dropout_p)?
        } else {
            att
        };
        
        // Attention output
        let y = att.matmul(&v)?; // [b, n_head, t, head_dim]
        
        // Transpose back to [b, t, n_head, head_dim]
        let y = y.transpose(1, 2)?;
        
        // Reshape to [b, t, c]
        let y = y.contiguous()?.reshape((b, t, c))?;
        
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
