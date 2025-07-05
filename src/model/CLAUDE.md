# CLAUDE.md - Model Implementation

## 📁 Directory Purpose
GPT-1 architecture implementation using Candle deep learning framework.

## 🏗️ Architecture Components
```
model/
├── mod.rs         # Public exports
├── gpt.rs         # Main GPT model struct
├── attention.rs   # Causal self-attention
└── block.rs       # Transformer blocks (attention + MLP)
```

## 🧠 Model Architecture

### GPT-1 Specifications
```rust
// Model sizes from config.rs
Small (12M):  { n_layer: 6,  n_head: 6,  n_embd: 384 }
Medium (33M): { n_layer: 8,  n_head: 8,  n_embd: 512 }
Large (117M): { n_layer: 12, n_head: 12, n_embd: 768 }

// Common parameters
vocab_size: 50257  // GPT-2 tokenizer
n_positions: 512   // Maximum sequence length
dropout: 0.1       // During training
```

### Layer Stack
```
Input tokens
    ↓
Token Embeddings + Position Embeddings
    ↓
Dropout (training only)
    ↓
Transformer Blocks × n_layer
    ├─> LayerNorm → Multi-Head Attention → Residual
    └─> LayerNorm → MLP → Residual
    ↓
Final LayerNorm
    ↓
Output projection (weight tied with embeddings)
```

## 🔑 Key Implementation Details

### Attention Mechanism (attention.rs)
```rust
// Causal mask prevents attending to future tokens
let mask = create_causal_mask(seq_length);

// Scaled dot-product attention
let scores = (Q @ K.T) / sqrt(d_k)
let masked_scores = scores + mask * -1e10
let weights = softmax(masked_scores)
let output = weights @ V
```

### Weight Tying
```rust
// GPT uses weight tying between input embeddings and output projection
// This saves vocab_size × n_embd parameters
let output_logits = hidden_states @ embedding_weights.transpose()
```

### Candle-specific Patterns
```rust
// Always specify device
let tensor = Tensor::new(&data, &device)?;

// Use explicit broadcasting
let result = tensor1.broadcast_add(&tensor2)?;

// Careful with tensor shapes
let x = x.reshape((batch_size, seq_len, hidden_dim))?;
```

## 🚀 Usage Examples

### Creating a Model
```rust
let config = Config::new(ModelSize::Small, seed);
let device = Device::cuda_if_available(0)?;
let varmap = VarMap::new();
let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
let model = GPT::new(&config, vb)?;
```

### Forward Pass
```rust
// For training (with targets)
let (logits, loss) = model.forward(&input_ids, Some(&targets), true)?;

// For inference (no targets)
let (logits, _) = model.forward(&input_ids, None, false)?;
```

### Text Generation
```rust
let generated = model.generate(
    &input_tensor,
    max_new_tokens: 100,
    temperature: 0.8,
    top_k: Some(40),
)?;
```

## ⚠️ Common Pitfalls

### Memory Issues
- Large models (117M) need ~2GB GPU memory
- Batch size affects memory usage significantly
- Use gradient accumulation for larger effective batches

### Tensor Shape Errors
```rust
// Common error: mismatched dimensions
// Always check: [batch, sequence, features]
assert_eq!(tensor.dims(), &[batch_size, seq_len, hidden_dim]);
```

### Device Mismatches
```rust
// Ensure all tensors are on same device
let tensor_a = tensor_a.to_device(&device)?;
let tensor_b = tensor_b.to_device(&device)?;
```

## 🧪 Testing the Model
```rust
// Quick sanity check
let input = Tensor::new(&[1i64, 2, 3], &device)?.unsqueeze(0)?;
let (output, _) = model.forward(&input, None, false)?;
assert_eq!(output.dims(), &[1, 3, vocab_size]);
```

## 🔄 Recent Changes
- Implemented weight tying for embeddings
- Fixed attention mask broadcasting
- Added top-k sampling for generation

## 🤖 AI Collaboration Notes
- When modifying attention, ensure causality is preserved
- Be careful with f32/f64 conversions (Candle prefers f32)
- Generation uses teacher forcing during training
- Remember to handle both training and inference modes

---
*This file documents the neural network implementation details.*