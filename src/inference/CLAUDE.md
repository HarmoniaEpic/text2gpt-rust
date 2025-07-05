# CLAUDE.md - Inference Module

## 📁 Directory Purpose
Text generation functionality for trained models, including loading saved models and generating text with various sampling strategies.

## 🏗️ Module Structure
```
inference/
├── mod.rs         # Public exports
└── generator.rs   # TextGenerator implementation
```

## 🎯 Core Functionality

### TextGenerator
```rust
pub struct TextGenerator {
    model: GPT,
    tokenizer: GPT2Tokenizer,
    config: Config,
    device: Device,
}
```

### Main Operations
1. **Load model** from saved directory
2. **Generate text** with customizable parameters
3. **Batch generation** for multiple prompts
4. **Stream generation** for real-time output

## 🚀 Usage Examples

### Loading a Model
```rust
// Standard loading
let generator = TextGenerator::load("models/cooking_20240123")?;

// Memory-mapped loading (faster for large models)
let generator = TextGenerator::load_mmaped("models/cooking_20240123")?;
```

### Basic Generation
```rust
let generated = generator.generate(
    prompt: "Today's recipe is",
    max_length: 100,
    temperature: 0.8,
    top_k: 40,
)?;
```

### Streaming Generation
```rust
// With live output callback
generator.generate_stream(
    prompt,
    max_length: 200,
    temperature: 0.9,
    top_k: 50,
    |token| {
        print!("{}", token);
        std::io::stdout().flush()?;
        Ok(())
    }
)?;
```

## 🎛️ Generation Parameters

### Temperature (Randomness)
```
0.1-0.4: Very focused, repetitive
0.5-0.7: Balanced creativity
0.8-1.0: Creative, diverse
1.1-2.0: Very random, potentially incoherent
```

### Top-k Sampling
```
0:     Greedy (always pick highest probability)
1-10:  Very constrained
20-40: Balanced (recommended)
50-100: More variety
```

### Future Parameters (TODO)
```rust
top_p: Option<f32>,        // Nucleus sampling
repetition_penalty: Option<f32>,  // Reduce repetition
```

## 🔧 Implementation Details

### Token Generation Loop
```rust
1. Encode prompt → token IDs
2. Create initial tensor
3. For each new token:
   - Forward pass through model
   - Get logits for last position
   - Apply temperature scaling
   - Apply top-k filtering
   - Sample from distribution
   - Append to sequence
4. Decode final tokens → text
```

### Sampling Implementation
```rust
// Temperature scaling
logits = logits / temperature

// Top-k filtering
top_k_values = logits.top_k(k)
filtered_logits = fill_with_neg_inf_except_top_k

// Probability distribution
probs = softmax(filtered_logits)

// Sample
cumulative_probs → sample_index
```

## 📁 Model Loading

### Expected Directory Structure
```
model_directory/
├── model.safetensors    # Required: model weights
├── tokenizer.json      # Required: tokenizer
├── config.json         # Required: model config
└── generation_info.json # Optional: training info
```

### Loading Process
```rust
1. Read config.json → Config struct
2. Load tokenizer.json → GPT2Tokenizer
3. Create VarMap and VarBuilder
4. Initialize GPT model structure
5. Load weights from safetensors
```

## ⚠️ Common Issues

### CUDA/CPU Mismatch
```rust
// Model on GPU, trying to generate on CPU
Solution: Ensure device consistency
let device = Device::cuda_if_available(0)?;
```

### Token Limit Exceeded
```rust
// Model max: 512 tokens
// Prompt + generation > 512
Solution: Truncate or use sliding window
```

### Memory Issues
```rust
// Large model + long generation
Solution: 
- Use smaller batch size (1)
- Reduce max_length
- Use CPU if GPU OOM
```

## 🧪 Testing Generation

### Quick Test
```rust
// Empty prompt (tests BOS token handling)
generator.generate("", 50, 0.8, 40)?

// Known prompt
generator.generate("The weather today", 50, 0.8, 40)?

// Edge cases
generator.generate("🎉", 50, 0.8, 40)?  // Unicode
```

### Quality Checks
```rust
// Coherence: Lower temperature
generate(prompt, 100, 0.5, 30)

// Creativity: Higher temperature
generate(prompt, 100, 1.0, 50)

// Deterministic: Greedy
generate(prompt, 100, 0.0, 1)
```

## 📊 Performance

### Generation Speed
```
GPU (RTX 3080):
- 12M model: ~100 tokens/second
- 33M model: ~50 tokens/second
- 117M model: ~20 tokens/second

CPU (Ryzen 5800X):
- 12M model: ~10 tokens/second
- 33M model: ~5 tokens/second
- 117M model: ~2 tokens/second
```

### Memory Usage
```
Model loading:
- 12M: ~500MB
- 33M: ~1GB
- 117M: ~2GB

During generation:
- Add ~100-500MB for activations
```

## 🤖 AI Collaboration Notes

### Current State
- ✅ Basic generation working
- ✅ Top-k sampling implemented
- ✅ Streaming output supported
- ❌ Top-p sampling not implemented
- ❌ Repetition penalty missing

### Integration Points
- CLI uses this for `infer` command
- Web API could use streaming
- Batch generation for evaluation

### Code Patterns
```rust
// Handle empty prompts
let input_ids = if prompt.is_empty() {
    vec![tokenizer.bos_token_id()]
} else {
    tokenizer.encode(prompt)?
};

// Device consistency
tensor.to_device(&self.device)?
```

---
*This file documents the inference and text generation functionality.*