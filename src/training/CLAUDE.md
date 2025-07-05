# CLAUDE.md - Training Pipeline

## 📁 Directory Purpose
Complete training pipeline from data generation to saved model, including optimization and evaluation.

## 🏗️ Module Structure
```
training/
├── mod.rs       # Public exports
└── trainer.rs   # Full pipeline implementation
```

## 🔄 Training Pipeline Flow
```
run_full_pipeline()
    ├─> 1. Initialize tokenizer
    ├─> 2. Generate data (DataGenerator)
    ├─> 3. Refine data (DataRefiner)
    ├─> 4. Create dataset (TextDataset)
    ├─> 5. Initialize model (GPT)
    ├─> 6. Train model (train_model)
    ├─> 7. Test generation
    └─> 8. Save everything
         ├─> model.safetensors
         ├─> tokenizer.json
         ├─> config.json
         ├─> dataset.json/txt
         └─> generation_info.json
```

## 🎯 Key Functions

### Main Pipeline Entry
```rust
pub async fn run_full_pipeline(
    prompt: &str,
    config: &Config,
    output_dir: &Path,
    generation_method: DataGenerationMethod,
) -> Result<TrainingResult>
```

### Training Loop
```rust
pub fn train_model(
    model: &GPT,
    dataset: &TextDataset,
    config: &Config,
    varmap: &Arc<VarMap>,
) -> Result<f32>
```

## 🔧 Training Configuration

### Optimizer Settings
```rust
AdamW {
    lr: 3e-4,          // Learning rate
    beta1: 0.9,        // First moment decay
    beta2: 0.999,      // Second moment decay
    eps: 1e-8,         // Epsilon
    weight_decay: 0.01, // L2 regularization
}
```

### Default Training Parameters
```rust
epochs: 20            // Can be customized
batch_size: 2         // Memory-efficient default
max_length: 128       // Sequence length
dropout: 0.1          // During training only
```

### Learning Schedule
- Currently: Fixed learning rate
- TODO: Cosine annealing or warmup

## 📊 Training Monitoring

### Progress Display
```
🔄 [00:05:23] [████████████████████░░░░░░] 156/200 steps | Loss: 3.4521 | ETA: 00:02:15
```

### Loss Tracking
- Smoothed loss over last 10 batches
- Epoch averages logged every 5 epochs
- Final loss returned for evaluation

### Memory Optimization
```rust
// Gradient accumulation (if needed)
// Currently: Direct update each batch
// TODO: Add accumulation for larger effective batch
```

## 💾 Model Saving

### Directory Structure
```
models/cooking_recipe_generator_20240123_143022/
├── model.safetensors      # Model weights (VarMap)
├── tokenizer.json         # GPT-2 tokenizer
├── config.json           # Model configuration
├── generation_info.json  # Training metadata
├── dataset.json          # Training data (JSON)
└── dataset.txt           # Training data (text)
```

### Folder Naming
```rust
// Using FolderNameBuilder
"cooking_recipe_generator_20240123_143022"
  ↑ domain/preset          ↑ timestamp
```

## 🧪 Post-Training Tests

### Generation Tests
```rust
// Test prompts
["", "The", "Today"]

// Parameters
max_tokens: 30
temperature: 0.8
top_k: 40
```

## ⚠️ Common Training Issues

### Out of Memory
```rust
// Solutions:
1. Reduce batch_size
2. Use smaller model_size
3. Reduce max_length
4. Enable gradient checkpointing (TODO)
```

### Loss Not Decreasing
```rust
// Check:
1. Learning rate (try 1e-4 to 1e-3)
2. Data quality (refiner thresholds)
3. Model initialization
4. Sufficient data amount
```

### Slow Training
```rust
// Optimize:
1. Ensure CUDA is used: config.device
2. Increase batch_size if memory allows
3. Use larger but fewer sequences
4. Profile with flamegraph
```

## 📈 Training Metrics

### Expected Loss Curves
```
Small (12M):
  Initial: ~8-10
  Final: ~2-4

Medium (33M):
  Initial: ~8-10
  Final: ~1.5-3

Large (117M):
  Initial: ~8-10
  Final: ~1-2.5
```

### Training Time Estimates
```
GPU (RTX 3080):
  12M model, 20 epochs: ~10 minutes
  33M model, 20 epochs: ~20 minutes
  117M model, 20 epochs: ~40 minutes

CPU:
  Multiply by ~10-20x
```

## 🔄 Recent Improvements
- Added VarMap for better weight management
- Implemented proper loss smoothing
- Added progress bar with ETA
- Improved error messages with context

## 🤖 AI Collaboration Notes

### Current Limitations
- No learning rate scheduling
- No gradient clipping
- No validation set splitting
- No early stopping

### Future Enhancements
- [ ] Mixed precision training (f16)
- [ ] Gradient accumulation
- [ ] Checkpoint resumption
- [ ] Distributed training support
- [ ] TensorBoard logging

### Code Patterns
```rust
// Always use context for errors
.context("Failed during model training")?

// Candle-specific optimizer pattern
optimizer.backward_step(&loss_tensor)?;

// Progress reporting pattern
pb.set_message(format!("{:.4}", smoothed_loss));
```

---
*This file documents the training pipeline implementation.*