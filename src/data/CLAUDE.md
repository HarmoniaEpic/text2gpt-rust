# CLAUDE.md - Data Processing Pipeline

## 📁 Directory Purpose
Data generation, refinement, and dataset management for training GPT models.

## 🏗️ Module Structure
```
data/
├── mod.rs         # Public API exports
├── generator.rs   # Data generation (template/Ollama)
├── refiner.rs     # Quality evaluation and filtering
└── dataset.rs     # PyTorch-style dataset and dataloader
```

## 🔄 Data Pipeline Flow
```
User Prompt → Domain Detection
    ↓
Data Generation (generator.rs)
    ├─> Template-based (fast, offline)
    └─> Ollama-powered (high quality, requires Ollama)
    ↓
Raw Samples (initial_tokens target)
    ↓
Quality Refinement (refiner.rs)
    ├─> Quality scoring
    └─> Sample selection
    ↓
Refined Dataset (final_tokens target)
    ↓
TextDataset (dataset.rs)
    └─> DataLoader for training
```

## 🎯 Key Components

### DataGenerator (generator.rs)
```rust
pub enum DataGenerationMethod {
    Template,  // Built-in templates
    Ollama {
        gen_model: String,      // e.g., "llama3.1:8b"
        refine_model: String,   // e.g., "qwen2.5:7b"
    },
}

// Domain detection
"cooking" → recipe templates
"poetry" → creative templates
"technical" → documentation templates
"general" → mixed templates
```

### Template Examples
```rust
// Cooking domain
"How to make pasta: First, prepare the tomato..."
"Simple curry recipe: Combine onion and spices..."

// Poetry domain
"In spring I feel joy, walking quietly..."
"Days of autumn wrapped in nostalgia..."
```

### Ollama Integration
```rust
// Generation request
OllamaRequest {
    model: "llama3.1:8b",
    prompt: "Generate text following: {user_prompt}",
    temperature: 0.8,
    num_predict: 150,
}

// Timeout configuration
CPU preset: 5 minutes per request
GPU preset: 30 seconds per request
```

### DataRefiner (refiner.rs)
```rust
// Quality scoring (0.0 - 1.0)
- Length appropriateness
- Punctuation presence
- Keyword relevance
- Ollama-based evaluation (optional)

// Selection strategy
1. Score all samples
2. Sort by quality
3. Select top samples until token target
```

### TextDataset (dataset.rs)
```rust
// PyTorch-style dataset
impl TextDataset {
    fn len(&self) -> usize
    fn get(&self, idx: usize) -> Result<(input, target)>
}

// Sliding window for sequence generation
[A, B, C, D, E] → 
  Input:  [A, B, C, D]
  Target: [B, C, D, E]
```

## 📊 Configuration

### Token Targets by Model Size
```rust
ModelSize::Small (12M):
  - initial: 30,000 tokens
  - final: 15,000 tokens

ModelSize::Medium (33M):
  - initial: 50,000 tokens
  - final: 25,000 tokens

ModelSize::Large (117M):
  - initial: 100,000 tokens
  - final: 50,000 tokens
```

### Generation Time Estimates
```
Template-based:
  - 10,000 tokens: < 1 minute
  - 100,000 tokens: ~5 minutes

Ollama-powered:
  - 10,000 tokens: ~5 minutes
  - 100,000 tokens: ~50 minutes
```

## 🚀 Usage Examples

### Generate with Templates
```rust
let generator = DataGenerator::new(prompt, tokenizer, ollama_host, timeouts)?;
let samples = generator.generate_samples(
    num_tokens: 50_000,
    method: &DataGenerationMethod::Template,
).await?;
```

### Generate with Ollama
```rust
let method = DataGenerationMethod::Ollama {
    gen_model: "llama3.1:8b".to_string(),
    refine_model: "qwen2.5:7b".to_string(),
};
let samples = generator.generate_samples(num_tokens, &method).await?;
```

### Refine Data
```rust
let refiner = DataRefiner::new(prompt, tokenizer, ollama_host, timeouts);
let refined = refiner.refine_samples(
    samples,
    target_tokens: 25_000,
    use_ollama: true,
    model: "llama3.1:8b",
).await?;
```

## ⚠️ Common Issues

### Ollama Timeouts
```rust
// Check if Ollama is running
let available = generator.check_ollama(model).await?;

// Fall back to templates if needed
if !available {
    log::warn!("Ollama not available, using templates");
}
```

### Memory Usage
- Large datasets can consume significant RAM
- Consider streaming for very large datasets
- Token counting is memory-intensive

### Quality Variability
- Template data is consistent but less diverse
- Ollama data is diverse but may need more filtering
- Balance quality vs. generation time

## 🧪 Testing Data Quality
```rust
// Quick quality check
let avg_length = samples.iter()
    .map(|s| s.len())
    .sum::<usize>() / samples.len();

let unique_starts = samples.iter()
    .map(|s| s.chars().take(10).collect::<String>())
    .collect::<HashSet<_>>()
    .len();
```

## 🤖 AI Collaboration Notes
- Domain detection is keyword-based, could be improved
- Template variety is limited but fast
- Ollama quality depends heavily on model choice
- Consider adding domain-specific quality metrics

---
*This file documents the data generation and processing pipeline.*