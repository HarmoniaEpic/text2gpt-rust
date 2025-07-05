# CLAUDE.md - Source Code Directory

## 📁 Directory Purpose
Main source code directory containing all Rust implementation files for Text2GPT1.

## 🏗️ Architecture Overview
```
src/
├── main.rs          # Entry point, CLI implementation
├── lib.rs           # Library root, public API
├── config.rs        # Configuration structures
├── model/           # Neural network implementation
├── tokenizer/       # Text tokenization
├── data/            # Dataset generation/processing
├── training/        # Training loop and optimization
├── inference/       # Text generation at inference
├── io/              # File I/O, model serialization
└── utils/           # Utility functions and helpers
```

## 🔑 Key Design Patterns

### Error Handling
```rust
use anyhow::{Result, Context};

// Preferred pattern
fn process_data(path: &Path) -> Result<Dataset> {
    let data = std::fs::read_to_string(path)
        .context("Failed to read dataset file")?;
    // ...
}
```

### Module Organization
- Each subdirectory has a `mod.rs` that re-exports public items
- Keep implementation details private, expose clean APIs
- Use `pub use` for convenience re-exports

### Type System
```rust
// Model sizes are strongly typed
pub enum ModelSize {
    Small,  // 12M
    Medium, // 33M
    Large,  // 117M
}

// Use newtype pattern for domain types
pub struct TokenCount(pub usize);
```

## 🚀 Entry Points

### CLI (main.rs)
- Interactive mode with dialoguer
- Subcommands: generate, infer, list
- Handles both Ollama and template-based generation

### Library API (lib.rs)
```rust
pub use config::{Config, ModelSize};
pub use model::GPT;
pub use tokenizer::GPT2Tokenizer;
```

## 🧩 Module Interactions
```
main.rs
  ├─> training/trainer.rs (run_full_pipeline)
  │     ├─> data/generator.rs (generate samples)
  │     ├─> data/refiner.rs (refine quality)
  │     ├─> model/gpt.rs (create model)
  │     └─> io/safetensors.rs (save model)
  │
  └─> inference/generator.rs (text generation)
        └─> model/gpt.rs (forward pass)
```

## 🔧 Feature Flags
```toml
# Cargo.toml (planned)
[features]
default = ["cpu"]
cpu = []
cuda = ["candle/cuda"]
```

## 📝 Common Tasks

### Adding a New Feature
1. Define interface in appropriate module
2. Implement core logic
3. Wire up in main.rs if user-facing
4. Add tests in same file or `tests/`

### Debugging Tips
- Use `RUST_LOG=debug` for verbose output
- Check `target/debug/` for intermediate files
- Use `cargo expand` to see macro expansions

## ⚠️ Important Constraints
- **Candle Tensors**: Always specify device (CPU/CUDA)
- **Memory**: Watch for large tensors, use views when possible
- **Async**: Only in data generation (Ollama API calls)
- **Dependencies**: Minimize external crates

## 🧪 Testing Strategy
```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_feature() {
        // Unit tests go in same file
    }
}
```

## 🤖 AI Pair Programming Notes
- When adding features, consider both CPU and GPU paths
- Error messages should be helpful with `.context()`
- Keep public APIs minimal and well-documented
- Use `cargo clippy` suggestions

---
*This file helps AI assistants understand the source code organization.*