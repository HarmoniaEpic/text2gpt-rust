# CLAUDE.md - Text2GPT1-Rust Project Context

## 🎯 Quick Context
- **Project**: Text2GPT1-Rust
- **Purpose**: Generate custom GPT models from prompts using Rust + Candle
- **Language**: Rust
- **Main Development Environment**: Manjaro Linux
- **Current Branch Strategy**: main branch for AI pair programming

## 📋 Project Overview
This tool automatically creates GPT-1 scale language models specialized for user-specified prompts (e.g., "A fun GPT that generates cooking recipes"). It's a complete Rust reimplementation of the Python Text2GPT1 prototype, achieving better performance and memory efficiency.

## 🛠️ Technology Stack
- **Deep Learning Framework**: Candle (Rust)
- **Tokenizer**: GPT-2 tokenizer via HuggingFace
- **Build Targets**:
  - Linux CPU: musl static binary (universal compatibility)
  - Linux GPU: AppImage format (CUDA dynamic detection)
  - Windows/macOS: Standard dynamic linking

## 📁 Key Files and Directories
```
src/
├── main.rs          # CLI entry point, interactive mode
├── config.rs        # Model configurations (12M/33M/117M)
├── model/           # GPT architecture implementation
├── data/            # Data generation (template/Ollama)
├── training/        # Training pipeline
├── inference/       # Text generation
└── utils/           # Helpers (Ollama detection, etc.)
```

## 🚀 Current Development Status
- ✅ Core GPT-1 architecture implemented
- ✅ Training pipeline with AdamW optimizer
- ✅ Data generation (template + Ollama integration)
- ✅ Interactive CLI interface
- 🔄 GitHub Actions CI/CD (buildable branch)
- 📝 TODO: Feature flags for CPU/CUDA builds
- 📝 TODO: AppImage packaging script

## 🔧 Build Commands
```bash
# Development (fast, local only)
cargo build
cargo run

# Release builds
# CPU (universal Linux)
cargo build --release --target x86_64-unknown-linux-musl --no-default-features --features cpu

# GPU (for AppImage)
cargo build --release --features cuda

# Run with specific model size
cargo run -- generate --prompt "A helpful cooking GPT" --model-size 12M
```

## 🤖 AI Collaboration Notes

### Last Session
- **Date**: 2024-01-15
- **Topics**: 
  - GitHub Actions configuration for buildable branch
  - musl static linking for CPU version
  - AppImage format for GPU version
- **Decisions**:
  - Use musl for maximum Linux compatibility
  - AppImage for GPU to handle library dependencies
  - main branch as primary development branch

### Current Focus
- [ ] Add feature flags to Cargo.toml
- [ ] Implement CUDA runtime detection
- [ ] Create AppImage packaging script
- [ ] Update README with new build instructions

### Important Context for AI
1. **Dependencies**: We use Candle (not PyTorch/TensorFlow)
2. **Error Handling**: Prefer `anyhow` for errors
3. **Async**: Use `tokio` for async operations
4. **Model Sizes**: 12M (small), 33M (medium), 117M (large)
5. **Data Generation**: Template-based or Ollama-powered

## 📝 Coding Conventions
- Use `Result<T>` with `anyhow::Result` for error handling
- Prefer `&str` over `String` for function parameters
- Use `log` crate for logging (not `println!`)
- Keep functions under 50 lines when possible
- Write tests for critical paths

## 🐛 Known Issues
- Ollama timeout on CPU can be long (5+ minutes)
- Windows builds need special handling for Candle
- GPU detection not yet implemented

## 📚 References
- [Candle Documentation](https://github.com/huggingface/candle)
- [GPT-1 Paper](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)
- [Rust Async Book](https://rust-lang.github.io/async-book/)

## 💡 Tips for Claude
- When modifying model architecture, check `src/model/gpt.rs`
- For CLI changes, focus on `src/main.rs`
- Data pipeline logic is in `src/data/`
- Always consider both CPU and GPU build targets
- Remember: we're targeting Linux primarily, with musl for portability

---
*This file is designed for AI pair programming. Update it when making significant changes.*