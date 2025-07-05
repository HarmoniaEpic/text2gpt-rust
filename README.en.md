# Text2GPT1-Rust 🦀

Generate custom GPT models tailored to specific behaviors from prompts - Rust + Candle implementation

[![Rust](https://img.shields.io/badge/rust-%23000000.svg?style=for-the-badge&logo=rust&logoColor=white)](https://www.rust-lang.org/)
[![Candle](https://img.shields.io/badge/Candle-Deep%20Learning-orange?style=for-the-badge)](https://github.com/huggingface/candle)

## 🌟 Overview

Text2GPT1-Rust is a tool that automatically creates GPT-1 scale language models specialized for user-specified prompts (e.g., "A fun GPT that generates cooking recipes"). This is a complete Rust + Candle reimplementation of the Python Text2GPT1 prototype, achieving better performance and memory efficiency.

### Key Features

- 🚀 **High Performance** - Fast model generation with Rust's zero-cost abstractions
- 💾 **Memory Efficient** - Efficient memory management with ownership system
- 🔒 **Type Safety** - Compile-time error detection for improved robustness
- 🤖 **Ollama Integration** - LLM-powered high-quality data generation
- 📊 **Real-time Progress** - Colorful progress bars for process visualization
- 🎯 **Domain Specific** - Support for multiple domains: cooking, poetry, technical, general
- 💼 **Standalone** - Distributable as a single binary
- ⏱️ **GPU/CPU Optimized** - Automatic timeout adjustment based on environment
- 🎨 **Smart Defaults** - Automatic optimal data size based on model size

## 🛠️ Installation

### Prerequisites

- Rust 1.70 or later
- CUDA-capable GPU (optional, CPU mode available)
- Ollama (optional, for high-quality data generation)

### Building from Source

```bash
# Clone the repository
git clone https://github.com/yourusername/text2gpt1-rust.git
cd text2gpt1-rust

# Build release version
cargo build --release

# Run
./target/release/text2gpt1
```

### Ollama Setup (Recommended)

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama service
ollama serve

# Download recommended models
ollama pull llama3.1:8b      # General purpose (8B)
ollama pull qwen2.5:7b       # Multilingual (7B)
ollama pull codellama:7b     # Code-focused (7B)
ollama pull tinyllama:1.1b   # Lightweight (1.1B)
```

## 📖 Usage

### Interactive Mode (Recommended)

```bash
# Start from main menu
text2gpt1
```

The interactive mode guides you through:

1. **Model Size Selection**
   - 12M - Fast, lightweight (~500MB memory)
   - 33M - Balanced (~1GB memory)
   - 117M - High quality (~2GB memory)

2. **Category Selection**
   - 🍳 Cooking & Recipes
   - ✍️ Poetry & Creative
   - 💻 Technical & Code
   - 📝 General Purpose

3. **Preset Selection or Custom Input**

4. **Training Parameters**
   - Epochs (20-500)
   - Data size (recommended or custom)

5. **Data Generation Method**
   - Template-based (fast, offline)
   - Ollama-powered (high quality, requires Ollama)

### Command Line Mode

```bash
# Basic usage (auto-adjusts token counts based on model size)
text2gpt1 generate --prompt "A fun GPT that generates cooking recipes" --model-size 12M
# → Auto settings: initial_tokens=30,000, final_tokens=15,000

# Large model with more training data
text2gpt1 generate --prompt "A GPT that writes poetic and creative text" --model-size 117M
# → Auto settings: initial_tokens=100,000, final_tokens=50,000

# High-quality generation with Ollama
text2gpt1 generate \
  --prompt "A GPT for technical documentation" \
  --model-size 33M \
  --ollama-gen-model llama3.1:8b \
  --ollama-refine-model qwen2.5:7b \
  --epochs 100

# Custom token counts (overrides auto settings)
text2gpt1 generate \
  --prompt "Business document generator GPT" \
  --model-size 33M \
  --initial-tokens 80000 \
  --final-tokens 40000

# CPU timeout settings
text2gpt1 generate \
  --prompt "Recipe GPT" \
  --model-size 12M \
  --ollama-timeout-preset cpu

# Inference with existing model
text2gpt1 infer --model-path models/cooking_20240123

# List saved models
text2gpt1 list
```

## ⚙️ Configuration Options

### Model Sizes and Default Settings

| Model Size | Parameters | Initial Tokens | Final Tokens | Recommended Use |
|------------|-----------|----------------|--------------|-----------------|
| **12M** | ~12 million | 30,000 | 15,000 | Fast prototyping, lightweight use |
| **33M** | ~33 million | 50,000 | 25,000 | Balanced, standard use |
| **117M** | ~117 million | 100,000 | 50,000 | High quality, production use |

### Data Generation Methods

1. **Template-based** - Offline capable, fast
2. **Ollama-powered** - High quality, diverse generation (recommended)

### Ollama Model Selection

#### Recommended Models by VRAM

**VRAM > 24GB**
- `llama3.1:70b` - Highest quality
- `qwen2.5:72b` - Excellent multilingual
- `mixtral:8x7b` - MoE architecture

**8GB < VRAM < 24GB**
- `llama3.1:8b` - Best balance
- `qwen2.5:7b` - Strong for non-English
- `mistral:7b-v0.3` - Fast and efficient

**VRAM < 8GB / CPU**
- `tinyllama:1.1b` - Ultra lightweight
- `phi3:mini` - Efficient small model
- `gemma2:2b` - Google's lightweight model

### Ollama Timeout Settings

#### Timeout Presets

| Preset | Connection Check | Text Generation | Quality Evaluation | Request Interval |
|--------|-----------------|-----------------|-------------------|------------------|
| `gpu` | 5s | 30s | 10s | 500ms |
| `cpu` | 5s | 300s (5min) | 60s (1min) | 1000ms (1s) |
| `auto` | Auto-detect based on device |

#### Timeout Options

- `--ollama-timeout-preset <auto|gpu|cpu>`: Timeout preset (default: auto)
- `--ollama-timeout-connection <seconds>`: Connection check timeout
- `--ollama-timeout-generation <seconds>`: Text generation timeout
- `--ollama-timeout-evaluation <seconds>`: Quality evaluation timeout
- `--ollama-request-interval <milliseconds>`: Request interval

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `TEXT2GPT1_OLLAMA_TIMEOUT_PRESET` | Timeout preset (auto/gpu/cpu) | auto |
| `TEXT2GPT1_OLLAMA_TIMEOUT_GENERATION` | Text generation timeout (seconds) | Depends on preset |
| `TEXT2GPT1_OLLAMA_TIMEOUT_EVALUATION` | Quality evaluation timeout (seconds) | Depends on preset |
| `TEXT2GPT1_OLLAMA_TIMEOUT_CONNECTION` | Connection check timeout (seconds) | 5 |
| `TEXT2GPT1_OLLAMA_REQUEST_INTERVAL` | Request interval (milliseconds) | Depends on preset |

### Main Parameters

- `--epochs`: Training epochs (default: 20)
- `--initial-tokens`: Initial token generation count (default: model size dependent)
- `--final-tokens`: Tokens after refinement (default: model size dependent)
- `--batch-size`: Batch size (default: 2)
- `--learning-rate`: Learning rate (default: 0.0003)

## 🎯 Examples

### Creating a Recipe GPT

```bash
text2gpt1 generate \
  --prompt "A GPT that generates simple and delicious home cooking recipes" \
  --model-size 12M \
  --epochs 50
```

### Creating a Poetry GPT

```bash
text2gpt1 generate \
  --prompt "A GPT that creates touching and beautiful poetry" \
  --model-size 117M \
  --ollama-gen-model llama3.1:70b \
  --epochs 100
```

### Creating a Technical Documentation GPT

```bash
text2gpt1 generate \
  --prompt "A GPT that creates clear API documentation" \
  --model-size 33M \
  --ollama-gen-model codellama:7b \
  --initial-tokens 80000 \
  --final-tokens 40000
```

### CPU Environment Usage

```bash
# Use CPU preset
text2gpt1 generate \
  --prompt "Recipe GPT" \
  --model-size 12M \
  --ollama-timeout-preset cpu \
  --ollama-gen-model tinyllama:1.1b

# Or set via environment variable
export TEXT2GPT1_OLLAMA_TIMEOUT_PRESET=cpu
text2gpt1 generate --prompt "Recipe GPT"
```

## 📊 Performance Benchmarks

### Data Generation Time

**With Ollama**
- 10,000 tokens: ~5 minutes
- 50,000 tokens: ~25 minutes
- 100,000 tokens: ~50 minutes

**With Templates**
- 10,000 tokens: < 1 minute
- 50,000 tokens: ~3 minutes
- 100,000 tokens: ~5 minutes

### Training Time (GPU, 20 epochs)
- 12M model + 15,000 tokens: ~10 minutes
- 33M model + 25,000 tokens: ~20 minutes
- 117M model + 50,000 tokens: ~40 minutes

## 📁 Output Files

Generated models are saved with the following structure:

```
models/
└── cooking_recipe_generator_20240123_143022/
    ├── model.safetensors      # Model weights
    ├── config.json            # Model configuration
    ├── tokenizer.json         # Tokenizer
    ├── generation_info.json   # Generation metadata
    ├── dataset.json           # Training data (JSON)
    └── dataset.txt            # Training data (text)
```

## 🔄 Python Version Compatibility

- **safetensors format**: Interoperable with PyTorch saved models
- **GPT2 tokenizer**: Uses the same tokenizer as HuggingFace
- **Architecture**: Same model structure as Python version

## 🐛 Troubleshooting

### Ollama Connection Issues

```bash
# Check if Ollama is running
ollama list

# Start with specific port
OLLAMA_HOST=0.0.0.0:11434 ollama serve
```

### Model Not Found

The interactive mode shows installed models when selecting Custom.
If no models are found, installation commands are displayed.

### GPU/CUDA Errors

```bash
# Run as CPU version
CUDA_VISIBLE_DEVICES="" text2gpt1 generate --prompt "..."
```

### Out of Memory

- Reduce batch size: `--batch-size 1`
- Use smaller model size: `--model-size 12M`
- Reduce data amount: `--initial-tokens 10000`

### CPU Timeout Issues

```bash
# Use CPU preset
text2gpt1 generate --prompt "..." --ollama-timeout-preset cpu

# Or extend timeouts individually
text2gpt1 generate --prompt "..." \
  --ollama-timeout-generation 600 \
  --ollama-timeout-evaluation 120

# Via environment variable
export TEXT2GPT1_OLLAMA_TIMEOUT_PRESET=cpu
```

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Fork the repository
- Create feature branches
- Submit pull requests
- Report issues

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Candle](https://github.com/huggingface/candle) project for the deep learning framework
- [HuggingFace Tokenizers](https://github.com/huggingface/tokenizers) for tokenization
- [Ollama](https://ollama.ai/) community for LLM integration
- Original GPT paper and research

## 📚 Related Links

- [Candle Documentation](https://github.com/huggingface/candle)
- [Rust Book](https://doc.rust-lang.org/book/)
- [GPT Paper](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)

## 🚀 Build Options

### Feature Flags (Planned)

```toml
# CPU-only build
cargo build --release --no-default-features --features cpu

# CUDA build
cargo build --release --features cuda

# Specific target builds
cargo build --release --target x86_64-unknown-linux-musl  # Static binary
```

### Deployment

- **Linux**: Static musl builds for maximum compatibility
- **GPU Version**: AppImage format for library management
- **Cross-platform**: Standard dynamic linking for Windows/macOS

---

**Note**: This is an active development project. For the latest updates, check the `buildable` branch.