# CLAUDE.md - I/O Module

## 📁 Directory Purpose
File input/output operations including model serialization, dataset storage, and metadata management.

## 🏗️ Module Structure
```
io/
├── mod.rs           # Public exports
├── safetensors.rs   # Model weight I/O
└── json.rs          # Dataset and metadata I/O
```

## 💾 Safetensors Module

### Purpose
Efficient model weight storage using HuggingFace's safetensors format.

### Key Functions
```rust
// Save model
pub fn save_model(varmap: &VarMap, path: P) -> Result<()>

// Load model
pub fn load_model(path: P, device: &Device) -> Result<(GPT, Config, Arc<VarMap>)>

// Memory-mapped loading (efficient for large models)
pub fn load_model_mmaped(path: P, device: &Device) -> Result<(GPT, Config)>

// Complete model folder operations
pub fn save_model_folder(varmap, tokenizer, config, folder) -> Result<()>
pub fn load_model_folder(folder, device) -> Result<(GPT, Tokenizer, Config, VarMap)>
```

### Safetensors Benefits
- **Safe**: No arbitrary code execution (unlike pickle)
- **Fast**: Zero-copy deserialization
- **Simple**: JSON header + raw tensor data
- **Portable**: Language-agnostic format

### File Structure
```
model.safetensors
├── Header (JSON)
│   ├── tensor names
│   ├── shapes
│   └── offsets
└── Data (raw bytes)
    ├── wte.weight
    ├── wpe.weight
    ├── h.0.attn.c_attn.weight
    └── ...
```

## 📄 JSON Module

### Dataset Storage
```rust
pub struct DatasetInfo {
    pub metadata: DatasetMetadata,
    pub samples: Vec<DatasetSample>,
}

pub struct DatasetMetadata {
    pub total_samples: usize,
    pub total_tokens: usize,
    pub creation_date: String,
    pub generation_method: String,
    pub ollama_used: bool,
    // ... more fields
}
```

### Generation Info
```rust
pub struct GenerationInfo {
    pub prompt: String,
    pub model_size: String,
    pub training_params: TrainingParams,
    pub data_stats: DataStats,
    pub creation_date: String,
    // ... more fields
}
```

### Model Listing
```rust
pub fn list_saved_models(models_dir: P) -> Result<Vec<ModelInfo>>
// Scans directory for saved models
// Sorts by creation date (newest first)
```

## 📁 Complete Model Package

### Directory Structure
```
cooking_recipe_generator_20240123_143022/
├── model.safetensors      # Model weights
├── tokenizer.json         # GPT-2 tokenizer config
├── config.json           # Model architecture config
├── generation_info.json  # Training metadata
├── dataset.json          # Training data (structured)
└── dataset.txt           # Training data (plain text)
```

### Loading Process
```rust
1. Check directory exists
2. Load config.json → determine architecture
3. Initialize empty model with config
4. Load weights from safetensors
5. Load tokenizer
6. Optionally load metadata
```

## 🔧 Implementation Details

### VarMap Integration
```rust
// Candle's VarMap handles the actual tensor storage
let varmap = VarMap::new();

// Save directly from VarMap
varmap.save(path)?;

// Load into VarMap
varmap.load(path)?;
```

### Error Handling
```rust
// Always provide context
.context("Failed to save model weights")?
.context("Failed to load tokenizer")?

// Check file existence
if !path.exists() {
    return Err(anyhow!("Model file not found: {}", path.display()));
}
```

### Path Handling
```rust
// Use Path trait for cross-platform compatibility
pub fn save_model<P: AsRef<Path>>(varmap: &VarMap, path: P)

// Build paths safely
let model_path = folder_path.join("model.safetensors");
let config_path = folder_path.join("config.json");
```

## 📊 File Sizes

### Model Weights
```
12M model:  ~50MB (safetensors)
33M model:  ~130MB
117M model: ~450MB
```

### Datasets
```
JSON format: ~2-10MB (depends on samples)
Text format: ~1-5MB (plain text)
```

### Compression
```bash
# Safetensors already efficient, but can compress for distribution
tar -czf model.tar.gz model_directory/
# Reduces size by ~30-50%
```

## ⚠️ Common Issues

### Path Errors
```rust
// Windows vs Unix paths
// Solution: Always use Path/PathBuf
let path = PathBuf::from("models").join("my_model");
```

### Missing Files
```rust
// Incomplete model directory
// Solution: Check all required files
const REQUIRED_FILES: &[&str] = &[
    "model.safetensors",
    "config.json",
    "tokenizer.json",
];
```

### Version Compatibility
```rust
// Different Candle versions
// Solution: Include version in metadata
"candle_version": "0.8.4"
```

## 🧪 Testing I/O

### Round-trip Test
```rust
// Save
save_model_folder(&varmap, &tokenizer, &config, "test_model")?;

// Load
let (model2, tok2, cfg2, vm2) = load_model_folder("test_model", &device)?;

// Verify
assert_eq!(config.n_embd, cfg2.n_embd);
```

### Corruption Detection
```rust
// Safetensors has built-in checksums
// Will error on corrupted files
```

## 🤖 AI Collaboration Notes

### Current Implementation
- ✅ Safetensors for model weights
- ✅ JSON for metadata and datasets
- ✅ Complete folder save/load
- ❌ Model sharding for very large models
- ❌ Incremental saves during training

### Best Practices
- Always save complete model folders
- Include all metadata for reproducibility
- Use meaningful folder names
- Consider version control for configs

### Future Enhancements
- [ ] Checkpoint saving during training
- [ ] Model quantization support
- [ ] Differential saves (only changed weights)
- [ ] Cloud storage integration

---
*This file documents the I/O operations for models and datasets.*