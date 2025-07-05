# CLAUDE.md - Utilities Module

## 📁 Directory Purpose
Helper functions and utilities that support the main application functionality.

## 🏗️ Module Structure
```
utils/
├── mod.rs              # Public exports
├── folder_naming.rs    # Smart folder name generation
└── ollama.rs          # Ollama service integration helpers
```

## 📂 Folder Naming Module

### Purpose
Generate meaningful, filesystem-safe folder names from prompts.

### FolderNameBuilder
```rust
pub struct FolderNameBuilder {
    max_length: usize,  // Default: 50
}

// Example outputs:
"料理レシピを生成するGPT" → "cooking_recipe_generator_20240123_143022"
"詩的な文章を創作するGPT" → "poetry_creative_20240123_143023"
"Custom prompt" → "custom_gpt_20240123_143024"
```

### Naming Strategy
```rust
1. Check for preset name (e.g., "cooking_recipe_generator")
2. Extract domain keywords
3. Translate key terms to English
4. Add timestamp
5. Ensure filesystem safety
```

### Domain Detection
```rust
Domains:
- "cooking" - レシピ, 料理, 食
- "poetry" - 詩, ポエム, 創作
- "technical" - 技術, プログラム, コード
- "general" - その他
```

### Safety Features
- Removes special characters
- Limits length to 50 chars
- Ensures uniqueness with timestamp
- ASCII-safe for cross-platform compatibility

## 🦙 Ollama Module

### Purpose
Integration with Ollama for LLM-powered data generation.

### Core Types
```rust
pub struct InstalledModel {
    pub name: String,
    pub size: String,
    pub modified: String,
    pub category: ModelCategory,
}

pub enum ModelCategory {
    General,      // llama, mistral, qwen
    Technical,    // codellama, deepseek-coder
    Lightweight,  // tinyllama, phi3:mini
    Large,        // 70B+ models
    Unknown,
}
```

### Key Functions
```rust
// Check if Ollama service is running
pub async fn check_ollama_running() -> Result<bool>

// Get list of installed models
pub async fn get_installed_models() -> Result<Vec<InstalledModel>>

// Format models for UI display
pub fn format_model_list_for_display(&[InstalledModel]) -> Vec<String>

// Show installation hints
pub fn show_installation_hints(category: &str) -> String
```

### Model Recommendations
```rust
// By hardware
High-end GPU (24GB+): llama3.1:70b, qwen2.5:72b
Standard GPU (8-16GB): llama3.1:8b, mistral:7b
Limited GPU/CPU: tinyllama:1.1b, phi3:mini

// By use case
Cooking: llama3.1:8b, qwen2.5:7b
Poetry: llama3.1:70b, mistral:7b
Technical: codellama:7b, deepseek-coder:6.7b
```

### Model Information
```rust
pub fn get_ollama_model_info(model: &str) -> Option<OllamaModelInfo> {
    // Returns:
    // - Display name
    // - Size
    // - Description
    // - Best use cases
}
```

## 🔧 Implementation Details

### Async Operations
```rust
// Ollama API calls use tokio
let rt = tokio::runtime::Runtime::new()?;
let models = rt.block_on(get_installed_models())?;
```

### Error Handling
```rust
// User-friendly error messages
show_ollama_not_running_error();
show_no_models_error();

// With recovery hints
"Please start Ollama with: ollama serve"
"Install a model with: ollama pull llama3.1:8b"
```

### Timeout Management
```rust
// API timeouts
Connection check: 5 seconds
Model list fetch: 10 seconds
```

## 🎨 UI Helpers

### Model Display Formatting
```rust
[General Purpose]
llama3.1:8b (4.7GB) - 2 days ago
qwen2.5:7b (4.5GB) - 1 week ago

[Technical/Code]
codellama:7b (3.8GB) - 3 days ago

[Lightweight]
tinyllama:1.1b (637MB) - 1 month ago
```

### Progress Indicators
```rust
// Used throughout the app
ProgressBar styling
Spinner animations
ETA calculations
```

## ⚠️ Common Issues

### Ollama Connection
```rust
// Service not running
Solution: check_ollama_running() before operations

// Wrong port
Default: http://localhost:11434
```

### Model Naming
```rust
// Model not found
"llama3.1" vs "llama3.1:8b"
Always use full name with tag
```

### Unicode Handling
```rust
// Japanese text in prompts
Properly detected and mapped to English domains
UTF-8 throughout
```

## 🧪 Testing Utilities

### Folder Name Tests
```rust
#[test]
fn test_japanese_domain_detection() {
    let builder = FolderNameBuilder::new();
    let name = builder.generate("料理レシピGPT", None);
    assert!(name.starts_with("cooking_"));
}
```

### Mock Ollama
```rust
// For testing without Ollama
const MOCK_MODELS: &[(&str, &str)] = &[
    ("llama3.1:8b", "4.7GB"),
    ("tinyllama:1.1b", "637MB"),
];
```

## 🤖 AI Collaboration Notes

### Current State
- ✅ Smart folder naming with domains
- ✅ Ollama model detection
- ✅ Hardware-based recommendations
- ❌ No caching of Ollama queries
- ❌ Limited domain keywords

### Extensibility
- Easy to add new domains
- Model info is data-driven
- UI helpers are reusable

### Future Enhancements
- [ ] Cache Ollama model list
- [ ] Auto-detect optimal model
- [ ] Progress upload to cloud
- [ ] Multi-language domain detection

---
*This file documents utility functions and helper modules.*