# CLAUDE.md - Tokenizer Module

## 📁 Directory Purpose
GPT-2 tokenizer wrapper for text encoding/decoding, using HuggingFace's tokenizers library.

## 🏗️ Module Structure
```
tokenizer/
├── mod.rs     # Public exports
└── gpt2.rs    # GPT2Tokenizer implementation
```

## 🔤 GPT2 Tokenizer

### Overview
- **Vocabulary Size**: 50,257 tokens
- **Special Tokens**: <|endoftext|> (id: 50256)
- **Encoding**: Byte-Pair Encoding (BPE)
- **Source**: HuggingFace's pretrained GPT-2 tokenizer

### Core Structure
```rust
pub struct GPT2Tokenizer {
    tokenizer: Tokenizer,  // HuggingFace tokenizer
    vocab_size: usize,     // 50,257
}
```

## 🚀 Usage

### Initialization
```rust
// Default (downloads from HuggingFace if needed)
let tokenizer = GPT2Tokenizer::new()?;

// From saved model directory
let tokenizer = GPT2Tokenizer::from_pretrained("models/my_model")?;
```

### Basic Operations
```rust
// Encode text to token IDs
let tokens: Vec<u32> = tokenizer.encode("Hello, world!")?;
// Result: [15496, 11, 995, 0]

// Decode tokens back to text
let text = tokenizer.decode(&tokens, skip_special_tokens: true)?;
// Result: "Hello, world!"
```

### Special Token IDs
```rust
tokenizer.pad_token_id()   // 50256 (same as EOS)
tokenizer.bos_token_id()   // 50256 (beginning of sequence)
tokenizer.eos_token_id()   // 50256 (end of sequence)
```

## 🔧 Implementation Details

### Loading Strategy
```rust
1. Try HuggingFace download
   - Requires internet on first use
   - Caches in ~/.cache/huggingface/

2. Fallback to local cache
   - Check dirs::cache_dir()
   - Look for tokenizers/gpt2.json

3. Error if not found
   - Clear error message with paths
```

### File Format
```json
// tokenizer.json structure
{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": { /* 50,257 entries */ },
    "merges": [ /* BPE merge rules */ ]
  },
  "pre_tokenizer": { /* byte-level */ },
  "post_processor": { /* special tokens */ }
}
```

### Tokenization Process
```
"Hello, world!" 
    ↓ Pre-tokenization (bytes)
[72, 101, 108, 108, 111, 44, 32, 119, 111, 114, 108, 100, 33]
    ↓ BPE merges
["Hello", ",", " world", "!"]
    ↓ Vocabulary lookup
[15496, 11, 995, 0]
```

## 📊 Token Statistics

### Common Patterns
```rust
// Single characters often = single tokens
"a" → [64]
"A" → [32]

// Common words = single tokens
"Hello" → [15496]
"the" → [262]

// Spaces usually attached to next word
" world" → [995]  // Note the leading space

// Numbers
"123" → [10163]
"2024" → [1238, 1731]  // Split into "20" + "24"
```

### Japanese Text
```rust
// Usually requires more tokens
"こんにちは" → [46036, 22174, 30298, 32015]
// ~1.5-3x more tokens than English
```

## ⚠️ Common Issues

### Token Limit
```rust
// Model max: 512 tokens
// But prompt + generation must fit
Solution: Check token count before generation

let prompt_tokens = tokenizer.encode(prompt)?.len();
let max_new_tokens = 512 - prompt_tokens;
```

### Unicode Handling
```rust
// GPT-2 uses byte-level BPE
"🎉" → [47728, 240, 137]  // Multiple tokens

// This is normal and expected
```

### Whitespace Sensitivity
```rust
// Different tokenization!
"Hello world" → [15496, 995]        // 2 tokens
"Hello  world" → [15496, 220, 995]  // 3 tokens (double space)
```

## 🔍 Debugging Tokenization

### Inspection Tools
```rust
// Check token count
let tokens = tokenizer.encode(text)?;
println!("Token count: {}", tokens.len());

// See individual tokens
for (i, &token_id) in tokens.iter().enumerate() {
    let token_text = tokenizer.decode(&[token_id], false)?;
    println!("Token {}: {} → '{}'", i, token_id, token_text);
}
```

### Gotchas
```rust
// Leading spaces matter!
"world" → [6894]
" world" → [995]

// Capitalization matters!
"Hello" → [15496]
"hello" → [31373]
```

## 🧪 Testing

### Round-trip Test
```rust
let original = "Hello, GPT-2! 🎉";
let tokens = tokenizer.encode(original)?;
let decoded = tokenizer.decode(&tokens, true)?;
assert_eq!(original, decoded);  // Should pass
```

### Edge Cases
```rust
// Empty string
tokenizer.encode("")?  // Returns: []

// Only special tokens
tokenizer.encode("<|endoftext|>")?  // Returns: [50256]

// Very long text
let long_text = "word ".repeat(1000);
let tokens = tokenizer.encode(&long_text)?;
assert!(tokens.len() <= 2000);  // ~2 tokens per "word "
```

## 🤖 AI Collaboration Notes

### Current Implementation
- ✅ Basic encode/decode
- ✅ Special token handling
- ✅ Pretrained model loading
- ✅ Model saving
- ❌ No custom vocabulary
- ❌ No training new tokenizer

### Integration Points
- Data generation needs token counting
- Training uses token IDs directly
- Inference converts both ways

### Important Notes
- We use u32 for token IDs (HF standard)
- But Candle uses i64 for tensors
- Always convert: `token as i64`

### Future Considerations
- [ ] Streaming tokenization
- [ ] Custom vocabularies for domains
- [ ] Subword regularization
- [ ] SentencePiece alternative

---
*This file documents the tokenizer implementation and usage.*