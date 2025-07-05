# English README is here.

[README.en.md](./README.en.md)

# buildable ブランチを使用して下さい / Use buildable branch

https://github.com/HarmoniaEpic/text2gpt-rust/tree/buildable

メインブランチは時折ビルド出来ない場合があります。 / Sometimes, main branch may not buildable.

# Text2GPT1-Rust 🦀

プロンプトから特定の性質を持つGPTモデルを自動生成するツール - Rust + Candle実装版

[![Rust](https://img.shields.io/badge/rust-%23000000.svg?style=for-the-badge&logo=rust&logoColor=white)](https://www.rust-lang.org/)
[![Candle](https://img.shields.io/badge/Candle-Deep%20Learning-orange?style=for-the-badge)](https://github.com/huggingface/candle)

## 🌟 概要

Text2GPT1-Rustは、ユーザーが指定したプロンプト（例：「料理レシピを生成する楽しいGPT」）から、その性質に特化したGPT-1スケールの言語モデルを自動的に作成するツールです。Python版のText2GPT1プロトタイプをRust + Candleで完全に再実装し、高速化とメモリ効率の向上を実現しました。

### 主な特徴

- 🚀 **高速実行** - Rustによるゼロコスト抽象化で高速なモデル生成
- 💾 **メモリ効率** - 所有権システムによる効率的なメモリ管理
- 🔒 **型安全性** - コンパイル時のエラー検出で堅牢性向上
- 🤖 **Ollama統合** - 高品質なデータ生成のためのLLM活用
- 📊 **リアルタイム進捗** - カラフルなプログレスバーで処理状況を可視化
- 🎯 **ドメイン特化** - 料理、詩、技術文書、汎用など複数のドメインに対応
- 💼 **スタンドアロン** - 単一バイナリとして配布可能
- ⏱️ **GPU/CPU最適化** - 環境に応じた自動タイムアウト調整
- 🎨 **スマートデフォルト** - モデルサイズに応じた最適なデータ量自動設定

## 🛠️ インストール

### 必要条件

- Rust 1.70以上
- CUDA対応GPU（オプション、CPUでも動作可能）
- Ollama（オプション、高品質データ生成用）

### ビルド方法

```bash
# リポジトリのクローン
git clone https://github.com/yourusername/text2gpt1-rust.git
cd text2gpt1-rust

# リリースビルド
cargo build --release

# 実行
./target/release/text2gpt1
```

### Ollamaのセットアップ（推奨）

```bash
# Ollamaのインストール
curl -fsSL https://ollama.ai/install.sh | sh

# Ollamaの起動
ollama serve

# 推奨モデルのダウンロード
ollama pull llama3.1:8b      # 汎用（8B）
ollama pull qwen2.5:7b       # 多言語対応（7B）
ollama pull codellama:7b     # コード特化（7B）
ollama pull tinyllama:1.1b   # 軽量（1.1B）
```

## 📖 使用方法

### 対話モード（推奨）

```bash
# メインメニューから開始
text2gpt1
```

対話モードでは以下の流れで進みます：

1. **モデルサイズ選択**
   - 12M - 高速、軽量（~500MB メモリ）
   - 33M - バランス型（~1GB メモリ）
   - 117M - 高品質（~2GB メモリ）

2. **カテゴリ選択**
   - 🍳 Cooking & Recipes
   - ✍️ Poetry & Creative
   - 💻 Technical & Code
   - 📝 General Purpose

3. **プリセット選択またはカスタム入力**

4. **学習パラメータ設定**
   - エポック数（20-500）
   - データサイズ（推奨値またはカスタム）

5. **データ生成方法の選択**
   - テンプレートベース（高速、オフライン）
   - Ollama統合（高品質、要Ollama）

### コマンドラインモード

```bash
# 基本的な使用例（モデルサイズに応じて自動的にトークン数が調整される）
text2gpt1 generate --prompt "料理レシピを生成する楽しいGPT" --model-size 12M
# → 自動設定: initial_tokens=30,000, final_tokens=15,000

# 大規模モデルを使用（より多くのデータで学習）
text2gpt1 generate --prompt "詩的で創造的な文章を書くGPT" --model-size 117M
# → 自動設定: initial_tokens=100,000, final_tokens=50,000

# Ollamaを使用した高品質生成
text2gpt1 generate \
  --prompt "技術文書を作成するGPT" \
  --model-size 33M \
  --ollama-gen-model llama3.1:8b \
  --ollama-refine-model qwen2.5:7b \
  --epochs 100

# カスタムトークン数を明示的に指定（自動設定を上書き）
text2gpt1 generate \
  --prompt "ビジネス文書生成GPT" \
  --model-size 33M \
  --initial-tokens 80000 \
  --final-tokens 40000

# CPU用タイムアウト設定
text2gpt1 generate \
  --prompt "料理レシピGPT" \
  --model-size 12M \
  --ollama-timeout-preset cpu

# 既存モデルで推論
text2gpt1 infer --model-path models/cooking_20240123

# モデル一覧表示
text2gpt1 list
```

## ⚙️ 設定オプション

### モデルサイズとデフォルト設定

| モデルサイズ | パラメータ数 | 初期トークン | 最終トークン | 推奨用途 |
|-------------|------------|------------|------------|---------|
| **12M** | 約1,200万 | 30,000 | 15,000 | 高速プロトタイピング、軽量用途 |
| **33M** | 約3,300万 | 50,000 | 25,000 | バランス型、標準的な用途 |
| **117M** | 約1億1,700万 | 100,000 | 50,000 | 高品質生成、プロダクション用途 |

### データ生成方法

1. **テンプレートベース** - オフライン対応、高速
2. **Ollama統合** - 高品質、多様性のある生成（推奨）

### Ollamaモデル選択

#### 推奨モデル（VRAM容量別）

**VRAM > 24GB**
- `llama3.1:70b` - 最高品質
- `qwen2.5:72b` - 優れた多言語対応
- `mixtral:8x7b` - MoEアーキテクチャ

**8GB < VRAM < 24GB**
- `llama3.1:8b` - バランス最良
- `qwen2.5:7b` - 日本語に強い
- `mistral:7b-v0.3` - 高速・効率的

**VRAM < 8GB / CPU**
- `tinyllama:1.1b` - 超軽量
- `phi3:mini` - 効率的な小型モデル
- `gemma2:2b` - Google製軽量モデル

### Ollamaタイムアウト設定

#### タイムアウトプリセット

| プリセット | 接続確認 | テキスト生成 | 品質評価 | リクエスト間隔 |
|-----------|---------|-------------|---------|---------------|
| `gpu` | 5秒 | 30秒 | 10秒 | 500ms |
| `cpu` | 5秒 | 300秒（5分） | 60秒（1分） | 1000ms（1秒） |
| `auto` | デバイスに応じて自動選択 |

#### タイムアウト関連オプション

- `--ollama-timeout-preset <auto|gpu|cpu>`: タイムアウトプリセット（デフォルト: auto）
- `--ollama-timeout-connection <秒>`: 接続確認タイムアウト
- `--ollama-timeout-generation <秒>`: テキスト生成タイムアウト
- `--ollama-timeout-evaluation <秒>`: 品質評価タイムアウト
- `--ollama-request-interval <ミリ秒>`: リクエスト間隔

### 環境変数

| 環境変数 | 説明 | デフォルト |
|---------|------|-----------|
| `TEXT2GPT1_OLLAMA_TIMEOUT_PRESET` | タイムアウトプリセット (auto/gpu/cpu) | auto |
| `TEXT2GPT1_OLLAMA_TIMEOUT_GENERATION` | テキスト生成タイムアウト（秒） | プリセットに依存 |
| `TEXT2GPT1_OLLAMA_TIMEOUT_EVALUATION` | 品質評価タイムアウト（秒） | プリセットに依存 |
| `TEXT2GPT1_OLLAMA_TIMEOUT_CONNECTION` | 接続確認タイムアウト（秒） | 5 |
| `TEXT2GPT1_OLLAMA_REQUEST_INTERVAL` | リクエスト間隔（ミリ秒） | プリセットに依存 |

### 主要パラメータ

- `--epochs`: 学習エポック数（デフォルト: 20）
- `--initial-tokens`: 初期生成トークン数（デフォルト: モデルサイズ依存）
- `--final-tokens`: 精選後のトークン数（デフォルト: モデルサイズ依存）
- `--batch-size`: バッチサイズ（デフォルト: 2）
- `--learning-rate`: 学習率（デフォルト: 0.0003）

## 🎯 使用例

### 料理レシピGPTの作成

```bash
text2gpt1 generate \
  --prompt "簡単で美味しい家庭料理のレシピを生成するGPT" \
  --model-size 12M \
  --epochs 50
```

### 詩的な文章生成GPTの作成

```bash
text2gpt1 generate \
  --prompt "感動的で美しい詩を創作するGPT" \
  --model-size 117M \
  --ollama-gen-model llama3.1:70b \
  --epochs 100
```

### 技術ドキュメント生成GPTの作成

```bash
text2gpt1 generate \
  --prompt "分かりやすいAPI仕様書を作成するGPT" \
  --model-size 33M \
  --ollama-gen-model codellama:7b \
  --initial-tokens 80000 \
  --final-tokens 40000
```

### CPU環境での使用例

```bash
# CPU用プリセットを使用
text2gpt1 generate \
  --prompt "料理レシピGPT" \
  --model-size 12M \
  --ollama-timeout-preset cpu \
  --ollama-gen-model tinyllama:1.1b

# または環境変数で設定
export TEXT2GPT1_OLLAMA_TIMEOUT_PRESET=cpu
text2gpt1 generate --prompt "料理レシピGPT"
```

## 📊 パフォーマンスの目安

### データ生成時間

**Ollama使用時**
- 10,000トークン: 約5分
- 50,000トークン: 約25分
- 100,000トークン: 約50分

**テンプレート使用時**
- 10,000トークン: < 1分
- 50,000トークン: 約3分
- 100,000トークン: 約5分

### 学習時間（GPU使用時、エポック数20の場合）
- 12Mモデル + 15,000トークン: 約10分
- 33Mモデル + 25,000トークン: 約20分
- 117Mモデル + 50,000トークン: 約40分

## 📁 出力ファイル

生成されたモデルは以下の構造で保存されます：

```
models/
└── cooking_recipe_generator_20240123_143022/
    ├── model.safetensors      # モデルの重み
    ├── config.json            # モデル設定
    ├── tokenizer.json         # トークナイザー
    ├── generation_info.json   # 生成情報
    ├── dataset.json           # 学習データ（JSON形式）
    └── dataset.txt            # 学習データ（テキスト形式）
```

## 🔄 Python版との互換性

- **safetensors形式**: PyTorchで保存したモデルとの相互運用が可能
- **GPT2トークナイザー**: HuggingFaceと同じトークナイザーを使用
- **同一アーキテクチャ**: Python版と同じモデル構造を採用

## 🐛 トラブルシューティング

### Ollamaが接続できない場合

```bash
# Ollamaが起動しているか確認
ollama list

# ポートを指定して起動
OLLAMA_HOST=0.0.0.0:11434 ollama serve
```

### モデルが見つからない場合

対話モードのCustom選択時に、インストール済みモデルが表示されます。
モデルがない場合は、推奨インストールコマンドが表示されます。

### GPU/CUDAエラーの場合

```bash
# CPU版として実行
CUDA_VISIBLE_DEVICES="" text2gpt1 generate --prompt "..."
```

### メモリ不足の場合

- バッチサイズを小さくする: `--batch-size 1`
- より小さいモデルサイズを使用: `--model-size 12M`
- データ量を減らす: `--initial-tokens 10000`

### CPU環境でタイムアウトが発生する場合

```bash
# CPU用プリセットを使用
text2gpt1 generate --prompt "..." --ollama-timeout-preset cpu

# または個別にタイムアウトを延長
text2gpt1 generate --prompt "..." \
  --ollama-timeout-generation 600 \
  --ollama-timeout-evaluation 120

# 環境変数での設定
export TEXT2GPT1_OLLAMA_TIMEOUT_PRESET=cpu
```

## 🤝 フォーク推奨

フォークを歓迎します！

## 📄 ライセンス

このプロジェクトはMITライセンスの下で公開されています。詳細は[LICENSE](LICENSE)ファイルを参照してください。

## 🙏 謝辞

- [Candle](https://github.com/huggingface/candle)プロジェクト
- [HuggingFace Tokenizers](https://github.com/huggingface/tokenizers)
- [Ollama](https://ollama.ai/)コミュニティ

## 📚 関連リンク

- [Candle Documentation](https://github.com/huggingface/candle)
- [Rust Book](https://doc.rust-lang.org/book/)
- [GPT論文](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)
