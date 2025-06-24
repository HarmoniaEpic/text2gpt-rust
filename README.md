# Text2GPT1-Rust 🦀

プロンプトから特定の性質を持つGPTモデルを自動生成するツール - Rust + Candle実装版

[![Rust](https://img.shields.io/badge/rust-%23000000.svg?style=for-the-badge&logo=rust&logoColor=white)](https://www.rust-lang.org/)
[![Candle](https://img.shields.io/badge/Candle-Deep%20Learning-orange?style=for-the-badge)](https://github.com/huggingface/candle)

## 🌟 概要

Text2GPT1-Rustは、ユーザーが指定したプロンプト（例：「料理レシピを生成する楽しいGPT」）から、その性質に特化したGPT-1スケールの言語モデルを自動的に作成するツールです。Python版の[Text2GPT1](https://github.com/example/text2gpt1)をRust + Candleで完全に再実装し、高速化とメモリ効率の向上を実現しました。

### 主な特徴

- 🚀 **高速実行** - Rustによるゼロコスト抽象化で高速なモデル生成
- 💾 **メモリ効率** - 所有権システムによる効率的なメモリ管理
- 🔒 **型安全性** - コンパイル時のエラー検出で堅牢性向上
- 🤖 **Ollama統合** - 高品質なデータ生成のためのLLM活用
- 📊 **リアルタイム進捗** - カラフルなプログレスバーで処理状況を可視化
- 🎯 **ドメイン特化** - 料理、詩、技術文書、汎用など複数のドメインに対応
- 💼 **スタンドアロン** - 単一バイナリとして配布可能

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

# モデルのダウンロード（推奨）
ollama pull llama3
ollama pull mistral
```

## 📖 使用方法

### 対話モード（推奨）

```bash
# メインメニューから開始
text2gpt1
```

対話モードでは以下の流れで進みます：

1. **カテゴリ選択**
   - 🍳 Cooking & Recipes
   - ✍️ Poetry & Creative
   - 💻 Technical & Code
   - 📝 General Purpose

2. **プリセット選択またはカスタム入力**
3. **学習パラメータ設定**
4. **データ生成方法の選択**

### コマンドラインモード

```bash
# 基本的な使用例
text2gpt1 generate --prompt "料理レシピを生成する楽しいGPT"

# Ollamaを使用した高品質生成
text2gpt1 generate \
  --prompt "詩的で創造的な文章を書くGPT" \
  --ollama-gen-model llama3 \
  --ollama-refine-model mistral \
  --epochs 100

# パラメータのカスタマイズ
text2gpt1 generate \
  --prompt "技術文書を作成するGPT" \
  --model-size 33M \
  --epochs 200 \
  --initial-tokens 3000 \
  --final-tokens 1500

# 既存モデルで推論
text2gpt1 infer --model-path models/cooking_20240123

# モデル一覧表示
text2gpt1 list
```

## ⚙️ 設定オプション

### モデルサイズ

| サイズ | パラメータ数 | 用途 |
|--------|------------|------|
| 12M | 約1200万 | 高速プロトタイピング、軽量用途 |
| 33M | 約3300万 | バランス型、標準的な用途 |
| 117M | 約1億1700万 | 高品質生成、リソースに余裕がある場合 |

### データ生成方法

1. **テンプレートベース** - オフライン対応、高速
2. **Ollama統合** - 高品質、多様性のある生成（推奨）

### 主要パラメータ

- `--epochs`: 学習エポック数（デフォルト: 20）
- `--initial-tokens`: 初期生成トークン数（デフォルト: 2000）
- `--final-tokens`: 精選後のトークン数（デフォルト: 1000）
- `--batch-size`: バッチサイズ（デフォルト: 2）
- `--learning-rate`: 学習率（デフォルト: 0.0003）

## 🎯 使用例

### 料理レシピGPTの作成

```bash
text2gpt1 generate \
  --prompt "簡単で美味しい家庭料理のレシピを生成するGPT" \
  --epochs 50 \
  --model-size 12M
```

### 詩的な文章生成GPTの作成

```bash
text2gpt1 generate \
  --prompt "感動的で美しい詩を創作するGPT" \
  --ollama-gen-model llama3:70b \
  --epochs 100 \
  --model-size 33M
```

### 技術ドキュメント生成GPTの作成

```bash
text2gpt1 generate \
  --prompt "分かりやすいAPI仕様書を作成するGPT" \
  --ollama-gen-model codellama \
  --epochs 150 \
  --model-size 33M
```

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

### GPU/CUDAエラーの場合

```bash
# CPU版として実行
CUDA_VISIBLE_DEVICES="" text2gpt1 generate --prompt "..."
```

### メモリ不足の場合

- バッチサイズを小さくする: `--batch-size 1`
- より小さいモデルサイズを使用: `--model-size 12M`

## 🤝 コントリビューション

プルリクエストを歓迎します！大きな変更を行う場合は、まずissueを開いて変更内容について議論してください。

## 📄 ライセンス

このプロジェクトはMITライセンスの下で公開されています。詳細は[LICENSE](LICENSE)ファイルを参照してください。

## 🙏 謝辞

- オリジナルのPython版Text2GPT1の作者
- [Candle](https://github.com/huggingface/candle)プロジェクト
- [HuggingFace Tokenizers](https://github.com/huggingface/tokenizers)
- [Ollama](https://ollama.ai/)コミュニティ

## 📚 関連リンク

- [Python版 Text2GPT1](https://github.com/example/text2gpt1)
- [Candle Documentation](https://github.com/huggingface/candle)
- [Rust Book](https://doc.rust-lang.org/book/)
- [GPT論文](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)