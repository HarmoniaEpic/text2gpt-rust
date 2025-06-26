#!/usr/bin/env python3
"""
Text2GPT1 Textual版: プロンプトから特定の性質を持つGPT-1モデルを自動生成
- Textualベースのモダンなリッチ TUI
- モデルごとのフォルダ管理機能
- カテゴリ選択UI（Cooking, Poetry, Technical, General）
- Ollama統合によるLLMベースのデータ生成
- 1kトークンのデータセット
- 12Mパラメータモデル（GPT-1の1/10スケール）
- safetensors形式での保存
- 作成済みモデルでの推論機能
"""

import os
import sys
import json
import random
import re
import time
import argparse
import requests
import asyncio
import unicodedata
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from safetensors.torch import save_file, load_file
from transformers import GPT2Tokenizer
import warnings
warnings.filterwarnings('ignore')

# Textual imports
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.css.query import NoMatches
from textual.message import Message
from textual.reactive import reactive
from textual.screen import Screen, ModalScreen
from textual.widgets import (
    Button, DataTable, Footer, Header, Input, Label, ListView, ListItem,
    LoadingIndicator, Log, OptionList, ProgressBar, RadioButton, RadioSet,
    RichLog, Static, TabbedContent, TabPane, TextArea
)
from textual.worker import Worker, get_current_worker
from rich.text import Text
from rich.panel import Panel
from rich.table import Table
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn


# ===== カテゴリとプリセットの定義 =====
CATEGORIES = {
    "cooking": {
        "name": "🍳 Cooking & Recipes",
        "name_ja": "料理・レシピ",
        "description": "料理レシピや食材の説明を生成",
        "icon": "🍳",
        "presets": [
            {
                "title": "家庭料理レシピジェネレーター",
                "title_en": "cooking_recipe_generator",
                "prompt": "簡単で美味しい家庭料理のレシピを生成する楽しいGPT",
                "description": "日常的な食材で作れる料理レシピを提供"
            },
            {
                "title": "プロフェッショナル料理アシスタント",
                "title_en": "professional_chef_assistant",
                "prompt": "本格的な料理技法と高度なレシピを提供する専門的なGPT",
                "description": "レストラン品質の料理を解説"
            },
            {
                "title": "ヘルシー＆ダイエットレシピ",
                "title_en": "healthy_diet_recipes",
                "prompt": "健康的で栄養バランスの良いレシピを生成するヘルシーGPT",
                "description": "カロリー控えめで栄養豊富なレシピ"
            },
            {
                "title": "お菓子・デザート専門",
                "title_en": "sweets_dessert_specialist",
                "prompt": "スイーツやデザートのレシピに特化した甘いGPT",
                "description": "ケーキ、クッキー、和菓子など"
            }
        ]
    },
    "poetry": {
        "name": "✍️ Poetry & Creative",
        "name_ja": "詩・創作",
        "description": "詩的で創造的な文章を生成",
        "icon": "✍️",
        "presets": [
            {
                "title": "現代詩ジェネレーター",
                "title_en": "modern_poetry_generator",
                "prompt": "現代的で感性豊かな詩を創作する詩人GPT",
                "description": "自由詩や散文詩を生成"
            },
            {
                "title": "俳句・短歌マスター",
                "title_en": "haiku_tanka_master",
                "prompt": "日本の伝統的な詩形を生成する和風GPT",
                "description": "5-7-5や5-7-5-7-7の形式"
            },
            {
                "title": "物語・小説クリエイター",
                "title_en": "story_novel_creator",
                "prompt": "短編小説や物語を創作するストーリーテラーGPT",
                "description": "創造的な物語を紡ぐ"
            },
            {
                "title": "歌詞・ソングライター",
                "title_en": "lyrics_songwriter",
                "prompt": "感動的な歌詞を書く音楽的なGPT",
                "description": "様々なジャンルの歌詞を創作"
            }
        ]
    },
    "technical": {
        "name": "💻 Technical & Code",
        "name_ja": "技術・プログラミング",
        "description": "技術文書やコード解説を生成",
        "icon": "💻",
        "presets": [
            {
                "title": "プログラミング解説者",
                "title_en": "programming_explainer",
                "prompt": "プログラミングの概念とコードを分かりやすく説明するGPT",
                "description": "初心者にも理解しやすい技術解説"
            },
            {
                "title": "API・ドキュメント作成",
                "title_en": "api_documentation_writer",
                "prompt": "技術仕様書やAPIドキュメントを作成する技術ライターGPT",
                "description": "構造化された技術文書"
            },
            {
                "title": "アルゴリズム・データ構造",
                "title_en": "algorithm_data_structures",
                "prompt": "アルゴリズムとデータ構造を解説する理論的なGPT",
                "description": "計算機科学の基礎概念"
            },
            {
                "title": "システム設計・アーキテクチャ",
                "title_en": "system_design_architecture",
                "prompt": "システム設計とソフトウェアアーキテクチャを説明するGPT",
                "description": "大規模システムの設計思想"
            }
        ]
    },
    "general": {
        "name": "📝 General Purpose",
        "name_ja": "汎用",
        "description": "様々な用途に対応する汎用的な文章生成",
        "icon": "📝",
        "presets": [
            {
                "title": "日常会話アシスタント",
                "title_en": "daily_conversation_assistant",
                "prompt": "親しみやすい日常会話を生成する友達のようなGPT",
                "description": "カジュアルな対話や雑談"
            },
            {
                "title": "ビジネス文書作成",
                "title_en": "business_document_writer",
                "prompt": "ビジネス文書やメールを作成するプロフェッショナルGPT",
                "description": "フォーマルなビジネス文章"
            },
            {
                "title": "教育・学習サポート",
                "title_en": "education_learning_support",
                "prompt": "分かりやすい説明で学習を支援する教育的なGPT",
                "description": "様々な分野の学習支援"
            },
            {
                "title": "ニュース・情報要約",
                "title_en": "news_information_summary",
                "prompt": "ニュースや情報を簡潔にまとめる要約GPT",
                "description": "重要なポイントを抽出"
            }
        ]
    }
}

# Ollamaモデルの説明情報
OLLAMA_MODEL_INFO = {
    "llama3": {
        "name": "Llama 3",
        "size": "8B",
        "description": "Meta's latest model - バランスの良い汎用モデル",
        "best_for": ["general", "cooking", "poetry"]
    },
    "llama3:70b": {
        "name": "Llama 3 Large",
        "size": "70B",
        "description": "最高品質の生成・評価（要高性能GPU）",
        "best_for": ["technical", "poetry"]
    },
    "mistral": {
        "name": "Mistral",
        "size": "7B",
        "description": "高速で効率的な生成",
        "best_for": ["general", "technical"]
    },
    "gemma": {
        "name": "Gemma",
        "size": "2B",
        "description": "Google製の軽量モデル - 高速処理",
        "best_for": ["general", "cooking"]
    },
    "gemma:7b": {
        "name": "Gemma Large",
        "size": "7B",
        "description": "Gemmaの大規模版 - より高品質",
        "best_for": ["general", "technical"]
    },
    "gemma2:2b": {
        "name": "Gemma 2",
        "size": "2B",
        "description": "Google製の最新軽量モデル",
        "best_for": ["general", "cooking"]
    },
    "gemma3:latest": {
        "name": "Gemma 3",
        "size": "Latest",
        "description": "Google製の最新版Gemmaモデル",
        "best_for": ["general", "technical"]
    },
    "qwen3:1.7b": {
        "name": "Qwen 3",
        "size": "1.7B",
        "description": "Alibaba製の軽量高性能モデル",
        "best_for": ["general", "technical"]
    },
    "codellama": {
        "name": "Code Llama",
        "size": "7B",
        "description": "プログラミング・技術文書に特化",
        "best_for": ["technical"]
    },
    "phi": {
        "name": "Phi-2",
        "size": "2.7B",
        "description": "Microsoft製の小型高性能モデル",
        "best_for": ["general", "technical"]
    },
    "neural-chat": {
        "name": "Neural Chat",
        "size": "7B",
        "description": "Intel製の対話特化モデル",
        "best_for": ["general", "poetry"]
    },
    "starling-lm": {
        "name": "Starling",
        "size": "7B",
        "description": "Berkeley製の指示追従モデル",
        "best_for": ["general", "cooking"]
    },
    "orca-mini": {
        "name": "Orca Mini",
        "size": "3B",
        "description": "Microsoft Orcaの軽量版",
        "best_for": ["general"]
    }
}


# ===== 1. 設定とハイパーパラメータ =====
class Config:
    """モデルとトレーニングの設定を管理するクラス"""
    # モデルアーキテクチャ（12Mパラメータ：GPT-1の1/10スケール）
    vocab_size = 50257  # GPT-2トークナイザーの語彙数
    n_embd = 384       # 埋め込み次元（GPT-1の768の半分）
    n_layer = 6        # Transformerレイヤー数（GPT-1の12の半分）
    n_head = 6         # アテンションヘッド数
    n_positions = 512  # 最大系列長
    
    # トレーニング設定
    batch_size = 2     # 小規模データセット用の小さいバッチサイズ
    learning_rate = 3e-4
    num_epochs = 20    # 高速プロトタイピング用の現実的なエポック数
    
    # データ設定
    initial_tokens = 2000  # 初期生成トークン数
    final_tokens = 1000    # 精選後のトークン数
    max_length = 128       # 個別サンプルの最大長
    
    # その他
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed = 42


# ===== フォルダ名生成ユーティリティ =====
def normalize_folder_name(name: str, max_length: int = 50) -> str:
    """フォルダ名を正規化（日本語→英語、特殊文字処理）"""
    # NFKDで分解して、アクセント記号を除去
    name = unicodedata.normalize('NFKD', name)
    name = ''.join([c for c in name if not unicodedata.combining(c)])
    
    # 英数字とアンダースコア以外を置換
    name = re.sub(r'[^a-zA-Z0-9_\s-]', '', name)
    
    # 連続するスペースやアンダースコアを一つに
    name = re.sub(r'[\s_-]+', '_', name)
    
    # 小文字化
    name = name.lower()
    
    # 前後のアンダースコアを削除
    name = name.strip('_')
    
    # 最大長制限
    if len(name) > max_length:
        name = name[:max_length]
    
    # 空になった場合のフォールバック
    if not name:
        name = "custom_model"
    
    return name


def create_model_folder_name(prompt: str, preset_info: Optional[Dict] = None) -> str:
    """モデルフォルダ名を生成"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if preset_info and 'title_en' in preset_info:
        # プリセット使用時
        base_name = preset_info['title_en']
    else:
        # カスタムプロンプト時
        # プロンプトから主要なキーワードを抽出
        if len(prompt) > 30:
            # 最初の30文字から生成
            base_name = "custom_" + normalize_folder_name(prompt[:30])
        else:
            base_name = "custom_" + normalize_folder_name(prompt)
    
    return f"{base_name}_{timestamp}"


# ===== 2. Textual メッセージ定義 =====
@dataclass
class MainMenuClicked(Message):
    """メインメニュー選択メッセージ"""
    action: str  # "create" or "inference" or "list"


@dataclass
class ModelCardClicked(Message):
    """モデルカードクリックメッセージ（推論用）"""
    model_path: str
    model_info: Dict


@dataclass
class CategoryCardClicked(Message):
    """カテゴリカードクリックメッセージ"""
    category_key: str


@dataclass
class PresetCardClicked(Message):
    """プリセットカードクリックメッセージ"""
    preset_data: Dict
    

@dataclass
class EpochOptionClicked(Message):
    """エポックオプションクリックメッセージ"""
    epochs: int


@dataclass
class GenerationMethodClicked(Message):
    """データ生成方法クリックメッセージ"""
    method: str  # "template" or "ollama"


@dataclass
class ModelCardOllamaClicked(Message):
    """Ollamaモデルカードクリックメッセージ"""
    model_name: str
    purpose: str  # "generation" or "refinement"


@dataclass
class ProgressUpdate(Message):
    """プログレス更新メッセージ"""
    step_key: str
    current: Optional[int] = None
    total: Optional[int] = None
    info: Optional[str] = None


@dataclass
class PreviewUpdate(Message):
    """プレビュー更新メッセージ"""
    content: Any


@dataclass
class SystemInfoUpdate(Message):
    """システム情報更新メッセージ"""
    key: str
    value: str


@dataclass
class StepChange(Message):
    """ステップ変更メッセージ"""
    step: int


@dataclass
class ProcessComplete(Message):
    """処理完了メッセージ"""
    folder_path: str
    domain: str


@dataclass
class ProcessError(Message):
    """エラーメッセージ"""
    error: str


# ===== 3. Textual ウィジェット =====
class MainMenuCard(Static):
    """メインメニュー選択用カード"""
    
    def __init__(self, action: str, title: str, description: str, icon: str, **kwargs):
        super().__init__(**kwargs)
        self.action = action
        self.title = title
        self.description = description
        self.icon = icon
        
    def compose(self) -> ComposeResult:
        yield Label(f"{self.icon} {self.title}", classes="menu-title")
        yield Label(self.description, classes="menu-description")
        
    def on_click(self) -> None:
        """クリック時にメッセージを送信"""
        self.post_message(MainMenuClicked(self.action))


class SavedModelCard(Static):
    """保存済みモデル選択用カード"""
    
    def __init__(self, model_path: str, model_info: Dict, **kwargs):
        super().__init__(**kwargs)
        self.model_path = model_path
        self.model_info = model_info
        
    def compose(self) -> ComposeResult:
        folder_name = os.path.basename(self.model_path)
        yield Label(f"📁 {folder_name}", classes="model-folder-name")
        yield Label(f"Domain: {self.model_info.get('category', 'Unknown')}", classes="model-info")
        yield Label(f"Prompt: {self.model_info.get('prompt', 'Unknown')[:50]}...", classes="model-info")
        yield Label(f"Size: {self.model_info.get('model_size', 'Unknown')}", classes="model-info")
        yield Label(f"Created: {self.model_info.get('creation_date', 'Unknown')}", classes="model-info")
        
    def on_click(self) -> None:
        """クリック時にメッセージを送信"""
        self.post_message(ModelCardClicked(self.model_path, self.model_info))


class CategoryCard(Static):
    """カテゴリ選択用カード"""
    
    def __init__(self, category_key: str, category_data: Dict, **kwargs):
        super().__init__(**kwargs)
        self.category_key = category_key
        self.category_data = category_data
        
    def compose(self) -> ComposeResult:
        icon = self.category_data['icon']
        name = self.category_data['name']
        name_ja = self.category_data['name_ja']
        description = self.category_data['description']
        
        yield Label(f"{icon} {name}", classes="category-title")
        yield Label(f"({name_ja})", classes="category-subtitle")
        yield Label(description, classes="category-description")
        
    def on_click(self) -> None:
        """クリック時にメッセージを送信"""
        self.post_message(CategoryCardClicked(self.category_key))


class PresetCard(Static):
    """プリセット選択用カード"""
    
    def __init__(self, preset_data: Dict, index: int, **kwargs):
        super().__init__(**kwargs)
        self.preset_data = preset_data
        self.index = index
        
    def compose(self) -> ComposeResult:
        yield Label(f"{self.index}. {self.preset_data['title']}", classes="preset-title")
        yield Label(f"「{self.preset_data['prompt']}」", classes="preset-prompt")
        yield Label(self.preset_data['description'], classes="preset-description")
        
    def on_click(self) -> None:
        """クリック時にメッセージを送信"""
        self.post_message(PresetCardClicked(self.preset_data))


class EpochOption(Static):
    """エポック数選択オプション"""
    
    def __init__(self, epochs: int, epoch_name: str, description: str, time_estimate: str, **kwargs):
        super().__init__(**kwargs)
        self.epochs = epochs
        self.epoch_name = epoch_name
        self.description = description
        self.time_estimate = time_estimate
        
    def compose(self) -> ComposeResult:
        if self.epochs > 0:
            yield Label(f"{self.epochs} epochs - {self.epoch_name}", classes="epoch-title")
        else:
            yield Label(self.epoch_name, classes="epoch-title")
        yield Label(self.description, classes="epoch-description")
        yield Label(f"予想時間: {self.time_estimate}", classes="epoch-time")
        
    def on_click(self) -> None:
        """クリック時にメッセージを送信"""
        self.post_message(EpochOptionClicked(self.epochs))


class GenerationMethodCard(Static):
    """データ生成方法選択用カード"""
    
    def __init__(self, method: str, title: str, description: str, advantages: List[str], **kwargs):
        super().__init__(**kwargs)
        self.method = method
        self.title = title
        self.description = description
        self.advantages = advantages
        
    def compose(self) -> ComposeResult:
        yield Label(self.title, classes="method-title")
        yield Label(self.description, classes="method-description")
        
        with Container(classes="advantages-list"):
            for advantage in self.advantages:
                yield Label(f"• {advantage}", classes="method-advantage")
        
    def on_click(self) -> None:
        """クリック時にメッセージを送信"""
        self.post_message(GenerationMethodClicked(self.method))


class ModelCardOllama(Static):
    """Ollamaモデル選択用カード"""
    
    def __init__(self, model_name: str, model_info: Dict, available: bool, recommended: bool, purpose: str, **kwargs):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.model_info = model_info
        self.available = available
        self.recommended = recommended
        self.purpose = purpose
        
    def compose(self) -> ComposeResult:
        status = "✓" if self.available else "×"
        recommend_mark = " ⭐" if self.recommended else ""
        
        title = f"[{'green' if self.available else 'red'}]{status}[/] {self.model_info['name']} ({self.model_info['size']}){recommend_mark}"
        yield Label(title, classes="model-title")
        yield Label(self.model_info['description'], classes="model-description")
        
        if not self.available:
            yield Label("インストールが必要: ollama pull " + self.model_name, classes="model-install-hint")
    
    def on_click(self) -> None:
        """クリック時にメッセージを送信（利用可能な場合のみ）"""
        if self.available:
            self.post_message(ModelCardOllamaClicked(self.model_name, self.purpose))


# ===== 4. 画面定義 =====
class MainMenuScreen(Screen):
    """メインメニュー画面"""
    
    CSS = """
    MainMenuScreen {
        align: center middle;
    }
    
    #menu-container {
        width: 80;
        height: auto;
        border: solid $primary;
        padding: 2;
    }
    
    .menu-card {
        margin: 2;
        padding: 2;
        border: solid $surface;
    }
    
    .menu-card:hover {
        border: solid $primary;
        background: $boost;
    }
    
    .menu-title {
        text-style: bold;
        color: $primary;
    }
    
    .menu-description {
        margin-top: 1;
    }
    
    #button-container {
        dock: bottom;
        height: 3;
        align: center middle;
    }
    """
    
    def compose(self) -> ComposeResult:
        yield Header()
        
        with Container(id="menu-container"):
            yield Label("Text2GPT1 - メインメニュー", classes="title")
            
            # 新規作成
            card1 = MainMenuCard(
                "create",
                "新しいGPTモデルを作成",
                "プロンプトから特定の性質を持つGPTモデルを自動生成します",
                "🔨",
                classes="menu-card"
            )
            card1.can_focus = True
            yield card1
            
            # 推論
            card2 = MainMenuCard(
                "inference",
                "既存のモデルで推論",
                "作成済みのGPTモデルを使用してテキストを生成します",
                "🤖",
                classes="menu-card"
            )
            card2.can_focus = True
            yield card2
                
        with Horizontal(id="button-container"):
            yield Button("終了", id="quit", variant="error")
            
        yield Footer()
        
    def on_mount(self) -> None:
        self.query_one(".menu-card").focus()
        
    async def on_main_menu_clicked(self, message: MainMenuClicked) -> None:
        """メニューカードがクリックされた時の処理"""
        if message.action == "create":
            await self.app.push_screen(CategorySelectionScreen())
        elif message.action == "inference":
            await self.app.push_screen(ModelSelectionScreen())
            
    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "quit":
            self.app.exit()


class ModelSelectionScreen(Screen):
    """モデル選択画面（推論用）"""
    
    CSS = """
    ModelSelectionScreen {
        align: center middle;
    }
    
    #model-container {
        width: 100;
        height: 80%;
        border: solid $primary;
        padding: 2;
    }
    
    .model-card {
        margin: 2;
        padding: 2;
        border: solid $surface;
    }
    
    .model-card:hover {
        border: solid $primary;
        background: $boost;
    }
    
    .model-folder-name {
        text-style: bold;
        color: $primary;
    }
    
    .model-info {
        margin-top: 1;
        color: $text-muted;
    }
    
    #button-container {
        dock: bottom;
        height: 3;
        align: center middle;
    }
    
    .no-models {
        text-align: center;
        color: $warning;
        margin: 4;
    }
    """
    
    def compose(self) -> ComposeResult:
        yield Header()
        
        with Container(id="model-container"):
            yield Label("作成済みモデル一覧", classes="title")
            
            with ScrollableContainer(id="model-list"):
                yield LoadingIndicator(id="loading")
                
        with Horizontal(id="button-container"):
            yield Button("戻る", id="back")
            
        yield Footer()
        
    def on_mount(self) -> None:
        """画面マウント時にモデル一覧を読み込み"""
        self.load_models()
        
    @work(thread=True)
    def load_models(self) -> None:
        """保存されたモデルを読み込み"""
        models = list_saved_models(self.app.args.output_dir)
        self.app.call_from_thread(self.display_models, models)
        
    def display_models(self, models: List[Dict]) -> None:
        """モデル一覧を表示"""
        # ローディングインジケータを削除
        try:
            self.query_one("#loading").remove()
        except:
            pass
        
        model_list = self.query_one("#model-list")
        
        if not models:
            model_list.mount(Label(
                "モデルが見つかりません。\n先に新しいモデルを作成してください。",
                classes="no-models"
            ))
        else:
            for model in models:
                card = SavedModelCard(
                    model['path'],
                    model['info'],
                    classes="model-card"
                )
                card.can_focus = True
                model_list.mount(card)
                
    async def on_model_card_clicked(self, message: ModelCardClicked) -> None:
        """モデルが選択された時の処理"""
        self.app.selected_model_path = message.model_path
        self.app.selected_model_info = message.model_info
        await self.app.push_screen(InferenceScreen())
            
    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "back":
            self.app.pop_screen()


class InferenceScreen(Screen):
    """推論画面"""
    
    BINDINGS = [
        Binding("ctrl+enter", "generate", "生成"),
        Binding("ctrl+l", "clear", "クリア"),
        Binding("escape", "back", "戻る"),
    ]
    
    CSS = """
    InferenceScreen {
        layout: grid;
        grid-size: 2 1;
        grid-columns: 2fr 3fr;
    }
    
    #left-panel {
        border: solid $primary;
        padding: 1;
        margin: 1;
    }
    
    #right-panel {
        border: solid $primary;
        padding: 1;
        margin: 1;
    }
    
    .panel-title {
        text-style: bold;
        color: $primary;
        margin-bottom: 1;
    }
    
    #model-info {
        margin-bottom: 2;
    }
    
    .param-container {
        margin: 1 0;
    }
    
    .param-label {
        margin-bottom: 1;
    }
    
    Input {
        width: 100%;
    }
    
    #prompt-area {
        height: 10;
        margin: 1 0;
    }
    
    #result-area {
        height: 20;
        margin: 1 0;
        border: solid $surface;
        padding: 1;
    }
    
    #button-container {
        dock: bottom;
        height: 3;
        align: center middle;
        margin-top: 1;
    }
    
    Button {
        margin: 0 1;
    }
    """
    
    def __init__(self):
        super().__init__()
        self.model = None
        self.tokenizer = None
        self.config = None
        self.temperature = reactive(0.8)
        self.max_length = reactive(100)
        self.top_k = reactive(40)
        
    def compose(self) -> ComposeResult:
        yield Header()
        
        # 左パネル - 設定
        with Container(id="left-panel"):
            yield Label("⚙️ 設定", classes="panel-title")
            
            with Container(id="model-info"):
                yield Label("読み込み中...", id="model-name")
                yield Label("", id="model-domain")
                yield Label("", id="model-size")
            
            # Temperature
            with Container(classes="param-container"):
                yield Label("Temperature (0.1-2.0):", classes="param-label")
                yield Input(value="0.8", id="temperature-input", type="number")
                
            # Max Length
            with Container(classes="param-container"):
                yield Label("Max Length (10-500):", classes="param-label")
                yield Input(value="100", id="max-length-input", type="integer")
                
            # Top-k
            with Container(classes="param-container"):
                yield Label("Top-k (1-100):", classes="param-label")
                yield Input(value="40", id="top-k-input", type="integer")
                
        # 右パネル - 生成
        with Container(id="right-panel"):
            yield Label("✍️ テキスト生成", classes="panel-title")
            
            yield Label("プロンプト:", classes="param-label")
            yield TextArea(id="prompt-area", language="markdown")
            
            yield Label("生成結果:", classes="param-label")
            yield RichLog(highlight=True, markup=True, id="result-area")
            
            with Horizontal(id="button-container"):
                yield Button("生成", id="generate", variant="primary")
                yield Button("クリア", id="clear")
                yield Button("別のモデル", id="change-model")
                yield Button("メニューに戻る", id="back-menu")
                
        yield Footer()
        
    def on_mount(self) -> None:
        """画面マウント時にモデルを読み込み"""
        self.load_model()
        
    @work(thread=True)
    def load_model(self) -> None:
        """モデルを読み込み"""
        try:
            model_path = self.app.selected_model_path
            self.model, self.tokenizer, self.config = load_model_safetensors(model_path)
            
            # UIを更新
            self.app.call_from_thread(self.update_model_info)
            
        except Exception as e:
            self.app.call_from_thread(
                self.post_message,
                ProcessError(f"モデルの読み込みに失敗しました: {str(e)}")
            )
            
    def update_model_info(self) -> None:
        """モデル情報を更新"""
        info = self.app.selected_model_info
        
        self.query_one("#model-name").update(f"モデル: {os.path.basename(self.app.selected_model_path)}")
        self.query_one("#model-domain").update(f"ドメイン: {info.get('category', 'Unknown')}")
        self.query_one("#model-size").update(f"サイズ: {info.get('model_size', 'Unknown')}")
        
        # プロンプトエリアにフォーカス
        self.query_one("#prompt-area").focus()
        
    def action_generate(self) -> None:
        """テキスト生成アクション"""
        self.generate_text()
        
    def action_clear(self) -> None:
        """クリアアクション"""
        self.query_one("#prompt-area").clear()
        self.query_one("#result-area").clear()
        
    def action_back(self) -> None:
        """戻るアクション"""
        self.app.pop_screen()
        
    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "generate":
            self.generate_text()
        elif event.button.id == "clear":
            self.action_clear()
        elif event.button.id == "change-model":
            self.app.pop_screen()
        elif event.button.id == "back-menu":
            # 全画面をクリアしてメインメニューを表示
            while len(self.app.screen_stack) > 1:
                self.app.pop_screen()
            await self.app.push_screen(MainMenuScreen())
            
    def generate_text(self) -> None:
        """テキスト生成を実行"""
        if not self.model:
            return
            
        prompt = self.query_one("#prompt-area").text.strip()
        if not prompt:
            return
            
        # パラメータを取得
        try:
            temperature = float(self.query_one("#temperature-input").value)
            temperature = max(0.1, min(2.0, temperature))
        except:
            temperature = 0.8
            
        try:
            max_length = int(self.query_one("#max-length-input").value)
            max_length = max(10, min(500, max_length))
        except:
            max_length = 100
            
        try:
            top_k = int(self.query_one("#top-k-input").value)
            top_k = max(1, min(100, top_k))
        except:
            top_k = 40
            
        # 生成を実行
        self.generate_text_worker(prompt, temperature, max_length, top_k)
        
    @work(thread=True)
    def generate_text_worker(self, prompt: str, temperature: float, max_length: int, top_k: int) -> None:
        """バックグラウンドでテキスト生成"""
        try:
            # 結果エリアをクリア
            self.app.call_from_thread(
                self.query_one("#result-area").clear
            )
            
            # 生成中メッセージ
            self.app.call_from_thread(
                self.query_one("#result-area").write,
                "[dim]生成中...[/dim]"
            )
            
            # 生成実行
            generated = generate_sample(
                self.model, 
                self.tokenizer, 
                self.config,
                prompt=prompt,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k
            )
            
            # 結果を表示
            self.app.call_from_thread(
                self.display_result,
                generated
            )
            
        except Exception as e:
            self.app.call_from_thread(
                self.query_one("#result-area").write,
                f"[red]エラー: {str(e)}[/red]"
            )
            
    def display_result(self, text: str) -> None:
        """生成結果を表示"""
        result_area = self.query_one("#result-area")
        result_area.clear()
        result_area.write(text)
        
    @on(ProcessError)
    def on_process_error(self, message: ProcessError) -> None:
        """エラーを処理"""
        result_area = self.query_one("#result-area")
        result_area.write(f"[bold red]❌ {message.error}[/bold red]")


class CategorySelectionScreen(Screen):
    """カテゴリ選択画面"""
    
    CSS = """
    CategorySelectionScreen {
        align: center middle;
    }
    
    #category-container {
        width: 80;
        height: auto;
        border: solid $primary;
        padding: 2;
    }
    
    .category-card {
        margin: 1;
        padding: 1;
        border: solid $surface;
    }
    
    .category-card:hover {
        border: solid $primary;
        background: $boost;
    }
    
    .category-title {
        text-style: bold;
        color: $primary;
    }
    
    .category-subtitle {
        color: $text-muted;
    }
    
    .category-description {
        margin-top: 1;
    }
    
    #button-container {
        dock: bottom;
        height: 3;
        align: center middle;
    }
    
    Button {
        margin: 0 1;
    }
    """
    
    def compose(self) -> ComposeResult:
        yield Header()
        
        with Container(id="category-container"):
            yield Label("どのタイプのGPTを作成しますか？", classes="title")
            
            for key, data in CATEGORIES.items():
                card = CategoryCard(key, data, classes="category-card")
                card.can_focus = True
                yield card
                
        with Horizontal(id="button-container"):
            yield Button("カスタム入力", id="custom", variant="primary")
            yield Button("戻る", id="back")
            
        yield Footer()
        
    def on_mount(self) -> None:
        self.query_one(".category-card").focus()
        
    async def on_category_card_clicked(self, message: CategoryCardClicked) -> None:
        """カテゴリカードがクリックされた時の処理"""
        self.app.selected_category = message.category_key
        await self.app.push_screen(PresetSelectionScreen(message.category_key))
            
    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "custom":
            await self.app.push_screen(CustomPromptScreen())
        elif event.button.id == "back":
            self.app.pop_screen()


class PresetSelectionScreen(Screen):
    """プリセット選択画面"""
    
    CSS = """
    PresetSelectionScreen {
        align: center middle;
    }
    
    #preset-container {
        width: 90;
        height: auto;
        max-height: 80%;
        border: solid $primary;
        padding: 2;
    }
    
    .preset-card {
        margin: 1;
        padding: 1;
        border: solid $surface;
    }
    
    .preset-card:hover {
        border: solid $primary;
        background: $boost;
    }
    
    .preset-title {
        text-style: bold;
        color: $primary;
    }
    
    .preset-prompt {
        color: $text-muted;
        margin-top: 1;
    }
    
    .preset-description {
        margin-top: 1;
    }
    
    #button-container {
        dock: bottom;
        height: 3;
        align: center middle;
    }
    """
    
    def __init__(self, category_key: str):
        super().__init__()
        self.category_key = category_key
        self.category = CATEGORIES[category_key]
        
    def compose(self) -> ComposeResult:
        yield Header()
        
        with ScrollableContainer(id="preset-container"):
            yield Label(f"{self.category['name']} - プリセット選択", classes="title")
            
            for i, preset in enumerate(self.category['presets'], 1):
                card = PresetCard(preset, i, classes="preset-card")
                card.can_focus = True
                yield card
                
        with Horizontal(id="button-container"):
            yield Button("カスタム入力", id="custom", variant="primary")
            yield Button("戻る", id="back")
            
        yield Footer()
        
    def on_mount(self) -> None:
        self.query_one(".preset-card").focus()
        
    async def on_preset_card_clicked(self, message: PresetCardClicked) -> None:
        """プリセットカードがクリックされた時の処理"""
        self.app.selected_prompt = message.preset_data['prompt']
        self.app.selected_preset_info = message.preset_data
        await self.app.push_screen(EpochSelectionScreen())
            
    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "custom":
            await self.app.push_screen(CustomPromptScreen())
        elif event.button.id == "back":
            self.app.pop_screen()


class EpochSelectionScreen(Screen):
    """エポック数選択画面"""
    
    CSS = """
    EpochSelectionScreen {
        align: center middle;
    }
    
    #epoch-container {
        width: 80;
        height: auto;
        border: solid $primary;
        padding: 2;
    }
    
    .epoch-option {
        margin: 1;
        padding: 1;
        border: solid $surface;
    }
    
    .epoch-option:hover {
        border: solid $primary;
        background: $boost;
    }
    
    .epoch-title {
        text-style: bold;
        color: $primary;
    }
    
    .epoch-description {
        margin-top: 1;
    }
    
    .epoch-time {
        color: $warning;
        margin-top: 1;
    }
    
    #button-container {
        dock: bottom;
        height: 3;
        align: center middle;
    }
    """
    
    EPOCH_OPTIONS = [
        {
            "epochs": 20,
            "name": "高速テスト",
            "description": "動作確認用。最小限の学習で素早く結果を確認",
            "time_estimate": "GPU: 約30秒 / CPU: 約2-3分"
        },
        {
            "epochs": 50,
            "name": "標準",
            "description": "バランスの取れた設定。基本的なパターンを学習",
            "time_estimate": "GPU: 約1-2分 / CPU: 約5-10分"
        },
        {
            "epochs": 100,
            "name": "品質重視",
            "description": "より深い学習。生成品質の向上を期待",
            "time_estimate": "GPU: 約2-5分 / CPU: 約15-20分"
        },
        {
            "epochs": 200,
            "name": "高品質",
            "description": "十分な学習時間。小規模データでも良好な結果",
            "time_estimate": "GPU: 約5-10分 / CPU: 約30-40分"
        },
        {
            "epochs": 500,
            "name": "最大品質",
            "description": "徹底的な学習。過学習のリスクあり",
            "time_estimate": "GPU: 約15-20分 / CPU: 約1-2時間"
        }
    ]
    
    def compose(self) -> ComposeResult:
        yield Header()
        
        with Container(id="epoch-container"):
            yield Label("学習エポック数を選択してください", classes="title")
            
            for option in self.EPOCH_OPTIONS:
                card = EpochOption(
                    option['epochs'],
                    option['name'],
                    option['description'],
                    option['time_estimate'],
                    classes="epoch-option"
                )
                card.can_focus = True
                yield card
                
        with Horizontal(id="button-container"):
            yield Button("カスタム入力", id="custom", variant="primary")
            yield Button("戻る", id="back")
            
        yield Footer()
        
    def on_mount(self) -> None:
        self.query_one(".epoch-option").focus()
        
    async def on_epoch_option_clicked(self, message: EpochOptionClicked) -> None:
        """エポックオプションがクリックされた時の処理"""
        self.app.selected_epochs = message.epochs
        await self.app.push_screen(DataGenerationMethodScreen())
            
    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "custom":
            await self.app.push_screen(CustomEpochScreen())
        elif event.button.id == "back":
            self.app.pop_screen()


class DataGenerationMethodScreen(Screen):
    """データ生成方法選択画面"""
    
    CSS = """
    DataGenerationMethodScreen {
        align: center middle;
    }
    
    #method-container {
        width: 90;
        max-height: 80%;
        border: solid $primary;
        padding: 2;
    }
    
    .method-card {
        margin: 2;
        padding: 2;
        border: solid $surface;
    }
    
    .method-card:hover {
        border: solid $primary;
        background: $boost;
    }
    
    .method-title {
        text-style: bold;
        color: $primary;
    }
    
    .method-description {
        margin-top: 1;
        color: $text;
    }
    
    .advantages-list {
        margin-top: 1;
        margin-left: 2;
    }
    
    .method-advantage {
        color: $success;
    }
    
    #button-container {
        dock: bottom;
        height: 3;
        align: center middle;
    }
    """
    
    def compose(self) -> ComposeResult:
        yield Header()
        
        with ScrollableContainer(id="method-container"):
            yield Label("データ生成方法を選択してください", classes="title")
            
            template_card = GenerationMethodCard(
                "template",
                "📝 テンプレートベース生成",
                "事前定義されたテンプレートを使用してデータを生成",
                [
                    "オフライン対応 - インターネット接続不要",
                    "高速生成 - 即座に結果を取得",
                    "安定動作 - 外部サービスに依存しない",
                    "軽量 - リソース使用量が少ない"
                ],
                classes="method-card"
            )
            template_card.can_focus = True
            yield template_card
            
            ollama_card = GenerationMethodCard(
                "ollama",
                "🤖 Ollamaベース生成（推奨）",
                "Ollamaの大規模言語モデルを使用して高品質なデータを生成",
                [
                    "高品質 - より自然で多様な文章生成",
                    "カスタマイズ可能 - モデル選択により特性を調整",
                    "プロンプトに忠実 - 指定した性質をより正確に反映",
                    "精選機能 - 生成したデータの品質評価も可能"
                ],
                classes="method-card"
            )
            ollama_card.can_focus = True
            yield ollama_card
                
        with Horizontal(id="button-container"):
            yield Button("戻る", id="back")
            
        yield Footer()
        
    def on_mount(self) -> None:
        self.query_one(".method-card").focus()
        
    async def on_generation_method_clicked(self, message: GenerationMethodClicked) -> None:
        """生成方法が選択された時の処理"""
        self.app.selected_generation_method = message.method
        
        if message.method == "ollama":
            # Ollamaモデル選択画面へ
            await self.app.push_screen(OllamaModelSelectionScreen("generation"))
        else:
            # テンプレートベースの場合は直接メイン処理へ
            await self.app.push_screen(MainProcessScreen())
            
    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "back":
            self.app.pop_screen()


class OllamaModelSelectionScreen(Screen):
    """Ollamaモデル選択画面"""
    
    CSS = """
    OllamaModelSelectionScreen {
        align: center middle;
    }
    
    #model-container {
        width: 100;
        height: 80%;
        border: solid $primary;
        padding: 2;
    }
    
    .model-card {
        margin: 1;
        padding: 1;
        border: solid $surface;
    }
    
    .model-card:hover {
        border: solid $primary;
        background: $boost;
    }
    
    .model-card.unavailable {
        opacity: 0.6;
    }
    
    .model-card.unavailable:hover {
        border: solid $surface;
        background: $background;
    }
    
    .model-title {
        text-style: bold;
    }
    
    .model-description {
        margin-top: 1;
    }
    
    .model-install-hint {
        color: $warning;
        margin-top: 1;
    }
    
    #button-container {
        dock: bottom;
        height: 3;
        align: center middle;
    }
    
    #loading {
        align: center middle;
    }
    """
    
    def __init__(self, purpose: str):
        super().__init__()
        self.purpose = purpose  # "generation" or "refinement"
        self.available_models = []
        self.checking_models = True
        
    def compose(self) -> ComposeResult:
        yield Header()
        
        with Container(id="model-container"):
            if self.purpose == "generation":
                yield Label("データ生成に使用するモデルを選択してください", classes="title")
            else:
                yield Label("データ精選（品質評価）に使用するモデルを選択してください", classes="title")
            
            with ScrollableContainer(id="model-list"):
                yield LoadingIndicator(id="loading")
                
        with Horizontal(id="button-container"):
            if self.purpose == "refinement":
                yield Button("生成と同じモデルを使用", id="same-model", variant="primary")
            yield Button("戻る", id="back")
            
        yield Footer()
        
    def on_mount(self) -> None:
        """画面マウント時にOllamaモデルをチェック"""
        self.check_ollama_models()
        
    @work(thread=True)
    def check_ollama_models(self) -> None:
        """利用可能なOllamaモデルをチェック"""
        try:
            response = requests.get(f"{self.app.args.ollama_host}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json()
                self.available_models = [m['name'] for m in models.get('models', [])]
            else:
                self.available_models = []
        except:
            self.available_models = []
        
        self.app.call_from_thread(self.display_models)
        
    def display_models(self) -> None:
        """モデルリストを表示"""
        self.checking_models = False
        
        # ローディングインジケータを削除
        try:
            self.query_one("#loading").remove()
        except:
            pass
        
        model_list = self.query_one("#model-list")
        
        # カテゴリに基づいて推奨モデルを決定
        category = self.app.selected_category or "general"
        
        # モデルカードを作成
        for model_name, model_info in OLLAMA_MODEL_INFO.items():
            available = model_name in self.available_models
            recommended = category in model_info.get('best_for', [])
            
            card = ModelCardOllama(
                model_name,
                model_info,
                available,
                recommended,
                self.purpose,
                classes="model-card" + (" unavailable" if not available else "")
            )
            if available:
                card.can_focus = True
            model_list.mount(card)
        
        # 利用可能なモデルがない場合の警告
        if not self.available_models:
            model_list.mount(Label(
                "[red]Ollamaモデルが見つかりません。[/red]\n"
                "Ollamaが起動していることを確認し、モデルをインストールしてください。\n"
                "例: ollama pull llama3",
                classes="warning"
            ))
        
    async def on_model_card_ollama_clicked(self, message: ModelCardOllamaClicked) -> None:
        """モデルが選択された時の処理"""
        if self.purpose == "generation":
            self.app.selected_generation_model = message.model_name
            # 精選用モデル選択へ
            await self.app.push_screen(OllamaModelSelectionScreen("refinement"))
        else:
            self.app.selected_refinement_model = message.model_name
            # メイン処理画面へ
            await self.app.push_screen(MainProcessScreen())
            
    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "same-model":
            # 生成と同じモデルを精選にも使用
            self.app.selected_refinement_model = self.app.selected_generation_model
            await self.app.push_screen(MainProcessScreen())
        elif event.button.id == "back":
            self.app.pop_screen()


class CustomPromptScreen(ModalScreen):
    """カスタムプロンプト入力画面"""
    
    CSS = """
    CustomPromptScreen {
        align: center middle;
    }
    
    #dialog {
        width: 60;
        height: auto;
        border: solid $primary;
        background: $surface;
        padding: 2;
    }
    
    Input {
        margin: 1 0;
    }
    
    #button-container {
        margin-top: 2;
        align: center middle;
    }
    """
    
    def compose(self) -> ComposeResult:
        with Container(id="dialog"):
            yield Label("カスタムプロンプトを入力してください", classes="title")
            yield Input(placeholder="例: 料理レシピを生成する楽しいGPT", id="prompt-input")
            
            with Horizontal(id="button-container"):
                yield Button("決定", id="ok", variant="primary")
                yield Button("キャンセル", id="cancel")
                
    def on_mount(self) -> None:
        self.query_one("#prompt-input").focus()
        
    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "ok":
            prompt = self.query_one("#prompt-input").value.strip()
            if prompt:
                self.app.selected_prompt = prompt
                self.app.selected_preset_info = None
                self.dismiss(True)
                # モーダル終了後に画面遷移
                await self.app.push_screen(EpochSelectionScreen())
        elif event.button.id == "cancel":
            self.dismiss(False)


class CustomEpochScreen(ModalScreen):
    """カスタムエポック数入力画面"""
    
    CSS = """
    CustomEpochScreen {
        align: center middle;
    }
    
    #dialog {
        width: 50;
        height: auto;
        border: solid $primary;
        background: $surface;
        padding: 2;
    }
    
    Input {
        margin: 1 0;
    }
    
    #button-container {
        margin-top: 2;
        align: center middle;
    }
    """
    
    def compose(self) -> ComposeResult:
        with Container(id="dialog"):
            yield Label("カスタムエポック数を入力してください", classes="title")
            yield Label("(1-1000の範囲)")
            yield Input(placeholder="例: 150", id="epoch-input", type="integer")
            
            with Horizontal(id="button-container"):
                yield Button("決定", id="ok", variant="primary")
                yield Button("キャンセル", id="cancel")
                
    def on_mount(self) -> None:
        self.query_one("#epoch-input").focus()
        
    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "ok":
            try:
                epochs = int(self.query_one("#epoch-input").value)
                if 1 <= epochs <= 1000:
                    self.app.selected_epochs = epochs
                    self.dismiss(True)
                    await self.app.push_screen(DataGenerationMethodScreen())
            except ValueError:
                pass
        elif event.button.id == "cancel":
            self.dismiss(False)


class MainProcessScreen(Screen):
    """メイン処理画面"""
    
    BINDINGS = [
        Binding("space", "toggle_pause", "一時停止/再開"),
        Binding("q", "quit", "終了"),
        Binding("f", "try_inference", "推論を試す", show=False),
    ]
    
    CSS = """
    MainProcessScreen {
        layout: grid;
        grid-size: 2 2;
        grid-rows: 1fr 1fr;
        grid-columns: 1fr 1fr;
    }
    
    #progress-panel {
        border: solid $primary;
        padding: 1;
        margin: 1;
    }
    
    #preview-panel {
        border: solid $primary;
        padding: 1;
        margin: 1;
    }
    
    #system-panel {
        border: solid $primary;
        padding: 1;
        margin: 1;
    }
    
    #status-panel {
        border: solid $primary;
        padding: 1;
        margin: 1;
    }
    
    .step-item {
        margin: 1 0;
    }
    
    .step-complete {
        color: $success;
    }
    
    .step-current {
        color: $warning;
        text-style: bold;
    }
    
    .step-pending {
        color: $text-muted;
    }
    
    ProgressBar {
        margin: 1 0;
    }
    """
    
    paused = reactive(False)
    current_step = reactive(0)
    
    def __init__(self):
        super().__init__()
        self.worker = None
        self.step_progress = {}
        self.system_info = {}
        self.completed_model_path = None
        self.inference_ready = False
        
    def compose(self) -> ComposeResult:
        yield Header()
        
        # Progress Panel
        with Container(id="progress-panel"):
            yield Label("📊 Progress", classes="panel-title")
            yield Static(id="steps-list")
            
        # Preview Panel  
        with Container(id="preview-panel"):
            yield Label("👁️ Live Preview", classes="panel-title")
            yield RichLog(highlight=True, markup=True, id="preview-log")
            
        # System Info Panel
        with Container(id="system-panel"):
            yield Label("🖥️ System Info", classes="panel-title")
            yield Static(id="system-info")
            
        # Status Panel
        with Container(id="status-panel"):
            yield Label("📋 Status", classes="panel-title")
            yield Static(id="status-info")
            
        yield Footer()
        
    def on_mount(self) -> None:
        """画面マウント時の処理"""
        # 初期状態を表示
        self.update_status_info()
        self.initialize_progress_display()
        self.initialize_system_info()
        self.initialize_preview()
        
        # メイン処理を開始
        self.start_main_process()
        
    def initialize_progress_display(self) -> None:
        """プログレス表示の初期化"""
        steps_widget = self.query_one("#steps-list")
        
        steps = [
            ("Data Generation", "data_gen"),
            ("Data Refinement", "data_refine"),
            ("Dataset Creation", "dataset"),
            ("Model Initialization", "model_init"),
            ("Model Training", "training"),
            ("Generation Test", "test"),
            ("Model & Dataset Saving", "save")
        ]
        
        content = []
        for i, (name, key) in enumerate(steps):
            style = "step-pending"
            status = " "
            content.append(f"[{style}][{status}] {i+1}. {name}[/{style}]")
        
        steps_widget.update("\n".join(content))
        
    def initialize_system_info(self) -> None:
        """システム情報の初期化"""
        system_widget = self.query_one("#system-info")
        
        content = []
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            content.append(f"[bold]GPU:[/bold] {device_name}")
            content.append(f"[bold]Memory:[/bold] 0.0/{memory_total:.1f} GB")
        else:
            content.append("[bold]Device:[/bold] CPU")
        
        content.append("[bold]Status:[/bold] Initializing...")
        system_widget.update("\n".join(content))
        
    def initialize_preview(self) -> None:
        """プレビューの初期化"""
        preview_log = self.query_one("#preview-log", RichLog)
        preview_log.write("[dim]Waiting for generation to start...[/dim]")
        
    def update_status_info(self) -> None:
        """ステータス情報を更新"""
        status_widget = self.query_one("#status-info")
        
        prompt = self.app.selected_prompt[:50] + "..." if len(self.app.selected_prompt) > 50 else self.app.selected_prompt
        epochs = self.app.selected_epochs if hasattr(self.app, 'selected_epochs') else 20
        
        # データ生成方法の情報を追加
        gen_method = getattr(self.app, 'selected_generation_method', 'template')
        if gen_method == "ollama":
            gen_model = getattr(self.app, 'selected_generation_model', 'llama3')
            ref_model = getattr(self.app, 'selected_refinement_model', gen_model)
            gen_info = f"Ollama ({gen_model} / {ref_model})"
        else:
            gen_info = "Template-based"
        
        status_text = f"""
[bold]Prompt:[/bold] {prompt}
[bold]Epochs:[/bold] {epochs}
[bold]Model Size:[/bold] {self.app.args.model_size}
[bold]Generation:[/bold] {gen_info}
[bold]Device:[/bold] {'CUDA' if torch.cuda.is_available() else 'CPU'}
[bold]Status:[/bold] {'⏸️ PAUSED' if self.paused else '▶️ Running'}
        """
        
        status_widget.update(status_text)
        
    def action_toggle_pause(self) -> None:
        """一時停止/再開"""
        self.paused = not self.paused
        self.update_status_info()
        
    def action_quit(self) -> None:
        """終了"""
        if self.worker:
            self.worker.cancel()
        self.app.exit()
        
    @work(thread=True)
    def start_main_process(self) -> None:
        """メイン処理を開始"""
        try:
            # デバッグログ
            self.app.call_from_thread(
                self.post_message,
                PreviewUpdate("Starting main process...")
            )
            
            # アプリケーションのコンテキストから必要な情報を取得
            app = self.app
            config = app.config
            args = app.args
            tokenizer = app.tokenizer
            user_prompt = app.selected_prompt
            preset_info = app.selected_preset_info
            
            # エポック数を設定
            if hasattr(app, 'selected_epochs'):
                config.num_epochs = app.selected_epochs
            
            # データ生成方法の設定を取得
            generation_method = getattr(app, 'selected_generation_method', 'template')
            generation_model = getattr(app, 'selected_generation_model', 'llama3')
            refinement_model = getattr(app, 'selected_refinement_model', generation_model)
            
            # TextualCallbacksを作成
            callbacks = TextualCallbacks(self)
            
            # メイン処理を実行
            folder_path, domain = run_main_process(
                config, args, user_prompt, tokenizer, preset_info, callbacks,
                generation_method=generation_method,
                generation_model=generation_model,
                refinement_model=refinement_model
            )
            
            # 完了したモデルのパスを保存
            self.completed_model_path = folder_path
            
            # 完了通知
            self.app.call_from_thread(
                self.post_message,
                ProcessComplete(folder_path, domain)
            )
            
        except Exception as e:
            self.app.call_from_thread(
                self.post_message,
                ProcessError(str(e))
            )
            
    @on(ProgressUpdate)
    def on_progress_update(self, message: ProgressUpdate) -> None:
        """プログレス更新を処理"""
        self.step_progress[message.step_key] = {
            'current': message.current,
            'total': message.total,
            'info': message.info
        }
        self.update_progress_display()
        
    @on(PreviewUpdate)
    def on_preview_update(self, message: PreviewUpdate) -> None:
        """プレビュー更新を処理"""
        preview_log = self.query_one("#preview-log", RichLog)
        
        if isinstance(message.content, dict):
            for key, value in message.content.items():
                preview_log.write(f"[bold]{key}:[/bold] {value}")
        else:
            preview_log.write(str(message.content))
            
    @on(SystemInfoUpdate)
    def on_system_info_update(self, message: SystemInfoUpdate) -> None:
        """システム情報更新を処理"""
        self.system_info[message.key] = message.value
        self.update_system_info_display()
        
    @on(StepChange)
    def on_step_change(self, message: StepChange) -> None:
        """ステップ変更を処理"""
        self.current_step = message.step
        self.update_progress_display()
        
    def update_progress_display(self) -> None:
        """プログレス表示を更新"""
        steps_widget = self.query_one("#steps-list")
        
        steps = [
            ("Data Generation", "data_gen"),
            ("Data Refinement", "data_refine"),
            ("Dataset Creation", "dataset"),
            ("Model Initialization", "model_init"),
            ("Model Training", "training"),
            ("Generation Test", "test"),
            ("Model & Dataset Saving", "save")
        ]
        
        content = []
        for i, (name, key) in enumerate(steps):
            if i < self.current_step:
                status = "✓"
                style = "step-complete"
            elif i == self.current_step:
                status = "►"
                style = "step-current"
            else:
                status = " "
                style = "step-pending"
                
            content.append(f"[{style}][{status}] {i+1}. {name}[/{style}]")
            
            # プログレスバー表示
            if key in self.step_progress and i <= self.current_step:
                progress = self.step_progress[key]
                if progress['current'] is not None and progress['total'] is not None:
                    percentage = progress['current'] / progress['total'] if progress['total'] > 0 else 0
                    bar_width = 30
                    filled = int(bar_width * percentage)
                    bar = "█" * filled + "░" * (bar_width - filled)
                    content.append(f"    {bar} {progress['current']}/{progress['total']}")
                    
                if progress['info']:
                    content.append(f"    {progress['info']}")
                    
        steps_widget.update("\n".join(content))
        
    def update_system_info_display(self) -> None:
        """システム情報表示を更新"""
        system_widget = self.query_one("#system-info")
        
        content = []
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            memory_used = torch.cuda.memory_allocated(0) / 1024**3
            memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            content.append(f"[bold]GPU:[/bold] {device_name}")
            content.append(f"[bold]Memory:[/bold] {memory_used:.1f}/{memory_total:.1f} GB")
        else:
            content.append("[bold]Device:[/bold] CPU")
            
        for key, value in self.system_info.items():
            content.append(f"[bold]{key}:[/bold] {value}")
            
        system_widget.update("\n".join(content))
        
    @on(ProcessComplete)
    def on_process_complete(self, message: ProcessComplete) -> None:
        """処理完了を処理"""
        preview_log = self.query_one("#preview-log", RichLog)
        preview_log.write("\n" + "="*50)
        preview_log.write("[bold green]✅ 処理が完了しました！[/bold green]")
        preview_log.write(f"[bold]モデルフォルダ:[/bold] {message.folder_path}")
        preview_log.write(f"[bold]ドメイン:[/bold] {message.domain}")
        preview_log.write("="*50)
        
        # 推論を試すボタンを追加
        if self.completed_model_path:
            preview_log.write("\n[bold cyan]このモデルで推論を試しますか？[/bold cyan]")
            preview_log.write("Fキーを押すと推論画面へ移動します")
            
            # 推論準備フラグを設定
            self.inference_ready = True
            
    def action_try_inference(self) -> None:
        """作成したモデルで推論を試す"""
        if not self.inference_ready or not self.completed_model_path:
            return
            
        # モデル情報を読み込み
        info_path = os.path.join(self.completed_model_path, "generation_info.json")
        with open(info_path, 'r', encoding='utf-8') as f:
            model_info = json.load(f)
        
        # アプリケーションに設定
        self.app.selected_model_path = self.completed_model_path
        self.app.selected_model_info = model_info
        
        # 全画面をポップしてメインメニューに戻る
        while len(self.app.screen_stack) > 1:
            self.app.pop_screen()
        
        # 推論画面へ
        self.app.push_screen(InferenceScreen())
        
    @on(ProcessError)
    def on_process_error(self, message: ProcessError) -> None:
        """エラーを処理"""
        preview_log = self.query_one("#preview-log", RichLog)
        preview_log.write(f"[bold red]❌ エラー: {message.error}[/bold red]")


# ===== 5. Textual対応コールバック =====
class TextualCallbacks:
    """Textual用のコールバック実装"""
    
    def __init__(self, screen):
        self.screen = screen
        self.app = screen.app if screen else None
        self.worker = None
        
        # Textualモードでのみワーカーを取得
        if screen is not None:
            try:
                self.worker = get_current_worker()
            except:
                # ワーカーコンテキスト外の場合はNoneのまま
                self.worker = None
        
    def update_progress(self, step_key: str, current: Optional[int] = None, 
                       total: Optional[int] = None, info: Optional[str] = None):
        """プログレスの更新"""
        if self.screen and self.app and self.worker and not self.worker.is_cancelled:
            self.app.call_from_thread(
                self.screen.post_message,
                ProgressUpdate(step_key, current, total, info)
            )
            
    def update_preview(self, content: Any):
        """プレビューの更新"""
        if self.screen and self.app and self.worker and not self.worker.is_cancelled:
            self.app.call_from_thread(
                self.screen.post_message,
                PreviewUpdate(content)
            )
            
    def update_system_info(self, key: str, value: str):
        """システム情報の更新"""
        if self.screen and self.app and self.worker and not self.worker.is_cancelled:
            self.app.call_from_thread(
                self.screen.post_message,
                SystemInfoUpdate(key, value)
            )
            
    def set_current_step(self, step: int):
        """現在のステップを設定"""
        if self.screen and self.app and self.worker and not self.worker.is_cancelled:
            self.app.call_from_thread(
                self.screen.post_message,
                StepChange(step)
            )
            
    def wait_if_paused(self):
        """一時停止中は待機"""
        if self.screen and hasattr(self.screen, 'paused'):
            while self.screen.paused:
                time.sleep(0.1)


# ===== 6. Ollama統合データ生成モジュール =====
class OllamaDataGenerator:
    """Ollamaを使用したデータ生成クラス"""
    
    def __init__(self, 
                 prompt: str, 
                 tokenizer,
                 ollama_host: str = "http://localhost:11434",
                 model: str = "llama3",
                 use_ollama: bool = True,
                 callbacks: Optional[TextualCallbacks] = None):
        """
        Args:
            prompt: ユーザーが指定するモデルの性質
            tokenizer: 使用するトークナイザー
            ollama_host: OllamaサーバーのURL
            model: 使用するOllamaモデル
            use_ollama: Ollamaを使用するかどうか
            callbacks: Textual更新用のコールバック
        """
        self.prompt = prompt
        self.tokenizer = tokenizer
        self.ollama_host = ollama_host
        self.model = model
        self.use_ollama = use_ollama
        self.callbacks = callbacks
        self.domain = self._extract_domain(prompt)
        
        if self.use_ollama:
            self.ollama_available = self._check_ollama_connection()
        else:
            self.ollama_available = False
            if self.callbacks:
                self.callbacks.update_system_info("Generation", "Template-based")
        
    def _check_ollama_connection(self) -> bool:
        """Ollamaサーバーへの接続を確認"""
        try:
            response = requests.get(f"{self.ollama_host}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json()
                model_names = [m['name'] for m in models.get('models', [])]
                
                # モデルの情報を表示
                model_info = OLLAMA_MODEL_INFO.get(self.model, {})
                model_display = f"{self.model} ({model_info.get('size', 'Unknown')})"
                
                if self.model in model_names:
                    status = f"✓ {model_display}"
                else:
                    status = f"× {self.model} not found"
                
                if self.callbacks:
                    self.callbacks.update_system_info("Generation Model", status)
                
                print(f"Ollama接続成功。利用可能なモデル: {model_names}")
                
                if self.model not in model_names:
                    print(f"警告: {self.model}が見つかりません。")
                    if model_names:
                        self.model = model_names[0]
                        print(f"代わりに{self.model}を使用します。")
                        return True
                    return False
                return True
            return False
        except:
            if self.callbacks:
                self.callbacks.update_system_info("Generation", "× Offline")
            return False
    
    def _extract_domain(self, prompt: str) -> str:
        """プロンプトからドメインを抽出"""
        for cat_key, cat_data in CATEGORIES.items():
            if cat_data['name_ja'] in prompt:
                return cat_key
            for preset in cat_data['presets']:
                if any(keyword in prompt for keyword in preset['prompt'].split()):
                    return cat_key
                    
        if "料理" in prompt or "レシピ" in prompt:
            return "cooking"
        elif "詩" in prompt or "ポエム" in prompt or "創作" in prompt:
            return "poetry"
        elif "技術" in prompt or "プログラ" in prompt:
            return "technical"
        else:
            return "general"
    
    def generate_with_ollama(self, prompt: str, max_tokens: int = 100) -> str:
        """Ollamaを使用してテキストを生成"""
        try:
            response = requests.post(
                f"{self.ollama_host}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_predict": max_tokens,
                        "temperature": 0.8,
                        "top_p": 0.9,
                    }
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['response'].strip()
            else:
                return self._fallback_generation()
                
        except Exception as e:
            return self._fallback_generation()
    
    def _fallback_generation(self) -> str:
        """Ollamaが使えない場合のフォールバック（テンプレートベース）"""
        if self.domain == "cooking":
            return self._generate_cooking_sample()
        elif self.domain == "poetry":
            return self._generate_poetry_sample()
        elif self.domain == "technical":
            return self._generate_technical_sample()
        else:
            return self._generate_general_sample()
    
    def _generate_cooking_sample(self) -> str:
        """料理レシピのサンプル生成（テンプレート）"""
        dishes = ["パスタ", "カレー", "サラダ", "スープ", "ケーキ", "寿司", "天ぷら", "ラーメン"]
        ingredients = ["トマト", "玉ねぎ", "にんじん", "じゃがいも", "肉", "魚", "卵", "チーズ"]
        
        dish = random.choice(dishes)
        ing1, ing2 = random.sample(ingredients, 2)
        
        templates = [
            f"{dish}の作り方：まず{ing1}を切ります。次に{ing2}を加えて炒めます。",
            f"簡単{dish}レシピ。{ing1}と{ing2}を使った美味しい料理です。",
            f"今日の献立は{dish}。材料は{ing1}、{ing2}などです。",
            f"プロが教える{dish}の極意。{ing1}の下処理がポイントです。"
        ]
        
        return random.choice(templates)
    
    def _generate_poetry_sample(self) -> str:
        """詩的なサンプル生成（テンプレート）"""
        themes = ["春", "夏", "秋", "冬", "愛", "希望", "夢", "時間", "記憶"]
        emotions = ["喜び", "悲しみ", "懐かしさ", "期待", "安らぎ", "切なさ", "感動"]
        
        theme = random.choice(themes)
        emotion = random.choice(emotions)
        
        templates = [
            f"{theme}の{emotion}を感じながら、私は静かに歩く。",
            f"{emotion}に包まれた{theme}の日々。心に響く瞬間。",
            f"{theme}よ、永遠に。{emotion}と共に生きていく。",
            f"言葉にできない{emotion}が、{theme}の中で揺れている。"
        ]
        
        return random.choice(templates)
    
    def _generate_technical_sample(self) -> str:
        """技術文書のサンプル生成（テンプレート）"""
        topics = ["Python", "機械学習", "API", "データベース", "アルゴリズム", "セキュリティ", "クラウド"]
        actions = ["実装", "最適化", "設計", "分析", "構築", "デバッグ", "テスト"]
        
        topic = random.choice(topics)
        action = random.choice(actions)
        
        templates = [
            f"{topic}の{action}について説明します。まず基本的な概念から始めます。",
            f"{action}を行う際の{topic}のベストプラクティスを紹介します。",
            f"効率的な{topic}の{action}方法。パフォーマンスを向上させるコツ。",
            f"{topic}における{action}の重要性と具体的な手順を解説します。"
        ]
        
        return random.choice(templates)
    
    def _generate_general_sample(self) -> str:
        """一般的なサンプル生成（テンプレート）"""
        subjects = ["今日", "明日", "人生", "世界", "私たち", "社会", "未来"]
        predicates = ["素晴らしい", "新しい", "大切な", "興味深い", "楽しい", "挑戦的な"]
        
        subj = random.choice(subjects)
        pred = random.choice(predicates)
        
        templates = [
            f"{subj}は{pred}。それが私の考えです。",
            f"{pred}な{subj}について考えてみましょう。",
            f"{subj}をより{pred}ものにするために、私たちができることは何でしょうか。"
        ]
        
        return random.choice(templates)
    
    def generate_samples(self, num_tokens: int) -> List[str]:
        """指定されたトークン数分のテキストサンプルを生成"""
        samples = []
        current_tokens = 0
        
        if self.use_ollama and self.ollama_available:
            model_info = OLLAMA_MODEL_INFO.get(self.model, {})
            print(f"\nOllamaを使用してドメイン '{self.domain}' のデータを生成中...")
            print(f"モデル: {self.model} - {model_info.get('description', 'No description')}")
            generation_prompts = self._create_generation_prompts()
        else:
            if not self.use_ollama:
                print(f"\nテンプレートベースで生成中...")
            else:
                print(f"\nOllamaが利用できないため、テンプレートベースで生成中...")
            print(f"ドメイン: {self.domain}")
        
        # Textual用のプログレス処理とtqdmの切り替え
        if self.callbacks and self.callbacks.app:
            total_iterations = 0
            prompt_index = 0
            
            while current_tokens < num_tokens:
                self.callbacks.wait_if_paused()
                
                if self.use_ollama and self.ollama_available:
                    current_prompt = generation_prompts[prompt_index % len(generation_prompts)]
                    sample = self.generate_with_ollama(current_prompt, max_tokens=150)
                    time.sleep(0.5)
                else:
                    sample = self._fallback_generation()
                
                if sample and len(sample) > 20:
                    samples.append(sample)
                    sample_tokens = len(self.tokenizer.encode(sample))
                    current_tokens += sample_tokens
                    
                    # Textual更新
                    self.callbacks.update_progress(
                        'data_gen',
                        current=current_tokens,
                        total=num_tokens,
                        info=f"Samples: {len(samples)} | Mode: {'Ollama' if self.use_ollama and self.ollama_available else 'Template'}"
                    )
                    
                    # プレビュー更新
                    if len(samples) % 5 == 0:
                        self.callbacks.update_preview({
                            "Latest Sample": sample[:100] + "...",
                            "Total Samples": len(samples),
                            "Tokens Generated": current_tokens
                        })
                
                prompt_index += 1
                total_iterations += 1
                
                if total_iterations > 100 and current_tokens < num_tokens * 0.5:
                    break
        else:
            # 非Textualモード（従来のtqdm使用）
            pbar = tqdm(total=num_tokens, desc="トークン生成", unit="tokens")
            prompt_index = 0
            
            while current_tokens < num_tokens:
                if self.use_ollama and self.ollama_available:
                    current_prompt = generation_prompts[prompt_index % len(generation_prompts)]
                    sample = self.generate_with_ollama(current_prompt, max_tokens=150)
                    time.sleep(0.5)
                else:
                    sample = self._fallback_generation()
                
                if sample and len(sample) > 20:
                    samples.append(sample)
                    sample_tokens = len(self.tokenizer.encode(sample))
                    current_tokens += sample_tokens
                    pbar.update(sample_tokens)
                
                prompt_index += 1
                
                if len(samples) > 100 and current_tokens < num_tokens * 0.5:
                    break
            
            pbar.close()
        
        print(f"生成完了: {len(samples)}個のサンプル、約{current_tokens}トークン")
        
        return samples
    
    def _create_generation_prompts(self) -> List[str]:
        """ドメインに応じた生成プロンプトのリストを作成"""
        base_instruction = f"以下の指示に従って、{self.prompt}ような文章を生成してください。\n\n"
        
        if self.domain == "cooking":
            return [
                base_instruction + "簡単な料理レシピを1つ書いてください。材料と手順を含めてください。",
                base_instruction + "季節の食材を使った料理のアイデアを説明してください。",
                base_instruction + "初心者向けの料理のコツを1つ教えてください。",
                base_instruction + "プロが使う調理技術を1つ紹介してください。",
                base_instruction + "健康的な料理のポイントを説明してください。"
            ]
        elif self.domain == "poetry":
            return [
                base_instruction + "自然をテーマにした短い詩を書いてください。",
                base_instruction + "感情を表現する詩的な文章を書いてください。",
                base_instruction + "季節の移ろいを表現する短文を書いてください。",
                base_instruction + "人生について考察する詩的な文章を書いてください。",
                base_instruction + "愛や友情についての詩を創作してください。"
            ]
        elif self.domain == "technical":
            return [
                base_instruction + "プログラミングの基本概念を1つ説明してください。",
                base_instruction + "技術的な問題解決の方法を1つ紹介してください。",
                base_instruction + "ソフトウェア開発のベストプラクティスを1つ説明してください。",
                base_instruction + "システム設計の重要な原則を1つ解説してください。",
                base_instruction + "最新の技術トレンドについて説明してください。"
            ]
        else:  # general
            return [
                base_instruction + "日常的な話題について自然な文章を書いてください。",
                base_instruction + "興味深い事実や知識を1つ共有してください。",
                base_instruction + "身近な出来事について説明してください。",
                base_instruction + "社会的な話題について意見を述べてください。",
                base_instruction + "人生のアドバイスを1つ書いてください。"
            ]


# ===== 7. データ精選モジュール =====
class DataRefiner:
    """生成されたデータを精選するクラス"""
    
    def __init__(self, prompt: str, tokenizer, use_ollama: bool = False,
                 ollama_host: str = "http://localhost:11434", model: str = "llama3",
                 callbacks: Optional[TextualCallbacks] = None):
        self.prompt = prompt
        self.tokenizer = tokenizer
        self.use_ollama = use_ollama
        self.ollama_host = ollama_host
        self.model = model
        self.callbacks = callbacks
        
        if self.use_ollama and callbacks:
            # モデル情報を表示
            model_info = OLLAMA_MODEL_INFO.get(self.model, {})
            model_display = f"{self.model} ({model_info.get('size', 'Unknown')})"
            callbacks.update_system_info("Refinement Model", model_display)
        
    def refine_samples(self, samples: List[str], target_tokens: int) -> List[str]:
        """サンプルを評価し、目標トークン数まで精選"""
        print("\nデータ精選中...")
        
        scored_samples = []
        
        # Textual用の進捗表示
        if self.callbacks and self.callbacks.app:
            for i, sample in enumerate(samples):
                self.callbacks.wait_if_paused()
                
                if self.use_ollama:
                    score = self._evaluate_with_ollama(sample)
                else:
                    score = self._calculate_score(sample)
                    
                scored_samples.append((score, sample))
                
                # 進捗更新
                self.callbacks.update_progress(
                    'data_refine',
                    current=i + 1,
                    total=len(samples),
                    info=f"Evaluating quality..."
                )
                
                # プレビュー更新（10サンプルごと）
                if (i + 1) % 10 == 0:
                    avg_score = sum(s[0] for s in scored_samples) / len(scored_samples)
                    self.callbacks.update_preview({
                        "Samples Evaluated": i + 1,
                        "Average Score": f"{avg_score:.2f}",
                        "Latest Score": f"{score:.2f}"
                    })
        else:
            # 非Textualモード
            for sample in tqdm(samples, desc="サンプル評価"):
                if self.use_ollama:
                    score = self._evaluate_with_ollama(sample)
                else:
                    score = self._calculate_score(sample)
                scored_samples.append((score, sample))
        
        # スコアの高い順にソート
        scored_samples.sort(reverse=True, key=lambda x: x[0])
        
        # 目標トークン数に達するまで上位サンプルを選択
        refined_samples = []
        current_tokens = 0
        
        for score, sample in scored_samples:
            sample_tokens = len(self.tokenizer.encode(sample))
            if current_tokens + sample_tokens <= target_tokens:
                refined_samples.append(sample)
                current_tokens += sample_tokens
            
            if current_tokens >= target_tokens * 0.95:
                break
        
        avg_score = sum(s[0] for s in scored_samples[:len(refined_samples)]) / len(refined_samples) if refined_samples else 0
        
        # 最終結果の更新
        if self.callbacks and self.callbacks.app:
            self.callbacks.update_progress(
                'data_refine',
                current=len(refined_samples),
                total=len(refined_samples),
                info=f"Quality: {avg_score:.2f} | Tokens: {current_tokens}"
            )
        
        print(f"精選完了: {len(samples)} → {len(refined_samples)} サンプル")
        print(f"トークン数: 約{current_tokens}")
        print(f"平均品質スコア: {avg_score:.2f}")
        
        return refined_samples
    
    def _evaluate_with_ollama(self, sample: str) -> float:
        """Ollamaを使用してサンプルを評価"""
        evaluation_prompt = f"""
以下の文章が「{self.prompt}」という目的にどれくらい適しているか評価してください。
0から10の数値で答えてください（0が最低、10が最高）。数値のみを返してください。

文章：
{sample}

評価（0-10）："""
        
        try:
            response = requests.post(
                f"{self.ollama_host}/api/generate",
                json={
                    "model": self.model,
                    "prompt": evaluation_prompt,
                    "stream": False,
                    "options": {"num_predict": 10, "temperature": 0.1}
                },
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                score_text = result['response'].strip()
                try:
                    score = float(score_text.split()[0])
                    return min(max(score / 10.0, 0.0), 1.0)
                except:
                    return 0.5
            return 0.5
        except:
            return self._calculate_score(sample)
    
    def _calculate_score(self, sample: str) -> float:
        """サンプルの品質スコアを計算（テンプレートベース）"""
        score = 0.0
        
        # 長さスコア
        length = len(sample)
        if 20 <= length <= 100:
            score += 1.0
        elif 10 <= length <= 150:
            score += 0.5
        
        # 句読点の適切な使用
        if sample.count('。') >= 1:
            score += 0.5
        if sample.count('、') >= 1:
            score += 0.3
        
        # プロンプトとの関連性（簡易的なキーワードマッチング）
        prompt_words = set(self.prompt.split())
        sample_words = set(sample.split())
        overlap = len(prompt_words & sample_words)
        score += overlap * 0.2
        
        # ランダム性を少し加える（多様性のため）
        score += random.random() * 0.3
        
        return score


# ===== 8. カスタムデータセット =====
class TextDataset(Dataset):
    """PyTorchのDatasetクラス"""
    
    def __init__(self, texts: List[str], tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # すべてのテキストをトークン化
        self.tokens = []
        for text in texts:
            encoded = tokenizer.encode(text, max_length=max_length, truncation=True)
            self.tokens.extend(encoded)
        
        print(f"データセット作成完了: {len(self.tokens)} トークン")
    
    def __len__(self):
        return max(1, len(self.tokens) - self.max_length)
    
    def __getitem__(self, idx):
        chunk = self.tokens[idx:idx + self.max_length + 1]
        
        if len(chunk) < self.max_length + 1:
            chunk = chunk + [self.tokenizer.pad_token_id] * (self.max_length + 1 - len(chunk))
        
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        
        return x, y


# ===== 9. GPTモデルの定義 =====
class CausalSelfAttention(nn.Module):
    """マルチヘッド自己注意機構"""
    
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(0.1)
        self.resid_dropout = nn.Dropout(0.1)
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        
        self.register_buffer("bias", torch.tril(torch.ones(config.n_positions, config.n_positions))
                                     .view(1, 1, config.n_positions, config.n_positions))
    
    def forward(self, x):
        B, T, C = x.size()
        
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        
        att = (q @ k.transpose(-2, -1)) * (1.0 / np.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = torch.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        y = self.resid_dropout(self.c_proj(y))
        
        return y


class Block(nn.Module):
    """Transformerブロック"""
    
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(0.1)
        )
    
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    """GPTモデル本体"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(0.1)
        self.h = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        self.apply(self._init_weights)
        
        n_params = sum(p.numel() for p in self.parameters())
        print(f"モデルパラメータ数: {n_params/1e6:.2f}M")
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(self, idx, targets=None):
        b, t = idx.size()
        assert t <= self.config.n_positions, f"系列長 {t} が最大値 {self.config.n_positions} を超えています"
        
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device).unsqueeze(0)
        
        tok_emb = self.wte(idx)
        pos_emb = self.wpe(pos)
        x = self.drop(tok_emb + pos_emb)
        
        for block in self.h:
            x = block(x)
        
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), 
                targets.view(-1),
                ignore_index=self.config.vocab_size - 1
            )
        
        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.n_positions else idx[:, -self.config.n_positions:]
            
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx


# ===== 10. トレーニング関数 =====
def train_model(model, train_loader, config, callbacks: Optional[TextualCallbacks] = None):
    """モデルのトレーニング"""
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
    
    print(f"\nトレーニング開始（{config.num_epochs}エポック）")
    
    callbacks = callbacks or TextualCallbacks(None)
    
    # 初期プログレスを設定（修正箇所）
    if callbacks and callbacks.app:
        callbacks.update_progress(
            'training',
            current=0,
            total=config.num_epochs,
            info="Preparing training..."
        )
    
    # 更新頻度の設定
    update_interval = max(1, config.num_epochs // 20)
    preview_interval = max(5, config.num_epochs // 4)
    
    for epoch in range(config.num_epochs):
        total_loss = 0
        num_batches = 0
        
        if callbacks and callbacks.app:
            # Textualモード
            for batch_idx, (x, y) in enumerate(train_loader):
                callbacks.wait_if_paused()
                
                x, y = x.to(config.device), y.to(config.device)
                
                optimizer.zero_grad()
                logits, loss = model(x, y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            # エポック終了後の更新
            if (epoch + 1) % update_interval == 0 or epoch == 0 or epoch == config.num_epochs - 1:
                # 進捗更新
                callbacks.update_progress(
                    'training',
                    current=epoch + 1,
                    total=config.num_epochs,
                    info=f"Loss: {total_loss/num_batches:.4f}"
                )
                
                # システム情報更新
                if (epoch + 1) % (update_interval * 2) == 0:
                    if torch.cuda.is_available():
                        callbacks.update_system_info(
                            "GPU Memory",
                            f"{torch.cuda.memory_allocated(0)/1024**3:.1f} GB"
                        )
                
                # プレビュー更新
                if (epoch + 1) % preview_interval == 0 or epoch == config.num_epochs - 1:
                    model.eval()
                    sample_text = generate_sample(model, train_loader.dataset.tokenizer, config)
                    callbacks.update_preview({
                        "Epoch": epoch + 1,
                        "Avg Loss": f"{total_loss/num_batches:.4f}",
                        "Sample": sample_text[:100] + "..."
                    })
                    model.train()
                
        else:
            # 非Textualモード（従来のtqdm使用）
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs}")
            for batch_idx, (x, y) in enumerate(pbar):
                x, y = x.to(config.device), y.to(config.device)
                
                optimizer.zero_grad()
                logits, loss = model(x, y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        
        # 定期的な生成サンプルの表示（非Textualモード）
        if not (callbacks and callbacks.app) and (epoch + 1) % update_interval == 0:
            print(f"\nEpoch {epoch+1} - 平均損失: {avg_loss:.4f}")
            model.eval()
            sample_text = generate_sample(model, train_loader.dataset.tokenizer, config)
            print(f"生成サンプル: {sample_text}")
            model.train()
    
    print("\nトレーニング完了！")


# ===== 11. テキスト生成関数 =====
def generate_sample(model, tokenizer, config, prompt="", max_length=50, temperature=1.0, top_k=40):
    """学習済みモデルでテキストを生成"""
    model.eval()
    
    if prompt:
        tokens = tokenizer.encode(prompt)
        idx = torch.tensor([tokens], dtype=torch.long).to(config.device)
    else:
        idx = torch.tensor([[tokenizer.bos_token_id]], dtype=torch.long).to(config.device)
    
    with torch.no_grad():
        generated = model.generate(idx, max_new_tokens=max_length, temperature=temperature, top_k=top_k)
    
    generated_text = tokenizer.decode(generated[0].tolist(), skip_special_tokens=True)
    return generated_text


# ===== 12. モデルの保存と読み込み =====
def save_model_safetensors(model, tokenizer, config, folder_path, metadata=None):
    """モデルをsafetensors形式で保存（フォルダ管理対応）"""
    print(f"\nモデルを保存中: {folder_path}")
    
    os.makedirs(folder_path, exist_ok=True)
    
    model_path = os.path.join(folder_path, "model.safetensors")
    
    state_dict = model.state_dict()
    
    default_metadata = {
        "model_type": "gpt",
        "vocab_size": str(config.vocab_size),
        "n_embd": str(config.n_embd),
        "n_layer": str(config.n_layer),
        "n_head": str(config.n_head),
        "n_positions": str(config.n_positions),
        "description": "Text2GPT1 Generated Model"
    }
    
    if metadata:
        default_metadata.update(metadata)
    
    save_file(state_dict, model_path, metadata=default_metadata)
    
    tokenizer.save_pretrained(folder_path)
    
    config_path = os.path.join(folder_path, "config.json")
    config_dict = {
        "vocab_size": config.vocab_size,
        "n_embd": config.n_embd,
        "n_layer": config.n_layer,
        "n_head": config.n_head,
        "n_positions": config.n_positions
    }
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)
    
    print(f"モデル保存完了: {model_path}")
    print(f"設定保存完了: {config_path}")
    
    return model_path


def load_model_safetensors(folder_path):
    """保存されたモデルを読み込む"""
    # 設定を読み込み
    config_path = os.path.join(folder_path, "config.json")
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = json.load(f)
    
    # Configオブジェクトを作成
    config = Config()
    config.vocab_size = config_dict['vocab_size']
    config.n_embd = config_dict['n_embd']
    config.n_layer = config_dict['n_layer']
    config.n_head = config_dict['n_head']
    config.n_positions = config_dict['n_positions']
    
    # モデルを初期化
    model = GPT(config).to(config.device)
    
    # モデルの重みを読み込み
    model_path = os.path.join(folder_path, "model.safetensors")
    state_dict = load_file(model_path)
    model.load_state_dict(state_dict)
    
    # トークナイザーを読み込み
    tokenizer = GPT2Tokenizer.from_pretrained(folder_path)
    
    return model, tokenizer, config


def save_generation_info(folder_path, info_dict):
    """生成情報をJSONファイルとして保存"""
    info_path = os.path.join(folder_path, "generation_info.json")
    
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(info_dict, f, indent=2, ensure_ascii=False)
    
    print(f"生成情報保存完了: {info_path}")


def save_dataset(folder_path, refined_samples, tokenizer, metadata=None):
    """データセットをJSON形式で保存"""
    os.makedirs(folder_path, exist_ok=True)
    
    dataset_path = os.path.join(folder_path, "dataset.json")
    
    samples_with_info = []
    total_tokens = 0
    
    for sample in refined_samples:
        tokens = tokenizer.encode(sample)
        token_count = len(tokens)
        total_tokens += token_count
        
        samples_with_info.append({
            "text": sample,
            "tokens": token_count
        })
    
    dataset_info = {
        "metadata": {
            "total_samples": len(refined_samples),
            "total_tokens": total_tokens,
            "average_tokens_per_sample": total_tokens / len(refined_samples) if refined_samples else 0,
            "creation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "tokenizer": "gpt2"
        },
        "samples": samples_with_info
    }
    
    if metadata:
        dataset_info["metadata"].update(metadata)
    
    with open(dataset_path, 'w', encoding='utf-8') as f:
        json.dump(dataset_info, f, indent=2, ensure_ascii=False)
    
    print(f"データセット保存完了: {dataset_path}")
    
    text_path = os.path.join(folder_path, "dataset.txt")
    with open(text_path, 'w', encoding='utf-8') as f:
        for i, sample in enumerate(refined_samples):
            f.write(f"=== Sample {i+1} ===\n")
            f.write(sample)
            f.write("\n\n")
    
    print(f"テキスト版データセット保存完了: {text_path}")
    
    return dataset_path


def list_saved_models(output_dir="models"):
    """保存されたモデルの一覧を取得"""
    models = []
    
    if not os.path.exists(output_dir):
        return models
    
    for folder in os.listdir(output_dir):
        folder_path = os.path.join(output_dir, folder)
        info_path = os.path.join(folder_path, "generation_info.json")
        
        if os.path.isdir(folder_path) and os.path.exists(info_path):
            try:
                with open(info_path, 'r', encoding='utf-8') as f:
                    info = json.load(f)
                models.append({
                    "path": folder_path,
                    "info": info
                })
            except:
                pass
    
    # 作成日時でソート（新しい順）
    models.sort(key=lambda x: x['info'].get('creation_date', ''), reverse=True)
    
    return models


# ===== 13. メイン処理 =====
def run_main_process(config, args, user_prompt, tokenizer, preset_info=None, 
                    callbacks: Optional[TextualCallbacks] = None,
                    generation_method: str = "template",
                    generation_model: str = "llama3",
                    refinement_model: str = "llama3"):
    """メイン処理をTextualコールバック付きで実行"""
    callbacks = callbacks or TextualCallbacks(None)
    start_time = time.time()
    
    data_stats = {
        "samples_generated": 0,
        "samples_refined": 0,
        "initial_tokens": config.initial_tokens,
        "final_tokens": config.final_tokens,
        "generation_method": generation_method,
        "generation_model": generation_model if generation_method == "ollama" else None,
        "refinement_model": refinement_model if generation_method == "ollama" else None
    }
    
    # ステップ1: データ生成
    callbacks.set_current_step(0)
    print(f"\n[ステップ1] {config.initial_tokens}トークンのデータを生成")
    
    use_ollama = (generation_method == "ollama" and not args.no_ollama)
    
    generator = OllamaDataGenerator(
        user_prompt, 
        tokenizer,
        ollama_host=args.ollama_host,
        model=generation_model,
        use_ollama=use_ollama,
        callbacks=callbacks
    )
    
    raw_samples = generator.generate_samples(config.initial_tokens)
    data_stats["samples_generated"] = len(raw_samples)
    print(f"生成完了: {len(raw_samples)}個のサンプル")
    
    # ステップ2: データ精選
    callbacks.set_current_step(1)
    print(f"\n[ステップ2] {config.final_tokens}トークンに精選")
    
    # 精選でOllamaを使用するかどうか
    use_ollama_refine = use_ollama and generator.ollama_available
    
    refiner = DataRefiner(
        user_prompt, 
        tokenizer,
        use_ollama=use_ollama_refine,
        ollama_host=args.ollama_host,
        model=refinement_model,
        callbacks=callbacks
    )
    refined_samples = refiner.refine_samples(raw_samples, config.final_tokens)
    data_stats["samples_refined"] = len(refined_samples)
    
    # ステップ3: データセット作成
    callbacks.set_current_step(2)
    print("\n[ステップ3] データセットを作成")
    dataset = TextDataset(refined_samples, tokenizer, config.max_length)
    train_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    callbacks.update_progress('dataset', current=1, total=1, info="Dataset ready")
    
    # ステップ4: モデルの初期化
    callbacks.set_current_step(3)
    print(f"\n[ステップ4] GPTモデルを初期化（{args.model_size}）")
    model = GPT(config).to(config.device)
    callbacks.update_progress('model_init', current=1, total=1, info=f"Model: {args.model_size}")
    
    # ステップ5: モデルのトレーニング
    callbacks.set_current_step(4)
    print("\n[ステップ5] モデルをトレーニング")
    train_model(model, train_loader, config, callbacks)
    
    # ステップ6: 最終的な生成テスト
    callbacks.set_current_step(5)
    print("\n[ステップ6] 学習済みモデルでテキスト生成テスト")
    test_results = []
    for i in range(3):
        generated = generate_sample(model, tokenizer, config, prompt="", max_length=30)
        test_results.append(generated)
        print(f"生成例{i+1}: {generated}")
        
        callbacks.update_progress('test', current=i+1, total=3, info="Testing generation")
        callbacks.update_preview({f"Test {i+1}": generated})
    
    # ステップ7: モデルの保存
    callbacks.set_current_step(6)
    print("\n[ステップ7] モデルとデータセットを保存")
    
    folder_name = create_model_folder_name(user_prompt, preset_info)
    folder_path = os.path.join(args.output_dir, folder_name)
    
    dataset_metadata = {
        "prompt": user_prompt,
        "domain": generator.domain,
        "generation_method": generation_method,
        "generation_model": generation_model if generation_method == "ollama" else None,
        "refinement_model": refinement_model if generation_method == "ollama" else None,
        "ollama_used": generator.ollama_available and generation_method == "ollama",
        "quality_threshold": "refined"
    }
    save_dataset(folder_path, refined_samples, tokenizer, dataset_metadata)
    
    callbacks.update_progress('save', current=50, total=100, info="Dataset saved, saving model...")
    
    end_time = time.time()
    generation_time = end_time - start_time
    
    metadata = {
        "prompt": user_prompt,
        "domain": generator.domain,
        "model_size": args.model_size,
        "training_tokens": str(config.final_tokens),
        "creation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "generation_method": generation_method,
        "ollama_used": str(generator.ollama_available and generation_method == "ollama")
    }
    
    save_model_safetensors(model, tokenizer, config, folder_path, metadata)
    
    generation_info = {
        "prompt": user_prompt,
        "category": generator.domain,
        "preset": preset_info['title'] if preset_info else "カスタム",
        "model_size": args.model_size,
        "training_params": {
            "epochs": config.num_epochs,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate
        },
        "data_stats": data_stats,
        "dataset_files": {
            "json": "dataset.json",
            "text": "dataset.txt"
        },
        "creation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "generation_method": generation_method,
        "generation_model": generation_model if generation_method == "ollama" else None,
        "refinement_model": refinement_model if generation_method == "ollama" else None,
        "ollama_used": generator.ollama_available and generation_method == "ollama",
        "generation_time": f"{generation_time:.1f} seconds"
    }
    
    save_generation_info(folder_path, generation_info)
    
    callbacks.update_progress('save', current=100, total=100, info="All files saved")
    
    return folder_path, generator.domain


# ===== 14. Textualアプリケーション =====
class Text2GPT1App(App):
    """Text2GPT1のTextualアプリケーション"""
    
    CSS = """
    Screen {
        background: $background;
    }
    
    .title {
        text-align: center;
        text-style: bold;
        color: $primary;
        margin: 1;
    }
    
    .panel-title {
        text-style: bold;
        color: $primary-lighten-2;
        margin-bottom: 1;
    }
    """
    
    BINDINGS = [
        Binding("ctrl+c", "quit", "終了", show=False),
    ]
    
    def __init__(self, config, args, tokenizer):
        super().__init__()
        self.config = config
        self.args = args
        self.tokenizer = tokenizer
        
        # 選択された値を保存
        self.selected_category = None
        self.selected_prompt = None
        self.selected_preset_info = None
        self.selected_epochs = config.num_epochs
        self.selected_generation_method = "template"
        self.selected_generation_model = "llama3"
        self.selected_refinement_model = "llama3"
        
        # 推論モード用
        self.selected_model_path = None
        self.selected_model_info = None
        
    async def on_mount(self) -> None:
        """アプリ起動時の処理"""
        self.title = "Text2GPT1 Textual Edition"
        self.sub_title = "プロンプトからカスタムGPTを自動生成"
        
        if self.args.inference and self.args.model_path:
            # 推論モードで起動
            self.selected_model_path = self.args.model_path
            info_path = os.path.join(self.args.model_path, "generation_info.json")
            if os.path.exists(info_path):
                with open(info_path, 'r', encoding='utf-8') as f:
                    self.selected_model_info = json.load(f)
            await self.push_screen(InferenceScreen())
        elif self.args.prompt:
            # コマンドライン引数でプロンプトが指定されている場合
            self.selected_prompt = self.args.prompt
            
            # コマンドライン引数から生成方法とモデルを設定
            if not self.args.no_ollama:
                self.selected_generation_method = "ollama"
                self.selected_generation_model = self.args.ollama_gen_model
                self.selected_refinement_model = self.args.ollama_refine_model
            else:
                self.selected_generation_method = "template"
            
            await self.push_screen(MainProcessScreen())
        else:
            # 対話形式で開始（メインメニュー）
            await self.push_screen(MainMenuScreen())
            
    def action_quit(self) -> None:
        """アプリを終了"""
        self.exit()


# ===== 15. コマンドライン引数パーサー =====
def create_parser():
    """コマンドライン引数パーサーを作成"""
    parser = argparse.ArgumentParser(
        description='Text2GPT1 Textual版 - プロンプトから特定の性質を持つGPT-1モデルを自動生成',
        epilog='''
使用例:
  # 基本的な使用方法（対話形式 with メインメニュー）
  %(prog)s
  
  # プロンプトを指定して実行（非対話形式）
  %(prog)s --prompt "料理レシピを生成する楽しいGPT"
  
  # 推論モード
  %(prog)s --inference --model-path models/cooking_recipe_generator_20240123_143022
  
  # TUIを無効化
  %(prog)s --no-tui
  
  # Ollamaを使用（デフォルト）
  %(prog)s --prompt "詩的な文章を書くGPT" --ollama-gen-model llama3 --ollama-refine-model mistral
  
  # Ollamaを使わずテンプレートベースで生成
  %(prog)s --prompt "技術文書を作成するGPT" --no-ollama
  
  # カスタムパラメータでの実行
  %(prog)s --prompt "会話が得意なGPT" --initial-tokens 3000 --final-tokens 1500 --epochs 50

必要な準備:
  1. PyTorchとTransformersのインストール: pip install torch transformers safetensors tqdm numpy requests
  2. Textualのインストール: pip install textual
  3. (推奨) Ollamaのインストールと起動: ollama serve
  4. (推奨) Ollamaモデルのダウンロード: ollama pull llama3
        ''',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '-p', '--prompt',
        type=str,
        help='生成したいGPTの性質を指定するプロンプト'
    )
    
    parser.add_argument(
        '--inference',
        action='store_true',
        help='推論モードで起動'
    )
    
    parser.add_argument(
        '--model-path',
        type=str,
        help='推論に使用するモデルのパス'
    )
    
    parser.add_argument(
        '--no-tui',
        action='store_true',
        help='TUI（Text User Interface）を無効化'
    )
    
    data_group = parser.add_argument_group('データ生成オプション')
    data_group.add_argument(
        '--initial-tokens',
        type=int,
        default=2000,
        help='初期生成するトークン数（デフォルト: 2000）'
    )
    data_group.add_argument(
        '--final-tokens',
        type=int,
        default=1000,
        help='精選後の最終トークン数（デフォルト: 1000）'
    )
    data_group.add_argument(
        '--no-ollama',
        action='store_true',
        help='Ollamaを使用せず、テンプレートベースで生成'
    )
    data_group.add_argument(
        '--ollama-model',
        type=str,
        default='llama3',
        help='[非推奨] --ollama-gen-modelと--ollama-refine-modelを使用してください'
    )
    data_group.add_argument(
        '--ollama-gen-model',
        type=str,
        default='llama3',
        help='データ生成に使用するOllamaモデル（デフォルト: llama3）'
    )
    data_group.add_argument(
        '--ollama-refine-model',
        type=str,
        default=None,
        help='データ精選に使用するOllamaモデル（デフォルト: 生成と同じモデル）'
    )
    data_group.add_argument(
        '--ollama-host',
        type=str,
        default='http://localhost:11434',
        help='OllamaサーバーのURL（デフォルト: http://localhost:11434）'
    )
    
    train_group = parser.add_argument_group('学習オプション')
    train_group.add_argument(
        '--epochs',
        type=int,
        default=20,
        help='学習エポック数（デフォルト: 20）'
    )
    train_group.add_argument(
        '--batch-size',
        type=int,
        default=2,
        help='バッチサイズ（デフォルト: 2）'
    )
    train_group.add_argument(
        '--learning-rate',
        type=float,
        default=3e-4,
        help='学習率（デフォルト: 3e-4）'
    )
    
    model_group = parser.add_argument_group('モデルオプション')
    model_group.add_argument(
        '--model-size',
        type=str,
        choices=['12M', '33M', '117M'],
        default='12M',
        help='モデルサイズ（デフォルト: 12M）'
    )
    
    output_group = parser.add_argument_group('出力オプション')
    output_group.add_argument(
        '--output-dir',
        type=str,
        default='models',
        help='モデルの保存先ディレクトリ（デフォルト: models）'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='詳細なログを表示'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='乱数シード（デフォルト: 42）'
    )
    parser.add_argument(
        '--version',
        action='version',
        version='%(prog)s 6.2.0 (Textual Edition with Inference)'
    )
    
    return parser


# ===== 16. メイン実行関数 =====
def main():
    """Text2GPT1のメイン処理"""
    parser = create_parser()
    args = parser.parse_args()
    
    # 古い--ollama-modelが指定されていて新しいオプションがない場合の互換性処理
    if args.ollama_model != 'llama3' and args.ollama_gen_model == 'llama3':
        args.ollama_gen_model = args.ollama_model
    if args.ollama_refine_model is None:
        args.ollama_refine_model = args.ollama_gen_model
    
    print("="*60)
    print("Text2GPT1 - プロンプトからカスタムGPTを自動生成")
    print("Version 6.2.0 - Textual Edition with Inference")
    print("="*60)
    
    config = Config()
    config.seed = args.seed
    config.num_epochs = args.epochs
    config.batch_size = args.batch_size
    config.learning_rate = args.learning_rate
    config.initial_tokens = args.initial_tokens
    config.final_tokens = args.final_tokens
    
    if args.model_size == '33M':
        config.n_embd = 512
        config.n_layer = 8
        config.n_head = 8
    elif args.model_size == '117M':
        config.n_embd = 768
        config.n_layer = 12
        config.n_head = 12
    
    torch.manual_seed(config.seed)
    
    print("\nトークナイザーを準備中...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    use_textual_tui = not args.prompt and not args.no_tui and not (args.inference and args.model_path)
    
    if use_textual_tui:
        # Textual TUIモード
        try:
            app = Text2GPT1App(config, args, tokenizer)
            app.run()
        except KeyboardInterrupt:
            print("\n\n処理が中断されました。")
        except Exception as e:
            print(f"\n\nエラーが発生しました: {e}")
            import traceback
            traceback.print_exc()
    else:
        # 非TUIモード
        if args.inference and args.model_path:
            # 推論モード
            print(f"\n推論モード: {args.model_path}")
            
            try:
                model, tokenizer, config = load_model_safetensors(args.model_path)
                print("モデルの読み込みに成功しました。")
                
                # インタラクティブな推論ループ
                print("\nテキスト生成を開始します。終了するには 'quit' と入力してください。")
                while True:
                    prompt = input("\nプロンプト> ").strip()
                    if prompt.lower() == 'quit':
                        break
                    
                    if not prompt:
                        continue
                    
                    generated = generate_sample(
                        model, tokenizer, config,
                        prompt=prompt,
                        max_length=100,
                        temperature=0.8,
                        top_k=40
                    )
                    
                    print(f"\n生成結果:\n{generated}")
                    
            except Exception as e:
                print(f"エラー: {e}")
                
        elif args.prompt:
            # 非対話的モデル作成
            user_prompt = args.prompt
            print(f"\n目標: 「{user_prompt}」GPTを作成")
            
            # 生成方法の設定
            generation_method = "template" if args.no_ollama else "ollama"
            
            folder_path, domain = run_main_process(
                config, args, user_prompt, tokenizer,
                generation_method=generation_method,
                generation_model=args.ollama_gen_model,
                refinement_model=args.ollama_refine_model
            )
            
            print("\n"+"="*60)
            print("Text2GPT1 完了！")
            print(f"モデルフォルダ: {folder_path}")
            print(f"性質: {user_prompt}")
            print(f"ドメイン: {domain}")
            print(f"サイズ: {args.model_size}")
            print(f"エポック数: {config.num_epochs}")
            print(f"データセット: {folder_path}/dataset.json")
            print(f"生成方法: {generation_method}")
            if generation_method == "ollama":
                print(f"生成モデル: {args.ollama_gen_model}")
                print(f"精選モデル: {args.ollama_refine_model}")
            print("="*60)
            
        else:
            # プロンプトなしの非TUIモード
            print("\nどのような性質のGPT-1を生成しますか？")
            print("\nカテゴリ:")
            for key, cat in CATEGORIES.items():
                print(f"  - {cat['name']} ({cat['name_ja']}): {cat['description']}")
            print("\n例: 料理レシピを生成する、詩的な文章を書く、技術文書を作成する")
            user_prompt = input("プロンプト: ").strip()
            
            if not user_prompt:
                user_prompt = "一般的な文章を生成する"
                print(f"デフォルトプロンプトを使用: {user_prompt}")
            
            print(f"\n目標: 「{user_prompt}」GPTを作成")
            
            # 生成方法の設定
            generation_method = "template" if args.no_ollama else "ollama"
            
            folder_path, domain = run_main_process(
                config, args, user_prompt, tokenizer,
                generation_method=generation_method,
                generation_model=args.ollama_gen_model,
                refinement_model=args.ollama_refine_model
            )
            
            print("\n"+"="*60)
            print("Text2GPT1 完了！")
            print(f"モデルフォルダ: {folder_path}")
            print(f"性質: {user_prompt}")
            print(f"ドメイン: {domain}")
            print(f"サイズ: {args.model_size}")
            print(f"エポック数: {config.num_epochs}")
            print(f"データセット: {folder_path}/dataset.json")
            print(f"生成方法: {generation_method}")
            if generation_method == "ollama":
                print(f"生成モデル: {args.ollama_gen_model}")
                print(f"精選モデル: {args.ollama_refine_model}")
            print("="*60)


if __name__ == "__main__":
    main()