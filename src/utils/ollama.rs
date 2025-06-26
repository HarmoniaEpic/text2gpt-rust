use anyhow::{anyhow, Context, Result};
use colored::*;
use serde::Deserialize;
use std::collections::HashMap;
use std::time::Duration;

/// Ollamaからのモデル情報レスポンス
#[derive(Debug, Deserialize)]
pub struct OllamaModelsResponse {
    pub models: Vec<OllamaModelDetails>,
}

/// Ollamaモデルの詳細情報
#[derive(Debug, Deserialize)]
pub struct OllamaModelDetails {
    pub name: String,
    pub size: u64,
    pub digest: String,
    pub modified_at: String,
}

/// インストール済みモデル情報
#[derive(Debug, Clone)]
pub struct InstalledModel {
    pub name: String,
    pub size: String,
    pub modified: String,
    pub category: ModelCategory,
}

/// モデルカテゴリ
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ModelCategory {
    General,      // 汎用モデル
    Technical,    // 技術/コード特化
    Lightweight,  // 軽量モデル (< 3GB)
    Large,        // 大規模モデル (> 30GB)
    Unknown,      // 不明/その他
}

impl std::fmt::Display for ModelCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelCategory::General => write!(f, "General Purpose"),
            ModelCategory::Technical => write!(f, "Technical/Code"),
            ModelCategory::Lightweight => write!(f, "Lightweight"),
            ModelCategory::Large => write!(f, "Large Scale"),
            ModelCategory::Unknown => write!(f, "Other"),
        }
    }
}

/// カテゴリごとの推奨モデル
const CATEGORY_RECOMMENDATIONS: &[(&str, &[&str])] = &[
    ("cooking", &["llama3.1:8b", "qwen2.5:7b", "gemma2:9b", "mistral:7b-v0.3"]),
    ("poetry", &["llama3.1:70b", "qwen2.5:14b", "mistral:7b-v0.3", "gemma2:9b"]),
    ("technical", &["codellama:7b", "deepseek-coder:6.7b", "qwen2.5:14b", "mixtral:8x7b"]),
    ("general", &["llama3.1:8b", "qwen2.5:7b", "mistral:7b-v0.3", "gemma2:2b"]),
];

/// Ollamaが実行中かチェック
pub async fn check_ollama_running() -> Result<bool> {
    let client = reqwest::Client::new();
    let url = "http://localhost:11434/api/tags";
    
    match client.get(url)
        .timeout(Duration::from_secs(5))
        .send()
        .await
    {
        Ok(response) => Ok(response.status().is_success()),
        Err(_) => Ok(false),
    }
}

/// インストール済みのOllamaモデルを取得
pub async fn get_installed_models() -> Result<Vec<InstalledModel>> {
    let client = reqwest::Client::new();
    let url = "http://localhost:11434/api/tags";
    
    let response = client.get(url)
        .timeout(Duration::from_secs(10))
        .send()
        .await
        .context("Failed to connect to Ollama")?;
    
    if !response.status().is_success() {
        return Err(anyhow!("Ollama API returned error status: {}", response.status()));
    }
    
    let models_response: OllamaModelsResponse = response.json()
        .await
        .context("Failed to parse Ollama response")?;
    
    let installed_models: Vec<InstalledModel> = models_response.models
        .into_iter()
        .map(|model| {
            let size = format_size(model.size);
            let modified = format_modified_time(&model.modified_at);
            let category = categorize_model(&model.name);
            
            InstalledModel {
                name: model.name,
                size,
                modified,
                category,
            }
        })
        .collect();
    
    Ok(installed_models)
}

/// モデル名からカテゴリを判定
pub fn categorize_model(model_name: &str) -> ModelCategory {
    let name_lower = model_name.to_lowercase();
    
    // Technical/Code models
    if name_lower.contains("code") || 
       name_lower.contains("deepseek-coder") ||
       name_lower.contains("starcoder") {
        return ModelCategory::Technical;
    }
    
    // Large models (by name pattern)
    if name_lower.contains(":70b") || 
       name_lower.contains(":72b") ||
       name_lower.contains("8x7b") ||
       name_lower.contains(":40b") {
        return ModelCategory::Large;
    }
    
    // Lightweight models
    if name_lower.contains("tinyllama") ||
       name_lower.contains(":1b") ||
       name_lower.contains(":2b") ||
       name_lower.contains(":3b") ||
       name_lower.contains("phi3:mini") ||
       name_lower.contains("orca-mini") {
        return ModelCategory::Lightweight;
    }
    
    // General purpose models
    if name_lower.contains("llama") ||
       name_lower.contains("mistral") ||
       name_lower.contains("qwen") ||
       name_lower.contains("gemma") ||
       name_lower.contains("phi") {
        return ModelCategory::General;
    }
    
    ModelCategory::Unknown
}

/// モデルリストを表示用にフォーマット
pub fn format_model_list_for_display(models: &[InstalledModel]) -> Vec<String> {
    // カテゴリ別にグループ化
    let mut categorized: HashMap<ModelCategory, Vec<&InstalledModel>> = HashMap::new();
    for model in models {
        categorized.entry(model.category.clone()).or_default().push(model);
    }
    
    let mut display_list = Vec::new();
    let categories = [
        ModelCategory::General,
        ModelCategory::Technical,
        ModelCategory::Large,
        ModelCategory::Lightweight,
        ModelCategory::Unknown,
    ];
    
    for category in &categories {
        if let Some(models) = categorized.get(category) {
            if !models.is_empty() {
                display_list.push(format!("\n[{}]", category).bright_cyan().to_string());
                for model in models {
                    display_list.push(format!(
                        "{} ({}) - {}",
                        model.name,
                        model.size,
                        model.modified
                    ));
                }
            }
        }
    }
    
    display_list
}

/// インストールのヒントを表示
pub fn show_installation_hints(category: &str) -> String {
    let mut hints = String::new();
    
    hints.push_str(&format!("\n{}\n", "📦 Recommended models to install:".bright_yellow()));
    
    // VRAM別の推奨
    hints.push_str(&format!("\n{}\n", "For high-end GPU (24GB+ VRAM):".bright_green()));
    hints.push_str("  $ ollama pull llama3.1:70b\n");
    hints.push_str("  $ ollama pull qwen2.5:72b\n");
    hints.push_str("  $ ollama pull mixtral:8x7b\n");
    
    hints.push_str(&format!("\n{}\n", "For standard GPU (8-16GB VRAM):".bright_green()));
    hints.push_str("  $ ollama pull llama3.1:8b\n");
    hints.push_str("  $ ollama pull qwen2.5:7b\n");
    hints.push_str("  $ ollama pull mistral:7b-v0.3\n");
    
    hints.push_str(&format!("\n{}\n", "For limited GPU/CPU (4-8GB):".bright_green()));
    hints.push_str("  $ ollama pull gemma2:2b\n");
    hints.push_str("  $ ollama pull phi3:mini\n");
    hints.push_str("  $ ollama pull tinyllama:1.1b\n");
    
    // カテゴリ別の推奨
    if let Some((_, models)) = CATEGORY_RECOMMENDATIONS.iter().find(|(cat, _)| *cat == category) {
        hints.push_str(&format!("\n{} {}:\n", 
            "Specifically for".bright_yellow(), 
            category.bright_cyan()
        ));
        for model in models.iter().take(3) {
            hints.push_str(&format!("  $ ollama pull {}\n", model));
        }
    }
    
    hints
}

/// カテゴリ別のWarningメッセージを生成
pub fn generate_category_warning(category: &str, available_models: &[InstalledModel]) -> Option<String> {
    let category_models: Vec<&InstalledModel> = available_models
        .iter()
        .filter(|m| matches!(m.category, ModelCategory::Technical) && category == "technical")
        .collect();
    
    if category_models.is_empty() && category == "technical" {
        let mut warning = String::new();
        warning.push_str(&format!("\n{} {}\n", 
            "⚠️ ".yellow(), 
            "Warning: No technical/code models found".yellow()
        ));
        warning.push_str(&format!("\n{}\n", 
            "For better results with technical documentation, consider installing:".bright_white()
        ));
        warning.push_str("  $ ollama pull codellama:7b\n");
        warning.push_str("  $ ollama pull deepseek-coder:6.7b\n");
        warning.push_str(&format!("\n{}\n", 
            "Continuing with available models...".bright_white()
        ));
        return Some(warning);
    }
    
    None
}

/// サイズをフォーマット (bytes -> human readable)
fn format_size(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
    let mut size = bytes as f64;
    let mut unit_index = 0;
    
    while size >= 1024.0 && unit_index < UNITS.len() - 1 {
        size /= 1024.0;
        unit_index += 1;
    }
    
    if unit_index >= 2 { // MB以上
        format!("{:.1}{}", size, UNITS[unit_index])
    } else {
        format!("{:.0}{}", size, UNITS[unit_index])
    }
}

/// 更新時刻をフォーマット
fn format_modified_time(iso_time: &str) -> String {
    // ISO 8601形式の時刻を解析
    if let Ok(parsed) = chrono::DateTime::parse_from_rfc3339(iso_time) {
        let now = chrono::Local::now();
        let duration = now.signed_duration_since(parsed);
        
        if duration.num_days() > 30 {
            format!("{} months ago", duration.num_days() / 30)
        } else if duration.num_days() > 0 {
            format!("{} days ago", duration.num_days())
        } else if duration.num_hours() > 0 {
            format!("{} hours ago", duration.num_hours())
        } else {
            "recently".to_string()
        }
    } else {
        "unknown".to_string()
    }
}

/// Ollamaが起動していない場合のエラーメッセージ
pub fn show_ollama_not_running_error() {
    eprintln!("{}", "\n❌ Error: Ollama is not running".bright_red().bold());
    eprintln!("\nPlease start Ollama with:");
    eprintln!("  $ {}", "ollama serve".bright_cyan());
    eprintln!("\nThen try again.");
}

/// モデルが一つもない場合のエラーメッセージ
pub fn show_no_models_error() {
    eprintln!("{}", "\n❌ Error: No Ollama models installed".bright_red().bold());
    eprintln!("{}", show_installation_hints("general"));
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_categorize_model() {
        assert_eq!(categorize_model("codellama:7b"), ModelCategory::Technical);
        assert_eq!(categorize_model("llama3.1:70b"), ModelCategory::Large);
        assert_eq!(categorize_model("tinyllama:1.1b"), ModelCategory::Lightweight);
        assert_eq!(categorize_model("llama3.1:8b"), ModelCategory::General);
        assert_eq!(categorize_model("custom-model"), ModelCategory::Unknown);
    }
    
    #[test]
    fn test_format_size() {
        assert_eq!(format_size(500), "500B");
        assert_eq!(format_size(1024), "1KB");
        assert_eq!(format_size(1536), "2KB");
        assert_eq!(format_size(1048576), "1.0MB");
        assert_eq!(format_size(5368709120), "5.0GB");
    }
}
