use chrono::Local;
use std::collections::HashMap;

/// フォルダ名生成のためのビルダー
pub struct FolderNameBuilder {
    max_length: usize,
}

impl FolderNameBuilder {
    /// 新しいFolderNameBuilderを作成
    pub fn new() -> Self {
        Self { max_length: 50 }
    }

    /// プロンプトとプリセット情報からフォルダ名を生成
    /// 
    /// # Arguments
    /// * `prompt` - ユーザーが入力したプロンプト
    /// * `preset_en` - プリセットの英語名（オプション）
    /// 
    /// # Returns
    /// ファイルシステムで使用可能なフォルダ名
    /// 
    /// # Example
    /// ```
    /// let builder = FolderNameBuilder::new();
    /// let name = builder.generate("料理レシピを生成するGPT", Some("cooking_recipe_generator"));
    /// // => "cooking_recipe_generator_20240123_143022"
    /// ```
    pub fn generate(&self, prompt: &str, preset_en: Option<&str>) -> String {
        let base_name = if let Some(preset) = preset_en {
            // プリセットがある場合はそれを使用
            preset.to_string()
        } else {
            // カスタムプロンプトの場合はキーワード抽出
            self.extract_base_name(prompt)
        };

        // タイムスタンプを追加
        let timestamp = Local::now().format("%Y%m%d_%H%M%S");
        let full_name = format!("{}_{}", base_name, timestamp);

        // 最大長に収める
        if full_name.len() > self.max_length {
            let timestamp_len = 16; // "YYYYMMDD_HHMMSS_" の長さ
            let max_base_len = self.max_length.saturating_sub(timestamp_len);
            let truncated_base = base_name.chars().take(max_base_len).collect::<String>();
            format!("{}_{}", truncated_base, timestamp)
        } else {
            full_name
        }
    }

    /// プロンプトから基本となる名前を抽出
    fn extract_base_name(&self, prompt: &str) -> String {
        // まず既知のドメインキーワードを探す
        if let Some(domain) = self.detect_domain(prompt) {
            // ドメインが見つかった場合、追加のキーワードを探す
            let keywords = self.extract_keywords(prompt, domain);
            if !keywords.is_empty() {
                return keywords.join("_");
            }
            return domain.to_string();
        }

        // ドメインが見つからない場合は、プロンプトをサニタイズして使用
        let sanitized = self.sanitize_for_filename(prompt);
        if sanitized.is_empty() {
            "custom_model".to_string()
        } else {
            format!("custom_{}", sanitized)
        }
    }

    /// プロンプトからドメインを検出
    fn detect_domain(&self, prompt: &str) -> Option<&'static str> {
        // シンプルなキーワードマッチング
        let domain_patterns = [
            (vec!["料理", "レシピ", "食"], "cooking"),
            (vec!["詩", "ポエム", "創作"], "poetry"),
            (vec!["技術", "プログラ", "コード"], "technical"),
            (vec!["会話", "対話", "チャット"], "conversation"),
        ];

        for (keywords, domain) in domain_patterns.iter() {
            if keywords.iter().any(|k| prompt.contains(k)) {
                return Some(domain);
            }
        }

        None
    }

    /// ドメインに応じたキーワードを抽出
    fn extract_keywords(&self, prompt: &str, domain: &str) -> Vec<String> {
        let mut keywords = vec![domain.to_string()];
    
        // ドメイン固有のキーワードマッピング
        let keyword_map = self.get_keyword_map();
    
        // プロンプトから追加のキーワードを探す
        for (jp, en) in keyword_map.iter() {
            // iter().any()を使用して存在チェック
            if prompt.contains(jp) && !keywords.iter().any(|k| k == en) {
                keywords.push(en.to_string());
                if keywords.len() >= 3 {
                    break; // 最大3つまで
                }
            }
        }
    
        keywords
    }    
    
    /// 日本語から英語へのキーワードマッピング
    fn get_keyword_map(&self) -> HashMap<&'static str, &'static str> {
        let mut map = HashMap::new();
        
        // 基本的なキーワードのみ
        map.insert("簡単", "simple");
        map.insert("高度", "advanced");
        map.insert("専門", "professional");
        map.insert("初心者", "beginner");
        map.insert("楽しい", "fun");
        map.insert("生成", "generator");
        map.insert("アシスタント", "assistant");
        map.insert("ヘルパー", "helper");
        
        map
    }

    /// 文字列をファイル名として使用可能な形式にサニタイズ
    fn sanitize_for_filename(&self, s: &str) -> String {
        s.chars()
            .filter(|c| c.is_ascii_alphanumeric() || *c == '_' || *c == '-')
            .take(30) // 最初の30文字まで
            .collect::<String>()
            .to_lowercase()
    }
}

impl Default for FolderNameBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_with_preset() {
        let builder = FolderNameBuilder::new();
        let name = builder.generate("任意のプロンプト", Some("cooking_recipe_generator"));
        assert!(name.starts_with("cooking_recipe_generator_"));
        assert!(name.len() <= 50);
    }

    #[test]
    fn test_cooking_domain() {
        let builder = FolderNameBuilder::new();
        let name = builder.generate("簡単な料理レシピを生成するGPT", None);
        assert!(name.starts_with("cooking_"));
        assert!(name.contains("simple") || name.contains("generator"));
    }

    #[test]
    fn test_poetry_domain() {
        let builder = FolderNameBuilder::new();
        let name = builder.generate("詩的な文章を創作するGPT", None);
        assert!(name.starts_with("poetry_"));
    }

    #[test]
    fn test_technical_domain() {
        let builder = FolderNameBuilder::new();
        let name = builder.generate("技術文書を生成するプログラム", None);
        assert!(name.starts_with("technical_"));
    }

    #[test]
    fn test_custom_prompt() {
        let builder = FolderNameBuilder::new();
        let name = builder.generate("特殊な用途のGPT", None);
        assert!(name.starts_with("custom_"));
    }

    #[test]
    fn test_empty_prompt() {
        let builder = FolderNameBuilder::new();
        let name = builder.generate("", None);
        assert!(name.starts_with("custom_model_"));
    }

    #[test]
    fn test_special_characters() {
        let builder = FolderNameBuilder::new();
        let name = builder.generate("テスト！＃＄％＆", None);
        // 特殊文字は除去される
        assert!(!name.contains('！'));
        assert!(!name.contains('＃'));
    }

    #[test]
    fn test_max_length() {
        let builder = FolderNameBuilder::new();
        let long_prompt = "a".repeat(100);
        let name = builder.generate(&long_prompt, None);
        assert!(name.len() <= 50);
    }
}
