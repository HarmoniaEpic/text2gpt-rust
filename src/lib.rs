pub mod config;
pub mod data;
pub mod inference;
pub mod io;
pub mod model;
pub mod tokenizer;
pub mod training;

pub use config::{Config, ModelSize};
pub use model::gpt::GPT;
pub use tokenizer::gpt2::GPT2Tokenizer;

/// Custom error type for Text2GPT1
#[derive(thiserror::Error, Debug)]
pub enum Text2GPT1Error {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("Model error: {0}")]
    Model(String),
    
    #[error("Tokenizer error: {0}")]
    Tokenizer(String),
    
    #[error("Training error: {0}")]
    Training(String),
    
    #[error("Data generation error: {0}")]
    DataGeneration(String),
    
    #[error("Candle error: {0}")]
    Candle(#[from] candle_core::Error),
    
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),
    
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    
    #[error("Other error: {0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, Text2GPT1Error>;