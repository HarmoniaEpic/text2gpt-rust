use anyhow::Result;
use clap::{Parser, Subcommand};
use colored::*;
use dialoguer::{theme::ColorfulTheme, Select};
use std::path::PathBuf;

mod config;
mod data;
mod inference;
mod io;
mod model;
mod tokenizer;
mod training;

use crate::config::{Config, ModelSize};
use crate::data::generator::DataGenerationMethod;

#[derive(Parser)]
#[command(
    name = "text2gpt1",
    version = "0.1.0",
    about = "Generate custom GPT models from prompts",
    long_about = "Text2GPT1 - Generate custom GPT-1 models with specific behaviors from prompts\n\
                  \n\
                  Examples:\n\
                    # Interactive mode\n\
                    text2gpt1\n\
                    \n\
                    # Generate a model directly\n\
                    text2gpt1 generate --prompt \"A helpful cooking recipe GPT\"\n\
                    \n\
                    # Inference with existing model\n\
                    text2gpt1 infer --model-path models/cooking_20240123"
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
    
    /// Enable verbose logging
    #[arg(short, long)]
    verbose: bool,
}

#[derive(Subcommand)]
enum Commands {
    /// Generate a new GPT model
    Generate {
        /// The prompt describing the desired GPT behavior
        #[arg(short, long)]
        prompt: Option<String>,
        
        /// Number of training epochs
        #[arg(long, default_value = "20")]
        epochs: usize,
        
        /// Initial tokens to generate
        #[arg(long, default_value = "2000")]
        initial_tokens: usize,
        
        /// Final tokens after refinement
        #[arg(long, default_value = "1000")]
        final_tokens: usize,
        
        /// Model size (12M, 33M, or 117M)
        #[arg(long, default_value = "12M")]
        model_size: ModelSize,
        
        /// Don't use Ollama for data generation
        #[arg(long)]
        no_ollama: bool,
        
        /// Ollama model for generation
        #[arg(long, default_value = "llama3")]
        ollama_gen_model: String,
        
        /// Ollama model for refinement (defaults to generation model)
        #[arg(long)]
        ollama_refine_model: Option<String>,
        
        /// Output directory for models
        #[arg(long, default_value = "models")]
        output_dir: PathBuf,
        
        /// Batch size for training
        #[arg(long, default_value = "2")]
        batch_size: usize,
        
        /// Learning rate
        #[arg(long, default_value = "0.0003")]
        learning_rate: f64,
        
        /// Random seed
        #[arg(long, default_value = "42")]
        seed: u64,
    },
    
    /// Run inference with an existing model
    Infer {
        /// Path to the model directory
        #[arg(short, long)]
        model_path: PathBuf,
        
        /// Initial prompt for generation
        #[arg(short, long)]
        prompt: Option<String>,
        
        /// Maximum tokens to generate
        #[arg(long, default_value = "100")]
        max_length: usize,
        
        /// Temperature for sampling
        #[arg(long, default_value = "0.8")]
        temperature: f64,
        
        /// Top-k sampling parameter
        #[arg(long, default_value = "40")]
        top_k: usize,
    },
    
    /// List saved models
    List {
        /// Models directory
        #[arg(long, default_value = "models")]
        models_dir: PathBuf,
    },
}

// Category definitions
const CATEGORIES: &[(&str, &str, &str)] = &[
    ("cooking", "🍳 Cooking & Recipes", "Generate cooking recipes and food descriptions"),
    ("poetry", "✍️ Poetry & Creative", "Create poetic and creative texts"),
    ("technical", "💻 Technical & Code", "Generate technical documentation and code explanations"),
    ("general", "📝 General Purpose", "General-purpose text generation"),
];

// Preset definitions for each category
fn get_presets(category: &str) -> Vec<(&str, &str)> {
    match category {
        "cooking" => vec![
            ("Simple home cooking recipes", "Generate easy and delicious home cooking recipes"),
            ("Professional chef assistant", "Provide advanced cooking techniques and recipes"),
            ("Healthy diet recipes", "Generate healthy and nutritious recipes"),
            ("Desserts specialist", "Specialize in sweets and dessert recipes"),
        ],
        "poetry" => vec![
            ("Modern poetry generator", "Create contemporary and expressive poetry"),
            ("Haiku master", "Generate traditional Japanese poetry forms"),
            ("Story creator", "Create short stories and narratives"),
            ("Songwriter", "Write emotional and creative song lyrics"),
        ],
        "technical" => vec![
            ("Programming explainer", "Explain programming concepts clearly"),
            ("API documentation writer", "Create technical specifications and docs"),
            ("Algorithm specialist", "Explain algorithms and data structures"),
            ("System architect", "Describe system design and architecture"),
        ],
        "general" => vec![
            ("Daily conversation assistant", "Generate friendly casual conversations"),
            ("Business writer", "Create professional business documents"),
            ("Education support", "Provide clear educational explanations"),
            ("News summarizer", "Summarize news and information concisely"),
        ],
        _ => vec![],
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    
    // Initialize logger
    if cli.verbose {
        env_logger::Builder::from_default_env()
            .filter_level(log::LevelFilter::Debug)
            .init();
    } else {
        env_logger::Builder::from_default_env()
            .filter_level(log::LevelFilter::Info)
            .init();
    }
    
    println!("{}", "=".repeat(60).bright_blue());
    println!("{}", "Text2GPT1 - Generate custom GPT models from prompts".bright_white().bold());
    println!("{}", "Version 0.1.0 - Rust + Candle Edition".bright_white());
    println!("{}", "=".repeat(60).bright_blue());
    println!();
    
    match cli.command {
        Some(Commands::Generate { 
            prompt,
            epochs,
            initial_tokens,
            final_tokens,
            model_size,
            no_ollama,
            ollama_gen_model,
            ollama_refine_model,
            output_dir,
            batch_size,
            learning_rate,
            seed,
        }) => {
            let mut config = Config::new(model_size, seed);
            config.num_epochs = epochs;
            config.initial_tokens = initial_tokens;
            config.final_tokens = final_tokens;
            config.batch_size = batch_size;
            config.learning_rate = learning_rate;
            
            let ollama_refine_model = ollama_refine_model.unwrap_or_else(|| ollama_gen_model.clone());
            
            if let Some(prompt) = prompt {
                // Non-interactive mode
                generate_model(
                    &prompt,
                    &config,
                    &output_dir,
                    !no_ollama,
                    &ollama_gen_model,
                    &ollama_refine_model,
                )?;
            } else {
                // Interactive mode
                interactive_generate(
                    config,
                    output_dir,
                    !no_ollama,
                    ollama_gen_model,
                    ollama_refine_model,
                )?;
            }
        }
        
        Some(Commands::Infer { 
            model_path,
            prompt,
            max_length,
            temperature,
            top_k,
        }) => {
            inference_mode(model_path, prompt, max_length, temperature, top_k)?;
        }
        
        Some(Commands::List { models_dir }) => {
            list_models(&models_dir)?;
        }
        
        None => {
            // Interactive main menu
            main_menu()?;
        }
    }
    
    Ok(())
}

fn main_menu() -> Result<()> {
    let selections = vec![
        "🔨 Create a new GPT model",
        "🤖 Run inference with existing model",
        "📋 List saved models",
        "❌ Exit",
    ];
    
    let selection = Select::with_theme(&ColorfulTheme::default())
        .with_prompt("What would you like to do?")
        .default(0)
        .items(&selections)
        .interact()?;
    
    match selection {
        0 => {
            let config = Config::default();
            interactive_generate(
                config,
                PathBuf::from("models"),
                true,
                "llama3".to_string(),
                "llama3".to_string(),
            )?;
        }
        1 => {
            let models_dir = PathBuf::from("models");
            if let Some(model_path) = select_model(&models_dir)? {
                inference_mode(model_path, None, 100, 0.8, 40)?;
            }
        }
        2 => {
            list_models(&PathBuf::from("models"))?;
            println!("\nPress Enter to continue...");
            let mut input = String::new();
            std::io::stdin().read_line(&mut input)?;
            main_menu()?;
        }
        _ => {}
    }
    
    Ok(())
}

fn interactive_generate(
    mut config: Config,
    output_dir: PathBuf,
    use_ollama: bool,
    ollama_gen_model: String,
    ollama_refine_model: String,
) -> Result<()> {
    // Category selection
    println!("{}", "Select a category:".bright_cyan());
    let category_names: Vec<String> = CATEGORIES.iter()
        .map(|(_, name, desc)| format!("{} - {}", name, desc))
        .collect();
    
    let category_idx = Select::with_theme(&ColorfulTheme::default())
        .with_prompt("Category")
        .default(0)
        .items(&category_names)
        .interact()?;
    
    let (category_key, _, _) = CATEGORIES[category_idx];
    
    // Preset or custom
    let presets = get_presets(category_key);
    let mut preset_names: Vec<String> = presets.iter()
        .map(|(name, desc)| format!("{} - {}", name, desc))
        .collect();
    preset_names.push("✏️ Custom prompt".to_string());
    
    println!("\n{}", "Select a preset or enter custom prompt:".bright_cyan());
    let preset_idx = Select::with_theme(&ColorfulTheme::default())
        .with_prompt("Preset")
        .default(0)
        .items(&preset_names)
        .interact()?;
    
    let prompt = if preset_idx < presets.len() {
        presets[preset_idx].0.to_string()
    } else {
        println!("\n{}", "Enter your custom prompt:".bright_cyan());
        dialoguer::Input::<String>::with_theme(&ColorfulTheme::default())
            .with_prompt("Prompt")
            .interact_text()?
    };
    
    // Epoch selection
    println!("\n{}", "Select training epochs:".bright_cyan());
    let epoch_options = vec![
        ("20", "Fast test - Quick results for testing"),
        ("50", "Standard - Balanced quality and speed"),
        ("100", "Quality - Better generation quality"),
        ("200", "High quality - Thorough training"),
        ("500", "Maximum quality - Extensive training (may overfit)"),
        ("Custom", "Enter custom epoch count"),
    ];
    
    let epoch_descriptions: Vec<String> = epoch_options.iter()
        .map(|(epochs, desc)| format!("{} epochs - {}", epochs, desc))
        .collect();
    
    let epoch_idx = Select::with_theme(&ColorfulTheme::default())
        .with_prompt("Epochs")
        .default(1)
        .items(&epoch_descriptions)
        .interact()?;
    
    config.num_epochs = match epoch_idx {
        0 => 20,
        1 => 50,
        2 => 100,
        3 => 200,
        4 => 500,
        _ => {
            dialoguer::Input::<usize>::with_theme(&ColorfulTheme::default())
                .with_prompt("Custom epochs (1-1000)")
                .validate_with(|input: &usize| {
                    if *input > 0 && *input <= 1000 {
                        Ok(())
                    } else {
                        Err("Epochs must be between 1 and 1000")
                    }
                })
                .interact_text()?
        }
    };
    
    // Data generation method selection
    println!("\n{}", "Select data generation method:".bright_cyan());
    let gen_methods = vec![
        ("Template-based", "Fast, offline generation using predefined templates"),
        ("Ollama-powered", "High-quality generation using LLM (requires Ollama)"),
    ];
    
    let gen_method_idx = Select::with_theme(&ColorfulTheme::default())
        .with_prompt("Generation method")
        .default(1)
        .items(&gen_methods.iter().map(|(name, desc)| format!("{} - {}", name, desc)).collect::<Vec<_>>())
        .interact()?;
    
    let use_ollama = gen_method_idx == 1;
    let (gen_model, ref_model) = if use_ollama {
        // Ollama method - select models
        println!("\n{}", "Select Ollama model for data generation:".bright_cyan());
        let ollama_models = vec![
            "llama3", "llama3:70b", "mistral", "gemma", "gemma:7b", 
            "codellama", "phi", "Custom"
        ];
        
        let gen_model_idx = Select::with_theme(&ColorfulTheme::default())
            .with_prompt("Generation model")
            .default(0)
            .items(&ollama_models)
            .interact()?;
        
        let gen_model = if gen_model_idx == ollama_models.len() - 1 {
            dialoguer::Input::<String>::with_theme(&ColorfulTheme::default())
                .with_prompt("Custom model name")
                .interact_text()?
        } else {
            ollama_models[gen_model_idx].to_string()
        };
        
        println!("\n{}", "Select model for data refinement (quality evaluation):".bright_cyan());
        let ref_model_options = vec![
            format!("Same as generation ({})", gen_model),
            "llama3".to_string(),
            "mistral".to_string(),
            "Custom".to_string(),
        ];
        
        let ref_model_idx = Select::with_theme(&ColorfulTheme::default())
            .with_prompt("Refinement model")
            .default(0)
            .items(&ref_model_options)
            .interact()?;
        
        let ref_model = match ref_model_idx {
            0 => gen_model.clone(),
            3 => dialoguer::Input::<String>::with_theme(&ColorfulTheme::default())
                .with_prompt("Custom model name")
                .interact_text()?,
            idx => ref_model_options[idx].clone(),
        };
        
        (gen_model, ref_model)
    } else {
        // Template-based - use the provided default models
        (ollama_gen_model.clone(), ollama_refine_model.clone())
    };
    
    // Generate the model
    generate_model(
        &prompt,
        &config,
        &output_dir,
        use_ollama,
        &gen_model,
        &ref_model,
    )?;
    
    Ok(())
}

fn generate_model(
    prompt: &str,
    config: &Config,
    output_dir: &PathBuf,
    use_ollama: bool,
    ollama_gen_model: &str,
    ollama_refine_model: &str,
) -> Result<()> {
    println!("\n{}", format!("Goal: Create a GPT that \"{}\"", prompt).bright_green());
    
    // Create tokio runtime for async operations
    let rt = tokio::runtime::Runtime::new()?;
    
    // Create the full pipeline
    let result = rt.block_on(crate::training::run_full_pipeline(
        prompt,
        config,
        output_dir,
        if use_ollama { 
            DataGenerationMethod::Ollama { 
                gen_model: ollama_gen_model.to_string(),
                refine_model: ollama_refine_model.to_string(),
            }
        } else {
            DataGenerationMethod::Template
        },
    ))?;
    
    println!("\n{}", "=".repeat(60).bright_blue());
    println!("{}", "Text2GPT1 Complete!".bright_green().bold());
    println!("{}", format!("Model folder: {}", result.model_path.display()).bright_white());
    println!("{}", format!("Domain: {}", result.domain).bright_white());
    println!("{}", format!("Model size: {:?}", config.model_size).bright_white());
    println!("{}", format!("Epochs: {}", config.num_epochs).bright_white());
    println!("{}", format!("Dataset: {}/dataset.json", result.model_path.display()).bright_white());
    println!("{}", "=".repeat(60).bright_blue());
    
    Ok(())
}

fn inference_mode(
    model_path: PathBuf,
    initial_prompt: Option<String>,
    max_length: usize,
    temperature: f64,
    top_k: usize,
) -> Result<()> {
    println!("{}", format!("Loading model from: {}", model_path.display()).bright_cyan());
    
    let generator = crate::inference::TextGenerator::load(&model_path)?;
    
    if let Some(prompt) = initial_prompt {
        // Single generation
        let generated = generator.generate(&prompt, max_length, temperature, top_k)?;
        println!("\n{}", "Generated text:".bright_green());
        println!("{}", generated);
    } else {
        // Interactive mode
        println!("\n{}", "Interactive inference mode. Type 'quit' to exit.".bright_cyan());
        
        loop {
            let prompt = dialoguer::Input::<String>::with_theme(&ColorfulTheme::default())
                .with_prompt("Prompt")
                .interact_text()?;
            
            if prompt.trim().eq_ignore_ascii_case("quit") {
                break;
            }
            
            let generated = generator.generate(&prompt, max_length, temperature, top_k)?;
            println!("\n{}", "Generated:".bright_green());
            println!("{}\n", generated);
        }
    }
    
    Ok(())
}

fn list_models(models_dir: &PathBuf) -> Result<()> {
    let models = crate::io::json::list_saved_models(models_dir)?;
    
    if models.is_empty() {
        println!("{}", "No models found.".yellow());
        return Ok(());
    }
    
    println!("{}", format!("Found {} models:", models.len()).bright_cyan());
    println!("{}", "-".repeat(80));
    
    for (i, model) in models.iter().enumerate() {
        println!("{}", format!("{}. {}", i + 1, model.folder_name).bright_white().bold());
        println!("   Path: {}", model.path.display());
        println!("   Prompt: {}", model.info.prompt);
        println!("   Domain: {}", model.info.category);
        println!("   Created: {}", model.info.creation_date);
        println!("   Model size: {}", model.info.model_size);
        println!("{}", "-".repeat(80));
    }
    
    Ok(())
}

fn select_model(models_dir: &PathBuf) -> Result<Option<PathBuf>> {
    let models = crate::io::json::list_saved_models(models_dir)?;
    
    if models.is_empty() {
        println!("{}", "No models found. Please create a model first.".yellow());
        return Ok(None);
    }
    
    let model_names: Vec<String> = models.iter()
        .map(|m| format!("{} - {} ({})", m.folder_name, m.info.prompt, m.info.creation_date))
        .collect();
    
    let selection = Select::with_theme(&ColorfulTheme::default())
        .with_prompt("Select a model")
        .default(0)
        .items(&model_names)
        .interact()?;
    
    Ok(Some(models[selection].path.clone()))
}