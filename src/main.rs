use anyhow::{anyhow, Result};
use clap::{Parser, Subcommand};
use colored::*;
use dialoguer::{theme::ColorfulTheme, Select};
use std::io::Write;
use std::path::PathBuf;

mod config;
mod data;
mod inference;
mod io;
mod model;
mod tokenizer;
mod training;
mod utils;

use crate::config::{Config, ModelSize, OllamaTimeoutPreset};
use crate::data::generator::DataGenerationMethod;
use crate::utils::{
    check_ollama_running, get_installed_models, format_model_list_for_display,
    generate_category_warning, show_ollama_not_running_error, show_no_models_error,
    show_installation_hints,
};

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
        
        /// Initial tokens to generate (defaults based on model size)
        #[arg(long)]
        initial_tokens: Option<usize>,
        
        /// Final tokens after refinement (defaults based on model size)
        #[arg(long)]
        final_tokens: Option<usize>,
        
        /// Model size (12M, 33M, or 117M)
        #[arg(long, default_value = "12M")]
        model_size: ModelSize,
        
        /// Don't use Ollama for data generation
        #[arg(long)]
        no_ollama: bool,
        
        /// Ollama model for generation
        #[arg(long, default_value = "llama3.1:8b")]
        ollama_gen_model: String,
        
        /// Ollama model for refinement (defaults to generation model)
        #[arg(long)]
        ollama_refine_model: Option<String>,
        
        /// Ollama timeout preset (auto, gpu, cpu)
        #[arg(long, default_value = "auto")]
        ollama_timeout_preset: OllamaTimeoutPreset,
        
        /// Ollama connection timeout in seconds
        #[arg(long)]
        ollama_timeout_connection: Option<u64>,
        
        /// Ollama generation timeout in seconds
        #[arg(long)]
        ollama_timeout_generation: Option<u64>,
        
        /// Ollama evaluation timeout in seconds
        #[arg(long)]
        ollama_timeout_evaluation: Option<u64>,
        
        /// Ollama request interval in milliseconds
        #[arg(long)]
        ollama_request_interval: Option<u64>,
        
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

// Category definitions with preset information
const CATEGORIES: &[(&str, &str, &str)] = &[
    ("cooking", "🍳 Cooking & Recipes", "Generate cooking recipes and food descriptions"),
    ("poetry", "✍️ Poetry & Creative", "Create poetic and creative texts"),
    ("technical", "💻 Technical & Code", "Generate technical documentation and code explanations"),
    ("general", "📝 General Purpose", "General-purpose text generation"),
];

// Enhanced preset structure with English names
#[derive(Clone)]
struct PresetInfo {
    title: &'static str,
    title_en: &'static str,  // English name for folder naming
    prompt: &'static str,
    description: &'static str,
}

// Preset definitions for each category with English names
fn get_presets(category: &str) -> Vec<PresetInfo> {
    match category {
        "cooking" => vec![
            PresetInfo {
                title: "Simple home cooking recipes",
                title_en: "simple_cooking_recipes",
                prompt: "Generate easy and delicious home cooking recipes",
                description: "Provides easy-to-follow recipes for everyday meals",
            },
            PresetInfo {
                title: "Professional chef assistant",
                title_en: "professional_chef_assistant",
                prompt: "Provide advanced cooking techniques and recipes",
                description: "Advanced culinary techniques and gourmet recipes",
            },
            PresetInfo {
                title: "Healthy diet recipes",
                title_en: "healthy_diet_recipes",
                prompt: "Generate healthy and nutritious recipes",
                description: "Focus on nutrition and balanced meals",
            },
            PresetInfo {
                title: "Desserts specialist",
                title_en: "desserts_specialist",
                prompt: "Specialize in sweets and dessert recipes",
                description: "Cakes, cookies, and various desserts",
            },
        ],
        "poetry" => vec![
            PresetInfo {
                title: "Modern poetry generator",
                title_en: "modern_poetry_generator",
                prompt: "Create contemporary and expressive poetry",
                description: "Free verse and modern poetic styles",
            },
            PresetInfo {
                title: "Haiku master",
                title_en: "haiku_master",
                prompt: "Generate traditional Japanese poetry forms",
                description: "5-7-5 and other traditional forms",
            },
            PresetInfo {
                title: "Story creator",
                title_en: "story_creator",
                prompt: "Create short stories and narratives",
                description: "Imaginative storytelling and narratives",
            },
            PresetInfo {
                title: "Songwriter",
                title_en: "songwriter",
                prompt: "Write emotional and creative song lyrics",
                description: "Lyrics for various music genres",
            },
        ],
        "technical" => vec![
            PresetInfo {
                title: "Programming explainer",
                title_en: "programming_explainer",
                prompt: "Explain programming concepts clearly",
                description: "Clear explanations of coding concepts",
            },
            PresetInfo {
                title: "API documentation writer",
                title_en: "api_documentation_writer",
                prompt: "Create technical specifications and docs",
                description: "Professional technical documentation",
            },
            PresetInfo {
                title: "Algorithm specialist",
                title_en: "algorithm_specialist",
                prompt: "Explain algorithms and data structures",
                description: "Computer science fundamentals",
            },
            PresetInfo {
                title: "System architect",
                title_en: "system_architect",
                prompt: "Describe system design and architecture",
                description: "Large-scale system design patterns",
            },
        ],
        "general" => vec![
            PresetInfo {
                title: "Daily conversation assistant",
                title_en: "daily_conversation_assistant",
                prompt: "Generate friendly casual conversations",
                description: "Natural everyday dialogue",
            },
            PresetInfo {
                title: "Business writer",
                title_en: "business_writer",
                prompt: "Create professional business documents",
                description: "Formal business communication",
            },
            PresetInfo {
                title: "Education support",
                title_en: "education_support",
                prompt: "Provide clear educational explanations",
                description: "Learning assistance across subjects",
            },
            PresetInfo {
                title: "News summarizer",
                title_en: "news_summarizer",
                prompt: "Summarize news and information concisely",
                description: "Extract key points efficiently",
            },
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
            ollama_timeout_preset,
            ollama_timeout_connection,
            ollama_timeout_generation,
            ollama_timeout_evaluation,
            ollama_request_interval,
            output_dir,
            batch_size,
            learning_rate,
            seed,
        }) => {
            let mut config = Config::new(model_size, seed);
            config.num_epochs = epochs;
            
            // Use model size-based defaults if not explicitly specified
            if let Some(tokens) = initial_tokens {
                config.initial_tokens = tokens;
            }
            if let Some(tokens) = final_tokens {
                config.final_tokens = tokens;
            }
            
            config.batch_size = batch_size;
            config.learning_rate = learning_rate as f32;
            
            // Initialize device first
            config.init_device()?;
            
            // Check environment variable for preset override
            let preset = std::env::var("TEXT2GPT1_OLLAMA_TIMEOUT_PRESET")
                .ok()
                .and_then(|s| s.parse::<OllamaTimeoutPreset>().ok())
                .unwrap_or(ollama_timeout_preset);
            
            // Initialize Ollama timeouts
            let timeouts = config.init_ollama_timeouts(preset);
            
            // Apply command line overrides
            let timeouts = timeouts.clone().with_overrides(
                ollama_timeout_connection,
                ollama_timeout_generation,
                ollama_timeout_evaluation,
                ollama_request_interval,
            );
            config.ollama_timeouts = Some(timeouts);
            
            // Log timeout settings
            config.ollama_timeouts().log_settings(preset);
            
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

/// Estimate data generation time based on token count and method
fn estimate_generation_time(tokens: usize, use_ollama: bool) -> String {
    let minutes = if use_ollama {
        // Ollama generation is slower but higher quality
        (tokens as f64 / 1000.0) * 0.5 // ~0.5 minutes per 1000 tokens
    } else {
        // Template generation is much faster
        (tokens as f64 / 1000.0) * 0.05 // ~0.05 minutes per 1000 tokens
    };
    
    if minutes < 1.0 {
        "< 1 minute".to_string()
    } else if minutes < 60.0 {
        format!("{:.0} minutes", minutes.ceil())
    } else {
        format!("{:.1} hours", minutes / 60.0)
    }
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
            let config = Config::new(ModelSize::Small, 42); // Will prompt for size in interactive mode
            interactive_generate(
                config,
                PathBuf::from("models"),
                true,
                "llama3.1:8b".to_string(),
                "llama3.1:8b".to_string(),
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
    // Model size selection
    println!("{}", "Select model size:".bright_cyan());
    let model_sizes = vec![
        ("12M", "Small (12M params) - Fast training, ~500MB memory", ModelSize::Small),
        ("33M", "Medium (33M params) - Balanced quality, ~1GB memory", ModelSize::Medium),
        ("117M", "Large (117M params) - Best quality, ~2GB memory", ModelSize::Large),
    ];
    
    let size_descriptions: Vec<String> = model_sizes.iter()
        .map(|(size, desc, _)| format!("{} - {}", size, desc))
        .collect();
    
    let size_idx = Select::with_theme(&ColorfulTheme::default())
        .with_prompt("Model size")
        .default(0)
        .items(&size_descriptions)
        .interact()?;
    
    let (_, _, selected_model_size) = model_sizes[size_idx];
    
    // Update config with new model size and corresponding token defaults
    config = Config::new(selected_model_size, config.seed);
    config.init_device()?;
    config.init_ollama_timeouts(OllamaTimeoutPreset::Auto);
    
    // Show recommended token counts
    let (rec_initial, rec_final) = Config::get_recommended_tokens(selected_model_size);
    println!("\n{}", format!("Recommended dataset size for {}: {} → {} tokens", 
        selected_model_size, rec_initial, rec_final).bright_white());
    
    // Category selection
    println!("\n{}", "Select a category:".bright_cyan());
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
        .map(|p| format!("{} - {}", p.title, p.description))
        .collect();
    preset_names.push("✏️ Custom prompt".to_string());
    
    println!("\n{}", "Select a preset or enter custom prompt:".bright_cyan());
    let preset_idx = Select::with_theme(&ColorfulTheme::default())
        .with_prompt("Preset")
        .default(0)
        .items(&preset_names)
        .interact()?;
    
    let (prompt, preset_en) = if preset_idx < presets.len() {
        let preset = &presets[preset_idx];
        (preset.prompt.to_string(), Some(preset.title_en))
    } else {
        println!("\n{}", "Enter your custom prompt:".bright_cyan());
        let custom_prompt = dialoguer::Input::<String>::with_theme(&ColorfulTheme::default())
            .with_prompt("Prompt")
            .interact_text()?;
        (custom_prompt, None)
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
    
    // Data size selection
    println!("\n{}", "Select dataset size:".bright_cyan());
    let data_size_options = vec![
        ("Recommended", format!("{} → {} tokens", config.initial_tokens, config.final_tokens), None),
        ("Quick test", "10,000 → 5,000 tokens - Very fast, lower quality".to_string(), Some((10_000, 5_000))),
        ("Light", "20,000 → 10,000 tokens - Fast training".to_string(), Some((20_000, 10_000))),
        ("Standard", "50,000 → 25,000 tokens - Good balance".to_string(), Some((50_000, 25_000))),
        ("High quality", "100,000 → 50,000 tokens - Better results".to_string(), Some((100_000, 50_000))),
        ("Custom", "Specify custom token counts".to_string(), None),
    ];
    
    let data_descriptions: Vec<String> = data_size_options.iter()
        .map(|(name, desc, _)| format!("{} - {}", name, desc))
        .collect();
    
    let data_idx = Select::with_theme(&ColorfulTheme::default())
        .with_prompt("Dataset size")
        .default(0)
        .items(&data_descriptions)
        .interact()?;
    
    match data_idx {
        0 => {}, // Keep recommended (already set)
        5 => {
            // Custom
            config.initial_tokens = dialoguer::Input::<usize>::with_theme(&ColorfulTheme::default())
                .with_prompt("Initial tokens (before refinement)")
                .default(config.initial_tokens)
                .validate_with(|input: &usize| {
                    if *input >= 1000 && *input <= 1_000_000 {
                        Ok(())
                    } else {
                        Err("Must be between 1,000 and 1,000,000")
                    }
                })
                .interact_text()?;
                
            config.final_tokens = dialoguer::Input::<usize>::with_theme(&ColorfulTheme::default())
                .with_prompt("Final tokens (after refinement)")
                .default(config.initial_tokens / 2)
                .validate_with(|input: &usize| {
                    if *input >= 500 && *input <= config.initial_tokens {
                        Ok(())
                    } else {
                        Err("Must be between 500 and initial tokens")
                    }
                })
                .interact_text()?;
        }
        _ => {
            // Predefined option
            if let Some((initial, final_t)) = data_size_options[data_idx].2 {
                config.initial_tokens = initial;
                config.final_tokens = final_t;
            }
        }
    }
    
    // Show estimated time
    let estimated_time = estimate_generation_time(config.initial_tokens, use_ollama);
    println!("\n{}", format!("Estimated data generation time: {}", estimated_time).bright_yellow());
    
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
    
    let use_ollama = gen_method_idx == 1 && use_ollama;
    let (gen_model, ref_model) = if use_ollama {
        // Ollama method - select models
        println!("\n{}", "Select Ollama model for data generation:".bright_cyan());
        
        // Updated model list with categories
        let ollama_models_with_info = vec![
            ("llama3.1:8b", "Llama 3.1 (8B) - Best overall balance", "general"),
            ("llama3.1:70b", "Llama 3.1 (70B) - Highest quality", "large"),
            ("qwen2.5:7b", "Qwen 2.5 (7B) - Excellent multilingual", "general"),
            ("qwen2.5:14b", "Qwen 2.5 (14B) - High quality multilingual", "medium"),
            ("qwen2.5:72b", "Qwen 2.5 (72B) - Top multilingual", "large"),
            ("mistral:7b-v0.3", "Mistral v0.3 (7B) - Fast & efficient", "general"),
            ("mixtral:8x7b", "Mixtral MoE (8x7B) - High quality", "large"),
            ("gemma2:2b", "Gemma 2 (2B) - Lightweight", "small"),
            ("gemma2:9b", "Gemma 2 (9B) - Balanced", "medium"),
            ("phi3:mini", "Phi-3 Mini (3.8B) - Efficient", "small"),
            ("codellama:7b", "Code Llama (7B) - Code specialist", "technical"),
            ("deepseek-coder:6.7b", "DeepSeek Coder (6.7B) - Code focused", "technical"),
            ("tinyllama:1.1b", "TinyLlama (1.1B) - Ultra light", "tiny"),
            ("Custom", "Enter custom model name", "custom"),
        ];
        
        let model_display: Vec<String> = ollama_models_with_info.iter()
            .map(|(_, desc, _)| desc.to_string())
            .collect();
        
        let gen_model_idx = Select::with_theme(&ColorfulTheme::default())
            .with_prompt("Generation model")
            .default(0)
            .items(&model_display)
            .interact()?;
        
        let gen_model = if gen_model_idx == ollama_models_with_info.len() - 1 {
            // Custom model selection - check installed models
            println!("\n{}", "Checking installed Ollama models...".bright_yellow());
            
            // Create a new runtime for this async operation
            let rt = tokio::runtime::Runtime::new()?;
            
            // Check if Ollama is running
            let ollama_running = rt.block_on(check_ollama_running())?;
            if !ollama_running {
                show_ollama_not_running_error();
                return Err(anyhow::anyhow!("Ollama is not running"));
            }
            
            // Get installed models
            let installed_models = rt.block_on(get_installed_models())?;
            if installed_models.is_empty() {
                show_no_models_error();
                return Err(anyhow::anyhow!("No Ollama models installed"));
            }
            
            // Show category warning if needed
            if let Some(warning) = generate_category_warning(category_key, &installed_models) {
                println!("{}", warning);
            }
            
            // Format models for display
            let model_list = format_model_list_for_display(&installed_models);
            let mut selection_items = model_list;
            selection_items.push("\n[Manual Input]".bright_magenta().to_string());
            selection_items.push("[Install New Model]".bright_green().to_string());
            selection_items.push("[Cancel]".bright_red().to_string());
            
            println!("\n{}", "Select an installed model:".bright_cyan());
            let model_selection = Select::with_theme(&ColorfulTheme::default())
                .with_prompt("Model")
                .items(&selection_items)
                .interact()?;
            
            // Calculate actual model index (accounting for category headers)
            let mut actual_model_idx = 0;
            let mut current_idx = 0;
            let mut selected_model_name = String::new();
            
            for item in &selection_items {
                if current_idx == model_selection {
                    if item.contains("[Manual Input]") {
                        selected_model_name = dialoguer::Input::<String>::with_theme(&ColorfulTheme::default())
                            .with_prompt("Enter model name")
                            .interact_text()?;
                        break;
                    } else if item.contains("[Install New Model]") {
                        println!("{}", show_installation_hints(category_key));
                        return Err(anyhow::anyhow!("Please install a model and try again"));
                    } else if item.contains("[Cancel]") {
                        return Err(anyhow::anyhow!("Model selection cancelled"));
                    } else if !item.starts_with("\n[") && !item.starts_with("[") {
                        // This is a model entry, extract the name
                        if let Some(space_pos) = item.find(' ') {
                            selected_model_name = item[..space_pos].to_string();
                            break;
                        }
                    }
                }
                current_idx += 1;
            }
            
            selected_model_name
        } else {
            ollama_models_with_info[gen_model_idx].0.to_string()
        };
        
        println!("\n{}", "Select model for data refinement (quality evaluation):".bright_cyan());
        let ref_model_options = vec![
            format!("Same as generation ({})", gen_model),
            "llama3.1:8b - Balanced evaluator".to_string(),
            "qwen2.5:7b - Multilingual evaluator".to_string(),
            "mistral:7b-v0.3 - Fast evaluator".to_string(),
            "Custom".to_string(),
        ];
        
        let ref_model_idx = Select::with_theme(&ColorfulTheme::default())
            .with_prompt("Refinement model")
            .default(0)
            .items(&ref_model_options)
            .interact()?;
        
        let ref_model = match ref_model_idx {
            0 => gen_model.clone(),
            4 => {
                // Custom selection for refinement model
                println!("\n{}", "Select refinement model:".bright_cyan());
                
                // Create a new runtime for this async operation
                let rt = tokio::runtime::Runtime::new()?;
                
                // Try to get installed models, but don't error if failed
                match rt.block_on(get_installed_models()) {
                    Ok(models) if !models.is_empty() => {
                        // Show simplified list
                        let model_names: Vec<String> = models.iter()
                            .map(|m| format!("{} ({})", m.name, m.size))
                            .collect();
                        
                        let mut options = vec![format!("Same as generation ({})", gen_model)];
                        options.extend(model_names);
                        options.push("[Manual Input]".to_string());
                        
                        let selection = Select::with_theme(&ColorfulTheme::default())
                            .with_prompt("Refinement model")
                            .default(0)
                            .items(&options)
                            .interact()?;
                        
                        if selection == 0 {
                            gen_model.clone()
                        } else if selection == options.len() - 1 {
                            dialoguer::Input::<String>::with_theme(&ColorfulTheme::default())
                                .with_prompt("Custom model name")
                                .interact_text()?
                        } else {
                            models[selection - 1].name.clone()
                        }
                    }
                    _ => {
                        // Fallback to manual input
                        dialoguer::Input::<String>::with_theme(&ColorfulTheme::default())
                            .with_prompt("Custom model name")
                            .interact_text()?
                    }
                }
            }
            idx => {
                // Extract model name from the option string
                match idx {
                    1 => "llama3.1:8b".to_string(),
                    2 => "qwen2.5:7b".to_string(),
                    3 => "mistral:7b-v0.3".to_string(),
                    _ => gen_model.clone(),
                }
            }
        };
        
        (gen_model, ref_model)
    } else {
        // Template-based - use the provided default models
        (ollama_gen_model.clone(), ollama_refine_model.clone())
    };
    
    // Generate the model with preset English name
    generate_model_with_preset(
        &prompt,
        preset_en,
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
    generate_model_with_preset(
        prompt,
        None,
        config,
        output_dir,
        use_ollama,
        ollama_gen_model,
        ollama_refine_model,
    )
}

fn generate_model_with_preset(
    prompt: &str,
    preset_en: Option<&str>,
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
    println!("{}", format!("Final loss: {:.4}", result.final_loss).bright_white());
    println!("{}", format!("Training time: {:.1}s", result.training_time_seconds).bright_white());
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
    
    // Use the improved loading method
    let generator = crate::inference::TextGenerator::load(&model_path)?;
    
    println!("{}", "Model loaded successfully!".bright_green());
    println!("\n{}", generator.model_info());
    
    if let Some(prompt) = initial_prompt {
        // Single generation
        println!("\n{}", "Generating text...".bright_yellow());
        let generated = generator.generate(&prompt, max_length, temperature, top_k)?;
        println!("\n{}", "Generated text:".bright_green());
        println!("{}", generated);
    } else {
        // Interactive mode
        println!("\n{}", "Interactive inference mode. Type 'quit' to exit.".bright_cyan());
        println!("{}", "You can also use special commands:".bright_cyan());
        println!("  /info    - Show model information");
        println!("  /params  - Show current generation parameters");
        println!("  /set     - Change generation parameters");
        
        let mut current_temp = temperature;
        let mut current_top_k = top_k;
        let mut current_max_length = max_length;
        
        loop {
            let prompt = dialoguer::Input::<String>::with_theme(&ColorfulTheme::default())
                .with_prompt("Prompt")
                .interact_text()?;
            
            if prompt.trim().eq_ignore_ascii_case("quit") {
                break;
            }
            
            match prompt.trim() {
                "/info" => {
                    println!("\n{}", generator.model_info());
                    continue;
                }
                "/params" => {
                    println!("\nCurrent parameters:");
                    println!("  Temperature: {}", current_temp);
                    println!("  Top-k: {}", current_top_k);
                    println!("  Max length: {}", current_max_length);
                    continue;
                }
                "/set" => {
                    // Interactive parameter setting
                    current_temp = dialoguer::Input::<f64>::with_theme(&ColorfulTheme::default())
                        .with_prompt("Temperature (0.1-2.0)")
                        .default(current_temp)
                        .validate_with(|input: &f64| {
                            if *input >= 0.1 && *input <= 2.0 {
                                Ok(())
                            } else {
                                Err("Temperature must be between 0.1 and 2.0")
                            }
                        })
                        .interact_text()?;
                    
                    current_top_k = dialoguer::Input::<usize>::with_theme(&ColorfulTheme::default())
                        .with_prompt("Top-k (1-100)")
                        .default(current_top_k)
                        .validate_with(|input: &usize| {
                            if *input >= 1 && *input <= 100 {
                                Ok(())
                            } else {
                                Err("Top-k must be between 1 and 100")
                            }
                        })
                        .interact_text()?;
                    
                    current_max_length = dialoguer::Input::<usize>::with_theme(&ColorfulTheme::default())
                        .with_prompt("Max length (10-500)")
                        .default(current_max_length)
                        .validate_with(|input: &usize| {
                            if *input >= 10 && *input <= 500 {
                                Ok(())
                            } else {
                                Err("Max length must be between 10 and 500")
                            }
                        })
                        .interact_text()?;
                    
                    println!("Parameters updated!");
                    continue;
                }
                _ => {}
            }
            
            // Stream generation with live output
            print!("\n{}", "Generated:".bright_green());
            print!(" ");
            std::io::stdout().flush()?;
            
            let _generated = generator.generate_stream(
                &prompt, 
                current_max_length, 
                current_temp, 
                current_top_k,
                |token| {
                    print!("{}", token);
                    std::io::stdout().flush()?;
                    Ok(())
                }
            )?;
            
            println!("\n");
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
        println!("   Epochs: {}", model.info.training_params.epochs);
        println!("   Final tokens: {}", model.info.data_stats.final_tokens);
        if model.info.ollama_used {
            if let Some(ref gen_model) = model.info.generation_model {
                println!("   Generation model: {}", gen_model);
            }
        }
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
