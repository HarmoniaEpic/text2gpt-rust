pub mod folder_naming;
pub mod ollama;

pub use folder_naming::FolderNameBuilder;
pub use ollama::{
    check_ollama_running,
    get_installed_models,
    categorize_model,
    format_model_list_for_display,
    show_installation_hints,
    generate_category_warning,
    show_ollama_not_running_error,
    show_no_models_error,
    InstalledModel,
    ModelCategory,
};
