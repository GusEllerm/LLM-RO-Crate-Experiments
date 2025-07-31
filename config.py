"""
Configuration settings for LLM experiments with RO-Crate manifests.
"""

import os
from typing import Dict, Any

# API Configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')

# Model Configuration
DEFAULT_MODELS = {
    'openai': 'gpt-3.5-turbo',
    'anthropic': 'claude-3-sonnet-20240229',
    'local': 'llama2'  # For local models
}

# Token Limits
TOKEN_LIMITS = {
    'gpt-3.5-turbo': 4096,
    'gpt-4': 8192,
    'gpt-4-turbo': 128000,
    'claude-3-sonnet': 200000,
    'claude-3-opus': 200000
}

# Experiment Configuration
EXPERIMENT_CONFIG = {
    'max_tokens_response': 1000,
    'temperature': 0.7,
    'top_p': 0.9,
    'max_retries': 3,
    'timeout_seconds': 30
}

# RO-Crate Analysis Settings
ROCRATE_SETTINGS = {
    'max_files_to_describe': 10,
    'max_description_length': 500,
    'include_technical_details': True,
    'summarize_large_crates': True,
    'large_crate_threshold': 50  # Number of entities
}

# Output Configuration
OUTPUT_CONFIG = {
    'save_intermediate_results': True,
    'output_directory': './results',
    'log_level': 'INFO',
    'include_timestamps': True
}

def get_model_config(model_name: str) -> Dict[str, Any]:
    """Get configuration for a specific model."""
    return {
        'name': model_name,
        'max_tokens': TOKEN_LIMITS.get(model_name, 4096),
        'temperature': EXPERIMENT_CONFIG['temperature'],
        'max_response_tokens': EXPERIMENT_CONFIG['max_tokens_response']
    }
