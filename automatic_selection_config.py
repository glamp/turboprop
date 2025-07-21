#!/usr/bin/env python3
"""
Automatic Tool Selection Configuration

This module contains all configurable constants and parameters for the 
automatic tool selection system. All hard-coded values are centralized 
here to allow easy customization and tuning.
"""

from typing import Dict, Any


# Complexity Scoring Configuration
COMPLEXITY_SCORING = {
    "base_complexity": 0.2,
    "complex_keyword_weight": 0.3,
    "advanced_keyword_weight": 0.2,
    "long_input_weight": 0.1,
    "long_input_threshold": 100,
    "max_complexity": 1.0,
}

# Confidence Calculation Weights
CONFIDENCE_WEIGHTS = {
    "pattern_weight": 0.6,
    "historical_weight": 0.4,
    "confidence_boost": 0.1,
    "default_confidence": 0.5,
    "max_confidence": 1.0,
}

# Tool Ranking Weights
RANKING_WEIGHTS = {
    "user_preference_weight": 0.4,
    "context_suitability_weight": 0.35,
    "historical_effectiveness_weight": 0.25,
}

# Tool Suitability Rules
TOOL_SUITABILITY_RULES: Dict[str, Dict[str, float]] = {
    "search": {
        "search_files": 0.9,
        "grep": 0.8,
        "find": 0.7,
        "analyze": 0.4
    },
    "file": {
        "read": 0.9,
        "write": 0.9,
        "edit": 0.8,
        "search": 0.6
    },
    "analysis": {
        "analyze": 0.9,
        "search": 0.7,
        "read": 0.6
    }
}

# Default Values
DEFAULT_VALUES = {
    "default_score": 0.5,
    "default_confidence": 0.0,
    "default_ranking_score": 0.5,
    "fallback_context_confidence": 0.0,
}

# Complexity Thresholds
COMPLEXITY_THRESHOLDS = {
    "high_complexity_threshold": 0.7,
    "low_complexity_threshold": 0.3,
    "complexity_bonus": 0.1,
}

# Score Limits
SCORE_LIMITS = {
    "min_score": 0.0,
    "max_score": 1.0,
}

# Storage Configuration
STORAGE_CONFIG = {
    "default_storage_dir": ".turboprop",
    "preferences_file": "preferences.json",
    "learning_model_file": "learning_model.json",
    "effectiveness_data_file": "effectiveness_data.json",
}

# Memory Management Configuration  
MEMORY_CONFIG = {
    "max_history_size": 1000,
    "history_cleanup_threshold": 1200,
    "batch_size": 100,
}

# Performance Configuration
PERFORMANCE_CONFIG = {
    "max_suggestions_to_process": 10,
    "debounce_time_seconds": 1.0,
    "cache_timeout_seconds": 300,  # 5 minutes
}


def get_config() -> Dict[str, Any]:
    """Get complete configuration dictionary."""
    return {
        "complexity_scoring": COMPLEXITY_SCORING,
        "confidence_weights": CONFIDENCE_WEIGHTS,
        "ranking_weights": RANKING_WEIGHTS,
        "tool_suitability_rules": TOOL_SUITABILITY_RULES,
        "default_values": DEFAULT_VALUES,
        "complexity_thresholds": COMPLEXITY_THRESHOLDS,
        "score_limits": SCORE_LIMITS,
        "storage_config": STORAGE_CONFIG,
        "memory_config": MEMORY_CONFIG,
        "performance_config": PERFORMANCE_CONFIG,
    }


def load_config_from_file(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file, falling back to defaults."""
    import json
    from pathlib import Path
    
    try:
        config_file = Path(config_path)
        if config_file.exists():
            with open(config_file, 'r') as f:
                file_config = json.load(f)
            
            # Merge with defaults (file config takes precedence)
            default_config = get_config()
            for key, value in file_config.items():
                if key in default_config and isinstance(value, dict):
                    default_config[key].update(value)
                else:
                    default_config[key] = value
            
            return default_config
    except Exception as e:
        # Log error and return defaults
        print(f"Warning: Could not load config from {config_path}: {e}")
    
    return get_config()


# Global configuration instance (can be overridden)
CONFIG = get_config()