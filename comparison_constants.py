#!/usr/bin/env python3
"""
comparison_constants.py: Centralized constants and configuration for tool comparison system.

This module defines all magic numbers, thresholds, and configuration values used
throughout the tool comparison and decision support systems.
"""

# Metric weights for tool comparison
METRIC_WEIGHTS = {
    "functionality": 0.25,
    "usability": 0.20,
    "reliability": 0.20,
    "performance": 0.15,
    "compatibility": 0.10,
    "documentation": 0.10,
}

# Score normalization and thresholds
SCORE_LIMITS = {
    "min_score": 0.0,
    "max_score": 1.0,
    "default_score": 0.5,
    "confidence_threshold": 0.7,
}

# Functionality scoring constants
FUNCTIONALITY_CONFIG = {
    "max_parameter_count_normalization": 20.0,
    "default_param_score": 0.2,
    "param_count_weight": 0.4,
    "complexity_weight": 0.3,
    "description_weight": 0.2,
    "category_weight": 0.1,
}

# Usability scoring constants
USABILITY_CONFIG = {
    "required_param_penalty_factor": 0.7,
    "jargon_penalty_max": 0.4,
    "jargon_normalization_factor": 10.0,
    "documentation_quality_weight": 0.7,
    "avg_quality_weight": 0.3,
}

# Performance scoring constants
PERFORMANCE_CONFIG = {
    "base_performance_score": 0.7,
    "high_complexity_threshold": 0.7,
    "lightweight_category_bonus": 0.7,
}

# Reliability scoring constants
RELIABILITY_CONFIG = {
    "base_safety_score": 0.7,
    "default_reliability_score": 0.7,
}

# Category-based scoring bonuses
CATEGORY_BONUSES = {
    "functionality": {
        "advanced": 0.8,
        "comprehensive": 0.9,
        "specialized": 0.7,
        "basic": 0.3,
        "simple": 0.2,
    },
    "performance": {
        "optimized": 0.9,
        "fast": 0.8,
        "standard": 0.7,
        "lightweight": 0.7,
        "basic": 0.5,
    },
}

# Parameter complexity penalties
PARAMETER_COMPLEXITY_PENALTIES = {
    "object": 0.3,
    "array": 0.2,
    "any": 0.4,
    "union": 0.25,
}

# Documentation quality scoring
DOCUMENTATION_SCORING = {
    "technical_term_bonus": 0.1,
    "example_mention_bonus": 0.15,
    "empty_penalty": -0.5,
}

# Decision support thresholds
DECISION_THRESHOLDS = {
    "excellence_threshold": 0.7,
    "competence_threshold": 0.6,
    "quality_threshold": 0.8,
    "usability_threshold": 0.6,
    "trade_off_threshold": 0.3,
    "significant_difference_threshold": 0.2,
    "performance_difference_threshold": 0.2,
    "complexity_difference_threshold": 0.25,
}

# Task-specific scoring weights
TASK_SCORING_WEIGHTS = {
    "task_fit_weight": 0.6,
    "quality_weight": 0.4,
    "priority_metrics_weight": 0.7,
    "all_metrics_weight": 0.3,
}

# Confidence calculation factors
CONFIDENCE_FACTORS = {
    "data_completeness_weight": 0.4,
    "score_consistency_weight": 0.4,
    "score_confidence_weight": 0.2,
    "confidence_boost": 1.2,
    "minimum_confidence": 0.1,
    "moderate_score_peak": 0.5,
}

# Cache and performance settings
PERFORMANCE_CONFIG_CACHE = {
    "max_cache_size": 1000,
    "cache_ttl_seconds": 3600,
    "lru_eviction_threshold": 0.8,
}

# Search and ranking constants
SEARCH_CONFIG = {
    "keyword_boost_per_match": 0.1,
    "max_keyword_boost": 0.3,
    "default_fit_score": 0.5,
}

# Error handling constants
ERROR_CONFIG = {
    "retry_attempts": 3,
    "retry_delay_seconds": 1.0,
    "timeout_seconds": 30.0,
}