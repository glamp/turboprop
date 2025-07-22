"""
YAML Configuration Loading Module for Turboprop

This module provides functionality to load configuration from .turboprop.yml files
with fallback to environment variables and defaults. It supports the directory-based
configuration pattern where each repository can have its own .turboprop.yml file.

Example usage:
    from yaml_config import load_yaml_config
    
    # Load config from current directory's .turboprop.yml
    config_data = load_yaml_config()
    
    # Load config from specific directory
    config_data = load_yaml_config("/path/to/repo")
    
    # Get a specific config value with fallback
    max_results = get_config_value(config_data, "search.default_max_results", 5)
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

try:
    import yaml
except ImportError:
    raise ImportError(
        "PyYAML is required for YAML configuration support. " "Please install it with: pip install PyYAML>=6.0"
    )

logger = logging.getLogger(__name__)


class YAMLConfigError(Exception):
    """Exception raised when YAML configuration loading fails."""

    pass


def find_config_file(directory: Optional[Union[str, Path]] = None) -> Optional[Path]:
    """
    Find .turboprop.yml configuration file in the specified directory.

    Args:
        directory: Directory to search in. Defaults to current working directory.

    Returns:
        Path to .turboprop.yml file if found, None otherwise.
    """
    if directory is None:
        directory = Path.cwd()
    else:
        directory = Path(directory)

    config_file = directory / ".turboprop.yml"

    if config_file.exists() and config_file.is_file():
        return config_file

    return None


def load_yaml_config(directory: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """
    Load YAML configuration from .turboprop.yml file.

    Args:
        directory: Directory containing .turboprop.yml. Defaults to current directory.

    Returns:
        Dictionary containing parsed YAML configuration, or empty dict if no file found.

    Raises:
        YAMLConfigError: If YAML file exists but cannot be parsed.
    """
    config_file = find_config_file(directory)

    if config_file is None:
        logger.debug("No .turboprop.yml file found, using environment variables and defaults")
        return {}

    try:
        with open(config_file, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f) or {}

        logger.info(f"Loaded YAML configuration from {config_file}")
        return config_data

    except yaml.YAMLError as e:
        raise YAMLConfigError(f"Failed to parse YAML configuration file {config_file}: {e}") from e
    except (OSError, IOError) as e:
        raise YAMLConfigError(f"Failed to read configuration file {config_file}: {e}") from e


def get_config_value(
    config_data: Dict[str, Any], key_path: str, default: Any = None, env_var: Optional[str] = None
) -> Any:
    """
    Get configuration value with fallback chain: environment -> YAML -> default.

    Args:
        config_data: Parsed YAML configuration data
        key_path: Dot-separated path to config value (e.g., "database.threads")
        default: Default value if not found in environment or YAML
        env_var: Environment variable name to check first

    Returns:
        Configuration value from environment, YAML, or default (in that order)
    """
    # First try environment variable if provided (highest priority)
    if env_var:
        env_value = os.getenv(env_var)
        if env_value is not None:
            return env_value

    # Then try to get from YAML config
    yaml_value = _get_nested_value(config_data, key_path)
    if yaml_value is not None:
        return yaml_value

    # Finally return default
    return default


def _get_nested_value(data: Dict[str, Any], key_path: str) -> Any:
    """
    Get a nested value from a dictionary using dot notation.

    Args:
        data: Dictionary to search
        key_path: Dot-separated path (e.g., "database.threads")

    Returns:
        Value if found, None otherwise
    """
    if not key_path:
        return None

    keys = key_path.split(".")
    current = data

    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]

    return current


def validate_yaml_structure(config_data: Dict[str, Any]) -> bool:
    """
    Validate that the YAML configuration has the expected structure.

    Args:
        config_data: Parsed YAML configuration data

    Returns:
        True if structure is valid, False otherwise
    """
    if not isinstance(config_data, dict):
        return False

    # Define expected top-level sections
    valid_sections = {"database", "file_processing", "search", "embedding", "server", "logging", "mcp"}

    # Check that all keys are valid section names
    for key in config_data.keys():
        if key not in valid_sections:
            logger.warning(f"Unknown configuration section: {key}")

    # Check that each section is a dictionary
    for key, value in config_data.items():
        if key in valid_sections and not isinstance(value, dict):
            logger.error(f"Configuration section '{key}' must be a dictionary")
            return False

    return True


def merge_configs(yaml_config: Dict[str, Any], env_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge YAML and environment-based configurations.
    Environment variables take precedence over YAML values.

    Args:
        yaml_config: Configuration loaded from YAML file
        env_config: Configuration derived from environment variables

    Returns:
        Merged configuration dictionary
    """

    def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge two dictionaries."""
        result = base.copy()

        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = deep_merge(result[key], value)
            else:
                result[key] = value

        return result

    return deep_merge(yaml_config, env_config)


def create_sample_config() -> str:
    """
    Create a sample .turboprop.yml configuration file content.

    Returns:
        String containing sample YAML configuration with comments
    """
    return """# Turboprop Configuration File
# This file configures the semantic code search and indexing system.
# All settings are optional - if not specified, environment variables 
# (TURBOPROP_*) and built-in defaults will be used.

# Database configuration
database:
  memory_limit: "1GB"              # Memory limit for DuckDB
  threads: 4                       # Number of threads for database operations
  max_retries: 3                   # Maximum connection retry attempts
  retry_delay: 0.1                 # Delay between retries (seconds)
  connection_timeout: 30.0         # Connection timeout (seconds)
  statement_timeout: 60.0          # SQL statement timeout (seconds)
  lock_timeout: 10.0               # File lock timeout (seconds)
  auto_vacuum: true                # Enable automatic database optimization

# File processing configuration
file_processing:
  max_file_size_mb: 1.0            # Maximum file size to index (MB)
  debounce_seconds: 5.0            # Debounce delay for file watching (seconds)
  preview_length: 200              # Length of file previews (characters)
  snippet_length: 300              # Length of search snippets (characters)
  batch_size: 100                  # Batch size for processing files
  max_workers: 4                   # Maximum parallel workers
  enable_language_detection: true  # Enable programming language detection

# Search configuration
search:
  default_max_results: 5           # Default number of search results
  max_results_limit: 20            # Maximum allowed search results
  min_similarity: 0.1              # Minimum similarity threshold
  high_confidence_threshold: 0.8   # High confidence similarity threshold
  medium_confidence_threshold: 0.6 # Medium confidence similarity threshold

# Embedding model configuration
embedding:
  model: "all-MiniLM-L6-v2"        # SentenceTransformer model name
  dimensions: 384                  # Embedding vector dimensions
  device: "cpu"                    # Device: "cpu", "cuda", or "mps"
  batch_size: 32                   # Batch size for embedding generation
  max_retries: 3                   # Maximum embedding retry attempts
  retry_base_delay: 1.0            # Base delay for embedding retries

# HTTP server configuration
server:
  host: "0.0.0.0"                  # Server bind address
  port: 8000                       # Server port
  watch_directory: "."             # Directory to watch for changes
  watch_max_file_size_mb: 1.0      # Max file size for watching (MB)
  watch_debounce_seconds: 5.0      # Debounce for file watching (seconds)
  request_timeout: 30.0            # Request timeout (seconds)
  max_concurrent_requests: 10      # Maximum concurrent requests

# Logging configuration
logging:
  level: "INFO"                    # Log level: DEBUG, INFO, WARNING, ERROR, CRITICAL
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: null                       # Log file path (null for console only)
  max_size: 10485760               # Maximum log file size in bytes (10MB)
  backup_count: 5                  # Number of backup log files to keep

# MCP (Model Context Protocol) configuration
mcp:
  default_max_file_size_mb: 1.0    # Default max file size for MCP operations
  default_debounce_seconds: 5.0    # Default debounce for MCP file watching
  max_files_list: 100              # Maximum files to list in MCP responses
  max_search_results: 20           # Maximum search results for MCP
  default_max_recommendations: 5   # Default max tool recommendations
  default_max_alternatives: 5      # Default max alternative tools
  max_task_description_length: 2000 # Maximum task description length
"""


def get_config_file_path(directory: Optional[Union[str, Path]] = None) -> Path:
    """
    Get the path where the .turboprop.yml configuration file should be located.

    Args:
        directory: Directory path. Defaults to current working directory.

    Returns:
        Path to .turboprop.yml file (may or may not exist)
    """
    if directory is None:
        directory = Path.cwd()
    else:
        directory = Path(directory)

    return directory / ".turboprop.yml"
