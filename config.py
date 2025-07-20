"""
Configuration module for Turboprop semantic code search system.

This module centralizes all configuration constants and default values
to make the system more maintainable and configurable.

Example usage:
    from config import config

    # Access configuration values
    print(f"Database path: {config.database.get_db_path()}")

    # Get configuration summary
    print(config.get_summary())

    # Validate configuration
    try:
        config.validate()
        print("Configuration is valid")
    except ValueError as e:
        print(f"Configuration error: {e}")
"""

import os
import re
from pathlib import Path
from typing import Optional


class ConfigValidationError(ValueError):
    """Exception raised when configuration validation fails."""

    pass


def validate_positive_int(value: str, var_name: str, default: int) -> int:
    """Validate and convert string to positive integer."""
    try:
        result = int(value)
    except ValueError as e:
        raise ConfigValidationError(f"{var_name} must be a valid integer, got '{value}'") from e

    if result <= 0:
        raise ConfigValidationError(f"{var_name} must be positive, got {result}")
    return result


def validate_positive_float(value: str, var_name: str, default: float) -> float:
    """Validate and convert string to positive float."""
    try:
        result = float(value)
    except ValueError as e:
        raise ConfigValidationError(f"{var_name} must be a valid float, got '{value}'") from e

    if result <= 0:
        raise ConfigValidationError(f"{var_name} must be positive, got {result}")
    return result


def validate_non_negative_float(value: str, var_name: str, default: float) -> float:
    """Validate and convert string to non-negative float."""
    try:
        result = float(value)
    except ValueError as e:
        raise ConfigValidationError(f"{var_name} must be a valid float, got '{value}'") from e

    if result < 0:
        raise ConfigValidationError(f"{var_name} must be non-negative, got {result}")
    return result


def validate_memory_limit(value: str, var_name: str) -> str:
    """Validate memory limit string format (e.g., '1GB', '512MB')."""
    pattern = r"^(\d+(?:\.\d+)?)\s*(GB|MB|KB|B)$"
    if not re.match(pattern, value.upper()):
        raise ConfigValidationError(
            f"{var_name} must be in format like '1GB', '512MB', '1024KB', got '{value}'"
        )
    return value


def validate_boolean(value: str, var_name: str) -> bool:
    """Validate and convert string to boolean."""
    lower_value = value.lower()
    if lower_value in ("true", "1", "yes", "on"):
        return True
    elif lower_value in ("false", "0", "no", "off"):
        return False
    else:
        raise ConfigValidationError(
            f"{var_name} must be 'true', 'false', '1', '0', 'yes', 'no', "
            f"'on', or 'off', got '{value}'"
        )


def validate_log_level(value: str, var_name: str) -> str:
    """Validate log level."""
    valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    if value.upper() not in valid_levels:
        raise ConfigValidationError(f"{var_name} must be one of {valid_levels}, got '{value}'")
    return value.upper()


def validate_device(value: str, var_name: str) -> str:
    """Validate device string."""
    valid_devices = ["cpu", "cuda", "mps"]
    if value.lower() not in valid_devices:
        raise ConfigValidationError(f"{var_name} must be one of {valid_devices}, got '{value}'")
    return value.lower()


def validate_range_float(value: str, var_name: str, min_val: float, max_val: float) -> float:
    """Validate float is within specified range."""
    try:
        result = float(value)
    except ValueError as e:
        raise ConfigValidationError(f"{var_name} must be a valid float, got '{value}'") from e

    if result < min_val or result > max_val:
        raise ConfigValidationError(
            f"{var_name} must be between {min_val} and {max_val}, got {result}"
        )
    return result


class DatabaseConfig:
    """Database-related configuration constants."""

    # DuckDB performance settings
    MEMORY_LIMIT: str = validate_memory_limit(
        os.getenv("TURBOPROP_DB_MEMORY_LIMIT", "1GB"), "TURBOPROP_DB_MEMORY_LIMIT"
    )
    THREADS: int = validate_positive_int(
        os.getenv("TURBOPROP_DB_THREADS", "4"), "TURBOPROP_DB_THREADS", 4
    )

    # Database connection and retry settings
    MAX_RETRIES: int = validate_positive_int(
        os.getenv("TURBOPROP_DB_MAX_RETRIES", "3"), "TURBOPROP_DB_MAX_RETRIES", 3
    )
    RETRY_DELAY: float = validate_positive_float(
        os.getenv("TURBOPROP_DB_RETRY_DELAY", "0.1"), "TURBOPROP_DB_RETRY_DELAY", 0.1
    )

    # Connection pool and timeout settings
    MAX_CONNECTIONS_PER_THREAD: int = validate_positive_int(
        os.getenv("TURBOPROP_DB_MAX_CONNECTIONS_PER_THREAD", "1"),
        "TURBOPROP_DB_MAX_CONNECTIONS_PER_THREAD", 1
    )
    CONNECTION_TIMEOUT: float = validate_positive_float(
        os.getenv("TURBOPROP_DB_CONNECTION_TIMEOUT", "30.0"),
        "TURBOPROP_DB_CONNECTION_TIMEOUT", 30.0
    )
    STATEMENT_TIMEOUT: float = validate_positive_float(
        os.getenv("TURBOPROP_DB_STATEMENT_TIMEOUT", "60.0"), "TURBOPROP_DB_STATEMENT_TIMEOUT", 60.0
    )

    # File lock settings
    LOCK_TIMEOUT: float = validate_positive_float(
        os.getenv("TURBOPROP_DB_LOCK_TIMEOUT", "10.0"), "TURBOPROP_DB_LOCK_TIMEOUT", 10.0
    )
    LOCK_RETRY_INTERVAL: float = validate_positive_float(
        os.getenv("TURBOPROP_DB_LOCK_RETRY_INTERVAL", "0.1"),
        "TURBOPROP_DB_LOCK_RETRY_INTERVAL", 0.1
    )

    # Database file configuration
    DEFAULT_DB_NAME: str = "code_index.duckdb"
    DEFAULT_DB_DIR: str = ".turboprop"
    TABLE_NAME: str = "code_files"

    # Database optimization settings
    CHECKPOINT_INTERVAL: int = validate_positive_int(
        os.getenv("TURBOPROP_DB_CHECKPOINT_INTERVAL", "1000"),
        "TURBOPROP_DB_CHECKPOINT_INTERVAL", 1000
    )
    AUTO_VACUUM: bool = validate_boolean(
        os.getenv("TURBOPROP_DB_AUTO_VACUUM", "true"), "TURBOPROP_DB_AUTO_VACUUM"
    )
    TEMP_DIRECTORY: Optional[str] = os.getenv("TURBOPROP_DB_TEMP_DIRECTORY")

    # Connection management settings
    CONNECTION_MAX_AGE: float = validate_positive_float(
        os.getenv("TURBOPROP_DB_CONNECTION_MAX_AGE", "3600.0"),
        "TURBOPROP_DB_CONNECTION_MAX_AGE", 3600.0
    )  # 1 hour
    CONNECTION_IDLE_TIMEOUT: float = validate_positive_float(
        os.getenv("TURBOPROP_DB_CONNECTION_IDLE_TIMEOUT", "300.0"),
        "TURBOPROP_DB_CONNECTION_IDLE_TIMEOUT", 300.0
    )  # 5 minutes
    CONNECTION_HEALTH_CHECK_INTERVAL: float = validate_positive_float(
        os.getenv("TURBOPROP_DB_CONNECTION_HEALTH_CHECK_INTERVAL", "60.0"),
        "TURBOPROP_DB_CONNECTION_HEALTH_CHECK_INTERVAL",
        60.0,
    )  # 1 minute

    @classmethod
    def get_db_path(cls, repo_path: Optional[Path] = None) -> Path:
        """Get the database path for a repository."""
        if repo_path is None:
            repo_path = Path.cwd()
        return repo_path / cls.DEFAULT_DB_DIR / cls.DEFAULT_DB_NAME


class FileProcessingConfig:
    """File processing and indexing configuration."""

    # File size limits (in MB)
    MAX_FILE_SIZE_MB: float = validate_positive_float(
        os.getenv("TURBOPROP_MAX_FILE_SIZE_MB", "1.0"), "TURBOPROP_MAX_FILE_SIZE_MB", 1.0
    )

    # File watching and debouncing
    DEBOUNCE_SECONDS: float = validate_positive_float(
        os.getenv("TURBOPROP_DEBOUNCE_SECONDS", "5.0"), "TURBOPROP_DEBOUNCE_SECONDS", 5.0
    )

    # Preview and snippet settings
    PREVIEW_LENGTH: int = validate_positive_int(
        os.getenv("TURBOPROP_PREVIEW_LENGTH", "200"), "TURBOPROP_PREVIEW_LENGTH", 200
    )
    SNIPPET_LENGTH: int = validate_positive_int(
        os.getenv("TURBOPROP_SNIPPET_LENGTH", "300"), "TURBOPROP_SNIPPET_LENGTH", 300
    )

    # Batch processing settings
    BATCH_SIZE: int = validate_positive_int(
        os.getenv("TURBOPROP_BATCH_SIZE", "100"), "TURBOPROP_BATCH_SIZE", 100
    )

    # File watching cleanup and health check intervals
    CLEANUP_INTERVAL: int = validate_positive_int(
        os.getenv("TURBOPROP_CLEANUP_INTERVAL", "300"), "TURBOPROP_CLEANUP_INTERVAL", 300
    )  # 5 minutes
    HEALTH_CHECK_INTERVAL: int = validate_positive_int(
        os.getenv("TURBOPROP_HEALTH_CHECK_INTERVAL", "3600"),
        "TURBOPROP_HEALTH_CHECK_INTERVAL", 3600
    )  # 1 hour

    # Parallel processing settings
    MAX_WORKERS: int = validate_positive_int(
        os.getenv("TURBOPROP_MAX_WORKERS", "4"), "TURBOPROP_MAX_WORKERS", 4
    )
    MIN_FILES_FOR_PARALLEL: int = validate_positive_int(
        os.getenv("TURBOPROP_MIN_FILES_FOR_PARALLEL", "10"), "TURBOPROP_MIN_FILES_FOR_PARALLEL", 10
    )
    PARALLEL_CHUNK_SIZE: int = validate_positive_int(
        os.getenv("TURBOPROP_PARALLEL_CHUNK_SIZE", "50"), "TURBOPROP_PARALLEL_CHUNK_SIZE", 50
    )

    # Language detection settings
    ENABLE_LANGUAGE_DETECTION: bool = validate_boolean(
        os.getenv("TURBOPROP_ENABLE_LANGUAGE_DETECTION", "true"),
        "TURBOPROP_ENABLE_LANGUAGE_DETECTION"
    )

    # File extensions that we consider to be code files worth indexing
    CODE_EXTENSIONS = {
        ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".c", ".cpp", ".h", ".cs",
        ".go", ".rs", ".swift", ".kt", ".m", ".rb", ".php", ".sh", ".html", ".css",
        ".json", ".yaml", ".yml", ".xml"
    }

    # Language detection fallback map for file extensions
    EXTENSION_TO_LANGUAGE_MAP = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".jsx": "javascript",
        ".tsx": "typescript",
        ".java": "java",
        ".cpp": "cpp",
        ".cc": "cpp",
        ".cxx": "cpp",
        ".c": "c",
        ".h": "c",
        ".hpp": "cpp",
        ".rs": "rust",
        ".go": "go",
        ".php": "php",
        ".rb": "ruby",
        ".swift": "swift",
        ".kt": "kotlin",
        ".scala": "scala",
        ".sh": "bash",
        ".bash": "bash",
        ".zsh": "zsh",
        ".fish": "fish",
        ".ps1": "powershell",
        ".sql": "sql",
        ".html": "html",
        ".htm": "html",
        ".css": "css",
        ".scss": "scss",
        ".sass": "sass",
        ".less": "less",
        ".xml": "xml",
        ".json": "json",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".toml": "toml",
        ".ini": "ini",
        ".cfg": "ini",
        ".conf": "ini",
        ".md": "markdown",
        ".markdown": "markdown",
        ".rst": "restructuredtext",
        ".tex": "latex",
        ".r": "r",
        ".R": "r",
        ".m": "matlab",
        ".pl": "perl",
        ".lua": "lua",
        ".vim": "vimscript",
        ".dockerfile": "dockerfile",
        ".makefile": "makefile",
    }


class SearchConfig:
    """Search-related configuration constants."""

    # Search result limits
    DEFAULT_MAX_RESULTS: int = validate_positive_int(
        os.getenv("TURBOPROP_DEFAULT_MAX_RESULTS", "5"), "TURBOPROP_DEFAULT_MAX_RESULTS", 5
    )
    MAX_RESULTS_LIMIT: int = validate_positive_int(
        os.getenv("TURBOPROP_MAX_RESULTS_LIMIT", "20"), "TURBOPROP_MAX_RESULTS_LIMIT", 20
    )

    # Similarity and distance thresholds
    MIN_SIMILARITY_THRESHOLD: float = validate_range_float(
        os.getenv("TURBOPROP_MIN_SIMILARITY", "0.1"), "TURBOPROP_MIN_SIMILARITY", 0.0, 1.0
    )

    # Confidence level thresholds for search results
    HIGH_CONFIDENCE_THRESHOLD: float = validate_range_float(
        os.getenv("TURBOPROP_HIGH_CONFIDENCE_THRESHOLD", "0.8"), "TURBOPROP_HIGH_CONFIDENCE_THRESHOLD", 0.0, 1.0
    )
    MEDIUM_CONFIDENCE_THRESHOLD: float = validate_range_float(
        os.getenv("TURBOPROP_MEDIUM_CONFIDENCE_THRESHOLD", "0.6"), "TURBOPROP_MEDIUM_CONFIDENCE_THRESHOLD", 0.0, 1.0
    )

    # Display formatting
    SEPARATOR_LENGTH: int = validate_positive_int(
        os.getenv("TURBOPROP_SEPARATOR_LENGTH", "50"), "TURBOPROP_SEPARATOR_LENGTH", 50
    )

    # Snippet length constants
    SNIPPET_DISPLAY_MAX_LENGTH: int = validate_positive_int(
        os.getenv("TURBOPROP_SNIPPET_DISPLAY_LENGTH", "100"), "TURBOPROP_SNIPPET_DISPLAY_LENGTH", 100
    )
    SNIPPET_CONTENT_MAX_LENGTH: int = validate_positive_int(
        os.getenv("TURBOPROP_SNIPPET_CONTENT_LENGTH", "200"), "TURBOPROP_SNIPPET_CONTENT_LENGTH", 200
    )

    # Snippet extraction relevance thresholds
    HIGH_RELEVANCE_THRESHOLD: float = validate_range_float(
        os.getenv("TURBOPROP_HIGH_RELEVANCE_THRESHOLD", "0.8"), "TURBOPROP_HIGH_RELEVANCE_THRESHOLD", 0.0, 1.0
    )
    CANDIDATE_FILTER_THRESHOLD: float = validate_range_float(
        os.getenv("TURBOPROP_CANDIDATE_FILTER_THRESHOLD", "0.7"), "TURBOPROP_CANDIDATE_FILTER_THRESHOLD", 0.0, 1.0
    )

    # Snippet extraction scoring bonuses
    EXACT_MATCH_BONUS: float = validate_range_float(
        os.getenv("TURBOPROP_EXACT_MATCH_BONUS", "0.8"), "TURBOPROP_EXACT_MATCH_BONUS", 0.0, 2.0
    )
    TITLE_MATCH_SCORE: float = validate_range_float(
        os.getenv("TURBOPROP_TITLE_MATCH_SCORE", "0.7"), "TURBOPROP_TITLE_MATCH_SCORE", 0.0, 2.0
    )


class EmbeddingConfig:
    """Embedding model and processing configuration."""

    # Model settings (from code_index.py)
    EMBED_MODEL: str = os.getenv("TURBOPROP_EMBED_MODEL", "all-MiniLM-L6-v2")
    DIMENSIONS: int = validate_positive_int(
        os.getenv("TURBOPROP_EMBEDDING_DIMENSIONS", "384"), "TURBOPROP_EMBEDDING_DIMENSIONS", 384
    )

    # Processing settings
    DEVICE: str = validate_device(
        os.getenv("TURBOPROP_DEVICE", "cpu"), "TURBOPROP_DEVICE"
    )  # or "cuda", "mps"
    BATCH_SIZE: int = validate_positive_int(
        os.getenv("TURBOPROP_EMBEDDING_BATCH_SIZE", "32"), "TURBOPROP_EMBEDDING_BATCH_SIZE", 32
    )

    # Retry settings
    MAX_RETRIES: int = validate_positive_int(
        os.getenv("TURBOPROP_EMBEDDING_MAX_RETRIES", "3"), "TURBOPROP_EMBEDDING_MAX_RETRIES", 3
    )
    RETRY_BASE_DELAY: float = validate_positive_float(
        os.getenv("TURBOPROP_EMBEDDING_RETRY_BASE_DELAY", "1.0"),
        "TURBOPROP_EMBEDDING_RETRY_BASE_DELAY", 1.0
    )


class ServerConfig:
    """Server and API configuration."""

    # FastAPI server settings
    HOST: str = os.getenv("TURBOPROP_HOST", "0.0.0.0")
    PORT: int = validate_positive_int(os.getenv("TURBOPROP_PORT", "8000"), "TURBOPROP_PORT", 8000)

    # Watcher settings for server mode
    WATCH_DIRECTORY: str = os.getenv("TURBOPROP_WATCH_DIR", ".")
    WATCH_MAX_FILE_SIZE_MB: float = validate_positive_float(
        os.getenv("TURBOPROP_WATCH_MAX_FILE_SIZE_MB", "1.0"),
        "TURBOPROP_WATCH_MAX_FILE_SIZE_MB", 1.0
    )
    WATCH_DEBOUNCE_SECONDS: float = validate_positive_float(
        os.getenv("TURBOPROP_WATCH_DEBOUNCE_SECONDS", "5.0"),
        "TURBOPROP_WATCH_DEBOUNCE_SECONDS", 5.0
    )

    # API rate limiting and timeouts
    REQUEST_TIMEOUT: float = validate_positive_float(
        os.getenv("TURBOPROP_REQUEST_TIMEOUT", "30.0"), "TURBOPROP_REQUEST_TIMEOUT", 30.0
    )
    MAX_CONCURRENT_REQUESTS: int = validate_positive_int(
        os.getenv("TURBOPROP_MAX_CONCURRENT_REQUESTS", "10"),
        "TURBOPROP_MAX_CONCURRENT_REQUESTS", 10
    )


class LoggingConfig:
    """Logging configuration."""

    # Log levels
    LOG_LEVEL: str = validate_log_level(
        os.getenv("TURBOPROP_LOG_LEVEL", "INFO"), "TURBOPROP_LOG_LEVEL"
    )

    # Log formatting
    LOG_FORMAT: str = os.getenv(
        "TURBOPROP_LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # File logging
    LOG_FILE: Optional[str] = os.getenv("TURBOPROP_LOG_FILE")
    LOG_MAX_SIZE: int = validate_positive_int(
        os.getenv("TURBOPROP_LOG_MAX_SIZE", "10485760"), "TURBOPROP_LOG_MAX_SIZE", 10485760
    )  # 10MB
    LOG_BACKUP_COUNT: int = validate_positive_int(
        os.getenv("TURBOPROP_LOG_BACKUP_COUNT", "5"), "TURBOPROP_LOG_BACKUP_COUNT", 5
    )


class MCPConfig:
    """MCP (Model Context Protocol) server configuration."""

    # Default settings for MCP operations
    DEFAULT_MAX_FILE_SIZE_MB: float = validate_positive_float(
        os.getenv("TURBOPROP_MCP_MAX_FILE_SIZE_MB", "1.0"), "TURBOPROP_MCP_MAX_FILE_SIZE_MB", 1.0
    )
    DEFAULT_DEBOUNCE_SECONDS: float = validate_positive_float(
        os.getenv("TURBOPROP_MCP_DEBOUNCE_SECONDS", "5.0"), "TURBOPROP_MCP_DEBOUNCE_SECONDS", 5.0
    )

    # MCP server limits
    MAX_FILES_LIST: int = validate_positive_int(
        os.getenv("TURBOPROP_MCP_MAX_FILES_LIST", "100"), "TURBOPROP_MCP_MAX_FILES_LIST", 100
    )
    MAX_SEARCH_RESULTS: int = validate_positive_int(
        os.getenv("TURBOPROP_MCP_MAX_SEARCH_RESULTS", "20"), "TURBOPROP_MCP_MAX_SEARCH_RESULTS", 20
    )


# Convenience class for accessing all configurations
class Config:
    """Main configuration class that provides access to all configuration sections."""

    database = DatabaseConfig()
    file_processing = FileProcessingConfig()
    search = SearchConfig()
    embedding = EmbeddingConfig()
    server = ServerConfig()
    logging = LoggingConfig()
    mcp = MCPConfig()

    @classmethod
    def validate(cls) -> bool:
        """Validate all configuration values."""
        try:
            # Validation happens during class initialization
            # All values are validated when the environment variables are read
            # If we get here, validation passed
            return True
        except ConfigValidationError:
            # Re-raise the validation error
            raise

    @classmethod
    def get_validation_status(cls) -> str:
        """Get validation status for all configuration values."""
        try:
            cls.validate()
            return "✅ All configuration values are valid"
        except ConfigValidationError as e:
            return f"❌ Configuration validation failed: {e}"

    @classmethod
    def get_summary(cls) -> str:
        """Get a summary of current configuration settings."""
        validation_status = cls.get_validation_status()
        return f"""
Turboprop Configuration Summary:
==============================

Validation Status: {validation_status}

Database:
  Memory Limit: {cls.database.MEMORY_LIMIT}
  Threads: {cls.database.THREADS}
  Max Retries: {cls.database.MAX_RETRIES}
  Retry Delay: {cls.database.RETRY_DELAY}s
  Connection Timeout: {cls.database.CONNECTION_TIMEOUT}s
  Statement Timeout: {cls.database.STATEMENT_TIMEOUT}s
  Lock Timeout: {cls.database.LOCK_TIMEOUT}s
  Auto Vacuum: {cls.database.AUTO_VACUUM}

File Processing:
  Max File Size: {cls.file_processing.MAX_FILE_SIZE_MB}MB
  Debounce: {cls.file_processing.DEBOUNCE_SECONDS}s
  Preview Length: {cls.file_processing.PREVIEW_LENGTH} chars
  Language Detection: {cls.file_processing.ENABLE_LANGUAGE_DETECTION}

Search:
  Default Max Results: {cls.search.DEFAULT_MAX_RESULTS}
  Max Results Limit: {cls.search.MAX_RESULTS_LIMIT}
  Min Similarity: {cls.search.MIN_SIMILARITY_THRESHOLD}

Embedding:
  Model: {cls.embedding.EMBED_MODEL}
  Dimensions: {cls.embedding.DIMENSIONS}
  Device: {cls.embedding.DEVICE}
  Batch Size: {cls.embedding.BATCH_SIZE}
  Max Retries: {cls.embedding.MAX_RETRIES}
  Retry Base Delay: {cls.embedding.RETRY_BASE_DELAY}s

Server:
  Host: {cls.server.HOST}
  Port: {cls.server.PORT}
  Watch Directory: {cls.server.WATCH_DIRECTORY}

Logging:
  Level: {cls.logging.LOG_LEVEL}
  File: {cls.logging.LOG_FILE or 'Console only'}
        """


# Create a global config instance for easy access
config = Config()
