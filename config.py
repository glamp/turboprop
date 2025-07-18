"""
Configuration module for Turboprop semantic code search system.

This module centralizes all configuration constants and default values
to make the system more maintainable and configurable.
"""

import os
from pathlib import Path
from typing import Optional


class DatabaseConfig:
    """Database-related configuration constants."""

    # DuckDB performance settings
    MEMORY_LIMIT: str = os.getenv("TURBOPROP_DB_MEMORY_LIMIT", "1GB")
    THREADS: int = int(os.getenv("TURBOPROP_DB_THREADS", "4"))

    # Database connection and retry settings
    MAX_RETRIES: int = int(os.getenv("TURBOPROP_DB_MAX_RETRIES", "3"))
    RETRY_DELAY: float = float(os.getenv("TURBOPROP_DB_RETRY_DELAY", "0.1"))

    # Database file configuration
    DEFAULT_DB_NAME: str = "code_index.duckdb"
    DEFAULT_DB_DIR: str = ".turboprop"

    @classmethod
    def get_db_path(cls, repo_path: Optional[Path] = None) -> Path:
        """Get the database path for a repository."""
        if repo_path is None:
            repo_path = Path.cwd()
        return repo_path / cls.DEFAULT_DB_DIR / cls.DEFAULT_DB_NAME


class FileProcessingConfig:
    """File processing and indexing configuration."""

    # File size limits (in MB)
    MAX_FILE_SIZE_MB: float = float(os.getenv("TURBOPROP_MAX_FILE_SIZE_MB", "1.0"))

    # File watching and debouncing
    DEBOUNCE_SECONDS: float = float(os.getenv("TURBOPROP_DEBOUNCE_SECONDS", "5.0"))

    # Preview and snippet settings
    PREVIEW_LENGTH: int = int(os.getenv("TURBOPROP_PREVIEW_LENGTH", "200"))
    SNIPPET_LENGTH: int = int(os.getenv("TURBOPROP_SNIPPET_LENGTH", "300"))

    # Batch processing settings
    BATCH_SIZE: int = int(os.getenv("TURBOPROP_BATCH_SIZE", "100"))


class SearchConfig:
    """Search-related configuration constants."""

    # Search result limits
    DEFAULT_MAX_RESULTS: int = int(os.getenv("TURBOPROP_DEFAULT_MAX_RESULTS", "5"))
    MAX_RESULTS_LIMIT: int = int(os.getenv("TURBOPROP_MAX_RESULTS_LIMIT", "20"))

    # Similarity and distance thresholds
    MIN_SIMILARITY_THRESHOLD: float = float(os.getenv("TURBOPROP_MIN_SIMILARITY", "0.1"))

    # Display formatting
    SEPARATOR_LENGTH: int = int(os.getenv("TURBOPROP_SEPARATOR_LENGTH", "50"))


class EmbeddingConfig:
    """Embedding model and processing configuration."""

    # Model settings (from code_index.py)
    EMBED_MODEL: str = os.getenv("TURBOPROP_EMBED_MODEL", "all-MiniLM-L6-v2")
    DIMENSIONS: int = int(os.getenv("TURBOPROP_EMBEDDING_DIMENSIONS", "384"))

    # Processing settings
    DEVICE: str = os.getenv("TURBOPROP_DEVICE", "cpu")  # or "cuda", "mps"
    BATCH_SIZE: int = int(os.getenv("TURBOPROP_EMBEDDING_BATCH_SIZE", "32"))


class ServerConfig:
    """Server and API configuration."""

    # FastAPI server settings
    HOST: str = os.getenv("TURBOPROP_HOST", "0.0.0.0")
    PORT: int = int(os.getenv("TURBOPROP_PORT", "8000"))

    # Watcher settings for server mode
    WATCH_DIRECTORY: str = os.getenv("TURBOPROP_WATCH_DIR", ".")
    WATCH_MAX_FILE_SIZE_MB: float = float(os.getenv("TURBOPROP_WATCH_MAX_FILE_SIZE_MB", "1.0"))
    WATCH_DEBOUNCE_SECONDS: float = float(os.getenv("TURBOPROP_WATCH_DEBOUNCE_SECONDS", "5.0"))

    # API rate limiting and timeouts
    REQUEST_TIMEOUT: float = float(os.getenv("TURBOPROP_REQUEST_TIMEOUT", "30.0"))
    MAX_CONCURRENT_REQUESTS: int = int(os.getenv("TURBOPROP_MAX_CONCURRENT_REQUESTS", "10"))


class LoggingConfig:
    """Logging configuration."""

    # Log levels
    LOG_LEVEL: str = os.getenv("TURBOPROP_LOG_LEVEL", "INFO")

    # Log formatting
    LOG_FORMAT: str = os.getenv("TURBOPROP_LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # File logging
    LOG_FILE: Optional[str] = os.getenv("TURBOPROP_LOG_FILE")
    LOG_MAX_SIZE: int = int(os.getenv("TURBOPROP_LOG_MAX_SIZE", "10485760"))  # 10MB
    LOG_BACKUP_COUNT: int = int(os.getenv("TURBOPROP_LOG_BACKUP_COUNT", "5"))


class MCPConfig:
    """MCP (Model Context Protocol) server configuration."""

    # Default settings for MCP operations
    DEFAULT_MAX_FILE_SIZE_MB: float = float(os.getenv("TURBOPROP_MCP_MAX_FILE_SIZE_MB", "1.0"))
    DEFAULT_DEBOUNCE_SECONDS: float = float(os.getenv("TURBOPROP_MCP_DEBOUNCE_SECONDS", "5.0"))

    # MCP server limits
    MAX_FILES_LIST: int = int(os.getenv("TURBOPROP_MCP_MAX_FILES_LIST", "100"))
    MAX_SEARCH_RESULTS: int = int(os.getenv("TURBOPROP_MCP_MAX_SEARCH_RESULTS", "20"))


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
    def get_summary(cls) -> str:
        """Get a summary of current configuration settings."""
        return f"""
Turboprop Configuration Summary:
==============================

Database:
  Memory Limit: {cls.database.MEMORY_LIMIT}
  Threads: {cls.database.THREADS}
  Max Retries: {cls.database.MAX_RETRIES}
  Retry Delay: {cls.database.RETRY_DELAY}s

File Processing:
  Max File Size: {cls.file_processing.MAX_FILE_SIZE_MB}MB
  Debounce: {cls.file_processing.DEBOUNCE_SECONDS}s
  Preview Length: {cls.file_processing.PREVIEW_LENGTH} chars

Search:
  Default Max Results: {cls.search.DEFAULT_MAX_RESULTS}
  Max Results Limit: {cls.search.MAX_RESULTS_LIMIT}
  Min Similarity: {cls.search.MIN_SIMILARITY_THRESHOLD}

Embedding:
  Model: {cls.embedding.EMBED_MODEL}
  Dimensions: {cls.embedding.DIMENSIONS}
  Device: {cls.embedding.DEVICE}

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
