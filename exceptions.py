"""
Custom exceptions for the Turboprop semantic code search system.

This module defines specific exception classes for different failure scenarios
to improve error handling and debugging throughout the codebase.
"""


class TurbopropError(Exception):
    """Base exception class for all Turboprop-related errors."""

    pass


class DatabaseError(TurbopropError):
    """Raised when database operations fail."""

    pass


class DatabaseConnectionError(DatabaseError):
    """Raised when database connection cannot be established."""

    pass


class DatabaseLockError(DatabaseError):
    """Raised when database is locked and retry attempts are exhausted."""

    pass


class DatabaseSchemaError(DatabaseError):
    """Raised when database schema operations fail."""

    pass


class EmbeddingError(TurbopropError):
    """Raised when embedding generation fails."""

    pass


class EmbeddingModelError(EmbeddingError):
    """Raised when embedding model initialization fails."""

    pass


class FileProcessingError(TurbopropError):
    """Raised when file processing operations fail."""

    pass


class FileNotFoundError(FileProcessingError):
    """Raised when a required file cannot be found."""

    pass


class FileSizeError(FileProcessingError):
    """Raised when a file exceeds the maximum size limit."""

    pass


class IndexingError(TurbopropError):
    """Raised when indexing operations fail."""

    pass


class SearchError(TurbopropError):
    """Raised when search operations fail."""

    pass


class ConfigurationError(TurbopropError):
    """Raised when configuration is invalid or missing."""

    pass


class MCPError(TurbopropError):
    """Raised when MCP (Model Context Protocol) operations fail."""

    pass


class WatcherError(TurbopropError):
    """Raised when file watcher operations fail."""

    pass
