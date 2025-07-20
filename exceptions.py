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


class DatabaseMigrationError(DatabaseError):
    """Raised when database migration operations fail."""

    pass


class DatabaseTimeoutError(DatabaseError):
    """Raised when database operations timeout."""

    pass


class DatabaseConnectionTimeoutError(DatabaseTimeoutError):
    """Raised when database connection establishment times out."""

    pass


class DatabaseStatementTimeoutError(DatabaseTimeoutError):
    """Raised when database statement execution times out."""

    pass


class DatabaseHealthCheckError(DatabaseError):
    """Raised when database connection health checks fail."""

    pass


class DatabaseTransactionError(DatabaseError):
    """Raised when database transaction operations fail."""

    pass


class DatabasePermissionError(DatabaseError):
    """Raised when database permission/access errors occur."""

    pass


class DatabaseCorruptionError(DatabaseError):
    """Raised when database corruption is detected."""

    pass


class DatabaseDiskSpaceError(DatabaseError):
    """Raised when database operations fail due to insufficient disk space."""

    pass


class DatabasePoolExhaustionError(DatabaseError):
    """Raised when database connection pool is exhausted."""

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


class GitError(TurbopropError):
    """Base exception class for Git-related operations."""

    pass


class GitRepositoryError(GitError):
    """Raised when Git repository is not found or invalid."""

    pass


class GitCommandError(GitError):
    """Raised when Git command execution fails."""

    pass


class GitTimeoutError(GitError):
    """Raised when Git operations timeout."""

    pass


class GitRemoteError(GitError):
    """Raised when Git remote operations fail."""

    pass


class GitBranchError(GitError):
    """Raised when Git branch operations fail."""

    pass


class AnalysisError(TurbopropError):
    """Raised when task analysis operations fail."""

    pass
