#!/usr/bin/env python3
"""
error_handling_utils.py: Common error handling utilities to reduce code duplication.

This module provides decorators and context managers for consistent error handling
across the turboprop codebase, particularly for database and search operations.
"""

import logging
from functools import wraps
from typing import Any, Callable, TypeVar

from .exceptions import DatabaseError, EmbeddingError, SearchError

logger = logging.getLogger(__name__)

T = TypeVar("T")


def handle_search_errors(operation_name: str, return_empty_list: bool = True):
    """
    Decorator to handle common search operation errors.

    Args:
        operation_name: Name of the operation for logging
        return_empty_list: If True, returns empty list on error; otherwise returns None
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except (DatabaseError, EmbeddingError) as error:
                logger.error("Error in %s: %s", operation_name, error)
                return [] if return_empty_list else None
            except Exception as error:
                logger.error("Unexpected error in %s: %s", operation_name, error)
                raise SearchError(f"{operation_name} failed: {error}") from error

        return wrapper

    return decorator


def handle_database_errors(operation_name: str, return_value: Any = None):
    """
    Decorator to handle database operation errors.

    Args:
        operation_name: Name of the operation for logging
        return_value: Value to return on error (default: None)
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except DatabaseError as error:
                logger.error("Database error in %s: %s", operation_name, error)
                return return_value
            except Exception as error:
                logger.error("Unexpected error in %s: %s", operation_name, error)
                raise SearchError(f"{operation_name} failed: {error}") from error

        return wrapper

    return decorator


def handle_statistics_errors(operation_name: str):
    """
    Decorator specifically for statistics operations that return error dict on failure.

    Args:
        operation_name: Name of the operation for logging
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except DatabaseError as error:
                logger.error("Error getting %s: %s", operation_name, error)
                return {"error": str(error)}
            except Exception as error:
                logger.error("Unexpected error getting %s: %s", operation_name, error)
                raise SearchError(f"{operation_name} retrieval failed: {error}") from error

        return wrapper

    return decorator


class ErrorHandler:
    """Context manager and utility class for consistent error handling."""

    def __init__(self, operation_name: str, return_value: Any = None, raise_on_unexpected: bool = True):
        """
        Initialize error handler.

        Args:
            operation_name: Name of the operation for logging
            return_value: Value to return on expected errors
            raise_on_unexpected: Whether to raise SearchError on unexpected exceptions
        """
        self.operation_name = operation_name
        self.return_value = return_value
        self.raise_on_unexpected = raise_on_unexpected

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            return False  # No exception occurred

        if issubclass(exc_type, (DatabaseError, EmbeddingError)):
            logger.error("Error in %s: %s", self.operation_name, exc_val)
            return True  # Suppress the exception
        elif self.raise_on_unexpected:
            logger.error("Unexpected error in %s: %s", self.operation_name, exc_val)
            # Let the exception propagate, but it will be a SearchError
            raise SearchError(f"{self.operation_name} failed: {exc_val}") from exc_val
        else:
            logger.error("Unexpected error in %s: %s", self.operation_name, exc_val)
            return True  # Suppress the exception

        return False  # Don't suppress other exceptions


def log_and_handle_error(operation_name: str, error: Exception, return_value: Any = None) -> Any:
    """
    Utility function to log and handle errors consistently.

    Args:
        operation_name: Name of the operation for logging
        error: The exception that occurred
        return_value: Value to return for expected errors

    Returns:
        The return_value for expected errors, raises SearchError for unexpected ones
    """
    if isinstance(error, (DatabaseError, EmbeddingError)):
        logger.error("Error in %s: %s", operation_name, error)
        return return_value
    else:
        logger.error("Unexpected error in %s: %s", operation_name, error)
        raise SearchError(f"{operation_name} failed: {error}") from error
