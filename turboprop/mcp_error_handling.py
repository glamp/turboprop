#!/usr/bin/env python3
"""
mcp_error_handling.py: Standardized error handling for MCP tools.

This module provides consistent error response formats, exception handling utilities,
and logging across all MCP tools to ensure uniform behavior and debugging experience.
"""

import json
import logging
import time
import traceback
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from . import response_config


class ErrorSeverity(Enum):
    """Error severity levels for categorization."""

    CRITICAL = "critical"  # Service unavailable, data corruption
    HIGH = "high"  # Tool unavailable, invalid configuration
    MEDIUM = "medium"  # Invalid parameters, missing data
    LOW = "low"  # Warnings, optional features unavailable


class ErrorCategory(Enum):
    """Error categories for better organization."""

    VALIDATION = "validation"  # Input parameter validation
    AUTHENTICATION = "authentication"  # Authentication/authorization
    RESOURCE = "resource"  # Database, file system, network
    BUSINESS_LOGIC = "business_logic"  # Domain-specific logic errors
    SYSTEM = "system"  # Infrastructure, configuration
    EXTERNAL = "external"  # Third-party service failures


@dataclass
class MCPError:
    """Standardized MCP error structure."""

    tool_name: str
    message: str
    category: ErrorCategory
    severity: ErrorSeverity
    context: str = ""
    error_code: str = ""
    timestamp: str = ""
    version: str = response_config.RESPONSE_VERSION

    # Additional debugging information
    stack_trace: Optional[str] = None
    request_data: Optional[Dict[str, Any]] = None
    suggestions: List[str] = None

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
        if self.suggestions is None:
            self.suggestions = []

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "success": False,
            "error": {
                "tool": self.tool_name,
                "message": self.message,
                "category": self.category.value,
                "severity": self.severity.value,
                "context": self.context,
                "error_code": self.error_code,
                "timestamp": self.timestamp,
                "version": self.version,
                "stack_trace": self.stack_trace,
                "request_data": self.request_data,
                "suggestions": self.suggestions,
            },
        }

    def to_json(self) -> str:
        """Convert to JSON string for MCP responses."""
        return json.dumps(self.to_dict(), indent=response_config.JSON_INDENT)


class MCPErrorHandler:
    """Centralized error handling for MCP tools."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def create_validation_error(
        self, tool_name: str, message: str, context: str = "", suggestions: Optional[List[str]] = None
    ) -> MCPError:
        """Create a parameter validation error."""
        return MCPError(
            tool_name=tool_name,
            message=message,
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.MEDIUM,
            context=context,
            error_code="VALIDATION_FAILED",
            suggestions=suggestions or [],
        )

    def create_resource_error(
        self, tool_name: str, message: str, context: str = "", severity: ErrorSeverity = ErrorSeverity.HIGH
    ) -> MCPError:
        """Create a resource access error."""
        return MCPError(
            tool_name=tool_name,
            message=message,
            category=ErrorCategory.RESOURCE,
            severity=severity,
            context=context,
            error_code="RESOURCE_ERROR",
        )

    def create_system_error(
        self, tool_name: str, message: str, context: str = "", include_trace: bool = False
    ) -> MCPError:
        """Create a system error with optional stack trace."""
        error = MCPError(
            tool_name=tool_name,
            message=message,
            category=ErrorCategory.SYSTEM,
            severity=ErrorSeverity.CRITICAL,
            context=context,
            error_code="SYSTEM_ERROR",
        )

        if include_trace:
            error.stack_trace = traceback.format_exc()

        return error

    def create_business_logic_error(
        self, tool_name: str, message: str, context: str = "", suggestions: Optional[List[str]] = None
    ) -> MCPError:
        """Create a business logic error."""
        return MCPError(
            tool_name=tool_name,
            message=message,
            category=ErrorCategory.BUSINESS_LOGIC,
            severity=ErrorSeverity.MEDIUM,
            context=context,
            error_code="BUSINESS_LOGIC_ERROR",
            suggestions=suggestions or [],
        )

    def handle_exception(
        self, tool_name: str, exception: Exception, context: str = "", request_data: Optional[Dict[str, Any]] = None
    ) -> MCPError:
        """Handle an exception and convert it to standardized MCP error."""
        error_message = str(exception)
        error_type = type(exception).__name__

        # Log the exception
        self.logger.error(
            f"Tool '{tool_name}' encountered {error_type}: {error_message}",
            extra={"context": context, "tool": tool_name},
            exc_info=True,
        )

        # Categorize exception type
        category, severity = self._categorize_exception(exception)

        error = MCPError(
            tool_name=tool_name,
            message=f"{error_type}: {error_message}",
            category=category,
            severity=severity,
            context=context,
            error_code=f"{category.value.upper()}_{error_type.upper()}",
            stack_trace=traceback.format_exc(),
            request_data=request_data,
        )

        # Add suggestions based on exception type
        error.suggestions = self._get_exception_suggestions(exception)

        return error

    def _categorize_exception(self, exception: Exception) -> Tuple[ErrorCategory, ErrorSeverity]:
        """Categorize exception type into category and severity."""
        if isinstance(exception, (ValueError, TypeError)):
            return ErrorCategory.VALIDATION, ErrorSeverity.MEDIUM
        elif isinstance(exception, (FileNotFoundError, PermissionError)):
            return ErrorCategory.RESOURCE, ErrorSeverity.HIGH
        elif isinstance(exception, (ConnectionError, TimeoutError)):
            return ErrorCategory.EXTERNAL, ErrorSeverity.HIGH
        elif isinstance(exception, (RuntimeError, SystemError)):
            return ErrorCategory.SYSTEM, ErrorSeverity.CRITICAL
        else:
            return ErrorCategory.SYSTEM, ErrorSeverity.HIGH

    def _get_exception_suggestions(self, exception: Exception) -> List[str]:
        """Get helpful suggestions based on exception type."""
        suggestions = []

        if isinstance(exception, ValueError):
            suggestions.append("Check that all required parameters are provided with valid values")
            suggestions.append("Verify parameter formats match expected patterns")
        elif isinstance(exception, FileNotFoundError):
            suggestions.append("Ensure the specified file or directory exists")
            suggestions.append("Check file permissions and access rights")
        elif isinstance(exception, ConnectionError):
            suggestions.append("Verify network connectivity")
            suggestions.append("Check if the service is running and accessible")
        elif isinstance(exception, TimeoutError):
            suggestions.append("Try reducing the request size or complexity")
            suggestions.append("Check system load and available resources")

        return suggestions


# Global error handler instance
error_handler = MCPErrorHandler()


# Convenience functions for common error patterns
def create_validation_error(
    tool_name: str, message: str, context: str = "", suggestions: Optional[List[str]] = None
) -> str:
    """Create and return JSON validation error response."""
    error = error_handler.create_validation_error(tool_name, message, context, suggestions)
    return error.to_json()


def create_resource_error(
    tool_name: str, message: str, context: str = "", severity: ErrorSeverity = ErrorSeverity.HIGH
) -> str:
    """Create and return JSON resource error response."""
    error = error_handler.create_resource_error(tool_name, message, context, severity)
    return error.to_json()


def handle_tool_exception(
    tool_name: str, exception: Exception, context: str = "", request_data: Optional[Dict[str, Any]] = None
) -> str:
    """Handle exception and return JSON error response."""
    error = error_handler.handle_exception(tool_name, exception, context, request_data)
    return error.to_json()


def create_error_response(tool_name: str, error_message: str, context: str = "") -> Dict[str, Any]:
    """
    Legacy function for backward compatibility.

    This maintains compatibility with existing code while providing
    the new standardized error structure.
    """
    error = MCPError(
        tool_name=tool_name,
        message=error_message,
        category=ErrorCategory.BUSINESS_LOGIC,
        severity=ErrorSeverity.MEDIUM,
        context=context,
        error_code="LEGACY_ERROR",
    )
    return error.to_dict()


# Decorator for automatic exception handling
def mcp_tool_exception_handler(func):
    """Decorator to automatically handle exceptions in MCP tool functions."""

    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            tool_name = func.__name__
            context = f"Arguments: {args}, Keyword arguments: {kwargs}"
            return handle_tool_exception(tool_name, e, context)

    return wrapper
