#!/usr/bin/env python3
"""
Standardized Error Handling - Consistent Error Response Patterns

This module provides standardized error handling utilities for consistent
error responses across all automatic tool selection components.
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from .logging_config import get_logger

logger = get_logger(__name__)


class ErrorSeverity(Enum):
    """Standard error severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class StandardError:
    """Standardized error response object."""

    error_code: str
    message: str
    severity: ErrorSeverity
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    recovery_suggestions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "error_code": self.error_code,
            "message": self.message,
            "severity": self.severity.value,
            "context": self.context,
            "timestamp": self.timestamp,
            "recovery_suggestions": self.recovery_suggestions,
        }


@dataclass
class ErrorResult:
    """Standard result container that can hold either success data or error."""

    success: bool
    data: Any = None
    error: Optional[StandardError] = None

    @classmethod
    def success_result(cls, data: Any) -> "ErrorResult":
        """Create a successful result."""
        return cls(success=True, data=data)

    @classmethod
    def error_result(cls, error: StandardError) -> "ErrorResult":
        """Create an error result."""
        return cls(success=False, error=error)


class StandardErrorHandler:
    """Standard error handler with consistent patterns."""

    def __init__(self, module_name: str):
        self.module_name = module_name
        self.logger = get_logger(module_name)

    def handle_with_default(
        self,
        operation: str,
        exception: Exception,
        default_value: Any,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Handle error by returning a default value."""
        # Log the error
        self.logger.error("Error in %s.%s: %s", self.module_name, operation, str(exception))

        # Return default value for backward compatibility
        return default_value

    def handle_with_result(
        self,
        operation: str,
        exception: Exception,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: Optional[Dict[str, Any]] = None,
        recovery_suggestions: Optional[List[str]] = None,
    ) -> ErrorResult:
        """Handle error by returning a StandardError result."""
        error_code = f"{self.module_name.upper()}_OPERATION_FAILED"

        # Log the error
        self.logger.error("Error in %s.%s: %s", self.module_name, operation, str(exception))

        # Create standardized error
        error = StandardError(
            error_code=error_code,
            message=f"Operation '{operation}' failed: {str(exception)}",
            severity=severity,
            context=context or {},
            recovery_suggestions=recovery_suggestions
            or [
                "Check system logs for more details",
                "Verify input parameters are valid",
                "Retry the operation after a brief delay",
            ],
        )

        return ErrorResult.error_result(error)


def create_error_handler(module_name: str) -> StandardErrorHandler:
    """Factory function to create standardized error handlers."""
    return StandardErrorHandler(module_name)
