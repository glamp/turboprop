#!/usr/bin/env python3
"""
ranking_exceptions.py: Custom exceptions for the ranking system.

This module defines specific exception types used throughout the ranking system
to provide better error handling and debugging capabilities.
"""


class RankingError(Exception):
    """Base exception class for all ranking-related errors."""
    pass


class InvalidRankingWeightsError(RankingError):
    """Raised when ranking weights are invalid or don't sum to 1.0."""
    pass


class FileAccessError(RankingError):
    """Raised when a file cannot be accessed for ranking purposes."""
    pass


class GitInfoError(RankingError):
    """Raised when Git information cannot be retrieved or is malformed."""
    pass


class MatchReasonGenerationError(RankingError):
    """Raised when match reasons cannot be generated for a result."""
    pass


class ConfidenceScoringError(RankingError):
    """Raised when confidence scoring fails."""
    pass


class ResultDeduplicationError(RankingError):
    """Raised when result deduplication fails."""
    pass


class InvalidSearchResultError(RankingError):
    """Raised when a search result is malformed or missing required data."""
    pass


class RankingContextError(RankingError):
    """Raised when ranking context is invalid or incomplete."""
    pass
