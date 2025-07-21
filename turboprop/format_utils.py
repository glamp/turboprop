#!/usr/bin/env python3
"""
format_utils.py: Utility functions for format conversion between legacy and enhanced formats.

This module provides adapter functions to convert between tuple-based legacy format
and the new structured CodeSearchResult format, enabling backward compatibility.
"""

from typing import List, Tuple

from .search_result_types import CodeSearchResult


def convert_results_to_legacy_format(enhanced_results: List[CodeSearchResult]) -> List[Tuple[str, str, float]]:
    """
    Convert CodeSearchResult objects to legacy tuple format.

    Args:
        enhanced_results: List of CodeSearchResult objects

    Returns:
        List of tuples in format (file_path, snippet_text, distance_score)
    """
    return [result.to_tuple() for result in enhanced_results]


def convert_legacy_to_enhanced_format(legacy_results: List[Tuple[str, str, float]]) -> List[CodeSearchResult]:
    """
    Convert legacy tuple results to CodeSearchResult objects.

    Args:
        legacy_results: List of tuples in format (file_path, snippet_text, distance_score)

    Returns:
        List of CodeSearchResult objects with rich metadata
    """
    return [CodeSearchResult.from_tuple(result) for result in legacy_results]
