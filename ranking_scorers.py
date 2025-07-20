#!/usr/bin/env python3
"""
ranking_scorers.py: Individual scoring components for result ranking.

This module contains the individual scorer classes that calculate relevance scores
for different aspects of search results.

Classes:
- FileTypeScorer: Scores results based on file type relevance
- ConstructTypeScorer: Scores results based on construct type matching
- RecencyScorer: Scores results based on file recency using Git information
- FileSizeScorer: Scores results based on file size optimization
"""

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

from ranking_config import FileTypeConstants, QueryTypeConstants, get_ranking_config
from ranking_exceptions import FileAccessError, GitInfoError, InvalidSearchResultError
from search_result_types import CodeSearchResult

logger = logging.getLogger(__name__)


class FileTypeScorer:
    """Scores results based on file type relevance."""

    @classmethod
    def score_file_type(cls, file_path: str, query: str) -> float:
        """
        Score file type relevance (0.0 to 1.0).

        Args:
            file_path: Path to the file
            query: Search query

        Returns:
            Relevance score based on file type
        """
        path_obj = Path(file_path)
        extension = path_obj.suffix.lower()
        filename = path_obj.name.lower()

        # Check if this is a test file
        is_test_file = any(pattern in filename for pattern in FileTypeConstants.TEST_FILE_PATTERNS)

        # Base score by file type
        if extension in FileTypeConstants.SOURCE_CODE_TYPES:
            base_score = 1.0
            # Slightly reduce score for test files unless query is about testing
            if is_test_file and "test" not in query.lower():
                base_score = 0.8
        elif extension in FileTypeConstants.DOC_FILE_TYPES:
            # Higher score for docs if query seems documentation-related
            if any(keyword in query.lower() for keyword in FileTypeConstants.DOC_QUERY_KEYWORDS):
                base_score = 0.9
            else:
                base_score = 0.6
        elif extension in FileTypeConstants.CONFIG_FILE_TYPES:
            # Higher score for config files if query is configuration-related
            if any(keyword in query.lower() for keyword in FileTypeConstants.CONFIG_QUERY_KEYWORDS):
                base_score = 0.8
            else:
                base_score = 0.4
        else:
            base_score = 0.5  # Unknown file type gets neutral score

        return base_score


class ConstructTypeScorer:
    """Scores results based on construct type matching."""

    @classmethod
    def detect_query_type(cls, query: str) -> Optional[str]:
        """
        Detect the likely construct type from query keywords.

        Args:
            query: Search query string

        Returns:
            Detected construct type or None
        """
        query_lower = query.lower()
        for construct_type, keywords in QueryTypeConstants.TYPE_KEYWORDS.items():
            if any(keyword in query_lower for keyword in keywords):
                return construct_type
        return None

    @classmethod
    def score_construct_match(cls, result: CodeSearchResult, query_type: Optional[str]) -> float:
        """
        Score construct type matching (0.0 to 1.0).

        Args:
            result: Search result to score
            query_type: Detected query type

        Returns:
            Score based on construct type matching
        """
        if not query_type:
            return 0.5  # Neutral score when query type is unknown

        # Check if result has construct information
        if hasattr(result, "file_metadata") and result.file_metadata:
            construct_context = result.file_metadata.get("construct_context")
            if construct_context:
                construct_types = construct_context.get("construct_types", [])
                if query_type in construct_types:
                    return 1.0
                # Partial match for related types
                if query_type == "function" and "method" in construct_types:
                    return 0.8
                if query_type == "class" and any(t in construct_types for t in ["interface", "struct"]):
                    return 0.8

        # Fallback: analyze snippet content for construct type indicators
        snippet_text = result.snippet.text.lower()

        if query_type == "function":
            if any(indicator in snippet_text for indicator in QueryTypeConstants.FUNCTION_INDICATORS):
                return 0.9
        elif query_type == "class":
            if any(indicator in snippet_text for indicator in QueryTypeConstants.CLASS_INDICATORS):
                return 0.9
        elif query_type == "variable":
            if any(indicator in snippet_text for indicator in QueryTypeConstants.VARIABLE_INDICATORS):
                return 0.7
        elif query_type == "import":
            if any(indicator in snippet_text for indicator in QueryTypeConstants.IMPORT_INDICATORS):
                return 0.9

        return 0.3  # Low score for mismatched types


class RecencyScorer:
    """Scores results based on file recency using Git information."""

    @classmethod
    def get_file_modification_time(cls, file_path: str, git_info: Optional[Dict] = None) -> Optional[datetime]:
        """
        Get file modification time, preferring Git info over filesystem.

        Args:
            file_path: Path to the file
            git_info: Optional Git information

        Returns:
            Modification datetime or None

        Raises:
            GitInfoError: If git_info is malformed
            FileAccessError: If file cannot be accessed
        """
        try:
            # Validate git_info structure if provided
            if git_info is not None:
                if not isinstance(git_info, dict):
                    raise GitInfoError(f"Git info must be a dictionary, got {type(git_info)}")

                if "file_modifications" in git_info:
                    file_mods = git_info["file_modifications"]
                    if not isinstance(file_mods, dict):
                        raise GitInfoError("file_modifications must be a dictionary")

                    git_time = file_mods.get(file_path)
                    if git_time:
                        try:
                            return datetime.fromisoformat(git_time)
                        except ValueError as e:
                            logger.warning(f"Invalid git timestamp format for {file_path}: {git_time}, error: {e}")
                            # Continue to filesystem fallback

            # Fallback to filesystem modification time
            if not file_path or not isinstance(file_path, str):
                raise FileAccessError(f"Invalid file path: {file_path}")

            try:
                stat = os.stat(file_path)
                return datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
            except OSError as e:
                logger.debug(f"Could not access file {file_path}: {e}")
                return None

        except (GitInfoError, FileAccessError):
            raise  # Re-raise specific errors
        except Exception as e:
            logger.warning(f"Unexpected error getting modification time for {file_path}: {e}")
            return None

    @classmethod
    def score_recency(cls, file_path: str, git_info: Optional[Dict] = None) -> float:
        """
        Score file recency (0.0 to 1.0).

        Args:
            file_path: Path to the file
            git_info: Optional Git information

        Returns:
            Recency score (higher for more recently modified files)
        """
        mod_time = cls.get_file_modification_time(file_path, git_info)
        if not mod_time:
            return 0.5  # Neutral score if we can't determine modification time

        now = datetime.now(tz=timezone.utc)
        age_days = (now - mod_time).total_seconds() / (24 * 3600)

        # Score based on age with exponential decay using configurable thresholds
        config = get_ranking_config()
        recency_thresholds = config.recency

        if age_days <= recency_thresholds.very_recent:
            return 1.0
        elif age_days <= recency_thresholds.recent:
            return 0.9
        elif age_days <= recency_thresholds.moderately_recent:
            return 0.7
        elif age_days <= recency_thresholds.somewhat_old:
            return 0.5
        elif age_days <= recency_thresholds.old:
            return 0.3
        else:
            return 0.1


class FileSizeScorer:
    """Scores results based on file size optimization."""

    @classmethod
    def score_file_size(cls, result: CodeSearchResult) -> float:
        """
        Score file size optimization (0.0 to 1.0).

        Args:
            result: Search result to score

        Returns:
            Size optimization score

        Raises:
            InvalidSearchResultError: If result is malformed
        """
        try:
            # Validate result structure
            if not result:
                raise InvalidSearchResultError("Search result cannot be None")

            if not hasattr(result, "file_metadata"):
                logger.debug(f"Result for {getattr(result, 'file_path', 'unknown')} missing file_metadata attribute")
                return 0.5

            if not result.file_metadata or "size" not in result.file_metadata:
                logger.debug(f"No size information for {getattr(result, 'file_path', 'unknown')}")
                return 0.5  # Neutral score if size is unknown

            size_bytes = result.file_metadata["size"]

            # Validate size value
            if not isinstance(size_bytes, (int, float)) or size_bytes < 0:
                logger.warning(f"Invalid file size for {getattr(result, 'file_path', 'unknown')}: {size_bytes}")
                return 0.5

            # Get configurable size thresholds
            config = get_ranking_config()
            size_thresholds = config.file_size

            # Very small files might be incomplete or not useful
            if size_bytes < size_thresholds.min_useful_size:
                return 0.2

            # Optimal size range gets highest score
            if size_thresholds.optimal_min <= size_bytes <= size_thresholds.optimal_max:
                return 1.0

            # Score based on distance from optimal range
            if size_bytes < size_thresholds.optimal_min:
                # Small files
                ratio = size_bytes / size_thresholds.optimal_min
                return 0.4 + 0.6 * ratio
            else:
                # Large files - diminishing returns with size
                if size_bytes <= size_thresholds.medium_max:
                    return 0.8
                elif size_bytes <= size_thresholds.large_max:
                    return 0.6
                elif size_bytes <= size_thresholds.very_large_max:
                    return 0.4
                else:
                    return 0.2  # Very large files

        except InvalidSearchResultError:
            raise  # Re-raise specific errors
        except Exception as e:
            logger.warning(f"Error scoring file size for result: {e}")
            return 0.5  # Neutral fallback score
