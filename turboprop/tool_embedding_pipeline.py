#!/usr/bin/env python3
"""
Tool Embedding Pipeline

This module manages embedding generation for tool catalog with batch processing
optimizations, validation, and quality checks. It provides a pipeline for
generating semantic embeddings for tool descriptions, parameters, and examples.
"""

import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from .embedding_helper import EmbeddingGenerator
from .logging_config import get_logger
from .mcp_metadata_types import MCPToolMetadata, ParameterAnalysis, ToolExample

logger = get_logger(__name__)

# Configuration constants for embedding pipeline
DEFAULT_EMBEDDING_DIMENSIONS = 384  # Expected embedding vector dimensions
DEFAULT_QUALITY_THRESHOLD = 0.5  # Minimum quality score for valid embeddings
MINIMUM_EMBEDDING_NORM = 0.1  # Minimum norm value for valid embeddings
MINIMUM_EMBEDDING_VARIANCE = 0.01  # Minimum variance for valid embeddings
BATCH_SIZE_LIMIT = 1000  # Maximum batch size for memory optimization


@dataclass
class EmbeddingResult:
    """Result of an embedding generation operation."""

    success: bool
    embeddings: List[np.ndarray]
    errors: List[str]
    execution_time: Optional[float] = None
    quality_scores: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "success": self.success,
            "embeddings_count": len(self.embeddings),
            "errors": self.errors,
            "execution_time": self.execution_time,
            "quality_scores": self.quality_scores,
        }


@dataclass
class EmbeddingQualityMetrics:
    """Metrics for embedding quality assessment."""

    dimension_count: int
    norm_score: float
    non_zero_ratio: float
    uniformity_score: float
    is_valid: bool

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class ToolEmbeddingPipeline:
    """Manages embedding generation for tool catalog."""

    def __init__(self, embedding_generator: EmbeddingGenerator):
        """
        Initialize the embedding pipeline.

        Args:
            embedding_generator: EmbeddingGenerator instance for creating embeddings
        """
        self.embedding_generator = embedding_generator
        self.quality_threshold = DEFAULT_QUALITY_THRESHOLD
        self.dimension_count = DEFAULT_EMBEDDING_DIMENSIONS

        logger.info("Initialized ToolEmbeddingPipeline")

    def _generate_batch_embeddings(
        self,
        descriptions: List[str],
        item_names: List[str],
        item_type: str,
        include_quality_scores: bool = False,
    ) -> tuple[List[np.ndarray], List[str], Optional[List[float]]]:
        """
        Generate embeddings for a batch of descriptions with common validation logic.

        Args:
            descriptions: List of text descriptions to embed
            item_names: List of item names for logging/error reporting
            item_type: Type of items being embedded (for error messages)
            include_quality_scores: Whether to calculate quality scores

        Returns:
            Tuple of (embeddings, errors, quality_scores)
        """
        embeddings = []
        errors = []
        quality_scores = [] if include_quality_scores else None

        try:
            # Generate embeddings (single vs batch)
            if len(descriptions) == 1:
                embedding = self.embedding_generator.encode(descriptions[0])
                batch_embeddings = [embedding]
            else:
                batch_embeddings = self.embedding_generator.encode_batch(descriptions)
                # Ensure it's a list of individual embeddings
                if len(batch_embeddings.shape) == 2:
                    batch_embeddings = [batch_embeddings[i] for i in range(batch_embeddings.shape[0])]

            # Validate each embedding
            for i, embedding in enumerate(batch_embeddings):
                quality_score = self._calculate_quality_score(embedding) if include_quality_scores else None

                if self.validate_embedding_quality(embedding):
                    embeddings.append(embedding)
                    if include_quality_scores:
                        quality_scores.append(quality_score)
                    logger.debug("Generated embedding for %s: %s", item_type, item_names[i])
                else:
                    if include_quality_scores:
                        error_msg = (
                            f"Low quality embedding for {item_type} {item_names[i]} (score: {quality_score:.3f})"
                        )
                    else:
                        error_msg = f"Invalid embedding for {item_type}: {item_names[i]}"
                    errors.append(error_msg)
                    logger.warning(error_msg)

                    # Add zero embedding as placeholder to maintain index alignment
                    embeddings.append(np.zeros(self.dimension_count, dtype=np.float32))
                    if include_quality_scores:
                        quality_scores.append(0.0)

        except Exception as e:
            error_msg = f"{item_type.title()} embedding generation failed: {e}"
            errors.append(error_msg)
            logger.error(error_msg)

            # Create placeholder embeddings for all items
            for name in item_names:
                embeddings.append(np.zeros(self.dimension_count, dtype=np.float32))
                if include_quality_scores:
                    quality_scores.append(0.0)

        return embeddings, errors, quality_scores

    def generate_tool_embeddings(self, tools: List[MCPToolMetadata]) -> EmbeddingResult:
        """
        Generate embeddings for tool descriptions.

        Processes tool descriptions in batches, handles embedding generation errors
        gracefully, stores embeddings with metadata, and validates embedding quality.

        Args:
            tools: List of tool metadata objects

        Returns:
            EmbeddingResult with generated embeddings and statistics
        """
        if not tools:
            logger.info("No tools provided for embedding generation")
            return EmbeddingResult(success=True, embeddings=[], errors=[], execution_time=0.0, quality_scores=[])

        logger.info("Generating embeddings for %d tools", len(tools))
        start_time = time.time()

        try:
            # Extract descriptions for batch processing
            descriptions = []
            tool_names = []

            for tool in tools:
                # Create rich description including name and category context
                description = self._create_rich_description(tool)
                descriptions.append(description)
                tool_names.append(tool.name)

            # Use helper method for common embedding logic
            embeddings, errors, quality_scores = self._generate_batch_embeddings(
                descriptions, tool_names, "tool", include_quality_scores=True
            )

            execution_time = time.time() - start_time
            success = len(errors) == 0

            logger.info(
                "Tool embedding generation completed: %d embeddings, %d errors in %.2fs",
                len(embeddings),
                len(errors),
                execution_time,
            )

            return EmbeddingResult(
                success=success,
                embeddings=embeddings,
                errors=errors,
                execution_time=execution_time,
                quality_scores=quality_scores,
            )

        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Critical error during tool embedding generation: {e}"
            logger.error(error_msg)

            return EmbeddingResult(
                success=False, embeddings=[], errors=[error_msg], execution_time=execution_time, quality_scores=[]
            )

    def generate_parameter_embeddings(self, parameters: List[ParameterAnalysis]) -> EmbeddingResult:
        """
        Generate embeddings for tool parameters.

        Creates embeddings for parameter descriptions including type and constraint
        information, handles complex parameter schemas, and links embeddings to parent tools.

        Args:
            parameters: List of parameter analysis objects

        Returns:
            EmbeddingResult with parameter embeddings
        """
        if not parameters:
            logger.info("No parameters provided for embedding generation")
            return EmbeddingResult(success=True, embeddings=[], errors=[], execution_time=0.0)

        logger.info("Generating embeddings for %d parameters", len(parameters))
        start_time = time.time()

        try:
            # Create enriched parameter descriptions
            param_descriptions = []
            param_names = []

            for param in parameters:
                # Create rich description including type, constraints, and context
                description = self._create_parameter_description(param)
                param_descriptions.append(description)
                param_names.append(param.name)

            # Use helper method for common embedding logic
            embeddings, errors, _ = self._generate_batch_embeddings(
                param_descriptions, param_names, "parameter", include_quality_scores=False
            )

            execution_time = time.time() - start_time
            success = len(errors) == 0

            logger.info(
                "Parameter embedding generation completed: %d embeddings, %d errors in %.2fs",
                len(embeddings),
                len(errors),
                execution_time,
            )

            return EmbeddingResult(success=success, embeddings=embeddings, errors=errors, execution_time=execution_time)

        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Critical error during parameter embedding generation: {e}"
            logger.error(error_msg)

            return EmbeddingResult(success=False, embeddings=[], errors=[error_msg], execution_time=execution_time)

    def generate_example_embeddings(self, examples: List[ToolExample]) -> EmbeddingResult:
        """
        Generate embeddings for usage examples.

        Creates embeddings for use case descriptions including context and expected
        outcomes, supports code snippet embeddings, and maintains example-to-tool relationships.

        Args:
            examples: List of tool example objects

        Returns:
            EmbeddingResult with example embeddings
        """
        if not examples:
            logger.info("No examples provided for embedding generation")
            return EmbeddingResult(success=True, embeddings=[], errors=[], execution_time=0.0)

        logger.info("Generating embeddings for %d examples", len(examples))
        start_time = time.time()

        try:
            # Create enriched example descriptions
            example_descriptions = []
            example_use_cases = []

            for example in examples:
                # Create comprehensive description including use case, context, and call
                description = self._create_example_description(example)
                example_descriptions.append(description)
                example_use_cases.append(example.use_case)

            # Use helper method for common embedding logic
            embeddings, errors, _ = self._generate_batch_embeddings(
                example_descriptions, example_use_cases, "example", include_quality_scores=False
            )

            execution_time = time.time() - start_time
            success = len(errors) == 0

            logger.info(
                "Example embedding generation completed: %d embeddings, %d errors in %.2fs",
                len(embeddings),
                len(errors),
                execution_time,
            )

            return EmbeddingResult(success=success, embeddings=embeddings, errors=errors, execution_time=execution_time)

        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Critical error during example embedding generation: {e}"
            logger.error(error_msg)

            return EmbeddingResult(success=False, embeddings=[], errors=[error_msg], execution_time=execution_time)

    def validate_embedding_quality(self, embedding: np.ndarray) -> bool:
        """
        Validate the quality of a generated embedding.

        Checks for zero embeddings, proper dimensions, reasonable norm values,
        and other quality indicators.

        Args:
            embedding: Embedding vector to validate

        Returns:
            True if embedding is valid, False otherwise
        """
        try:
            if embedding is None:
                return False

            # Check dimensions
            if len(embedding.shape) != 1 or embedding.shape[0] != self.dimension_count:
                logger.debug("Invalid embedding dimensions: %s", embedding.shape)
                return False

            # Check for zero embedding (invalid)
            if np.allclose(embedding, 0.0, atol=1e-6):
                logger.debug("Zero embedding detected")
                return False

            # Check for NaN or infinite values
            if not np.isfinite(embedding).all():
                logger.debug("Non-finite values in embedding")
                return False

            # Check norm (should not be too small)
            norm = np.linalg.norm(embedding)
            if norm < MINIMUM_EMBEDDING_NORM:
                logger.debug("Embedding norm too small: %.6f", norm)
                return False

            # Check for reasonable variance (not all values the same)
            if np.std(embedding) < MINIMUM_EMBEDDING_VARIANCE:
                logger.debug("Embedding has low variance")
                return False

            return True

        except Exception as e:
            logger.error("Error validating embedding quality: %s", e)
            return False

    def get_embedding_quality_metrics(self, embedding: np.ndarray) -> EmbeddingQualityMetrics:
        """
        Get detailed quality metrics for an embedding.

        Args:
            embedding: Embedding to analyze

        Returns:
            EmbeddingQualityMetrics with detailed analysis
        """
        try:
            dimension_count = len(embedding) if embedding is not None else 0

            if embedding is None or dimension_count == 0:
                return EmbeddingQualityMetrics(
                    dimension_count=0, norm_score=0.0, non_zero_ratio=0.0, uniformity_score=0.0, is_valid=False
                )

            # Calculate norm score
            norm = np.linalg.norm(embedding)
            norm_score = min(norm, 1.0)  # Normalize to 0-1

            # Calculate non-zero ratio
            non_zero_count = np.count_nonzero(embedding)
            non_zero_ratio = non_zero_count / dimension_count if dimension_count > 0 else 0.0

            # Calculate uniformity score (inverse of how uniform the values are)
            std_dev = np.std(embedding)
            uniformity_score = min(std_dev * 10, 1.0)  # Scale and cap at 1.0

            # Overall validity
            is_valid = self.validate_embedding_quality(embedding)

            return EmbeddingQualityMetrics(
                dimension_count=dimension_count,
                norm_score=norm_score,
                non_zero_ratio=non_zero_ratio,
                uniformity_score=uniformity_score,
                is_valid=is_valid,
            )

        except Exception as e:
            logger.error("Error calculating embedding quality metrics: %s", e)
            return EmbeddingQualityMetrics(
                dimension_count=0, norm_score=0.0, non_zero_ratio=0.0, uniformity_score=0.0, is_valid=False
            )

    def _create_rich_description(self, tool: MCPToolMetadata) -> str:
        """
        Create rich description for tool embedding generation.

        Args:
            tool: Tool metadata

        Returns:
            Enhanced description string
        """
        parts = []

        # Add tool name and category
        parts.append(f"Tool: {tool.name}")
        parts.append(f"Category: {tool.category}")

        # Add main description
        if tool.description:
            parts.append(f"Description: {tool.description}")

        # Add parameter information
        if hasattr(tool, "parameters") and tool.parameters:
            param_names = [p.name for p in tool.parameters[:5]]  # Limit to first 5
            parts.append(f"Parameters: {', '.join(param_names)}")

        return " ".join(parts)

    def _create_parameter_description(self, param: ParameterAnalysis) -> str:
        """
        Create enhanced description for parameter embedding.

        Args:
            param: Parameter analysis

        Returns:
            Enhanced parameter description
        """
        parts = []

        parts.append(f"Parameter: {param.name}")
        parts.append(f"Type: {param.type}")
        parts.append(f"Required: {param.required}")

        if param.description:
            parts.append(f"Description: {param.description}")

        # Add constraint information if available
        if hasattr(param, "constraints") and param.constraints:
            constraint_info = []
            for key, value in param.constraints.items():
                constraint_info.append(f"{key}: {value}")
            if constraint_info:
                parts.append(f"Constraints: {', '.join(constraint_info)}")

        return " ".join(parts)

    def _create_example_description(self, example: ToolExample) -> str:
        """
        Create enhanced description for example embedding.

        Args:
            example: Tool example

        Returns:
            Enhanced example description
        """
        parts = []

        parts.append(f"Use case: {example.use_case}")

        if hasattr(example, "context") and example.context:
            parts.append(f"Context: {example.context}")

        if hasattr(example, "example_call") and example.example_call:
            parts.append(f"Example: {example.example_call}")

        if hasattr(example, "expected_output") and example.expected_output:
            parts.append(f"Expected output: {example.expected_output}")

        return " ".join(parts)

    def _calculate_quality_score(self, embedding: np.ndarray) -> float:
        """
        Calculate a quality score for an embedding.

        Args:
            embedding: Embedding to score

        Returns:
            Quality score between 0.0 and 1.0
        """
        try:
            if embedding is None or len(embedding) == 0:
                return 0.0

            # Check for zero embedding
            if np.allclose(embedding, 0.0, atol=1e-6):
                return 0.0

            # Check for non-finite values
            if not np.isfinite(embedding).all():
                return 0.0

            # Calculate components of quality score
            norm = np.linalg.norm(embedding)
            norm_score = min(norm, 1.0)  # Prefer reasonable norms

            non_zero_ratio = np.count_nonzero(embedding) / len(embedding)

            variance_score = min(np.std(embedding) * 5, 1.0)  # Prefer some variance

            # Combine scores
            quality_score = norm_score * 0.4 + non_zero_ratio * 0.3 + variance_score * 0.3

            return max(0.0, min(1.0, quality_score))

        except Exception as e:
            logger.error("Error calculating quality score: %s", e)
            return 0.0
