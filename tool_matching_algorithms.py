#!/usr/bin/env python3
"""
Tool Matching Algorithms

This module implements algorithms for matching tools to search queries, calculating
relevance scores, and generating explanations for matches.
"""

import math
from typing import Any, Dict, List, Optional, Tuple

from logging_config import get_logger
from mcp_metadata_types import MCPToolMetadata
from tool_search_results import ProcessedQuery, ToolSearchResult

logger = get_logger(__name__)

# Configuration constants for matching algorithms
DEFAULT_SIMILARITY_THRESHOLD = 0.1
CATEGORY_MATCH_BOOST = 0.15
TYPE_MATCH_BOOST = 0.10
PARAMETER_COMPATIBILITY_BOOST = 0.05
COMPLEXITY_PREFERENCE_BOOST = 0.08
EXAMPLE_QUALITY_BOOST = 0.03

# Cosine similarity normalization constants
COSINE_SIMILARITY_MIN_BOUND = 0.0
COSINE_SIMILARITY_MAX_BOUND = 1.0
COSINE_SIMILARITY_RANGE_OFFSET = 1.0
COSINE_SIMILARITY_RANGE_DIVISOR = 2.0

# Ranking strategy configurations
RANKING_STRATEGIES = {
    "relevance": {"semantic_weight": 0.6, "metadata_weight": 0.25, "quality_weight": 0.15},
    "similarity": {"semantic_weight": 0.9, "metadata_weight": 0.05, "quality_weight": 0.05},
    "popularity": {"semantic_weight": 0.3, "metadata_weight": 0.2, "quality_weight": 0.5},
    "simplicity": {"semantic_weight": 0.4, "metadata_weight": 0.3, "quality_weight": 0.3},
}


class ToolMatchingAlgorithms:
    """Algorithms for matching tools to search queries."""

    def __init__(self, similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD):
        """
        Initialize the tool matching algorithms.

        Args:
            similarity_threshold: Minimum similarity score to consider a match
        """
        self.similarity_threshold = similarity_threshold

    def calculate_semantic_similarity(self, query_embedding: List[float], tool_embedding: List[float]) -> float:
        """
        Calculate cosine similarity between query and tool embeddings.

        Args:
            query_embedding: Query embedding vector
            tool_embedding: Tool embedding vector

        Returns:
            Similarity score between 0.0 and 1.0
        """
        if not query_embedding or not tool_embedding:
            return 0.0

        if len(query_embedding) != len(tool_embedding):
            logger.warning(
                "Embedding dimension mismatch: query=%d, tool=%d",
                len(query_embedding),
                len(tool_embedding),
            )
            return 0.0

        try:
            # Calculate cosine similarity
            dot_product = sum(q * t for q, t in zip(query_embedding, tool_embedding))

            query_magnitude = math.sqrt(sum(q * q for q in query_embedding))
            tool_magnitude = math.sqrt(sum(t * t for t in tool_embedding))

            if query_magnitude == 0.0 or tool_magnitude == 0.0:
                return 0.0

            similarity = dot_product / (query_magnitude * tool_magnitude)

            # Normalize to [0, 1] range (cosine similarity is in [-1, 1])
            normalized_similarity = max(
                COSINE_SIMILARITY_MIN_BOUND,
                min(
                    COSINE_SIMILARITY_MAX_BOUND,
                    (similarity + COSINE_SIMILARITY_RANGE_OFFSET) / COSINE_SIMILARITY_RANGE_DIVISOR,
                ),
            )

            return normalized_similarity

        except (TypeError, ValueError, ZeroDivisionError) as e:
            logger.error("Error calculating semantic similarity - invalid embedding data: %s", e)
            return 0.0
        except (AttributeError, IndexError) as e:
            logger.error("Error calculating semantic similarity - malformed embedding structure: %s", e)
            return 0.0
        except Exception as e:
            logger.error("Unexpected error calculating semantic similarity: %s", e)
            return 0.0

    def calculate_relevance_score(
        self,
        semantic_score: float,
        tool_metadata: MCPToolMetadata,
        query_context: ProcessedQuery,
        additional_scores: Optional[Dict[str, float]] = None,
    ) -> float:
        """
        Calculate overall relevance combining multiple factors.

        Args:
            semantic_score: Base semantic similarity score
            tool_metadata: Tool metadata for contextual matching
            query_context: Processed query with intent and context
            additional_scores: Optional additional scoring factors

        Returns:
            Combined relevance score between 0.0 and 1.0
        """
        if semantic_score < self.similarity_threshold:
            return 0.0

        relevance = semantic_score
        additional_scores = additional_scores or {}

        try:
            # Category match boost
            if (
                query_context.detected_category
                and tool_metadata.category
                and query_context.detected_category.lower() == tool_metadata.category.lower()
            ):
                relevance += CATEGORY_MATCH_BOOST
                logger.debug("Category match boost applied: %s", tool_metadata.category)

            # Tool type boost (if we can infer tool type from metadata)
            if query_context.detected_tool_type and hasattr(tool_metadata, "tool_type"):
                if query_context.detected_tool_type == getattr(tool_metadata, "tool_type", None):
                    relevance += TYPE_MATCH_BOOST

            # Parameter compatibility boost
            parameter_compatibility = self._calculate_parameter_compatibility(tool_metadata, query_context)
            relevance += parameter_compatibility * PARAMETER_COMPATIBILITY_BOOST

            # Complexity preference boost
            complexity_match = self._calculate_complexity_match(tool_metadata, query_context)
            relevance += complexity_match * COMPLEXITY_PREFERENCE_BOOST

            # Example quality boost
            example_quality = self._calculate_example_quality_score(tool_metadata)
            relevance += example_quality * EXAMPLE_QUALITY_BOOST

            # Apply additional scores
            for score_name, score_value in additional_scores.items():
                relevance += score_value * 0.05  # Small boost for additional factors

            # Normalize to [0, 1] range
            relevance = max(0.0, min(1.0, relevance))

            return relevance

        except (AttributeError, KeyError) as e:
            logger.error("Error calculating relevance score - missing metadata attributes: %s", e)
            return semantic_score  # Fall back to semantic score
        except (TypeError, ValueError) as e:
            logger.error("Error calculating relevance score - invalid numeric values: %s", e)
            return max(0.0, min(1.0, semantic_score))
        except Exception as e:
            logger.error("Unexpected error calculating relevance score: %s", e)
            return semantic_score  # Fall back to semantic score

    def explain_match_reasons(
        self,
        tool: MCPToolMetadata,
        query: ProcessedQuery,
        scores: Dict[str, float],
    ) -> List[str]:
        """
        Generate human-readable explanations for why tool matched.

        Args:
            tool: Tool metadata
            query: Processed query
            scores: Dictionary of various scores

        Returns:
            List of match explanation strings
        """
        reasons = []

        try:
            semantic_score = scores.get("semantic_similarity", 0.0)
            relevance_score = scores.get("relevance_score", 0.0)

            # Semantic similarity explanation
            if semantic_score > 0.8:
                reasons.append(f"Very strong semantic match (similarity: {semantic_score:.1%})")
            elif semantic_score > 0.6:
                reasons.append(f"Strong semantic match (similarity: {semantic_score:.1%})")
            elif semantic_score > 0.4:
                reasons.append(f"Good semantic match (similarity: {semantic_score:.1%})")
            elif semantic_score > 0.2:
                reasons.append(f"Moderate semantic match (similarity: {semantic_score:.1%})")

            # Category match explanation
            if query.detected_category and tool.category and query.detected_category.lower() == tool.category.lower():
                reasons.append(f"Category match: {tool.category}")

            # Functionality match explanation
            if query.search_intent and query.search_intent.target_functionality:
                matched_functionality = []
                tool_desc_lower = tool.description.lower()
                for func in query.search_intent.target_functionality:
                    if func in tool_desc_lower:
                        matched_functionality.append(func)

                if matched_functionality:
                    reasons.append(f"Functionality match: {', '.join(matched_functionality)}")

            # Parameter compatibility explanation
            param_compatibility = scores.get("parameter_compatibility", 0.0)
            if param_compatibility > 0.7:
                reasons.append("Excellent parameter compatibility")
            elif param_compatibility > 0.5:
                reasons.append("Good parameter compatibility")

            # Complexity match explanation
            complexity_match = scores.get("complexity_match", 0.0)
            if complexity_match > 0.8:
                if tool.complexity_analysis and tool.complexity_analysis.overall_complexity < 0.3:
                    reasons.append("Simple tool matching preference")
                elif tool.complexity_analysis and tool.complexity_analysis.overall_complexity > 0.7:
                    reasons.append("Advanced tool matching preference")

            # Example quality explanation
            example_quality = scores.get("example_quality", 0.0)
            if example_quality > 0.8:
                reasons.append(f"Excellent usage examples ({len(tool.examples)} examples)")
            elif example_quality > 0.6:
                reasons.append(f"Good usage examples ({len(tool.examples)} examples)")

            # Constraint satisfaction explanation
            if query.search_intent and query.search_intent.constraints:
                satisfied_constraints = []
                for constraint, value in query.search_intent.constraints.items():
                    if constraint == "error_handling" and value:
                        if "error" in tool.description.lower() or "exception" in tool.description.lower():
                            satisfied_constraints.append("error handling")
                    elif constraint == "timeout" and value:
                        if "timeout" in tool.description.lower():
                            satisfied_constraints.append("timeout support")
                    elif constraint == "performance" and value == "high":
                        if any(term in tool.description.lower() for term in ["fast", "efficient", "performance"]):
                            satisfied_constraints.append("performance optimized")

                if satisfied_constraints:
                    reasons.append(f"Satisfies requirements: {', '.join(satisfied_constraints)}")

            # If no specific reasons found, provide general explanation
            if not reasons and relevance_score > 0.3:
                reasons.append("General match based on tool description")

            return reasons[:5]  # Return top 5 reasons

        except (AttributeError, KeyError) as e:
            logger.error("Error generating match reasons - missing tool or query attributes: %s", e)
            return ["Tool matched based on semantic similarity"]
        except (TypeError, ValueError) as e:
            logger.error("Error generating match reasons - invalid data types in scores: %s", e)
            return ["Tool matched with limited analysis"]
        except Exception as e:
            logger.error("Unexpected error generating match reasons: %s", e)
            return ["Tool matched based on semantic similarity"]

    def rank_results(
        self,
        results: List[ToolSearchResult],
        ranking_strategy: str = "relevance",
    ) -> List[ToolSearchResult]:
        """
        Rank search results by specified strategy.

        Args:
            results: List of search results to rank
            ranking_strategy: Ranking strategy ('relevance', 'similarity', 'popularity', 'simplicity')

        Returns:
            Sorted list of search results
        """
        if not results:
            return results

        strategy_config = RANKING_STRATEGIES.get(ranking_strategy, RANKING_STRATEGIES["relevance"])

        try:

            def calculate_ranking_score(result: ToolSearchResult) -> float:
                """Calculate ranking score based on strategy."""
                semantic_component = result.similarity_score * strategy_config["semantic_weight"]

                # Metadata component (category match, parameter count, etc.)
                metadata_component = (
                    (1.0 if result.category else 0.5) * 0.3
                    + (1.0 - min(result.complexity_score, 1.0)) * 0.4
                    + (min(result.parameter_count / 10.0, 1.0))  # Favor simpler tools in metadata
                    * 0.3  # Reasonable parameter count
                ) * strategy_config["metadata_weight"]

                # Quality component (examples, match reasons, confidence)
                quality_component = (
                    result.get_confidence_score() * 0.4
                    + min(len(result.examples) / 3.0, 1.0) * 0.3
                    + min(len(result.match_reasons) / 3.0, 1.0) * 0.3  # Good examples  # Clear match reasons
                ) * strategy_config["quality_weight"]

                return semantic_component + metadata_component + quality_component

            # Sort results by ranking score (descending)
            ranked_results = sorted(results, key=calculate_ranking_score, reverse=True)

            logger.debug(
                "Ranked %d results using %s strategy",
                len(ranked_results),
                ranking_strategy,
            )

            return ranked_results

        except KeyError as e:
            logger.error("Error ranking results - unknown ranking strategy '%s': %s", ranking_strategy, e)
            # Fall back to simple relevance score ranking
            return sorted(results, key=lambda r: r.relevance_score, reverse=True)
        except (AttributeError, TypeError) as e:
            logger.error("Error ranking results - invalid result objects or scores: %s", e)
            return results  # Return unranked results
        except Exception as e:
            logger.error("Unexpected error ranking results: %s", e)
            # Fall back to simple relevance score ranking
            return sorted(results, key=lambda r: r.relevance_score, reverse=True)

    def filter_results_by_threshold(
        self, results: List[ToolSearchResult], min_relevance: float = 0.2
    ) -> List[ToolSearchResult]:
        """
        Filter results by minimum relevance threshold.

        Args:
            results: List of search results
            min_relevance: Minimum relevance score threshold

        Returns:
            Filtered list of results
        """
        filtered_results = [r for r in results if r.relevance_score >= min_relevance]

        logger.debug(
            "Filtered results: %d -> %d (threshold: %.2f)",
            len(results),
            len(filtered_results),
            min_relevance,
        )

        return filtered_results

    def apply_diversity_filtering(
        self, results: List[ToolSearchResult], max_per_category: int = 3
    ) -> List[ToolSearchResult]:
        """
        Apply diversity filtering to ensure varied results across categories.

        Args:
            results: List of search results
            max_per_category: Maximum results per category

        Returns:
            Diversified list of results
        """
        category_counts = {}
        diversified_results = []

        for result in results:
            category = result.category or "unknown"
            current_count = category_counts.get(category, 0)

            if current_count < max_per_category:
                diversified_results.append(result)
                category_counts[category] = current_count + 1

        logger.debug(
            "Applied diversity filtering: %d -> %d results",
            len(results),
            len(diversified_results),
        )

        return diversified_results

    def _calculate_parameter_compatibility(
        self, tool_metadata: MCPToolMetadata, query_context: ProcessedQuery
    ) -> float:
        """
        Calculate parameter compatibility score.

        Args:
            tool_metadata: Tool metadata
            query_context: Processed query context

        Returns:
            Parameter compatibility score between 0.0 and 1.0
        """
        if not query_context.search_intent or not query_context.search_intent.constraints:
            return 0.5  # Neutral score when no constraints

        constraints = query_context.search_intent.constraints
        compatibility_score = 0.5

        try:
            # Check complexity preference
            if "complexity" in constraints:
                preferred_complexity = constraints["complexity"]
                if tool_metadata.complexity_analysis:
                    tool_complexity = tool_metadata.complexity_analysis.overall_complexity

                    if preferred_complexity == "low" and tool_complexity < 0.3:
                        compatibility_score += 0.3
                    elif preferred_complexity == "high" and tool_complexity > 0.7:
                        compatibility_score += 0.3
                    elif preferred_complexity == "medium" and 0.3 <= tool_complexity <= 0.7:
                        compatibility_score += 0.3

            # Check parameter requirements
            if "has_required_params" in constraints:
                required_count = len([p for p in tool_metadata.parameters if p.required])
                if constraints["has_required_params"] == "true" and required_count > 0:
                    compatibility_score += 0.2
                elif constraints["has_required_params"] == "false" and required_count == 0:
                    compatibility_score += 0.2

            return min(compatibility_score, 1.0)

        except (AttributeError, KeyError) as e:
            logger.error("Error calculating parameter compatibility - missing metadata or constraints: %s", e)
            return 0.5
        except (TypeError, ValueError) as e:
            logger.error("Error calculating parameter compatibility - invalid constraint values: %s", e)
            return 0.5
        except Exception as e:
            logger.error("Unexpected error calculating parameter compatibility: %s", e)
            return 0.5

    def _calculate_complexity_match(self, tool_metadata: MCPToolMetadata, query_context: ProcessedQuery) -> float:
        """
        Calculate complexity match score.

        Args:
            tool_metadata: Tool metadata
            query_context: Processed query context

        Returns:
            Complexity match score between 0.0 and 1.0
        """
        if not tool_metadata.complexity_analysis:
            return 0.5  # Neutral score when no complexity info

        query_lower = query_context.cleaned_query.lower()
        tool_complexity = tool_metadata.complexity_analysis.overall_complexity

        try:
            # Detect complexity preference from query
            if any(term in query_lower for term in ["simple", "basic", "easy"]):
                # User wants simple tools
                return 1.0 - tool_complexity  # Favor low complexity

            elif any(term in query_lower for term in ["advanced", "complex", "sophisticated"]):
                # User wants complex tools
                return tool_complexity  # Favor high complexity

            else:
                # No explicit preference, moderate tools are good
                # Peak at 0.5 complexity, decline toward extremes
                return 1.0 - abs(tool_complexity - 0.5) * 2

        except (AttributeError, KeyError) as e:
            logger.error("Error calculating complexity match - missing complexity analysis: %s", e)
            return 0.5
        except (TypeError, ValueError) as e:
            logger.error("Error calculating complexity match - invalid complexity values: %s", e)
            return 0.5
        except Exception as e:
            logger.error("Unexpected error calculating complexity match: %s", e)
            return 0.5

    def _calculate_example_quality_score(self, tool_metadata: MCPToolMetadata) -> float:
        """
        Calculate example quality score.

        Args:
            tool_metadata: Tool metadata

        Returns:
            Example quality score between 0.0 and 1.0
        """
        if not tool_metadata.examples:
            return 0.0

        try:
            # Calculate average effectiveness score
            if all(hasattr(ex, "effectiveness_score") for ex in tool_metadata.examples):
                avg_effectiveness = sum(ex.effectiveness_score for ex in tool_metadata.examples) / len(
                    tool_metadata.examples
                )
            else:
                avg_effectiveness = 0.5  # Default if no effectiveness scores

            # Factor in number of examples (more is better, up to a point)
            example_count_factor = min(len(tool_metadata.examples) / 3.0, 1.0)

            # Combine factors
            quality_score = (avg_effectiveness * 0.7) + (example_count_factor * 0.3)

            return min(quality_score, 1.0)

        except (AttributeError, KeyError) as e:
            logger.error("Error calculating example quality score - missing example data: %s", e)
            return 0.0
        except (TypeError, ValueError, ZeroDivisionError) as e:
            logger.error("Error calculating example quality score - invalid example metrics: %s", e)
            return 0.0
        except Exception as e:
            logger.error("Unexpected error calculating example quality score: %s", e)
            return 0.0
