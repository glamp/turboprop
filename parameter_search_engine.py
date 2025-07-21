#!/usr/bin/env python3
"""
Parameter Search Engine

This module provides the main parameter-aware search engine that integrates
parameter analysis, type compatibility, filtering, and ranking for sophisticated
tool search capabilities.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from advanced_filters import AdvancedFilters, ParameterFilterSet
from logging_config import get_logger
from mcp_metadata_types import ParameterAnalysis, ToolId
from mcp_tool_search_engine import MCPToolSearchEngine
from parameter_analyzer import ParameterAnalyzer, ParameterRequirements
from parameter_ranking import ComplexityPreference, ParameterRanking, RankingContext
from tool_search_results import ToolSearchResult
from type_compatibility_analyzer import ConversionChain, TypeCompatibilityAnalyzer

logger = get_logger(__name__)


@dataclass
class ToolChainStep:
    """A single step in a tool chain."""

    tool_id: ToolId
    tool_name: str
    description: str
    input_types: List[str] = field(default_factory=list)
    output_types: List[str] = field(default_factory=list)
    parameters: List[ParameterAnalysis] = field(default_factory=list)
    compatibility_score: float = 0.0
    step_explanation: str = ""


@dataclass
class ToolChainResult:
    """Result representing a chain of compatible tools."""

    chain_id: str
    source_description: str
    target_description: str
    steps: List[ToolChainStep] = field(default_factory=list)
    overall_compatibility: float = 0.0
    total_complexity: int = 0
    execution_feasibility: float = 0.0
    chain_explanation: str = ""
    suggested_parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolCompatibilityResult:
    """Result of tool compatibility analysis."""

    reference_tool_id: ToolId
    compatible_tools: List[ToolSearchResult] = field(default_factory=list)
    compatibility_matrix: Dict[str, float] = field(default_factory=dict)
    suggested_combinations: List[Tuple[str, str, float]] = field(default_factory=list)
    integration_patterns: List[str] = field(default_factory=list)
    compatibility_explanation: str = ""


class ParameterSearchEngine:
    """Search engine with comprehensive parameter awareness."""

    def __init__(
        self,
        tool_search_engine: MCPToolSearchEngine,
        parameter_analyzer: Optional[ParameterAnalyzer] = None,
        type_analyzer: Optional[TypeCompatibilityAnalyzer] = None,
    ):
        """
        Initialize parameter-aware search engine.

        Args:
            tool_search_engine: Base MCP tool search engine
            parameter_analyzer: Parameter analysis component
            type_analyzer: Type compatibility analyzer
        """
        self.tool_search_engine = tool_search_engine
        self.parameter_analyzer = parameter_analyzer or ParameterAnalyzer()
        self.type_analyzer = type_analyzer or TypeCompatibilityAnalyzer()
        self.advanced_filters = AdvancedFilters()
        self.parameter_ranking = ParameterRanking(self.parameter_analyzer, self.type_analyzer)

        # Performance tracking
        self.search_statistics = {
            "total_searches": 0,
            "parameter_searches": 0,
            "data_flow_searches": 0,
            "compatibility_searches": 0,
            "average_execution_time": 0.0,
        }

        logger.info("Initialized ParameterSearchEngine with all components")

    def search_by_parameters(
        self,
        input_types: Optional[List[str]] = None,
        output_types: Optional[List[str]] = None,
        required_parameters: Optional[List[str]] = None,
        optional_parameters: Optional[List[str]] = None,
        parameter_constraints: Optional[Dict[str, Any]] = None,
        k: int = 10,
        complexity_preference: ComplexityPreference = ComplexityPreference.ANY,
        enable_filtering: bool = True,
        boost_weight: float = 0.4,
    ) -> List[ToolSearchResult]:
        """
        Search tools by parameter specifications.

        Args:
            input_types: Expected input data types
            output_types: Expected output data types
            required_parameters: Parameter names that must be present
            optional_parameters: Parameter names that are preferred
            parameter_constraints: Specific parameter constraints
            k: Maximum number of results
            complexity_preference: Preferred tool complexity level
            enable_filtering: Whether to apply advanced filtering
            boost_weight: Weight for parameter-based ranking boost

        Returns:
            List of ToolSearchResult objects ranked by parameter compatibility
        """
        start_time = time.time()
        self.search_statistics["parameter_searches"] += 1

        try:
            logger.info("Starting parameter-based search with types: input=%s, output=%s", input_types, output_types)

            # Build parameter requirements
            requirements = ParameterRequirements(
                input_types=input_types or [],
                output_types=output_types or [],
                required_parameters=required_parameters or [],
                optional_parameters=optional_parameters or [],
                parameter_constraints=parameter_constraints or {},
                complexity_preference=complexity_preference.value
                if isinstance(complexity_preference, ComplexityPreference)
                else complexity_preference,
            )

            # Build search query from parameter requirements
            search_query = self._build_parameter_search_query(requirements)

            # Perform initial semantic search with expanded results
            initial_results = self.tool_search_engine.search_by_functionality(
                query=search_query, k=min(k * 3, 50)  # Get more results for filtering
            )

            if not initial_results.results:
                logger.info("No initial results found for parameter search")
                return []

            # Apply parameter-based filtering if enabled
            filtered_results = initial_results.results
            if enable_filtering:
                filtered_results = self._apply_parameter_filtering(
                    filtered_results, requirements, complexity_preference
                )

            # Apply parameter-based ranking boost
            if boost_weight > 0:
                ranking_context = RankingContext(
                    parameter_requirements=requirements,
                    complexity_preference=complexity_preference,
                    boost_weight=boost_weight,
                )
                ranked_results = self.parameter_ranking.apply_parameter_ranking_boost(
                    filtered_results, requirements, boost_weight, ranking_context
                )
            else:
                ranked_results = filtered_results

            # Apply complexity preference ranking
            final_results = self.parameter_ranking.rank_by_complexity_preference(ranked_results, complexity_preference)

            # Limit to requested count and add parameter explanations
            final_results = final_results[:k]
            self._add_parameter_explanations(final_results, requirements)

            execution_time = time.time() - start_time
            self._update_search_statistics(execution_time)

            logger.info("Parameter search completed: %d results in %.3fs", len(final_results), execution_time)

            return final_results

        except Exception as e:
            logger.error("Error in parameter search: %s", e)
            return []

    def search_by_data_flow(
        self,
        input_description: str,
        desired_output: str,
        allow_chaining: bool = True,
        max_chain_length: int = 3,
        k: int = 10,
    ) -> List[ToolChainResult]:
        """
        Find tools or tool chains for data transformation.

        Args:
            input_description: Description of input data/format
            desired_output: Description of desired output
            allow_chaining: Whether to consider tool chains
            max_chain_length: Maximum length of tool chains
            k: Maximum number of chain results

        Returns:
            List of ToolChainResult objects with viable data flow solutions
        """
        start_time = time.time()
        self.search_statistics["data_flow_searches"] += 1

        try:
            logger.info("Starting data flow search: %s -> %s", input_description, desired_output)

            # Extract types from descriptions
            input_types = self._extract_types_from_description(input_description)
            output_types = self._extract_types_from_description(desired_output)

            # Find direct tool matches
            direct_matches = self.search_by_parameters(input_types=input_types, output_types=output_types, k=k * 2)

            chain_results = []

            # Create single-tool chains from direct matches
            for tool in direct_matches[:k]:
                chain_result = self._create_single_tool_chain(
                    tool, input_description, desired_output, input_types, output_types
                )
                chain_results.append(chain_result)

            # Find multi-tool chains if enabled
            if allow_chaining and len(direct_matches) < k:
                multi_tool_chains = self._find_multi_tool_chains(
                    input_types,
                    output_types,
                    input_description,
                    desired_output,
                    max_chain_length,
                    k - len(direct_matches),
                )
                chain_results.extend(multi_tool_chains)

            # Sort by overall compatibility and feasibility
            chain_results.sort(
                key=lambda c: (c.overall_compatibility * 0.6 + c.execution_feasibility * 0.4), reverse=True
            )

            execution_time = time.time() - start_time
            self._update_search_statistics(execution_time)

            logger.info("Data flow search completed: %d chains found in %.3fs", len(chain_results), execution_time)

            return chain_results[:k]

        except Exception as e:
            logger.error("Error in data flow search: %s", e)
            return []

    def find_compatible_tools(
        self, reference_tool: str, compatibility_type: str = "input_output", k: int = 10
    ) -> List[ToolCompatibilityResult]:
        """
        Find tools compatible with a reference tool.

        Args:
            reference_tool: ID or name of reference tool
            compatibility_type: Type of compatibility ('input_output', 'chaining', 'alternative')
            k: Maximum number of compatible tools

        Returns:
            List of ToolCompatibilityResult objects
        """
        start_time = time.time()
        self.search_statistics["compatibility_searches"] += 1

        try:
            logger.info("Finding tools compatible with: %s (type: %s)", reference_tool, compatibility_type)

            # Get reference tool metadata
            ref_tool_data = self._get_reference_tool_data(reference_tool)
            if not ref_tool_data:
                logger.warning("Reference tool not found: %s", reference_tool)
                return []

            # Analyze compatibility based on type
            if compatibility_type == "input_output":
                compatible_results = self._find_io_compatible_tools(ref_tool_data, k)
            elif compatibility_type == "chaining":
                compatible_results = self._find_chainable_tools(ref_tool_data, k)
            elif compatibility_type == "alternative":
                compatible_results = self._find_alternative_tools(ref_tool_data, k)
            else:
                logger.warning("Unknown compatibility type: %s", compatibility_type)
                return []

            execution_time = time.time() - start_time
            self._update_search_statistics(execution_time)

            logger.info("Compatibility search completed: %d results in %.3fs", len(compatible_results), execution_time)

            return compatible_results

        except Exception as e:
            logger.error("Error finding compatible tools: %s", e)
            return []

    # Helper methods

    def _build_parameter_search_query(self, requirements: ParameterRequirements) -> str:
        """Build search query from parameter requirements."""
        query_parts = []

        # Add type-based terms
        if requirements.input_types:
            type_terms = " ".join(requirements.input_types)
            query_parts.append(f"handles {type_terms}")

        if requirements.output_types:
            output_terms = " ".join(requirements.output_types)
            query_parts.append(f"returns {output_terms}")

        # Add parameter-based terms
        if requirements.required_parameters:
            param_terms = " ".join(requirements.required_parameters)
            query_parts.append(f"with parameters {param_terms}")

        if requirements.optional_parameters:
            optional_terms = " ".join(requirements.optional_parameters[:3])  # Limit for query length
            query_parts.append(f"optional {optional_terms}")

        # Add complexity preference
        if requirements.complexity_preference and requirements.complexity_preference != "any":
            query_parts.append(f"{requirements.complexity_preference} tool")

        # Combine parts or use fallback
        if query_parts:
            return " ".join(query_parts)
        else:
            return "tool with parameters"

    def _apply_parameter_filtering(
        self,
        results: List[ToolSearchResult],
        requirements: ParameterRequirements,
        complexity_preference: ComplexityPreference,
    ) -> List[ToolSearchResult]:
        """Apply parameter-based filtering to search results."""
        # Build parameter filter set
        param_filters = ParameterFilterSet()

        # Set parameter count filters if implied by requirements
        if requirements.required_parameters:
            param_filters.min_required_parameters = len(requirements.required_parameters)

        # Set complexity filter based on preference
        if complexity_preference != ComplexityPreference.ANY:
            if complexity_preference == ComplexityPreference.SIMPLE:
                param_filters.max_complexity = 0.4
            elif complexity_preference == ComplexityPreference.MODERATE:
                param_filters.min_complexity = 0.3
                param_filters.max_complexity = 0.7
            elif complexity_preference == ComplexityPreference.COMPLEX:
                param_filters.min_complexity = 0.6

        # Set parameter name requirements
        if requirements.required_parameters:
            param_filters.required_parameter_names = requirements.required_parameters

        # Apply filters
        return self.advanced_filters.apply_parameter_filters(results, param_filters)

    def _add_parameter_explanations(self, results: List[ToolSearchResult], requirements: ParameterRequirements) -> None:
        """Add parameter matching explanations to search results."""
        for result in results:
            try:
                # Generate parameter match explanation
                match_result = self.parameter_analyzer.match_parameter_requirements(requirements, result.parameters)

                # Add explanation to match reasons
                if match_result.compatibility_explanation:
                    if hasattr(result, "match_reasons"):
                        result.match_reasons.append(f"Parameters: {match_result.compatibility_explanation}")

                # Add parameter score details
                param_scores = self.parameter_ranking.calculate_parameter_match_score(result, requirements)

                if param_scores.overall_parameter_score > 0.7:
                    explanation = "High parameter compatibility"
                elif param_scores.overall_parameter_score > 0.4:
                    explanation = "Moderate parameter compatibility"
                else:
                    explanation = "Limited parameter compatibility"

                if hasattr(result, "match_reasons"):
                    result.match_reasons.append(explanation)

            except Exception as e:
                logger.debug("Error adding parameter explanation for %s: %s", result.tool_id, e)

    def _extract_types_from_description(self, description: str) -> List[str]:
        """Extract likely data types from natural language description."""
        description_lower = description.lower()

        # Type mapping patterns
        type_patterns = {
            "string": ["text", "string", "content", "message", "name"],
            "file": ["file", "path", "document", "image", "video"],
            "number": ["number", "count", "size", "amount", "quantity"],
            "object": ["data", "structure", "object", "information", "metadata"],
            "array": ["list", "collection", "items", "multiple", "series"],
            "boolean": ["flag", "status", "true", "false", "enabled"],
        }

        detected_types = []
        for type_name, patterns in type_patterns.items():
            if any(pattern in description_lower for pattern in patterns):
                detected_types.append(type_name)

        # Default fallback
        if not detected_types:
            detected_types = ["string"]

        return detected_types

    def _create_single_tool_chain(
        self, tool: ToolSearchResult, input_desc: str, output_desc: str, input_types: List[str], output_types: List[str]
    ) -> ToolChainResult:
        """Create a single-tool chain result."""
        step = ToolChainStep(
            tool_id=tool.tool_id,
            tool_name=tool.name,
            description=tool.description,
            input_types=input_types,
            output_types=output_types,
            parameters=tool.parameters,
            compatibility_score=tool.relevance_score,
            step_explanation=f"Direct transformation using {tool.name}",
        )

        return ToolChainResult(
            chain_id=f"single_{tool.tool_id.value}",
            source_description=input_desc,
            target_description=output_desc,
            steps=[step],
            overall_compatibility=tool.relevance_score,
            total_complexity=1,
            execution_feasibility=tool.relevance_score * 0.9,  # Single tool is easier to execute
            chain_explanation=f"Single-step transformation using {tool.name}",
            suggested_parameters=self._extract_suggested_parameters(tool),
        )

    def _find_multi_tool_chains(
        self,
        input_types: List[str],
        output_types: List[str],
        input_desc: str,
        output_desc: str,
        max_length: int,
        k: int,
    ) -> List[ToolChainResult]:
        """Find multi-tool chains for complex transformations."""
        # Simplified implementation - in practice would use more sophisticated chaining algorithms
        chains = []

        try:
            # Find intermediate conversion possibilities
            conversion_chains = self.type_analyzer.find_type_conversion_chain(input_types, output_types)

            for conversion_chain in conversion_chains[:k]:
                if len(conversion_chain.steps) <= max_length:
                    # Build tool chain based on conversion steps
                    tool_chain = self._build_tool_chain_from_conversion(conversion_chain, input_desc, output_desc)
                    if tool_chain:
                        chains.append(tool_chain)

        except Exception as e:
            logger.debug("Error finding multi-tool chains: %s", e)

        return chains

    def _build_tool_chain_from_conversion(
        self, conversion_chain: ConversionChain, input_desc: str, output_desc: str
    ) -> Optional[ToolChainResult]:
        """Build a tool chain from a type conversion chain."""
        # Simplified implementation - would need more sophisticated mapping
        steps = []

        for i, conversion_step in enumerate(conversion_chain.steps):
            # Find tools that can perform this conversion
            tools = self.search_by_parameters(
                input_types=[conversion_step.from_type], output_types=[conversion_step.to_type], k=1
            )

            if tools:
                tool = tools[0]
                step = ToolChainStep(
                    tool_id=tool.tool_id,
                    tool_name=tool.name,
                    description=tool.description,
                    input_types=[conversion_step.from_type],
                    output_types=[conversion_step.to_type],
                    parameters=tool.parameters,
                    compatibility_score=conversion_step.confidence,
                    step_explanation=conversion_step.description,
                )
                steps.append(step)
            else:
                # Chain breaks if no tool found for conversion step
                return None

        if steps:
            return ToolChainResult(
                chain_id=f"chain_{conversion_chain.source_type}_{conversion_chain.target_type}",
                source_description=input_desc,
                target_description=output_desc,
                steps=steps,
                overall_compatibility=conversion_chain.reliability_score,
                total_complexity=len(steps),
                execution_feasibility=conversion_chain.reliability_score * (0.8 ** len(steps)),  # Decreases with length
                chain_explanation=f"Multi-step chain: {' -> '.join([s.tool_name for s in steps])}",
            )

        return None

    def _get_reference_tool_data(self, reference_tool: str) -> Optional[ToolSearchResult]:
        """Get reference tool data for compatibility analysis."""
        # Try to find tool by ID or name
        try:
            # Search for the tool
            search_results = self.tool_search_engine.search_by_functionality(query=reference_tool, k=5)

            # Return the best match
            if search_results.results:
                return search_results.results[0]

        except Exception as e:
            logger.debug("Error getting reference tool data: %s", e)

        return None

    def _find_io_compatible_tools(self, ref_tool: ToolSearchResult, k: int) -> List[ToolCompatibilityResult]:
        """Find tools with compatible input/output types."""
        # Simplified implementation
        compatible_tools = []

        # Extract types from reference tool parameters
        ref_input_types = [p.type for p in ref_tool.parameters if p.required]

        # Find tools with compatible inputs
        if ref_input_types:
            results = self.search_by_parameters(input_types=ref_input_types, k=k)
            compatible_tools.extend(results)

        # Create compatibility result
        if compatible_tools:
            return [
                ToolCompatibilityResult(
                    reference_tool_id=ref_tool.tool_id,
                    compatible_tools=compatible_tools,
                    compatibility_explanation=f"Found {len(compatible_tools)} tools with compatible input types",
                )
            ]

        return []

    def _find_chainable_tools(self, ref_tool: ToolSearchResult, k: int) -> List[ToolCompatibilityResult]:
        """Find tools that can be chained with the reference tool."""
        # Simplified chaining logic
        chainable_tools = []

        # Find tools that could consume the output of ref_tool
        # This is simplified - would need more sophisticated output type inference
        search_results = self.tool_search_engine.search_by_functionality(
            query=f"processes output from {ref_tool.name}", k=k
        )

        chainable_tools.extend(search_results.results)

        if chainable_tools:
            return [
                ToolCompatibilityResult(
                    reference_tool_id=ref_tool.tool_id,
                    compatible_tools=chainable_tools,
                    compatibility_explanation=f"Found {len(chainable_tools)} potentially chainable tools",
                )
            ]

        return []

    def _find_alternative_tools(self, ref_tool: ToolSearchResult, k: int) -> List[ToolCompatibilityResult]:
        """Find alternative tools with similar functionality."""
        # Use the existing alternative finding functionality
        alternatives_response = self.tool_search_engine.get_tool_alternatives(ref_tool.tool_id, k)

        if alternatives_response.results:
            return [
                ToolCompatibilityResult(
                    reference_tool_id=ref_tool.tool_id,
                    compatible_tools=alternatives_response.results,
                    compatibility_explanation=f"Found {len(alternatives_response.results)} alternative tools",
                )
            ]

        return []

    def _extract_suggested_parameters(self, tool: ToolSearchResult) -> Dict[str, Any]:
        """Extract suggested parameter values from tool analysis."""
        suggestions = {}

        for param in tool.parameters:
            if hasattr(param, "default_value") and param.default_value is not None:
                suggestions[param.name] = param.default_value
            elif hasattr(param, "examples") and param.examples:
                suggestions[param.name] = param.examples[0]

        return suggestions

    def _update_search_statistics(self, execution_time: float) -> None:
        """Update search performance statistics."""
        self.search_statistics["total_searches"] += 1

        # Update rolling average execution time
        current_avg = self.search_statistics["average_execution_time"]
        total_searches = self.search_statistics["total_searches"]

        new_avg = ((current_avg * (total_searches - 1)) + execution_time) / total_searches
        self.search_statistics["average_execution_time"] = new_avg

    def get_search_statistics(self) -> Dict[str, Any]:
        """Get search performance statistics."""
        return self.search_statistics.copy()
