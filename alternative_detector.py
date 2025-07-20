#!/usr/bin/env python3
"""
alternative_detector.py: Advanced alternative tool detection system.

This module provides sophisticated algorithms for detecting functionally similar
tools, analyzing their advantages and disadvantages, and grouping tools by
functional similarity.
"""

import hashlib
import itertools
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

from logging_config import get_logger
from mcp_tool_search_engine import MCPToolSearchEngine
from tool_search_results import ToolSearchResult

logger = get_logger(__name__)


@dataclass
class AlternativeAnalysis:
    """Analysis of an alternative tool compared to a reference tool."""

    tool_id: str
    tool_name: str
    similarity_score: float  # Overall functional similarity (0-1)
    functional_overlap: float  # How much functionality overlaps (0-1)

    # Differentiation analysis
    shared_capabilities: List[str]
    unique_capabilities: List[str]  # Capabilities this tool has that reference doesn't
    capability_gaps: List[str]  # Capabilities reference has that this tool doesn't

    # Suitability analysis
    when_to_prefer: List[str]  # Scenarios where this alternative is better
    advantages: List[str]  # Key advantages over reference tool
    disadvantages: List[str]  # Key disadvantages vs reference tool

    # Metadata
    confidence: float  # Confidence in the analysis (0-1)
    complexity_comparison: str  # 'simpler', 'similar', 'more_complex'
    learning_curve: str  # 'easy', 'moderate', 'difficult'

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass
class AlternativeAdvantageAnalysis:
    """Analysis of specific advantages between two alternatives."""

    primary_tool: str
    alternative_tool: str

    # Capability comparison
    shared_capabilities: List[str]
    primary_unique_capabilities: List[str]
    alternative_unique_capabilities: List[str]

    # Contextual analysis
    primary_advantages: List[str]
    alternative_advantages: List[str]
    use_case_fit: Dict[str, str]  # use_case -> better_tool

    # Switching analysis
    switching_cost: str  # 'low', 'medium', 'high'
    migration_considerations: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass
class FunctionalGroup:
    """A group of tools with similar functionality."""

    group_name: str
    core_functionality: List[str]
    tools: List[str]
    similarity_threshold: float
    group_characteristics: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


class SimilarityAnalyzer:
    """Analyzes functional similarity between tools."""

    def __init__(self):
        """Initialize the similarity analyzer."""
        self.capability_cache = {}
        self.similarity_cache = {}

    def calculate_functional_similarity(self, tool_a: str, tool_b: str) -> float:
        """
        Calculate functional similarity between two tools.

        Args:
            tool_a: First tool identifier
            tool_b: Second tool identifier

        Returns:
            Similarity score (0-1)
        """
        # Use cache if available
        cache_key = tuple(sorted([tool_a, tool_b]))
        if cache_key in self.similarity_cache:
            return self.similarity_cache[cache_key]

        # Mock implementation - would use actual tool analysis
        similarity = self._calculate_mock_similarity(tool_a, tool_b)

        self.similarity_cache[cache_key] = similarity
        return similarity

    def _calculate_mock_similarity(self, tool_a: str, tool_b: str) -> float:
        """Mock similarity calculation for testing."""
        # Simple heuristic based on name similarity and category
        name_similarity = self._calculate_name_similarity(tool_a, tool_b)

        # Category-based similarity
        category_similarity = self._infer_category_similarity(tool_a, tool_b)

        return name_similarity * 0.3 + category_similarity * 0.7

    def _calculate_name_similarity(self, name_a: str, name_b: str) -> float:
        """Calculate similarity based on tool names."""
        # Simple Jaccard similarity on characters
        set_a = set(name_a.lower())
        set_b = set(name_b.lower())

        intersection = len(set_a & set_b)
        union = len(set_a | set_b)

        return intersection / union if union > 0 else 0.0

    def _infer_category_similarity(self, tool_a: str, tool_b: str) -> float:
        """Infer category similarity based on tool names."""
        # Mock categorization
        file_ops = {"read", "write", "edit", "multiedit", "file"}
        search_ops = {"grep", "glob", "search", "find"}
        exec_ops = {"bash", "execute", "run", "task"}

        def get_category(tool_name: str):
            tool_lower = tool_name.lower()
            if any(op in tool_lower for op in file_ops):
                return "file_operations"
            elif any(op in tool_lower for op in search_ops):
                return "search_operations"
            elif any(op in tool_lower for op in exec_ops):
                return "execution_operations"
            return "other"

        cat_a = get_category(tool_a)
        cat_b = get_category(tool_b)

        return 0.8 if cat_a == cat_b else 0.2


class FunctionalAnalyzer:
    """Analyzes functional capabilities of tools."""

    def __init__(self):
        """Initialize the functional analyzer."""
        self.capability_patterns = self._load_capability_patterns()

    def extract_capabilities(self, tool: ToolSearchResult) -> List[str]:
        """
        Extract capabilities from tool metadata.

        Args:
            tool: Tool to analyze

        Returns:
            List of capability strings
        """
        capabilities = []

        # Extract from description
        if tool.description:
            capabilities.extend(self._extract_from_description(tool.description))

        # Extract from parameters
        capabilities.extend(self._extract_from_parameters(tool.parameters))

        # Extract from tool name
        capabilities.extend(self._extract_from_name(tool.name))

        return list(set(capabilities))  # Remove duplicates

    def _extract_from_description(self, description: str) -> List[str]:
        """Extract capabilities from tool description."""
        capabilities = []
        desc_lower = description.lower()

        # File operations
        if any(word in desc_lower for word in ["read", "reads", "reading"]):
            capabilities.append("file_reading")
        if any(word in desc_lower for word in ["write", "writes", "writing"]):
            capabilities.append("file_writing")
        if any(word in desc_lower for word in ["edit", "modify", "update"]):
            capabilities.append("content_modification")

        # Search operations
        if any(word in desc_lower for word in ["search", "find", "locate"]):
            capabilities.append("content_search")
        if any(word in desc_lower for word in ["pattern", "regex", "match"]):
            capabilities.append("pattern_matching")

        # Data operations
        if any(word in desc_lower for word in ["process", "transform", "convert"]):
            capabilities.append("data_processing")
        if any(word in desc_lower for word in ["analyze", "analysis"]):
            capabilities.append("data_analysis")

        # Execution operations
        if any(word in desc_lower for word in ["execute", "run", "launch"]):
            capabilities.append("command_execution")

        return capabilities

    def _extract_from_parameters(self, parameters: List[Any]) -> List[str]:
        """Extract capabilities from parameter structure."""
        capabilities = []

        if not parameters:
            return capabilities

        param_names = [p.name.lower() for p in parameters if hasattr(p, "name")]

        # Input/output capabilities
        if any("file" in name or "path" in name for name in param_names):
            capabilities.append("file_operations")
        if any("output" in name for name in param_names):
            capabilities.append("output_generation")
        if any("input" in name for name in param_names):
            capabilities.append("input_processing")

        # Configuration capabilities
        if any("config" in name or "option" in name for name in param_names):
            capabilities.append("configurable_behavior")
        if any("format" in name for name in param_names):
            capabilities.append("format_control")

        return capabilities

    def _extract_from_name(self, name: str) -> List[str]:
        """Extract capabilities from tool name."""
        capabilities = []
        name_lower = name.lower()

        # Direct capability mapping
        capability_map = {
            "read": ["file_reading", "content_access"],
            "write": ["file_writing", "content_creation"],
            "edit": ["content_modification", "file_editing"],
            "search": ["content_search", "pattern_matching"],
            "grep": ["content_search", "pattern_matching"],
            "glob": ["file_discovery", "pattern_matching"],
            "bash": ["command_execution", "shell_operations"],
            "task": ["task_execution", "workflow_control"],
        }

        for keyword, caps in capability_map.items():
            if keyword in name_lower:
                capabilities.extend(caps)

        return capabilities

    def _load_capability_patterns(self) -> Dict[str, List[str]]:
        """Load patterns for capability detection."""
        return {
            "file_operations": ["file", "path", "directory", "folder"],
            "text_processing": ["text", "string", "content", "data"],
            "search_operations": ["search", "find", "query", "match"],
            "transformation": ["transform", "convert", "process", "modify"],
        }


class AlternativeDetector:
    """Advanced system for detecting and analyzing tool alternatives."""

    def __init__(
        self, tool_search_engine: MCPToolSearchEngine, similarity_analyzer: Optional[SimilarityAnalyzer] = None
    ):
        """
        Initialize the alternative detector.

        Args:
            tool_search_engine: Engine for searching similar tools
            similarity_analyzer: Optional custom similarity analyzer
        """
        self.tool_search_engine = tool_search_engine
        self.similarity_analyzer = similarity_analyzer or SimilarityAnalyzer()
        self.functional_analyzer = FunctionalAnalyzer()

        # Caches for performance
        self.alternative_cache = {}
        self.capability_cache = {}

        logger.info("Alternative detector initialized")

    def find_alternatives(
        self, reference_tool: str, similarity_threshold: float = 0.7, max_alternatives: int = 10
    ) -> List[AlternativeAnalysis]:
        """
        Find tools that serve similar functions to the reference tool.

        Args:
            reference_tool: Tool to find alternatives for
            similarity_threshold: Minimum similarity score (0-1)
            max_alternatives: Maximum number of alternatives to return

        Returns:
            List of AlternativeAnalysis objects, sorted by similarity
        """
        try:
            logger.info(f"Finding alternatives for '{reference_tool}' with threshold {similarity_threshold}")

            # Check cache first
            cache_key = self._generate_cache_key(reference_tool, similarity_threshold, max_alternatives)
            if cache_key in self.alternative_cache:
                logger.debug("Returning cached alternatives")
                return self.alternative_cache[cache_key]

            # Step 1: Get reference tool information
            reference_info = self._get_tool_info(reference_tool)
            if not reference_info:
                logger.warning(f"Could not find information for reference tool: {reference_tool}")
                return []

            # Step 2: Search for potentially similar tools
            candidates = self._find_candidate_alternatives(reference_tool, max_alternatives * 3)

            # Step 3: Analyze each candidate
            alternatives = []
            reference_capabilities = self._get_tool_capabilities(reference_tool)

            for candidate in candidates:
                if str(candidate.tool_id) == reference_tool:
                    continue  # Skip self

                try:
                    analysis = self._analyze_alternative(
                        reference_tool, reference_info, reference_capabilities, candidate
                    )

                    if analysis.similarity_score >= similarity_threshold:
                        alternatives.append(analysis)

                except Exception as e:
                    logger.warning(f"Error analyzing candidate {candidate.tool_id}: {e}")
                    continue

            # Step 4: Sort by similarity and limit results
            alternatives.sort(key=lambda a: a.similarity_score, reverse=True)
            alternatives = alternatives[:max_alternatives]

            # Cache results
            self.alternative_cache[cache_key] = alternatives

            logger.info(f"Found {len(alternatives)} alternatives for '{reference_tool}'")
            return alternatives

        except Exception as e:
            logger.error(f"Error finding alternatives for '{reference_tool}': {e}")
            return []

    def detect_functional_groups(self, all_tools: List[str]) -> Dict[str, List[str]]:
        """
        Group tools by functional similarity.

        Args:
            all_tools: List of tool identifiers to group

        Returns:
            Dictionary mapping group names to lists of tool IDs
        """
        try:
            logger.info(f"Detecting functional groups for {len(all_tools)} tools")

            # Calculate similarity matrix
            similarity_matrix = self._build_similarity_matrix(all_tools)

            # Perform clustering
            groups = self._cluster_by_similarity(all_tools, similarity_matrix, threshold=0.6)

            # Name the groups based on their tools
            named_groups = {}
            for i, group_tools in enumerate(groups):
                group_name = self._generate_group_name(group_tools)
                named_groups[group_name] = group_tools

            logger.info(f"Detected {len(named_groups)} functional groups")
            return named_groups

        except Exception as e:
            logger.error(f"Error detecting functional groups: {e}")
            return {}

    def analyze_alternative_advantages(self, primary_tool: str, alternative_tool: str) -> AlternativeAdvantageAnalysis:
        """
        Analyze specific advantages of each tool in comparison.

        Args:
            primary_tool: First tool for comparison
            alternative_tool: Second tool for comparison

        Returns:
            AlternativeAdvantageAnalysis with detailed comparison
        """
        try:
            logger.info(f"Analyzing advantages: {primary_tool} vs {alternative_tool}")

            # Get capabilities for both tools
            primary_capabilities = self._get_tool_capabilities(primary_tool)
            alternative_capabilities = self._get_tool_capabilities(alternative_tool)

            # Find shared and unique capabilities
            shared = list(set(primary_capabilities) & set(alternative_capabilities))
            primary_unique = list(set(primary_capabilities) - set(alternative_capabilities))
            alternative_unique = list(set(alternative_capabilities) - set(primary_capabilities))

            # Analyze advantages
            primary_advantages = self._analyze_tool_advantages(primary_tool, primary_unique)
            alternative_advantages = self._analyze_tool_advantages(alternative_tool, alternative_unique)

            # Use case fit analysis
            use_case_fit = self._analyze_use_case_fit(
                primary_tool, alternative_tool, primary_capabilities, alternative_capabilities
            )

            # Switching cost analysis
            switching_cost, migration_considerations = self._analyze_switching_cost(primary_tool, alternative_tool)

            analysis = AlternativeAdvantageAnalysis(
                primary_tool=primary_tool,
                alternative_tool=alternative_tool,
                shared_capabilities=shared,
                primary_unique_capabilities=primary_unique,
                alternative_unique_capabilities=alternative_unique,
                primary_advantages=primary_advantages,
                alternative_advantages=alternative_advantages,
                use_case_fit=use_case_fit,
                switching_cost=switching_cost,
                migration_considerations=migration_considerations,
            )

            logger.info("Alternative advantage analysis completed")
            return analysis

        except Exception as e:
            logger.error(f"Error analyzing alternative advantages: {e}")
            # Return minimal analysis
            return AlternativeAdvantageAnalysis(
                primary_tool=primary_tool,
                alternative_tool=alternative_tool,
                shared_capabilities=[],
                primary_unique_capabilities=[],
                alternative_unique_capabilities=[],
                primary_advantages=[],
                alternative_advantages=[],
                use_case_fit={},
                switching_cost="unknown",
                migration_considerations=[],
            )

    # Helper methods

    def _get_tool_info(self, tool_id: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive tool information."""
        try:
            # This would integrate with the existing database/search system
            # For now, return mock data structure
            return {
                "id": tool_id,
                "name": tool_id,
                "description": f"Mock description for {tool_id}",
                "category": self._infer_category(tool_id),
                "parameters": [],
            }
        except Exception as e:
            logger.warning(f"Could not get info for tool {tool_id}: {e}")
            return None

    def _find_candidate_alternatives(self, reference_tool: str, max_candidates: int) -> List[ToolSearchResult]:
        """Find potential alternative tools using search."""
        try:
            # Use existing search engine to find similar tools
            search_query = f"alternative to {reference_tool} similar functionality"
            search_response = self.tool_search_engine.search_by_functionality(query=search_query, k=max_candidates)

            return search_response.results

        except Exception as e:
            logger.warning(f"Error finding candidates for {reference_tool}: {e}")
            return []

    def _get_tool_capabilities(self, tool_id: str) -> List[str]:
        """Get cached capabilities for a tool."""
        if tool_id in self.capability_cache:
            return self.capability_cache[tool_id]

        # Mock capability extraction
        capabilities = self._extract_mock_capabilities(tool_id)
        self.capability_cache[tool_id] = capabilities
        return capabilities

    def _extract_mock_capabilities(self, tool_id: str) -> List[str]:
        """Extract mock capabilities for testing."""
        return self._get_capability_mapping().get(tool_id, ["general_functionality"])

    def _get_capability_mapping(self) -> Dict[str, List[str]]:
        """Get the capability mapping for tools."""
        return {
            "read": ["file_reading", "content_access", "text_display"],
            "write": ["file_writing", "content_creation", "output_generation"],
            "edit": ["file_editing", "content_modification", "text_replacement"],
            "multiedit": ["batch_editing", "multiple_file_operations", "content_modification"],
            "grep": ["content_search", "pattern_matching", "text_filtering"],
            "glob": ["file_discovery", "path_matching", "file_listing"],
            "bash": ["command_execution", "shell_operations", "system_interaction"],
            "task": ["task_execution", "workflow_control", "process_management"],
        }

    def _analyze_alternative(
        self,
        reference_tool: str,
        reference_info: Dict[str, Any],
        reference_capabilities: List[str],
        candidate: ToolSearchResult,
    ) -> AlternativeAnalysis:
        """Analyze a candidate alternative tool."""
        candidate_id = str(candidate.tool_id)
        candidate_capabilities = self._get_tool_capabilities(candidate_id)

        # Calculate similarity scores
        similarity_score = self.similarity_analyzer.calculate_functional_similarity(reference_tool, candidate_id)

        # Calculate functional overlap
        shared_caps = set(reference_capabilities) & set(candidate_capabilities)
        total_caps = set(reference_capabilities) | set(candidate_capabilities)
        functional_overlap = len(shared_caps) / len(total_caps) if total_caps else 0.0

        # Analyze capabilities
        unique_capabilities = list(set(candidate_capabilities) - set(reference_capabilities))
        capability_gaps = list(set(reference_capabilities) - set(candidate_capabilities))
        shared_capabilities = list(shared_caps)

        # Generate advantages and scenarios
        advantages = self._generate_advantages(candidate, unique_capabilities)
        disadvantages = self._generate_disadvantages(candidate, capability_gaps)
        when_to_prefer = self._generate_preference_scenarios(candidate, unique_capabilities)

        # Assess complexity and learning curve
        complexity_comparison = self._compare_complexity(reference_tool, candidate)
        learning_curve = self._assess_learning_curve(candidate)

        return AlternativeAnalysis(
            tool_id=candidate_id,
            tool_name=candidate.name,
            similarity_score=similarity_score,
            functional_overlap=functional_overlap,
            shared_capabilities=shared_capabilities,
            unique_capabilities=unique_capabilities,
            capability_gaps=capability_gaps,
            when_to_prefer=when_to_prefer,
            advantages=advantages,
            disadvantages=disadvantages,
            confidence=min(similarity_score + 0.1, 1.0),
            complexity_comparison=complexity_comparison,
            learning_curve=learning_curve,
        )

    def _generate_advantages(self, tool: ToolSearchResult, unique_capabilities: List[str]) -> List[str]:
        """Generate advantage descriptions."""
        advantages = []

        # Based on unique capabilities
        capability_advantages = self._get_capability_advantage_descriptions()
        for capability in unique_capabilities:
            if capability in capability_advantages:
                advantages.append(capability_advantages[capability])

        # Based on tool characteristics
        if tool.parameter_count and tool.parameter_count <= 3:
            advantages.append("Simple interface with few parameters")
        elif tool.parameter_count and tool.parameter_count > 5:
            advantages.append("Highly configurable with many options")

        if not advantages:
            advantages.append("Provides alternative approach to the task")

        return advantages

    def _generate_disadvantages(self, tool: ToolSearchResult, capability_gaps: List[str]) -> List[str]:
        """Generate disadvantage descriptions."""
        disadvantages = []

        # Based on capability gaps
        capability_disadvantages = self._get_capability_disadvantage_descriptions()
        for gap in capability_gaps:
            if gap in capability_disadvantages:
                disadvantages.append(capability_disadvantages[gap])

        # Based on complexity
        if tool.parameter_count and tool.parameter_count > 6:
            disadvantages.append("More complex parameter structure")

        if not disadvantages and capability_gaps:
            disadvantages.append("Missing some capabilities of the reference tool")

        return disadvantages

    def _generate_preference_scenarios(self, tool: ToolSearchResult, unique_capabilities: List[str]) -> List[str]:
        """Generate scenarios when to prefer this alternative."""
        scenarios = []

        capability_scenarios = self._get_capability_scenario_descriptions()
        for capability in unique_capabilities:
            if capability in capability_scenarios:
                scenarios.append(capability_scenarios[capability])

        # Based on tool name patterns
        name_patterns = self._get_name_pattern_scenarios()
        for pattern, scenario in name_patterns.items():
            if pattern in tool.name.lower():
                scenarios.append(scenario)

        if not scenarios:
            scenarios.append("When the reference tool doesn't meet specific needs")

        return scenarios

    def _get_capability_advantage_descriptions(self) -> Dict[str, str]:
        """Get mapping of capabilities to advantage descriptions."""
        return {
            "batch_editing": "Supports batch operations on multiple files",
            "pattern_matching": "Advanced pattern matching capabilities",
            "workflow_control": "Better workflow integration and control",
            "system_interaction": "Direct system command execution",
            "content_search": "Powerful content search capabilities",
            "file_discovery": "Advanced file discovery and listing",
            "multiple_file_operations": "Can handle multiple files simultaneously",
        }

    def _get_capability_disadvantage_descriptions(self) -> Dict[str, str]:
        """Get mapping of capability gaps to disadvantage descriptions."""
        return {
            "file_writing": "Cannot modify files directly",
            "pattern_matching": "Limited pattern matching capabilities",
            "batch_editing": "No batch operation support",
            "system_interaction": "Cannot execute system commands",
            "content_creation": "Limited content creation abilities",
            "file_discovery": "Basic file discovery features",
        }

    def _get_capability_scenario_descriptions(self) -> Dict[str, str]:
        """Get mapping of capabilities to preference scenario descriptions."""
        return {
            "batch_editing": "When editing multiple files simultaneously",
            "pattern_matching": "When complex pattern matching is required",
            "workflow_control": "When workflow integration is important",
            "system_interaction": "When system command execution is needed",
            "content_search": "When advanced search capabilities are required",
            "file_discovery": "When comprehensive file discovery is needed",
        }

    def _get_name_pattern_scenarios(self) -> Dict[str, str]:
        """Get mapping of name patterns to scenarios."""
        return {
            "multi": "When handling multiple items at once",
            "advanced": "When advanced features are needed",
            "batch": "When batch processing is required",
            "bulk": "When bulk operations are necessary",
        }

    def _compare_complexity(self, reference_tool: str, candidate: ToolSearchResult) -> str:
        """Compare complexity between reference and candidate."""
        # Simple heuristic based on parameter count
        ref_complexity = self._estimate_tool_complexity(reference_tool)
        candidate_complexity = candidate.parameter_count if candidate.parameter_count else 2

        if candidate_complexity < ref_complexity * 0.8:
            return "simpler"
        elif candidate_complexity > ref_complexity * 1.2:
            return "more_complex"
        else:
            return "similar"

    def _estimate_tool_complexity(self, tool_id: str) -> int:
        """Estimate tool complexity (mock implementation)."""
        # Mock complexity mapping
        complexity_map = {"read": 2, "write": 3, "edit": 4, "multiedit": 6, "grep": 5, "glob": 3, "bash": 8, "task": 7}
        return complexity_map.get(tool_id, 4)

    def _assess_learning_curve(self, tool: ToolSearchResult) -> str:
        """Assess learning curve difficulty."""
        # Simple heuristic
        if tool.parameter_count and tool.parameter_count <= 2:
            return "easy"
        elif tool.parameter_count and tool.parameter_count <= 5:
            return "moderate"
        else:
            return "difficult"

    def _build_similarity_matrix(self, tools: List[str]) -> Dict[Tuple[str, str], float]:
        """Build similarity matrix for all tool pairs."""
        matrix = {}

        for tool_a, tool_b in itertools.combinations(tools, 2):
            similarity = self.similarity_analyzer.calculate_functional_similarity(tool_a, tool_b)
            matrix[(tool_a, tool_b)] = similarity
            matrix[(tool_b, tool_a)] = similarity  # Symmetric

        return matrix

    def _cluster_by_similarity(
        self, tools: List[str], similarity_matrix: Dict[Tuple[str, str], float], threshold: float
    ) -> List[List[str]]:
        """Cluster tools by similarity using simple threshold-based clustering."""
        # Simple clustering algorithm
        clusters = []
        assigned = set()

        for tool in tools:
            if tool in assigned:
                continue

            # Start new cluster
            cluster = [tool]
            assigned.add(tool)

            # Find similar tools to add to cluster
            for other_tool in tools:
                if other_tool in assigned:
                    continue

                similarity = similarity_matrix.get((tool, other_tool), 0.0)
                if similarity >= threshold:
                    cluster.append(other_tool)
                    assigned.add(other_tool)

            clusters.append(cluster)

        return clusters

    def _generate_group_name(self, group_tools: List[str]) -> str:
        """Generate a descriptive name for a functional group."""
        if not group_tools:
            return "empty_group"

        # Look for common patterns
        if all(any(op in tool.lower() for op in ["read", "write", "edit"]) for tool in group_tools):
            return "file_operations"
        elif all(any(op in tool.lower() for op in ["search", "find", "grep", "glob"]) for tool in group_tools):
            return "search_operations"
        elif all(any(op in tool.lower() for op in ["bash", "execute", "run", "task"]) for tool in group_tools):
            return "execution_operations"
        else:
            # Use first tool as representative
            return f"{group_tools[0]}_group"

    def _analyze_tool_advantages(self, tool_id: str, unique_capabilities: List[str]) -> List[str]:
        """Analyze advantages of a tool based on its unique capabilities."""
        advantages = []

        for capability in unique_capabilities:
            if capability == "batch_editing":
                advantages.append(f"{tool_id} excels at batch operations")
            elif capability == "pattern_matching":
                advantages.append(f"{tool_id} offers superior pattern matching")
            elif capability == "workflow_control":
                advantages.append(f"{tool_id} provides better workflow integration")
            else:
                advantages.append(f"{tool_id} offers {capability.replace('_', ' ')}")

        return advantages

    def _analyze_use_case_fit(
        self,
        primary_tool: str,
        alternative_tool: str,
        primary_capabilities: List[str],
        alternative_capabilities: List[str],
    ) -> Dict[str, str]:
        """Analyze which tool fits better for different use cases."""
        use_case_fit = {}

        # File operations
        if "file_writing" in primary_capabilities and "file_writing" not in alternative_capabilities:
            use_case_fit["file_modification"] = primary_tool
        elif "file_writing" in alternative_capabilities and "file_writing" not in primary_capabilities:
            use_case_fit["file_modification"] = alternative_tool

        # Batch operations
        if "batch_editing" in primary_capabilities:
            use_case_fit["multiple_files"] = primary_tool
        elif "batch_editing" in alternative_capabilities:
            use_case_fit["multiple_files"] = alternative_tool

        # Search operations
        if "pattern_matching" in primary_capabilities and len(primary_capabilities) > len(alternative_capabilities):
            use_case_fit["complex_search"] = primary_tool
        elif "pattern_matching" in alternative_capabilities and len(alternative_capabilities) > len(
            primary_capabilities
        ):
            use_case_fit["complex_search"] = alternative_tool

        return use_case_fit

    def _analyze_switching_cost(self, primary_tool: str, alternative_tool: str) -> Tuple[str, List[str]]:
        """Analyze the cost and considerations for switching between tools."""
        # Simple heuristic based on tool similarity
        similarity = self.similarity_analyzer.calculate_functional_similarity(primary_tool, alternative_tool)

        if similarity >= 0.8:
            cost = "low"
            considerations = ["Similar interface and parameters", "Minimal learning curve required"]
        elif similarity >= 0.6:
            cost = "medium"
            considerations = [
                "Some interface differences to learn",
                "May need to adjust workflows",
                "Parameter mapping required",
            ]
        else:
            cost = "high"
            considerations = [
                "Significant interface differences",
                "Substantial learning curve",
                "Major workflow adjustments needed",
                "Thorough testing recommended",
            ]

        return cost, considerations

    def _infer_category(self, tool_id: str) -> str:
        """Infer tool category from ID."""
        tool_lower = tool_id.lower()

        if any(op in tool_lower for op in ["read", "write", "edit"]):
            return "file_operations"
        elif any(op in tool_lower for op in ["search", "find", "grep", "glob"]):
            return "search_operations"
        elif any(op in tool_lower for op in ["bash", "execute", "run", "task"]):
            return "execution_operations"
        else:
            return "general"

    def _generate_cache_key(self, reference_tool: str, threshold: float, max_alternatives: int) -> str:
        """Generate cache key for alternatives lookup."""
        key_string = f"{reference_tool}_{threshold}_{max_alternatives}"
        return hashlib.md5(key_string.encode()).hexdigest()
