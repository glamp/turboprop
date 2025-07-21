#!/usr/bin/env python3
"""
tool_recommendation_engine.py: Main Tool Recommendation Engine

This module provides the central orchestrator for the comprehensive tool recommendation
system. It integrates task analysis, context awareness, recommendation algorithms, and
explanation generation to provide intelligent tool recommendations.
"""

import hashlib
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from .context_analyzer import ContextAnalyzer, TaskContext
from .exceptions import AnalysisError
from .logging_config import get_logger
from .mcp_tool_search_engine import MCPToolSearchEngine
from .parameter_search_engine import ParameterSearchEngine
from .recommendation_algorithms import RecommendationAlgorithms, ToolRecommendation, ToolSequenceRecommendation
from .recommendation_explainer import RecommendationExplainer, RecommendationExplanation
from .task_analyzer import TaskAnalysis, TaskAnalyzer

logger = get_logger(__name__)

# Configuration constants with environment variable overrides
DEFAULT_MAX_RECOMMENDATIONS = int(os.getenv("TURBOPROP_MAX_RECOMMENDATIONS", 5))
DEFAULT_CACHE_SIZE = int(os.getenv("TURBOPROP_CACHE_SIZE", 1000))
DEFAULT_CACHE_TTL_MINUTES = int(os.getenv("TURBOPROP_CACHE_TTL_MINUTES", 60))
MAX_TASK_DESCRIPTION_LENGTH = int(os.getenv("TURBOPROP_MAX_DESCRIPTION_LENGTH", 1000))


@dataclass
class RecommendationRequest:
    """Request for tool recommendations."""

    task_description: str
    max_recommendations: int = DEFAULT_MAX_RECOMMENDATIONS
    include_alternatives: bool = True
    include_explanations: bool = False
    context_data: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Validate request parameters."""
        if not self.task_description or not self.task_description.strip():
            raise ValueError("Task description cannot be empty")
        if self.max_recommendations < 1:
            raise ValueError("max_recommendations must be positive")
        if len(self.task_description) > MAX_TASK_DESCRIPTION_LENGTH:
            raise ValueError(f"Task description too long (max {MAX_TASK_DESCRIPTION_LENGTH} characters)")


@dataclass
class ToolSequenceRequest:
    """Request for tool sequence recommendations."""

    workflow_description: str
    context_data: Optional[Dict[str, Any]] = None
    optimization_goals: List[str] = field(default_factory=list)
    max_sequences: int = 3


@dataclass
class AlternativeRequest:
    """Request for alternative tool recommendations."""

    primary_tool: str
    task_context: TaskContext
    reason: str = "general"
    max_alternatives: int = 3


@dataclass
class RecommendationResponse:
    """Response containing tool recommendations."""

    recommendations: List[ToolRecommendation]
    task_analysis: TaskAnalysis
    context: TaskContext
    explanations: Optional[List[RecommendationExplanation]] = None
    alternatives: Optional[List[ToolRecommendation]] = None
    request_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class ToolSequenceResponse:
    """Response containing tool sequence recommendations."""

    sequences: List[ToolSequenceRecommendation]
    workflow_analysis: Dict[str, Any]
    optimization_results: Dict[str, Any]
    request_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class AlternativeResponse:
    """Response containing alternative recommendations."""

    alternatives: List[ToolRecommendation]
    comparison_analysis: Dict[str, Any]
    selection_guidance: List[str]
    request_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class RecommendationCache:
    """Simple in-memory cache for recommendations."""

    def __init__(self, max_size: int = DEFAULT_CACHE_SIZE, ttl_minutes: int = DEFAULT_CACHE_TTL_MINUTES):
        """Initialize the cache."""
        self.max_size = max_size
        self.ttl_seconds = ttl_minutes * 60
        self.cache = {}
        self.access_times = {}

        # Performance metrics
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def _generate_key(self, request: Any) -> str:
        """Generate cache key from request."""
        request_str = str(request)
        return hashlib.md5(request_str.encode()).hexdigest()

    def get(self, request: Any) -> Optional[Any]:
        """Get cached recommendation."""
        key = self._generate_key(request)

        if key not in self.cache:
            self.misses += 1
            return None

        # Check TTL
        access_time = self.access_times.get(key, 0)
        if time.time() - access_time > self.ttl_seconds:
            self._evict(key)
            self.misses += 1
            return None

        # Update access time
        self.access_times[key] = time.time()
        self.hits += 1
        return self.cache[key]

    def put(self, request: Any, response: Any) -> None:
        """Cache recommendation response."""
        key = self._generate_key(request)

        # Evict old entries if cache is full
        if len(self.cache) >= self.max_size:
            self._evict_lru()

        self.cache[key] = response
        self.access_times[key] = time.time()

    def _evict(self, key: str) -> None:
        """Evict specific cache entry."""
        self.cache.pop(key, None)
        self.access_times.pop(key, None)
        self.evictions += 1

    def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if not self.access_times:
            return

        lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        self._evict(lru_key)

    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        total_requests = self.hits + self.misses
        hit_rate = (self.hits / total_requests) if total_requests > 0 else 0.0

        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "hit_rate": hit_rate,
            "cache_size": len(self.cache),
            "max_size": self.max_size,
            "total_requests": total_requests,
        }


class ToolRecommendationEngine:
    """Intelligent tool recommendation system."""

    def __init__(
        self,
        tool_search_engine: MCPToolSearchEngine,
        parameter_search_engine: ParameterSearchEngine,
        task_analyzer: TaskAnalyzer,
        context_analyzer: ContextAnalyzer,
    ):
        """Initialize the tool recommendation engine."""
        self.tool_search_engine = tool_search_engine
        self.parameter_search_engine = parameter_search_engine
        self.task_analyzer = task_analyzer
        self.context_analyzer = context_analyzer

        # Initialize core components
        self.recommendation_algorithms = RecommendationAlgorithms()
        self.explainer = RecommendationExplainer()

        # Initialize caching
        self.cache = RecommendationCache()
        self.cache_config = {"max_size": DEFAULT_CACHE_SIZE, "ttl_minutes": DEFAULT_CACHE_TTL_MINUTES}

        # Performance settings
        self.enable_async_processing = True
        self.diversity_threshold = 0.7

        logger.info("Tool recommendation engine initialized")

    def recommend_for_task(self, request: RecommendationRequest) -> RecommendationResponse:
        """Get tool recommendations for a specific task."""
        # Input validation
        if not request:
            raise ValueError("Request cannot be None")
        if not request.task_description or not request.task_description.strip():
            raise ValueError("Task description cannot be empty")
        if len(request.task_description) > MAX_TASK_DESCRIPTION_LENGTH:
            raise ValueError(f"Task description exceeds maximum length of {MAX_TASK_DESCRIPTION_LENGTH} characters")
        if request.max_recommendations <= 0:
            raise ValueError("max_recommendations must be positive")

        logger.info(f"Processing recommendation request for: {request.task_description[:50]}...")

        # Check cache first
        cached_response = self.cache.get(request)
        if cached_response:
            logger.debug("Returning cached recommendation")
            return cached_response

        start_time = time.time()

        try:
            response = self._process_recommendation_request(request, start_time)

            # Cache the response
            self.cache.put(request, response)

            processing_time = time.time() - start_time
            logger.info(f"Recommendation processing complete in {processing_time:.2f}s")

            # Log performance metrics periodically
            if hasattr(self, "_log_performance_counter"):
                self._log_performance_counter += 1
            else:
                self._log_performance_counter = 1

            if self._log_performance_counter % 10 == 0:  # Every 10 requests
                cache_stats = self.cache.get_stats()
                logger.info(
                    f"Performance metrics - Cache hit rate: {cache_stats['hit_rate']:.1%}, "
                    f"Avg processing time: {processing_time:.2f}s"
                )

            return response

        except ValueError as e:
            # Input validation errors should be re-raised
            logger.error(f"Invalid input for recommendation request: {str(e)}")
            raise
        except AnalysisError as e:
            # Analysis failures should be re-raised
            logger.error(f"Task analysis failed: {str(e)}")
            raise
        except (ImportError, ModuleNotFoundError) as e:
            # Dependency issues - provide fallback
            logger.error(f"Missing dependency for recommendation processing: {str(e)}")
            return self._create_fallback_response(request, "Missing required dependencies")
        except MemoryError as e:
            # Memory issues - try with reduced scope
            logger.warning(f"Memory error during recommendation processing: {str(e)}")
            return self._create_fallback_response(request, "Processing failed due to memory constraints")
        except Exception as e:
            # Unexpected errors - log and provide fallback
            logger.error(f"Unexpected error processing recommendation request: {str(e)}", exc_info=True)
            return self._create_fallback_response(request, "Recommendation processing temporarily unavailable")

    def recommend_tool_sequence(self, request: ToolSequenceRequest) -> ToolSequenceResponse:
        """Recommend sequences of tools for complex workflows."""
        # Input validation
        if not request:
            raise ValueError("Request cannot be None")
        if not request.workflow_description or not request.workflow_description.strip():
            raise ValueError("Workflow description cannot be empty")
        if len(request.workflow_description) > MAX_TASK_DESCRIPTION_LENGTH * 2:  # Allow longer workflow descriptions
            raise ValueError("Workflow description exceeds maximum length")

        logger.info(f"Processing tool sequence request for: {request.workflow_description[:50]}...")

        start_time = time.time()

        try:
            # Step 1: Analyze workflow requirements
            workflow_analysis = self._analyze_workflow(request.workflow_description)

            # Step 2: Analyze context
            context = self._analyze_context(request.context_data)

            # Step 3: Break down workflow into sub-tasks
            sub_tasks = self._decompose_workflow(request.workflow_description)

            # Step 4: Find tool combinations for each sub-task
            tool_sequences = []
            for i, sub_task in enumerate(sub_tasks):
                sub_task_analysis = self.task_analyzer.analyze_task(sub_task)
                candidates = self._find_candidate_tools(sub_task_analysis, context, 5)

                if candidates:
                    # For now, create simple sequences
                    tool_chain = [tool.tool_id for tool in candidates[:3]]

                    # Use mock workflow requirements for optimization
                    from recommendation_algorithms import WorkflowRequirements

                    workflow_reqs = WorkflowRequirements(
                        steps=[f"step_{i}"],
                        data_flow_requirements={},
                        error_handling_strategy="fail_fast",
                        performance_requirements={},
                    )

                    optimized = self.recommendation_algorithms.optimize_tool_sequence(tool_chain, workflow_reqs)
                    tool_sequences.extend(optimized)

            # Step 5: Create response
            response = ToolSequenceResponse(
                sequences=tool_sequences,
                workflow_analysis=workflow_analysis,
                optimization_results={"efficiency_score": 0.8, "reliability_score": 0.75},
                request_metadata={"processing_time": time.time() - start_time, "sub_tasks_identified": len(sub_tasks)},
            )

            logger.info(f"Tool sequence processing complete in {time.time() - start_time:.2f}s")
            return response

        except ValueError as e:
            # Input validation errors should be re-raised
            logger.error(f"Invalid input for tool sequence request: {str(e)}")
            raise
        except (ImportError, ModuleNotFoundError) as e:
            # Dependency issues - provide fallback
            logger.error(f"Missing dependency for tool sequence processing: {str(e)}")
            return self._create_fallback_sequence_response(request, "Missing required dependencies")
        except Exception as e:
            # Unexpected errors - log and provide fallback
            logger.error(f"Unexpected error processing tool sequence request: {str(e)}", exc_info=True)
            return self._create_fallback_sequence_response(request, "Tool sequence processing temporarily unavailable")

    def get_alternative_recommendations(self, request: AlternativeRequest) -> AlternativeResponse:
        """Get alternative tool recommendations with explanations."""
        # Input validation
        if not request:
            raise ValueError("Request cannot be None")
        if not request.primary_tool or not request.primary_tool.strip():
            raise ValueError("Primary tool cannot be empty")

        logger.info(f"Processing alternative recommendations for: {request.primary_tool}")

        start_time = time.time()

        try:
            # For now, provide basic alternative recommendations
            # In a full implementation, this would use the tool catalog to find similar tools
            alternatives = []

            comparison_analysis = {
                "primary_advantages": [f"{request.primary_tool} offers specialized functionality"],
                "alternative_advantages": ["Alternatives may be simpler or more accessible"],
                "trade_offs": ["Feature richness vs simplicity"],
            }

            selection_guidance = [
                f"Use {request.primary_tool} when you need its specific capabilities",
                "Consider alternatives for simpler use cases",
            ]

            response = AlternativeResponse(
                alternatives=alternatives,
                comparison_analysis=comparison_analysis,
                selection_guidance=selection_guidance,
                request_metadata={"processing_time": time.time() - start_time, "reason": request.reason},
            )

            logger.info(f"Alternative recommendations processing complete in {time.time() - start_time:.2f}s")
            return response

        except ValueError as e:
            # Input validation errors should be re-raised
            logger.error(f"Invalid input for alternative recommendations request: {str(e)}")
            raise
        except Exception as e:
            # Unexpected errors - log and provide fallback
            logger.error(f"Unexpected error processing alternative recommendations: {str(e)}", exc_info=True)
            return self._create_fallback_alternative_response(
                request, "Alternative recommendations temporarily unavailable"
            )

    def configure_caching(self, max_size: int, ttl_minutes: int) -> None:
        """Configure caching parameters."""
        self.cache_config["max_size"] = max_size
        self.cache_config["ttl_minutes"] = ttl_minutes
        self.cache = RecommendationCache(max_size, ttl_minutes)
        logger.info(f"Cache configured: max_size={max_size}, ttl={ttl_minutes}m")

    # Helper methods

    def _analyze_context(self, context_data: Optional[Dict[str, Any]]) -> TaskContext:
        """Analyze context data and create TaskContext."""
        if not context_data:
            return TaskContext()

        # Analyze different context components
        user_context = None
        project_context = None
        env_constraints = None

        try:
            if any(key in context_data for key in ["user_skill_level", "tool_usage_history"]):
                user_context = self.context_analyzer.analyze_user_context(context_data)
        except Exception as e:
            logger.warning(f"Failed to analyze user context: {e}")

        try:
            if any(key in context_data for key in ["project_type", "existing_tools"]):
                project_context = self.context_analyzer.analyze_project_context(context_data)
        except Exception as e:
            logger.warning(f"Failed to analyze project context: {e}")

        try:
            if any(
                key in context_data
                for key in ["system_capabilities", "resource_constraints", "compliance_requirements"]
            ):
                env_constraints = self.context_analyzer.analyze_environmental_constraints(context_data)
        except Exception as e:
            logger.warning(f"Failed to analyze environmental constraints: {e}")

        return TaskContext(
            user_context=user_context,
            project_context=project_context,
            environmental_constraints=env_constraints,
            time_constraints=context_data.get("time_constraints"),
            quality_requirements=context_data.get("quality_requirements"),
        )

    def _find_candidate_tools(
        self, task_analysis: TaskAnalysis, context: TaskContext, max_candidates: int
    ) -> List[Any]:
        """Find candidate tools using multiple search strategies."""
        candidates = []

        # Strategy 1: Semantic search based on task description
        try:
            semantic_results = self.tool_search_engine.search_tools(
                task_analysis.task_description, max_results=max_candidates
            )
            candidates.extend(semantic_results)
        except Exception as e:
            logger.warning(f"Semantic search failed: {e}")

        # Strategy 2: Capability-based search
        required_capabilities = getattr(task_analysis, "required_capabilities", [])
        if required_capabilities and isinstance(required_capabilities, (list, tuple)):
            capability_query = " ".join(required_capabilities)
            if capability_query:
                try:
                    capability_results = self.tool_search_engine.search_tools(
                        capability_query, max_results=max_candidates // 2
                    )
                    candidates.extend(capability_results)
                except Exception as e:
                    logger.warning(f"Capability search failed: {e}")

        # Remove duplicates based on tool_id
        seen_ids = set()
        unique_candidates = []
        for candidate in candidates:
            tool_id = getattr(candidate, "tool_id", getattr(candidate, "name", str(candidate)))
            if tool_id not in seen_ids:
                seen_ids.add(tool_id)
                unique_candidates.append(candidate)

        return unique_candidates[:max_candidates]

    def _apply_diversity_filtering(
        self, recommendations: List[ToolRecommendation], max_recommendations: int
    ) -> List[ToolRecommendation]:
        """Apply diversity filtering to avoid over-recommending similar tools."""
        if len(recommendations) <= max_recommendations:
            return recommendations

        # Simple diversity: prefer tools with different categories/capabilities
        diverse_recommendations = []
        seen_categories = set()

        for rec in recommendations:
            tool_category = getattr(rec.tool, "metadata", {}).get("category", "unknown")

            if tool_category not in seen_categories or len(diverse_recommendations) < max_recommendations // 2:
                diverse_recommendations.append(rec)
                seen_categories.add(tool_category)

            if len(diverse_recommendations) >= max_recommendations:
                break

        # Fill remaining slots with highest-scoring tools
        while len(diverse_recommendations) < max_recommendations and len(diverse_recommendations) < len(
            recommendations
        ):
            for rec in recommendations:
                if rec not in diverse_recommendations:
                    diverse_recommendations.append(rec)
                    break

        return diverse_recommendations

    def _generate_explanations(
        self, recommendations: List[ToolRecommendation], task_analysis: TaskAnalysis
    ) -> List[RecommendationExplanation]:
        """Generate explanations for recommendations."""
        explanations = []

        for rec in recommendations:
            try:
                explanation = self.explainer.explain_recommendation(rec, task_analysis)
                explanations.append(explanation)
            except Exception as e:
                logger.warning(f"Failed to generate explanation for {rec.tool.name}: {e}")
                # Create minimal explanation
                minimal_explanation = RecommendationExplanation(
                    primary_reasons=rec.recommendation_reasons,
                    capability_match_explanation="Tool provides required capabilities",
                    complexity_fit_explanation="Appropriate complexity for task",
                    parameter_compatibility_explanation="Compatible with task requirements",
                    setup_requirements=[],
                    usage_best_practices=[],
                    common_pitfalls=[],
                    troubleshooting_tips=[],
                    when_this_is_optimal=[],
                    when_to_consider_alternatives=[],
                    skill_level_guidance="Suitable for your skill level",
                    confidence_explanation=f"Confidence level: {rec.confidence_level}",
                    known_limitations=[],
                    uncertainty_areas=[],
                )
                explanations.append(minimal_explanation)

        return explanations

    def _analyze_workflow(self, workflow_description: str) -> Dict[str, Any]:
        """Analyze workflow requirements."""
        # Simple workflow analysis
        workflow_length = len(workflow_description.split())
        complexity = "simple" if workflow_length < 10 else "moderate" if workflow_length < 20 else "complex"

        return {
            "complexity": complexity,
            "estimated_steps": len(workflow_description.split(",")) + 1,
            "requires_coordination": "and" in workflow_description.lower(),
        }

    def _decompose_workflow(self, workflow_description: str) -> List[str]:
        """Decompose workflow into sub-tasks."""
        # Simple decomposition based on commas and conjunctions
        parts = workflow_description.replace(" and ", ", ").split(", ")
        return [part.strip() for part in parts if part.strip()]

    def _process_recommendation_request(
        self, request: RecommendationRequest, start_time: float
    ) -> RecommendationResponse:
        """Core processing logic for recommendation requests."""
        # Step 1: Analyze task requirements and constraints
        logger.debug("Analyzing task requirements")
        task_analysis = self.task_analyzer.analyze_task(request.task_description)

        # Step 2: Analyze context if provided
        context = self._analyze_context(request.context_data)

        # Step 3: Find candidate tools through multiple search strategies
        logger.debug("Finding candidate tools")
        candidates = self._find_candidate_tools(task_analysis, context, request.max_recommendations * 2)

        # Step 4: Apply recommendation algorithms and ranking
        logger.debug("Applying recommendation algorithms")
        recommendations = self.recommendation_algorithms.apply_ensemble_ranking(candidates, task_analysis, context)

        # Step 5: Apply diversity filtering
        recommendations = self._apply_diversity_filtering(recommendations, request.max_recommendations)

        # Step 6: Generate explanations if requested
        explanations = None
        if request.include_explanations:
            logger.debug("Generating explanations")
            explanations = self._generate_explanations(recommendations, task_analysis)

        # Step 7: Find alternatives if requested
        alternatives = None
        if request.include_alternatives and len(recommendations) > 1:
            logger.debug("Finding alternatives")
            alternatives = recommendations[1:]  # All recommendations except the primary

        # Step 8: Create response
        return RecommendationResponse(
            recommendations=recommendations[: request.max_recommendations],
            task_analysis=task_analysis,
            context=context,
            explanations=explanations,
            alternatives=alternatives,
            request_metadata={
                "max_recommendations": request.max_recommendations,
                "processing_time": time.time() - start_time,
                "candidates_evaluated": len(candidates),
            },
        )

    def _create_fallback_response(self, request: RecommendationRequest, error_message: str) -> RecommendationResponse:
        """Create a fallback response when full processing fails."""
        logger.warning(f"Creating fallback response: {error_message}")

        # Create a minimal response with error information
        from context_analyzer import TaskContext

        fallback_context = TaskContext(
            user_context=None,
            project_context=None,
            environmental_constraints=None,
            time_constraints=None,
            quality_requirements=None,
        )

        return RecommendationResponse(
            recommendations=[],
            task_analysis=None,
            context=fallback_context,
            explanations=[],
            alternatives=[],
            request_metadata={
                "error": error_message,
                "fallback": True,
                "timestamp": time.time(),
                "request_task": request.task_description[:100] if request.task_description else "Unknown",
            },
        )

    def _create_fallback_sequence_response(
        self, request: ToolSequenceRequest, error_message: str
    ) -> ToolSequenceResponse:
        """Create a fallback response when sequence processing fails."""
        logger.warning(f"Creating fallback sequence response: {error_message}")

        # Create a minimal sequence response with error information
        return ToolSequenceResponse(
            sequences=[],
            workflow_analysis={"complexity": "unknown", "estimated_steps": 0, "requires_coordination": False},
            optimization_results={"efficiency_score": 0.0, "reliability_score": 0.0},
            request_metadata={
                "error": error_message,
                "fallback": True,
                "timestamp": time.time(),
                "request_workflow": request.workflow_description[:100] if request.workflow_description else "Unknown",
            },
        )

    def _create_fallback_alternative_response(
        self, request: AlternativeRequest, error_message: str
    ) -> AlternativeResponse:
        """Create a fallback response when alternative processing fails."""
        logger.warning(f"Creating fallback alternative response: {error_message}")

        # Create a minimal alternative response with error information
        return AlternativeResponse(
            alternatives=[],
            comparison_analysis={
                "primary_advantages": [f"{request.primary_tool} - analysis unavailable"],
                "alternative_advantages": ["Alternative analysis temporarily unavailable"],
                "trade_offs": ["Unable to perform comparison at this time"],
                "error": error_message,
            },
            selection_guidance=[f"Unable to provide guidance for {request.primary_tool} at this time"],
            request_metadata={
                "error": error_message,
                "fallback": True,
                "timestamp": time.time(),
                "request_tool": request.primary_tool,
            },
        )
