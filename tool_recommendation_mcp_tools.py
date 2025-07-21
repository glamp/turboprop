#!/usr/bin/env python3
"""
tool_recommendation_mcp_tools.py: MCP tools for intelligent tool recommendations.

This module implements sophisticated MCP tools that provide intelligent tool
recommendations for development tasks. It integrates task analysis, context
awareness, and recommendation algorithms to help users select optimal tools.

Core MCP Tools:
- recommend_tools_for_task: Main recommendation function with explanations
- analyze_task_requirements: Deep task analysis and requirement extraction
- suggest_tool_alternatives: Alternative tool suggestions with comparisons
- recommend_tool_sequence: Multi-tool workflow recommendations
"""

import time
from typing import Any, Dict, Optional

from config import config
from context_analyzer import EnvironmentalConstraints, TaskContext, UserContext
from logging_config import get_logger
from mcp_error_handling import create_validation_error, handle_tool_exception
from mcp_response_standardizer import standardize_mcp_tool_response
from recommendation_explainer_mcp import (
    AlternativeComparisonFormatter,
    MCPExplanationFormatter,
    TaskDescriptionSuggestionGenerator,
    WorkflowAnalysisFormatter,
)
from task_analysis_response_types import create_error_response  # Keep for backward compatibility
from task_analysis_response_types import (
    AlternativesResponse,
    AlternativeTool,
    AlternativeToolCore,
    AlternativeToolDetails,
    AnalysisMetrics,
    RecommendationCore,
    RecommendationEnhancements,
    RequirementsBreakdown,
    ResponseMetadata,
    TaskAnalysisCore,
    TaskAnalysisResponse,
    TaskRecommendationResponse,
    ToolRecommendation,
    ToolSequence,
    ToolSequenceResponse,
    ToolSequenceStep,
)
from task_analyzer import TaskAnalyzer
from tool_recommendation_engine import (
    AlternativeRequest,
    RecommendationRequest,
    ToolRecommendationEngine,
    ToolSequenceRequest,
)

logger = get_logger(__name__)

# Global instances - will be initialized by the calling module
_recommendation_engine: Optional[ToolRecommendationEngine] = None
_task_analyzer: Optional[TaskAnalyzer] = None
_explanation_formatter: Optional[MCPExplanationFormatter] = None
_suggestion_generator: Optional[TaskDescriptionSuggestionGenerator] = None
_comparison_formatter: Optional[AlternativeComparisonFormatter] = None
_workflow_formatter: Optional[WorkflowAnalysisFormatter] = None


def initialize_recommendation_tools(
    recommendation_engine: ToolRecommendationEngine, task_analyzer: TaskAnalyzer
) -> None:
    """
    Initialize the recommendation tools with required dependencies.

    Args:
        recommendation_engine: Main recommendation engine instance
        task_analyzer: Task analysis engine instance
    """
    global _recommendation_engine, _task_analyzer
    global _explanation_formatter, _suggestion_generator
    global _comparison_formatter, _workflow_formatter

    _recommendation_engine = recommendation_engine
    _task_analyzer = task_analyzer

    # Initialize formatters
    _explanation_formatter = MCPExplanationFormatter()
    _suggestion_generator = TaskDescriptionSuggestionGenerator()
    _comparison_formatter = AlternativeComparisonFormatter()
    _workflow_formatter = WorkflowAnalysisFormatter()

    logger.info("Tool recommendation MCP tools initialized successfully")


@standardize_mcp_tool_response
def recommend_tools_for_task(
    task_description: str,
    context: Optional[str] = None,
    max_recommendations: int = config.mcp.DEFAULT_MAX_RECOMMENDATIONS,
    include_alternatives: bool = True,
    complexity_preference: str = "balanced",
    explain_reasoning: bool = True,
) -> dict:
    """
    Get intelligent tool recommendations for a specific development task.

    This tool analyzes a task description and recommends the most appropriate MCP tools
    based on functionality, complexity, and context. It provides detailed explanations
    and alternative options to help choose the optimal approach.

    Args:
        task_description: Natural language description of the task to accomplish
                         Examples: "read configuration file and parse JSON data",
                                  "search codebase for specific function implementations"
        context: Additional context about environment, constraints, or preferences
                Examples: "performance critical", "beginner user", "large repository"
        max_recommendations: Maximum number of primary recommendations (1-10)
        include_alternatives: Whether to include alternative tool options
        complexity_preference: Tool complexity preference ('simple', 'balanced', 'powerful')
        explain_reasoning: Whether to include detailed explanations for recommendations

    Returns:
        Ranked tool recommendations with explanations and alternatives

    Examples:
        recommend_tools_for_task("read and modify configuration files")
        recommend_tools_for_task("search for functions in Python code", context="large codebase")
        recommend_tools_for_task("execute tests with timeout", complexity_preference="simple")
    """
    start_time = time.time()

    try:
        # Validate inputs
        if not task_description or not isinstance(task_description, str):
            return create_validation_error(
                "recommend_tools_for_task",
                "Task description must be a non-empty string",
                "Parameter validation failed",
                ["Provide a string value for task_description parameter"],
            )

        task_description = task_description.strip()
        if not task_description:
            return create_validation_error(
                "recommend_tools_for_task",
                "Task description cannot be empty",
                "Empty or whitespace-only input provided",
                ["Provide a meaningful task description", "Include specific details about what you want to accomplish"],
            )

        if len(task_description) > config.mcp.TASK_DESCRIPTION_MAX_LENGTH:
            return create_validation_error(
                "recommend_tools_for_task",
                f"Task description too long (max {config.mcp.TASK_DESCRIPTION_MAX_LENGTH} characters)",
                f"Input length: {len(task_description)} characters",
                [
                    "Shorten your task description",
                    "Focus on the essential requirements",
                    "Break complex tasks into smaller parts",
                ],
            )

        # Validate max_recommendations
        max_recommendations = max(1, min(max_recommendations, 10))

        # Validate complexity_preference
        if complexity_preference not in ["simple", "balanced", "powerful"]:
            complexity_preference = "balanced"

        # Check if recommendation engine is initialized
        if not _recommendation_engine or not _task_analyzer:
            return create_error_response("recommend_tools_for_task", "Recommendation engine not initialized")

        logger.info(f"Processing tool recommendation request: {task_description[:50]}...")

        # Analyze task requirements
        task_analysis = _task_analyzer.analyze_task(task_description)

        # Create task context from parameters
        task_context = create_task_context(
            context_description=context,
            complexity_preference=complexity_preference,
            user_preferences={"explain_reasoning": explain_reasoning},
        )

        # Create recommendation request
        request = RecommendationRequest(
            task_description=task_description,
            max_recommendations=max_recommendations,
            include_alternatives=include_alternatives,
            include_explanations=explain_reasoning,
            context_data=task_context.to_dict() if task_context else None,
        )

        # Get recommendations from engine
        engine_response = _recommendation_engine.recommend_for_task(request)

        # Convert engine recommendations to MCP format
        mcp_recommendations = []
        for rec in engine_response.recommendations:
            # Create core recommendation data
            core = RecommendationCore(
                tool_id=getattr(rec.tool, "tool_id", getattr(rec.tool, "name", "unknown")),
                tool_name=getattr(rec.tool, "name", "Unknown Tool"),
                confidence_score=rec.confidence_level if isinstance(rec.confidence_level, float) else 0.7,
                relevance_score=rec.relevance_score,
                task_alignment=rec.task_alignment,
                complexity_fit=rec.complexity_assessment or "moderate",
                skill_level_match=getattr(rec, "skill_level_match", "intermediate"),
            )

            # Create enhancement data
            enhancements = RecommendationEnhancements(
                recommendation_reasons=rec.recommendation_reasons,
                usage_guidance=rec.usage_guidance or [],
                parameter_suggestions=getattr(rec, "parameter_suggestions", {}),
                alternatives_available=include_alternatives and len(engine_response.recommendations) > 1,
                alternative_count=len(engine_response.recommendations) - 1 if include_alternatives else 0,
            )

            # Create composed recommendation
            mcp_rec = ToolRecommendation(core=core, enhancements=enhancements)
            mcp_recommendations.append(mcp_rec)

        # Create structured response
        # Create analysis metrics from task analysis
        analysis_metrics = None
        if task_analysis:
            analysis_metrics = AnalysisMetrics(
                complexity_assessment=task_analysis.complexity_level,
                confidence_score=task_analysis.confidence,
                estimated_steps=task_analysis.estimated_steps,
                skill_level_required=task_analysis.skill_level_required,
            )

        # Create metadata with processing time
        metadata = ResponseMetadata(processing_time=time.time() - start_time)

        response = TaskRecommendationResponse(
            task_description=task_description,
            recommendations=mcp_recommendations,
            analysis_metrics=analysis_metrics,
            context_factors=task_context.to_dict() if task_context else None,
            metadata=metadata,
        )

        # Add explanations if requested
        if explain_reasoning and _explanation_formatter:
            explanations = _explanation_formatter.generate_recommendation_explanations(
                mcp_recommendations, task_analysis
            )
            response.add_explanations(explanations)

        # Add task description improvement suggestions
        if _suggestion_generator and task_analysis:
            suggestions = _suggestion_generator.generate_task_description_suggestions(task_description, task_analysis)
            for suggestion in suggestions:
                response.add_refinement_suggestion(suggestion)

        logger.info(f"Tool recommendation completed in {time.time() - start_time:.2f}s")
        return response.to_dict()

    except Exception as e:
        return handle_tool_exception(
            "recommend_tools_for_task",
            e,
            f"Task: {task_description[:100]}...",
            {"task_description": task_description, "max_recommendations": max_recommendations},
        )


@standardize_mcp_tool_response
def analyze_task_requirements(
    task_description: str, detail_level: str = "standard", include_suggestions: bool = True
) -> dict:
    """
    Analyze a task description to understand requirements and constraints.

    This tool provides detailed analysis of what a task requires, helping to understand
    the complexity, required capabilities, and potential approaches before selecting tools.

    Args:
        task_description: Description of the task to analyze
        detail_level: Level of analysis detail ('basic', 'standard', 'comprehensive')
        include_suggestions: Whether to include improvement suggestions for task description

    Returns:
        Comprehensive task analysis with requirements, constraints, and insights

    Examples:
        analyze_task_requirements("process CSV files and generate reports")
        analyze_task_requirements("deploy application with monitoring", detail_level="comprehensive")
    """
    start_time = time.time()

    try:
        # Validate inputs
        if not task_description or not isinstance(task_description, str):
            return create_error_response("analyze_task_requirements", "Task description must be a non-empty string")

        task_description = task_description.strip()
        if not task_description:
            return create_error_response("analyze_task_requirements", "Task description cannot be empty")

        if len(task_description) > config.mcp.TASK_DESCRIPTION_MAX_LENGTH:
            return create_error_response(
                "analyze_task_requirements",
                f"Task description too long (max {config.mcp.TASK_DESCRIPTION_MAX_LENGTH} characters)",
            )

        # Validate detail_level
        if detail_level not in ["basic", "standard", "comprehensive"]:
            detail_level = "standard"

        # Check if task analyzer is initialized
        if not _task_analyzer:
            return create_error_response("analyze_task_requirements", "Task analyzer not initialized")

        logger.info(f"Analyzing task requirements: {task_description[:50]}...")

        # Perform task analysis
        analysis = _task_analyzer.analyze_task(task_description)

        # Extract detailed requirements
        requirements = _task_analyzer.extract_task_requirements(task_description)

        # Create detailed analysis response
        # Create core analysis data
        core = TaskAnalysisCore(
            task_description=task_description,
            detail_level=detail_level,
            required_capabilities=analysis.required_capabilities if analysis else [],
            potential_challenges=[],  # Will be populated below
        )

        # Create response metadata with processing time
        metadata = ResponseMetadata(processing_time=time.time() - start_time)

        # Create analysis metrics
        metrics = AnalysisMetrics(
            complexity_assessment=analysis.complexity_level if analysis else "moderate",
            confidence_score=analysis.confidence if analysis else 0.7,
            estimated_steps=analysis.estimated_steps if analysis else 3,
            skill_level_required=analysis.skill_level_required if analysis else "intermediate",
        )

        # Create requirements breakdown
        requirements_breakdown = RequirementsBreakdown(
            functional_requirements=requirements.functional_requirements,
            non_functional_requirements=requirements.non_functional_requirements,
            input_specifications=requirements.input_types,
            output_specifications=requirements.output_types,
        )

        response = TaskAnalysisResponse(
            core=core,
            metadata=metadata,
            metrics=metrics,
            requirements=requirements_breakdown,
            analysis=analysis.to_dict() if analysis else None,
        )

        # Add potential challenges based on analysis
        if analysis and detail_level in ["standard", "comprehensive"]:
            if analysis.complexity_level == "complex":
                response.add_challenge("Task complexity may require advanced tools or multiple steps")

            if analysis.estimated_steps > 5:
                response.add_challenge("Multi-step process may require workflow coordination")

            if analysis.confidence < 0.6:
                response.add_challenge("Task description may be too vague for optimal tool selection")

        # Add capabilities based on requirements
        for cap in requirements.functional_requirements:
            response.add_capability(cap)

        # Add suggestions if requested
        if include_suggestions and _suggestion_generator:
            suggestions = _suggestion_generator.generate_task_description_suggestions(task_description, analysis)
            response.add_suggestions(suggestions)

        # Add alternative approaches for comprehensive analysis
        if detail_level == "comprehensive":
            if "file" in task_description.lower():
                response.alternative_approaches.append("Consider batch processing for multiple files")
                response.alternative_approaches.append("Evaluate streaming vs. in-memory processing")

            if "data" in task_description.lower():
                response.alternative_approaches.append("Consider data validation and error handling approaches")
                response.alternative_approaches.append("Evaluate different data formats and serialization options")

        logger.info(f"Task analysis completed in {time.time() - start_time:.2f}s")
        return response.to_dict()

    except Exception as e:
        logger.error(f"Error in analyze_task_requirements: {str(e)}", exc_info=True)
        return create_error_response("analyze_task_requirements", str(e), task_description)


@standardize_mcp_tool_response
def suggest_tool_alternatives(
    primary_tool: str,
    task_context: Optional[str] = None,
    max_alternatives: int = config.mcp.DEFAULT_MAX_ALTERNATIVES,
    include_comparisons: bool = True,
) -> dict:
    """
    Suggest alternative tools for a given primary tool choice.

    This tool finds alternative tools that could accomplish similar tasks,
    providing comparisons and guidance on when each alternative might be preferred.

    Args:
        primary_tool: The primary tool to find alternatives for
        task_context: Context about the specific task or use case
        max_alternatives: Maximum number of alternatives to suggest
        include_comparisons: Whether to include detailed comparisons

    Returns:
        Alternative tool suggestions with comparisons and usage guidance

    Examples:
        suggest_tool_alternatives("bash", task_context="file processing")
        suggest_tool_alternatives("read", max_alternatives=3)
    """
    start_time = time.time()

    try:
        # Validate inputs
        if not primary_tool or not isinstance(primary_tool, str):
            return create_error_response("suggest_tool_alternatives", "Primary tool must be a non-empty string")

        primary_tool = primary_tool.strip()
        if not primary_tool:
            return create_error_response("suggest_tool_alternatives", "Primary tool name cannot be empty")

        # Validate max_alternatives
        max_alternatives = max(1, min(max_alternatives, 10))

        # Check if recommendation engine is initialized
        if not _recommendation_engine:
            return create_error_response("suggest_tool_alternatives", "Recommendation engine not initialized")

        logger.info(f"Finding alternatives for tool: {primary_tool}")

        # Create task context for alternatives
        context = create_task_context(task_context) if task_context else TaskContext()

        # Create alternative request
        request = AlternativeRequest(
            primary_tool=primary_tool, task_context=context, reason="user_requested", max_alternatives=max_alternatives
        )

        # Get alternative recommendations from engine
        engine_response = _recommendation_engine.get_alternative_recommendations(request)

        # Convert to MCP format
        mcp_alternatives = []
        for alt in engine_response.alternatives:
            # Create core alternative data
            core = AlternativeToolCore(
                tool_id=getattr(alt.tool, "tool_id", getattr(alt.tool, "name", "unknown")),
                tool_name=getattr(alt.tool, "name", "Unknown Tool"),
                similarity_score=alt.relevance_score,
                complexity_comparison=_compare_complexity(primary_tool, alt),
            )

            # Create detailed alternative data
            details = AlternativeToolDetails(
                advantages=alt.recommendation_reasons[:3],  # Limit to top 3 advantages
                use_cases=alt.usage_guidance[:3] if alt.usage_guidance else [],
                when_to_prefer=alt.usage_guidance[:2] if alt.usage_guidance else [],
            )

            # Create composed alternative
            alt_meta = AlternativeTool(core=core, details=details)
            mcp_alternatives.append(alt_meta)

        # Create alternatives response
        # Create metadata with processing time
        metadata = ResponseMetadata(processing_time=time.time() - start_time)

        response = AlternativesResponse(
            primary_tool=primary_tool,
            alternatives=mcp_alternatives,
            task_context=task_context,
            primary_tool_advantages=[f"{primary_tool} offers specialized functionality for your specific use case"],
            metadata=metadata,
        )

        # Add comparisons if requested
        if include_comparisons and mcp_alternatives and _comparison_formatter:
            comparisons = _comparison_formatter.generate_alternative_comparisons(primary_tool, mcp_alternatives)
            response.add_comparisons(comparisons)

        # Add selection criteria
        response.add_selection_criterion("Consider your technical skill level and requirements")
        response.add_selection_criterion("Evaluate setup complexity vs. functionality needs")
        response.add_selection_criterion("Think about long-term maintenance and support")

        # Categorize alternatives
        for alt in mcp_alternatives:
            if "simple" in " ".join(alt.advantages).lower():
                response.categorize_alternative("Simple alternatives", alt.tool_id)
            elif "performance" in " ".join(alt.advantages).lower():
                response.categorize_alternative("Performance-focused", alt.tool_id)
            else:
                response.categorize_alternative("Feature-rich alternatives", alt.tool_id)

        logger.info(f"Alternative suggestions completed in {time.time() - start_time:.2f}s")
        return response.to_dict()

    except Exception as e:
        logger.error(f"Error in suggest_tool_alternatives: {str(e)}", exc_info=True)
        return create_error_response("suggest_tool_alternatives", str(e), primary_tool)


@standardize_mcp_tool_response
def recommend_tool_sequence(
    workflow_description: str,
    optimization_goal: str = "balanced",
    max_sequence_length: int = config.mcp.DEFAULT_MAX_SEQUENCE_LENGTH,
    allow_parallel_tools: bool = False,
) -> dict:
    """
    Recommend sequences of tools for complex workflows.

    This tool analyzes complex workflows and recommends optimal sequences of tools
    to accomplish multi-step tasks efficiently and reliably.

    Args:
        workflow_description: Description of the complete workflow or process
        optimization_goal: What to optimize for ('speed', 'reliability', 'simplicity', 'balanced')
        max_sequence_length: Maximum number of tools in sequence
        allow_parallel_tools: Whether to suggest parallel tool execution

    Returns:
        Recommended tool sequences with explanations and alternatives

    Examples:
        recommend_tool_sequence("read config file, validate data, process and save results")
        recommend_tool_sequence("search code, analyze results, generate report", optimization_goal="speed")
    """
    start_time = time.time()

    try:
        # Validate inputs
        if not workflow_description or not isinstance(workflow_description, str):
            return create_error_response("recommend_tool_sequence", "Workflow description must be a non-empty string")

        workflow_description = workflow_description.strip()
        if not workflow_description:
            return create_error_response("recommend_tool_sequence", "Workflow description cannot be empty")

        if len(workflow_description) > config.mcp.TASK_DESCRIPTION_MAX_LENGTH * 2:
            return create_error_response(
                "recommend_tool_sequence",
                f"Workflow description too long (max {config.mcp.TASK_DESCRIPTION_MAX_LENGTH * 2} characters)",
            )

        # Validate optimization_goal
        if optimization_goal not in ["speed", "reliability", "simplicity", "balanced"]:
            optimization_goal = "balanced"

        # Validate max_sequence_length
        max_sequence_length = max(2, min(max_sequence_length, 20))

        # Check if engines are initialized
        if not _recommendation_engine or not _workflow_formatter:
            return create_error_response("recommend_tool_sequence", "Required engines not initialized")

        logger.info(f"Analyzing workflow: {workflow_description[:50]}...")

        # Analyze workflow requirements
        workflow_analysis = _workflow_formatter.analyze_workflow(workflow_description)

        # Create workflow context
        workflow_context = _workflow_formatter.create_workflow_context(
            optimization_goal=optimization_goal, max_length=max_sequence_length, allow_parallel=allow_parallel_tools
        )

        # Create sequence request
        request = ToolSequenceRequest(
            workflow_description=workflow_description,
            context_data=workflow_context,
            optimization_goals=[optimization_goal],
            max_sequences=3,
        )

        # Get sequence recommendations from engine
        engine_response = _recommendation_engine.recommend_tool_sequence(request)

        # Convert to MCP format
        mcp_sequences = []
        for i, seq in enumerate(engine_response.sequences):
            # Create sequence steps
            steps = []
            for j, tool_id in enumerate(
                getattr(
                    seq, "tool_chain", [seq.recommended_tool_id if hasattr(seq, "recommended_tool_id") else "unknown"]
                )
            ):
                step = ToolSequenceStep(
                    step_number=j + 1,
                    tool_id=tool_id,
                    tool_name=tool_id.replace("_", " ").title(),
                    purpose=f"Step {j + 1} of workflow",
                    estimated_time="1-2 minutes",
                    complexity="moderate",
                )
                steps.append(step)

            # Create sequence metadata
            sequence_meta = ToolSequence(
                sequence_id=f"seq_{i+1}",
                sequence_name=f"Workflow Sequence {i+1}",
                steps=steps,
                estimated_duration=f"{len(steps) * 2}-{len(steps) * 3} minutes",
                complexity_level=workflow_analysis.get("complexity", "moderate"),
                parallel_execution_possible=allow_parallel_tools,
                reliability_score=0.85,
                efficiency_score=0.8 if optimization_goal == "speed" else 0.75,
                maintainability_score=0.9 if optimization_goal == "simplicity" else 0.8,
            )

            # Add prerequisites and tips based on optimization goal
            if optimization_goal == "reliability":
                sequence_meta.prerequisites.append("Ensure proper error handling at each step")
                sequence_meta.optimization_tips.append("Add validation between steps")
            elif optimization_goal == "speed":
                sequence_meta.optimization_tips.append("Consider parallel execution where possible")
                sequence_meta.optimization_tips.append("Use streaming for large data sets")
            elif optimization_goal == "simplicity":
                sequence_meta.prerequisites.append("Familiarize yourself with basic tool usage")
                sequence_meta.optimization_tips.append("Start with default parameters")

            mcp_sequences.append(sequence_meta)

        # Create sequence response
        # Create metadata with processing time
        metadata = ResponseMetadata(processing_time=time.time() - start_time)

        # Create analysis metrics with complexity assessment
        metrics = AnalysisMetrics(complexity_assessment=workflow_analysis.get("complexity", "moderate"))

        response = ToolSequenceResponse(
            workflow_description=workflow_description,
            sequences=mcp_sequences,
            workflow_analysis=workflow_analysis,
            optimization_goal=optimization_goal,
            metadata=metadata,
            metrics=metrics,
        )

        # Add workflow insights
        if workflow_analysis.get("requires_coordination"):
            response.add_parallel_opportunity("Steps involving data transformation can potentially run in parallel")

        if workflow_analysis.get("estimated_steps", 0) > 5:
            response.add_bottleneck("Complex workflows may have bottlenecks in data processing steps")

        # Add customization suggestions
        response.add_customization_suggestion(f"Adjust sequence based on your {optimization_goal} priorities")
        response.add_customization_suggestion("Consider adding intermediate validation steps for reliability")

        logger.info(f"Tool sequence recommendation completed in {time.time() - start_time:.2f}s")
        return response.to_dict()

    except Exception as e:
        logger.error(f"Error in recommend_tool_sequence: {str(e)}", exc_info=True)
        return create_error_response("recommend_tool_sequence", str(e), workflow_description)


# Helper functions for context creation and management


def create_task_context(
    context_description: Optional[str] = None,
    complexity_preference: str = "balanced",
    user_preferences: Optional[Dict[str, Any]] = None,
) -> TaskContext:
    """Create TaskContext from MCP tool parameters."""

    # Parse context description for environmental constraints
    environmental_constraints = None
    if context_description:
        environmental_constraints = parse_context_description(context_description)

    # Determine skill level from context or preferences
    skill_level = "intermediate"  # Default
    if context_description:
        context_lower = context_description.lower()
        if "beginner" in context_lower:
            skill_level = "beginner"
        elif "advanced" in context_lower:
            skill_level = "advanced"

    # Create user context with required parameters
    user_context = UserContext(skill_level=skill_level, complexity_preference=complexity_preference)

    return TaskContext(user_context=user_context, environmental_constraints=environmental_constraints)


def parse_context_description(context_desc: str) -> EnvironmentalConstraints:
    """Parse context description into structured constraints."""
    # Create constraints with default values for required parameters
    constraints = EnvironmentalConstraints(
        cpu_cores=4,  # Default assumption
        memory_gb=8,  # Default assumption
        gpu_available=False,  # Conservative default
        operating_system="unknown",  # Will be detected if needed
    )

    context_lower = context_desc.lower()

    # Parse common constraint patterns
    if "performance critical" in context_lower or "fast" in context_lower:
        constraints.resource_limits = {"performance_priority": "high"}

    # Parse resource constraints
    if "memory" in context_lower:
        constraints.resource_limits = {"memory_sensitive": True}

    if "large" in context_lower:
        constraints.resource_limits.update({"scale": "large"})
    elif "small" in context_lower:
        constraints.resource_limits.update({"scale": "small"})

    # Parse security requirements
    if "secure" in context_lower or "security" in context_lower:
        constraints.security_requirements = ["standard_security"]

    return constraints


def tool_exists(tool_name: str) -> bool:
    """Check if a tool exists in the available tool catalog."""
    # This is a simplified implementation
    # In a real system, this would check against the actual tool catalog
    common_tools = [
        "bash",
        "read",
        "write",
        "edit",
        "glob",
        "grep",
        "search_code",
        "index_repository",
        "list_indexed_files",
        "watch_repository",
        "search_mcp_tools",
        "get_tool_details",
    ]
    return tool_name.lower() in common_tools


def _compare_complexity(primary_tool: str, alternative) -> str:
    """Compare complexity between primary tool and alternative."""
    # Simplified complexity comparison
    simple_tools = ["read", "write", "ls"]
    complex_tools = ["search_code", "index_repository", "watch_repository"]

    primary_simple = primary_tool.lower() in simple_tools
    primary_complex = primary_tool.lower() in complex_tools

    alt_name = getattr(alternative.tool, "name", "unknown").lower()
    alt_simple = alt_name in simple_tools
    alt_complex = alt_name in complex_tools

    if primary_simple and alt_complex:
        return "more_complex"
    elif primary_complex and alt_simple:
        return "simpler"
    else:
        return "similar"


# Tool availability check
def get_tool_availability_status() -> Dict[str, bool]:
    """Get the current availability status of recommendation tools."""
    return {
        "recommendation_engine": _recommendation_engine is not None,
        "task_analyzer": _task_analyzer is not None,
        "explanation_formatter": _explanation_formatter is not None,
        "suggestion_generator": _suggestion_generator is not None,
        "comparison_formatter": _comparison_formatter is not None,
        "workflow_formatter": _workflow_formatter is not None,
    }
