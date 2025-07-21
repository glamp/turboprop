#!/usr/bin/env python3
"""
Tests for recommendation_algorithms.py - Tool Recommendation Algorithms

This module tests the recommendation algorithms that score and rank tools
based on task requirements and context.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional

import pytest
from turboprop.recommendation_algorithms import (
    RecommendationAlgorithms,
    ToolRecommendation,
    ToolSequenceRecommendation,
    WorkflowRequirements,
)
from turboprop.task_analyzer import TaskAnalysis


# Mock classes for testing
@dataclass
class MockToolSearchResult:
    """Mock tool search result for testing."""

    tool_id: str
    name: str
    description: str
    score: float
    metadata: Optional[Dict] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class MockTaskContext:
    """Mock task context for testing."""

    user_skill_level: str = "intermediate"
    project_type: str = "general"
    time_constraints: Dict = field(default_factory=dict)
    quality_requirements: Dict = field(default_factory=dict)


class TestRecommendationAlgorithms:
    """Test cases for RecommendationAlgorithms class."""

    @pytest.fixture
    def recommendation_algorithms(self):
        """Create a RecommendationAlgorithms instance for testing."""
        return RecommendationAlgorithms()

    @pytest.fixture
    def simple_task_analysis(self):
        """Create a simple task analysis for testing."""
        return TaskAnalysis(
            task_description="Read a CSV file",
            task_intent="read",
            task_category="file_operation",
            required_capabilities=["read", "file_handling"],
            input_specifications=["csv"],
            output_specifications=[],
            performance_constraints={},
            quality_requirements=[],
            error_handling_needs=[],
            complexity_level="simple",
            estimated_steps=1,
            skill_level_required="beginner",
            confidence=0.9,
            analysis_notes=[],
        )

    @pytest.fixture
    def complex_task_analysis(self):
        """Create a complex task analysis for testing."""
        return TaskAnalysis(
            task_description="Analyze data with ML and generate report",
            task_intent="analyze",
            task_category="machine_learning",
            required_capabilities=["analysis", "machine_learning", "reporting"],
            input_specifications=["data"],
            output_specifications=["report"],
            performance_constraints={"speed": "fast"},
            quality_requirements=["accuracy > 95%"],
            error_handling_needs=["validation", "monitoring"],
            complexity_level="complex",
            estimated_steps=8,
            skill_level_required="advanced",
            confidence=0.85,
            analysis_notes=[],
        )

    @pytest.fixture
    def csv_reader_tool(self):
        """Create a CSV reader tool for testing."""
        return MockToolSearchResult(
            tool_id="csv_reader",
            name="CSV Reader",
            description="Read and parse CSV files",
            score=0.9,
            metadata={
                "category": "file_operation",
                "complexity": "simple",
                "input_types": ["csv"],
                "capabilities": ["read", "parse", "file_handling"],
            },
        )

    @pytest.fixture
    def ml_analyzer_tool(self):
        """Create an ML analyzer tool for testing."""
        return MockToolSearchResult(
            tool_id="ml_analyzer",
            name="ML Data Analyzer",
            description="Advanced machine learning data analysis",
            score=0.85,
            metadata={
                "category": "machine_learning",
                "complexity": "complex",
                "capabilities": ["analysis", "machine_learning", "statistics"],
                "skill_level": "advanced",
            },
        )

    def test_calculate_task_tool_fit_perfect_match(
        self, recommendation_algorithms, simple_task_analysis, csv_reader_tool
    ):
        """Test task-tool fit calculation for a perfect match."""
        fit_score = recommendation_algorithms.calculate_task_tool_fit(simple_task_analysis, csv_reader_tool)

        assert 0.8 <= fit_score <= 1.0  # Should be high for perfect match
        assert isinstance(fit_score, float)

    def test_calculate_task_tool_fit_poor_match(
        self, recommendation_algorithms, simple_task_analysis, ml_analyzer_tool
    ):
        """Test task-tool fit calculation for a poor match."""
        fit_score = recommendation_algorithms.calculate_task_tool_fit(simple_task_analysis, ml_analyzer_tool)

        assert 0.0 <= fit_score <= 0.5  # Should be low for poor match

    def test_calculate_task_tool_fit_with_context(
        self, recommendation_algorithms, complex_task_analysis, ml_analyzer_tool
    ):
        """Test task-tool fit calculation with context."""
        context = MockTaskContext(user_skill_level="advanced")

        fit_score = recommendation_algorithms.calculate_task_tool_fit(complex_task_analysis, ml_analyzer_tool, context)

        assert 0.7 <= fit_score <= 1.0  # Should be high for advanced user + complex tool

    def test_apply_ensemble_ranking_single_tool(self, recommendation_algorithms, simple_task_analysis, csv_reader_tool):
        """Test ensemble ranking with a single tool."""
        candidates = [csv_reader_tool]

        recommendations = recommendation_algorithms.apply_ensemble_ranking(candidates, simple_task_analysis)

        assert len(recommendations) == 1
        assert isinstance(recommendations[0], ToolRecommendation)
        assert recommendations[0].tool.name == "CSV Reader"
        assert recommendations[0].recommendation_score > 0

    def test_apply_ensemble_ranking_multiple_tools(
        self, recommendation_algorithms, simple_task_analysis, csv_reader_tool, ml_analyzer_tool
    ):
        """Test ensemble ranking with multiple tools."""
        candidates = [csv_reader_tool, ml_analyzer_tool]

        recommendations = recommendation_algorithms.apply_ensemble_ranking(candidates, simple_task_analysis)

        assert len(recommendations) == 2
        # CSV reader should rank higher for simple file operation task
        assert recommendations[0].tool.name == "CSV Reader"
        assert recommendations[0].recommendation_score > recommendations[1].recommendation_score

    def test_calculate_recommendation_confidence_high_alignment(self, recommendation_algorithms, simple_task_analysis):
        """Test confidence calculation for high task-tool alignment."""
        recommendation = ToolRecommendation(
            tool=MockToolSearchResult("test_tool", "Test Tool", "Test description", 0.9),
            recommendation_score=0.9,
            confidence_level="",  # Will be set by method
            task_alignment=0.95,
            capability_match=0.9,
            complexity_alignment=0.85,
            parameter_compatibility=0.8,
            recommendation_reasons=[],
            potential_issues=[],
            usage_guidance=[],
            alternative_tools=[],
            when_to_use="",
            when_not_to_use="",
            recommendation_strategy="",
            context_factors=[],
        )

        confidence = recommendation_algorithms.calculate_recommendation_confidence(recommendation, simple_task_analysis)

        assert 0.8 <= confidence <= 1.0
        assert isinstance(confidence, float)

    def test_calculate_recommendation_confidence_low_alignment(self, recommendation_algorithms, simple_task_analysis):
        """Test confidence calculation for low task-tool alignment."""
        recommendation = ToolRecommendation(
            tool=MockToolSearchResult("test_tool", "Test Tool", "Test description", 0.3),
            recommendation_score=0.3,
            confidence_level="",  # Will be set by method
            task_alignment=0.3,
            capability_match=0.4,
            complexity_alignment=0.2,
            parameter_compatibility=0.3,
            recommendation_reasons=[],
            potential_issues=[],
            usage_guidance=[],
            alternative_tools=[],
            when_to_use="",
            when_not_to_use="",
            recommendation_strategy="",
            context_factors=[],
        )

        confidence = recommendation_algorithms.calculate_recommendation_confidence(recommendation, simple_task_analysis)

        assert 0.0 <= confidence <= 0.5

    def test_optimize_tool_sequence_simple_workflow(self, recommendation_algorithms):
        """Test tool sequence optimization for simple workflow."""
        tool_chain = ["csv_reader", "data_processor", "report_generator"]
        workflow_requirements = WorkflowRequirements(
            steps=["read", "process", "generate"],
            data_flow_requirements={"csv": "processed_data", "processed_data": "report"},
            error_handling_strategy="fail_fast",
            performance_requirements={},
        )

        optimized_sequence = recommendation_algorithms.optimize_tool_sequence(tool_chain, workflow_requirements)

        assert len(optimized_sequence) > 0
        assert isinstance(optimized_sequence[0], ToolSequenceRecommendation)

    def test_tool_recommendation_dataclass_completeness(self):
        """Test that ToolRecommendation dataclass has all required fields."""
        recommendation = ToolRecommendation(
            tool=MockToolSearchResult("test", "Test", "Description", 0.8),
            recommendation_score=0.85,
            confidence_level="high",
            task_alignment=0.9,
            capability_match=0.8,
            complexity_alignment=0.85,
            parameter_compatibility=0.9,
            recommendation_reasons=["Perfect capability match"],
            potential_issues=["None identified"],
            usage_guidance=["Use with default parameters"],
            alternative_tools=["alternative_tool"],
            when_to_use="For CSV file operations",
            when_not_to_use="For binary file operations",
            recommendation_strategy="capability_based",
            context_factors=["user_skill_level"],
        )

        # Verify all fields are accessible
        assert recommendation.tool.name == "Test"
        assert recommendation.recommendation_score == 0.85
        assert recommendation.confidence_level == "high"
        assert len(recommendation.recommendation_reasons) == 1


class TestToolRecommendationScoring:
    """Test cases for tool recommendation scoring algorithms."""

    @pytest.fixture
    def algorithms(self):
        """Create algorithms instance for testing."""
        return RecommendationAlgorithms()

    def test_capability_match_scoring_perfect_match(self, algorithms):
        """Test capability matching for perfect overlap."""
        required_capabilities = ["read", "parse", "file_handling"]
        tool_capabilities = ["read", "parse", "file_handling", "validation"]

        score = algorithms._calculate_capability_match_score(required_capabilities, tool_capabilities)

        assert score == 1.0  # Perfect match (all required capabilities present)

    def test_capability_match_scoring_partial_match(self, algorithms):
        """Test capability matching for partial overlap."""
        required_capabilities = ["read", "parse", "file_handling", "validation"]
        tool_capabilities = ["read", "parse"]

        score = algorithms._calculate_capability_match_score(required_capabilities, tool_capabilities)

        assert 0.4 <= score <= 0.6  # 50% match (2 out of 4 capabilities)

    def test_capability_match_scoring_no_match(self, algorithms):
        """Test capability matching for no overlap."""
        required_capabilities = ["read", "parse"]
        tool_capabilities = ["write", "upload"]

        score = algorithms._calculate_capability_match_score(required_capabilities, tool_capabilities)

        assert score == 0.0  # No match

    def test_complexity_alignment_scoring_perfect_match(self, algorithms):
        """Test complexity alignment for perfect match."""
        task_complexity = "simple"
        tool_complexity = "simple"
        user_skill = "beginner"

        score = algorithms._calculate_complexity_alignment_score(task_complexity, tool_complexity, user_skill)

        assert score >= 0.9  # Should be very high for perfect alignment

    def test_complexity_alignment_scoring_mismatch(self, algorithms):
        """Test complexity alignment for mismatch."""
        task_complexity = "simple"
        tool_complexity = "complex"
        user_skill = "beginner"

        score = algorithms._calculate_complexity_alignment_score(task_complexity, tool_complexity, user_skill)

        assert score <= 0.4  # Should be low for mismatch


class TestWorkflowRequirements:
    """Test cases for WorkflowRequirements dataclass."""

    def test_workflow_requirements_creation(self):
        """Test creating WorkflowRequirements."""
        requirements = WorkflowRequirements(
            steps=["step1", "step2"],
            data_flow_requirements={"input": "output"},
            error_handling_strategy="retry",
            performance_requirements={"latency": "low"},
        )

        assert requirements.steps == ["step1", "step2"]
        assert requirements.data_flow_requirements == {"input": "output"}
        assert requirements.error_handling_strategy == "retry"
        assert requirements.performance_requirements == {"latency": "low"}
