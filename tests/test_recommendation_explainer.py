#!/usr/bin/env python3
"""
Tests for recommendation_explainer.py - Recommendation Explanation System

This module tests the system that generates explanations for tool recommendations,
including comparison between alternatives and usage guidance.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pytest

from recommendation_algorithms import AlternativeRecommendation, ToolRecommendation
from recommendation_explainer import (
    AlternativeComparison,
    ComparisonAnalyzer,
    ExplanationGenerator,
    GuidanceGenerator,
    RecommendationExplainer,
    RecommendationExplanation,
    UsageGuidance,
)
from task_analyzer import TaskAnalysis


# Mock classes for testing
@dataclass
class MockTaskContext:
    """Mock task context for testing."""

    user_skill_level: str = "intermediate"
    project_type: str = "general"
    constraints: Dict = field(default_factory=dict)


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


class TestRecommendationExplainer:
    """Test cases for RecommendationExplainer class."""

    @pytest.fixture
    def explainer(self):
        """Create a RecommendationExplainer instance for testing."""
        return RecommendationExplainer()

    @pytest.fixture
    def task_analysis(self):
        """Create a task analysis for testing."""
        return TaskAnalysis(
            task_description="Read and process CSV data",
            task_intent="process",
            task_category="data_processing",
            required_capabilities=["read", "csv", "data_processing"],
            input_specifications=["csv"],
            output_specifications=["processed_data"],
            performance_constraints={"speed": "fast"},
            quality_requirements=["accuracy"],
            error_handling_needs=["validation"],
            complexity_level="moderate",
            estimated_steps=3,
            skill_level_required="intermediate",
            confidence=0.85,
            analysis_notes=["CSV processing task"],
        )

    @pytest.fixture
    def high_score_recommendation(self):
        """Create a high-score recommendation for testing."""
        tool = MockToolSearchResult(
            tool_id="csv_processor",
            name="CSV Processor Pro",
            description="Advanced CSV processing tool",
            score=0.92,
            metadata={
                "category": "data_processing",
                "complexity": "moderate",
                "input_types": ["csv", "tsv"],
                "output_types": ["json", "xml", "parquet"],
                "capabilities": ["read", "csv", "data_processing", "validation"],
                "parameters": {
                    "delimiter": {"type": "string", "default": ",", "required": False},
                    "encoding": {"type": "string", "default": "utf-8", "required": False},
                    "skip_rows": {"type": "integer", "default": 0, "required": False},
                },
            },
        )

        return ToolRecommendation(
            tool=tool,
            recommendation_score=0.92,
            confidence_level="high",
            task_alignment=0.95,
            capability_match=0.90,
            complexity_alignment=0.88,
            parameter_compatibility=0.85,
            recommendation_reasons=["Perfect CSV processing capabilities", "Excellent complexity match"],
            potential_issues=[],
            usage_guidance=[],
            alternative_tools=["basic_csv_reader", "pandas_processor"],
            when_to_use="For complex CSV processing tasks",
            when_not_to_use="For simple file reading",
            recommendation_strategy="capability_based",
            context_factors=["user_skill:intermediate"],
        )

    @pytest.fixture
    def alternative_recommendation(self):
        """Create an alternative recommendation for testing."""
        tool = MockToolSearchResult(
            tool_id="basic_csv_reader",
            name="Basic CSV Reader",
            description="Simple CSV file reader",
            score=0.75,
            metadata={"category": "file_operation", "complexity": "simple", "capabilities": ["read", "csv"]},
        )

        return ToolRecommendation(
            tool=tool,
            recommendation_score=0.75,
            confidence_level="medium",
            task_alignment=0.78,
            capability_match=0.70,
            complexity_alignment=0.65,
            parameter_compatibility=0.80,
            recommendation_reasons=["Simple CSV reading"],
            potential_issues=["Limited processing capabilities"],
            usage_guidance=[],
            alternative_tools=[],
            when_to_use="For basic CSV reading",
            when_not_to_use="For complex processing",
            recommendation_strategy="simplicity_based",
            context_factors=[],
        )

    def test_explain_recommendation_high_score(self, explainer, high_score_recommendation, task_analysis):
        """Test explanation generation for high-score recommendation."""
        explanation = explainer.explain_recommendation(high_score_recommendation, task_analysis)

        assert isinstance(explanation, RecommendationExplanation)
        assert len(explanation.primary_reasons) > 0
        assert "Perfect CSV processing capabilities" in explanation.primary_reasons
        assert explanation.capability_match_explanation != ""
        assert explanation.complexity_fit_explanation != ""
        assert explanation.confidence_explanation != ""
        assert len(explanation.setup_requirements) >= 0
        assert len(explanation.usage_best_practices) >= 0

    def test_explain_recommendation_parameter_compatibility(self, explainer, high_score_recommendation, task_analysis):
        """Test parameter compatibility explanation."""
        explanation = explainer.explain_recommendation(high_score_recommendation, task_analysis)

        assert explanation.parameter_compatibility_explanation != ""
        # Should mention available parameters
        assert "parameter" in explanation.parameter_compatibility_explanation.lower()

    def test_compare_alternatives_basic(self, explainer, high_score_recommendation, alternative_recommendation):
        """Test basic alternative comparison."""
        primary = high_score_recommendation
        alternatives = [alternative_recommendation]

        comparison = explainer.compare_alternatives(primary, alternatives)

        assert isinstance(comparison, AlternativeComparison)
        assert comparison.primary_tool.tool.name == "CSV Processor Pro"
        assert len(comparison.alternatives) == 1
        assert comparison.alternatives[0].tool.name == "Basic CSV Reader"
        assert len(comparison.key_differences) > 0
        assert len(comparison.decision_factors) > 0

    def test_compare_alternatives_trade_offs(self, explainer, high_score_recommendation, alternative_recommendation):
        """Test trade-off analysis in alternative comparison."""
        primary = high_score_recommendation
        alternatives = [alternative_recommendation]

        comparison = explainer.compare_alternatives(primary, alternatives)

        # Should identify trade-offs between complexity and capability
        trade_offs = [factor["trade_off"] for factor in comparison.decision_factors if "trade_off" in factor]
        assert len(trade_offs) > 0

    def test_generate_usage_guidance_basic_parameters(self, explainer, high_score_recommendation, task_analysis):
        """Test usage guidance generation for basic parameters."""
        context = MockTaskContext(user_skill_level="intermediate")

        guidance = explainer.generate_usage_guidance(high_score_recommendation.tool, context)

        assert isinstance(guidance, UsageGuidance)
        assert len(guidance.parameter_recommendations) > 0
        assert len(guidance.configuration_suggestions) > 0
        assert guidance.complexity_guidance != ""
        assert len(guidance.common_pitfalls) >= 0
        assert len(guidance.optimization_tips) >= 0

    def test_generate_usage_guidance_beginner_user(self, explainer, high_score_recommendation):
        """Test usage guidance for beginner users."""
        context = MockTaskContext(user_skill_level="beginner")

        guidance = explainer.generate_usage_guidance(high_score_recommendation.tool, context)

        # Should provide more detailed guidance for beginners
        assert (
            "beginner" in guidance.complexity_guidance.lower()
            or "simple" in guidance.complexity_guidance.lower()
            or "basic" in guidance.complexity_guidance.lower()
        )
        assert len(guidance.step_by_step_instructions) > 0

    def test_generate_usage_guidance_advanced_user(self, explainer, high_score_recommendation):
        """Test usage guidance for advanced users."""
        context = MockTaskContext(user_skill_level="advanced")

        guidance = explainer.generate_usage_guidance(high_score_recommendation.tool, context)

        # Should provide more concise, technical guidance for advanced users
        assert len(guidance.optimization_tips) > 0
        assert len(guidance.advanced_features) > 0

    def test_recommendation_explanation_completeness(self, explainer, high_score_recommendation, task_analysis):
        """Test that recommendation explanation covers all required aspects."""
        explanation = explainer.explain_recommendation(high_score_recommendation, task_analysis)

        # Verify all explanation components are present
        required_fields = [
            "primary_reasons",
            "capability_match_explanation",
            "complexity_fit_explanation",
            "parameter_compatibility_explanation",
            "setup_requirements",
            "usage_best_practices",
            "common_pitfalls",
            "troubleshooting_tips",
            "when_this_is_optimal",
            "when_to_consider_alternatives",
            "skill_level_guidance",
            "confidence_explanation",
            "known_limitations",
            "uncertainty_areas",
        ]

        for field in required_fields:
            assert hasattr(explanation, field), f"Missing field: {field}"

    def test_explanation_serialization(self, explainer, high_score_recommendation, task_analysis):
        """Test that explanations can be serialized to dict."""
        explanation = explainer.explain_recommendation(high_score_recommendation, task_analysis)

        explanation_dict = explanation.to_dict()

        assert isinstance(explanation_dict, dict)
        assert "primary_reasons" in explanation_dict
        assert "confidence_explanation" in explanation_dict


class TestExplanationGenerator:
    """Test cases for ExplanationGenerator class."""

    @pytest.fixture
    def generator(self):
        """Create an ExplanationGenerator instance for testing."""
        return ExplanationGenerator()

    @pytest.fixture
    def task_analysis(self):
        """Create a task analysis for testing."""
        return TaskAnalysis(
            task_description="Process data files",
            task_intent="process",
            task_category="data_processing",
            required_capabilities=["processing", "files"],
            input_specifications=["files"],
            output_specifications=["processed"],
            performance_constraints={},
            quality_requirements=[],
            error_handling_needs=[],
            complexity_level="moderate",
            estimated_steps=2,
            skill_level_required="intermediate",
            confidence=0.8,
            analysis_notes=[],
        )

    def test_generate_capability_explanation_high_match(self, generator, task_analysis):
        """Test capability explanation for high match score."""
        capabilities_score = 0.9
        tool_capabilities = ["processing", "files", "validation"]

        explanation = generator.generate_capability_explanation(
            task_analysis.required_capabilities, tool_capabilities, capabilities_score
        )

        assert "excellent" in explanation.lower() or "perfect" in explanation.lower()
        assert explanation != ""

    def test_generate_capability_explanation_low_match(self, generator, task_analysis):
        """Test capability explanation for low match score."""
        capabilities_score = 0.3
        tool_capabilities = ["different", "capabilities"]

        explanation = generator.generate_capability_explanation(
            task_analysis.required_capabilities, tool_capabilities, capabilities_score
        )

        assert "limited" in explanation.lower() or "partial" in explanation.lower() or "missing" in explanation.lower()

    def test_generate_complexity_explanation_perfect_match(self, generator):
        """Test complexity explanation for perfect alignment."""
        task_complexity = "moderate"
        tool_complexity = "moderate"
        alignment_score = 1.0

        explanation = generator.generate_complexity_explanation(task_complexity, tool_complexity, alignment_score)

        assert "perfect" in explanation.lower() or "ideal" in explanation.lower()
        assert explanation != ""

    def test_generate_complexity_explanation_mismatch(self, generator):
        """Test complexity explanation for misalignment."""
        task_complexity = "simple"
        tool_complexity = "complex"
        alignment_score = 0.4

        explanation = generator.generate_complexity_explanation(task_complexity, tool_complexity, alignment_score)

        assert "complex" in explanation.lower() or "mismatch" in explanation.lower()


class TestComparisonAnalyzer:
    """Test cases for ComparisonAnalyzer class."""

    @pytest.fixture
    def analyzer(self):
        """Create a ComparisonAnalyzer instance for testing."""
        return ComparisonAnalyzer()

    def test_analyze_capability_differences(self, analyzer):
        """Test analysis of capability differences between tools."""
        primary_capabilities = ["read", "process", "validate", "export"]
        alternative_capabilities = ["read", "basic_process"]

        differences = analyzer.analyze_capability_differences(primary_capabilities, alternative_capabilities)

        assert "primary_only" in differences
        assert "alternative_only" in differences
        assert "common" in differences
        assert "validate" in differences["primary_only"]
        assert "export" in differences["primary_only"]

    def test_generate_trade_off_analysis(self, analyzer):
        """Test trade-off analysis generation."""
        primary_scores = {"capability": 0.9, "complexity": 0.7, "usability": 0.6}
        alternative_scores = {"capability": 0.6, "complexity": 0.9, "usability": 0.8}

        trade_offs = analyzer.generate_trade_off_analysis(primary_scores, alternative_scores)

        assert len(trade_offs) > 0
        # Should identify that primary has better capability, alternative has better usability
        capability_trade_off = next((t for t in trade_offs if "capability" in t.lower()), None)
        assert capability_trade_off is not None


class TestGuidanceGenerator:
    """Test cases for GuidanceGenerator class."""

    @pytest.fixture
    def generator(self):
        """Create a GuidanceGenerator instance for testing."""
        return GuidanceGenerator()

    def test_generate_parameter_recommendations_with_defaults(self, generator):
        """Test parameter recommendations with default values."""
        tool_parameters = {
            "delimiter": {"type": "string", "default": ",", "required": False},
            "encoding": {"type": "string", "default": "utf-8", "required": False},
            "validate": {"type": "boolean", "default": True, "required": False},
        }

        recommendations = generator.generate_parameter_recommendations(tool_parameters)

        assert len(recommendations) > 0
        # Should recommend keeping defaults for simple use cases
        delimiter_rec = next((r for r in recommendations if "delimiter" in r), None)
        assert delimiter_rec is not None

    def test_generate_step_by_step_instructions_beginner(self, generator):
        """Test step-by-step instruction generation for beginners."""
        tool_name = "CSV Processor"
        user_skill = "beginner"

        instructions = generator.generate_step_by_step_instructions(tool_name, user_skill)

        assert len(instructions) > 0
        assert isinstance(instructions, list)
        # Should have multiple steps for beginners
        assert len(instructions) >= 3

    def test_generate_optimization_tips_advanced(self, generator):
        """Test optimization tips for advanced users."""
        tool_metadata = {
            "performance_features": ["parallel_processing", "memory_optimization"],
            "advanced_parameters": ["batch_size", "thread_count"],
        }
        user_skill = "advanced"

        tips = generator.generate_optimization_tips(tool_metadata, user_skill)

        assert len(tips) > 0
        # Should mention performance features for advanced users
        performance_tip = next((t for t in tips if "performance" in t.lower()), None)
        assert performance_tip is not None
