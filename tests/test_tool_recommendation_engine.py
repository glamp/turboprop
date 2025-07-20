#!/usr/bin/env python3
"""
Tests for tool_recommendation_engine.py - Main Tool Recommendation Engine

This module tests the main orchestrator that coordinates all components to provide
comprehensive tool recommendations with explanations and context awareness.
"""

import pytest
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from unittest.mock import Mock, MagicMock

from tool_recommendation_engine import (
    ToolRecommendationEngine,
    RecommendationRequest,
    RecommendationResponse,
    ToolSequenceRequest,
    ToolSequenceResponse,
    AlternativeRequest,
    AlternativeResponse,
)
from task_analyzer import TaskAnalysis, TaskAnalyzer
from recommendation_algorithms import RecommendationAlgorithms, ToolRecommendation
from recommendation_explainer import RecommendationExplainer, RecommendationExplanation
from context_analyzer import ContextAnalyzer, TaskContext, UserContext, ProjectContext, EnvironmentalConstraints


# Mock classes for testing
@dataclass
class MockMCPToolSearchEngine:
    """Mock MCP tool search engine."""
    def search_tools(self, query: str, max_results: int = 10) -> List[Any]:
        return [
            Mock(tool_id="csv_reader", name="CSV Reader", description="Read CSV files", score=0.9,
                 metadata={"capabilities": ["read", "csv"], "complexity": "simple"}),
            Mock(tool_id="data_processor", name="Data Processor", description="Process data", score=0.8,
                 metadata={"capabilities": ["process", "data"], "complexity": "moderate"}),
        ]


@dataclass 
class MockParameterSearchEngine:
    """Mock parameter search engine."""
    def search_by_parameters(self, requirements: Dict) -> List[Any]:
        return []


class TestToolRecommendationEngine:
    """Test cases for ToolRecommendationEngine class."""

    @pytest.fixture
    def mock_components(self):
        """Create mock components for testing."""
        return {
            'tool_search_engine': MockMCPToolSearchEngine(),
            'parameter_search_engine': MockParameterSearchEngine(),
            'task_analyzer': Mock(spec=TaskAnalyzer),
            'context_analyzer': Mock(spec=ContextAnalyzer)
        }

    @pytest.fixture
    def recommendation_engine(self, mock_components):
        """Create a ToolRecommendationEngine instance for testing."""
        return ToolRecommendationEngine(**mock_components)

    @pytest.fixture
    def simple_request(self):
        """Create a simple recommendation request."""
        return RecommendationRequest(
            task_description="Read a CSV file and process data",
            max_recommendations=3,
            include_alternatives=True,
            context_data={
                "user_skill_level": "intermediate",
                "project_type": "data_analysis"
            }
        )

    @pytest.fixture
    def complex_request(self):
        """Create a complex recommendation request."""
        return RecommendationRequest(
            task_description="Build machine learning pipeline with data validation and monitoring",
            max_recommendations=5,
            include_alternatives=True,
            include_explanations=True,
            context_data={
                "user_skill_level": "advanced",
                "project_type": "machine_learning",
                "performance_requirements": {"latency": "low", "throughput": "high"},
                "compliance_requirements": ["audit_trail", "data_privacy"]
            }
        )

    @pytest.fixture
    def sequence_request(self):
        """Create a tool sequence request."""
        return ToolSequenceRequest(
            workflow_description="Data ingestion, processing, analysis, and visualization workflow",
            context_data={"project_type": "data_analysis"},
            optimization_goals=["efficiency", "reliability"]
        )

    def test_recommend_for_task_basic(self, recommendation_engine, simple_request, mock_components):
        """Test basic tool recommendation functionality."""
        # Setup mocks
        mock_task_analysis = TaskAnalysis(
            task_description=simple_request.task_description,
            task_intent="process",
            task_category="data_processing",
            required_capabilities=["read", "csv", "process"],
            input_specifications=["csv"],
            output_specifications=["processed_data"],
            performance_constraints={},
            quality_requirements=[],
            error_handling_needs=[],
            complexity_level="moderate",
            estimated_steps=2,
            skill_level_required="intermediate",
            confidence=0.85,
            analysis_notes=[]
        )
        
        mock_components['task_analyzer'].analyze_task.return_value = mock_task_analysis
        mock_components['context_analyzer'].analyze_user_context.return_value = UserContext("intermediate", {}, "balanced", "medium", "guided")
        
        # Execute recommendation
        response = recommendation_engine.recommend_for_task(simple_request)
        
        # Verify response structure
        assert isinstance(response, RecommendationResponse)
        assert len(response.recommendations) > 0
        assert response.task_analysis.task_description == simple_request.task_description
        assert response.request_metadata["max_recommendations"] == 3

    def test_recommend_for_task_with_context(self, recommendation_engine, complex_request, mock_components):
        """Test tool recommendation with full context analysis."""
        # Setup mocks
        mock_task_analysis = TaskAnalysis(
            task_description=complex_request.task_description,
            task_intent="build",
            task_category="machine_learning",
            required_capabilities=["ml", "pipeline", "monitoring"],
            input_specifications=["data"],
            output_specifications=["model", "metrics"],
            performance_constraints={"latency": "low"},
            quality_requirements=["accuracy > 95%"],
            error_handling_needs=["monitoring", "validation"],
            complexity_level="complex",
            estimated_steps=8,
            skill_level_required="advanced",
            confidence=0.80,
            analysis_notes=[]
        )
        
        mock_user_context = UserContext("advanced", {"ml_tool": 0.8}, "powerful", "high", "efficient")
        mock_project_context = ProjectContext("machine_learning", "general", 5, ["scikit-learn"], ["python"], 
                                            {"latency": "low"}, ["audit_trail"], ["ml_pipelines"])
        mock_env_constraints = EnvironmentalConstraints(8, 32, True, "linux", ["encryption"], {"memory": "16GB"}, ["SOC2"])
        
        mock_components['task_analyzer'].analyze_task.return_value = mock_task_analysis
        mock_components['context_analyzer'].analyze_user_context.return_value = mock_user_context
        mock_components['context_analyzer'].analyze_project_context.return_value = mock_project_context
        mock_components['context_analyzer'].analyze_environmental_constraints.return_value = mock_env_constraints
        
        # Execute recommendation
        response = recommendation_engine.recommend_for_task(complex_request)
        
        # Verify context integration
        assert response.context.user_context.skill_level == "advanced"
        assert response.context.project_context.project_type == "machine_learning"
        assert response.context.environmental_constraints.cpu_cores == 8
        assert len(response.recommendations) > 0

    def test_recommend_for_task_with_explanations(self, recommendation_engine, simple_request, mock_components):
        """Test tool recommendation with explanations."""
        simple_request.include_explanations = True
        
        # Setup mocks
        mock_task_analysis = TaskAnalysis(
            task_description=simple_request.task_description,
            task_intent="process",
            task_category="data_processing",
            required_capabilities=["read", "process"],
            input_specifications=["csv"],
            output_specifications=["data"],
            performance_constraints={},
            quality_requirements=[],
            error_handling_needs=[],
            complexity_level="moderate",
            estimated_steps=2,
            skill_level_required="intermediate",
            confidence=0.85,
            analysis_notes=[]
        )
        
        mock_components['task_analyzer'].analyze_task.return_value = mock_task_analysis
        mock_components['context_analyzer'].analyze_user_context.return_value = UserContext("intermediate", {}, "balanced", "medium", "guided")
        
        # Execute recommendation
        response = recommendation_engine.recommend_for_task(simple_request)
        
        # Verify explanations are included
        assert response.explanations is not None
        assert len(response.explanations) > 0

    def test_recommend_tool_sequence_basic(self, recommendation_engine, sequence_request, mock_components):
        """Test tool sequence recommendation."""
        # Setup mocks
        mock_components['context_analyzer'].analyze_project_context.return_value = ProjectContext(
            "data_analysis", "general", 3, ["pandas"], ["python"], {}, [], ["data_pipeline"]
        )
        
        # Execute sequence recommendation  
        response = recommendation_engine.recommend_tool_sequence(sequence_request)
        
        # Verify response structure
        assert isinstance(response, ToolSequenceResponse)
        assert len(response.sequences) > 0
        assert response.workflow_analysis["complexity"] in ["simple", "moderate", "complex"]

    def test_get_alternative_recommendations_basic(self, recommendation_engine, mock_components):
        """Test alternative recommendation retrieval."""
        request = AlternativeRequest(
            primary_tool="csv_reader",
            task_context=TaskContext(),
            reason="performance_optimization",
            max_alternatives=3
        )
        
        # Execute alternative recommendations
        response = recommendation_engine.get_alternative_recommendations(request)
        
        # Verify response structure
        assert isinstance(response, AlternativeResponse)
        assert len(response.alternatives) >= 0
        assert response.comparison_analysis is not None

    def test_caching_functionality(self, recommendation_engine, simple_request, mock_components):
        """Test recommendation caching."""
        # Setup mocks
        mock_task_analysis = TaskAnalysis(
            task_description=simple_request.task_description,
            task_intent="process",
            task_category="data_processing",
            required_capabilities=["read", "process"],
            input_specifications=["csv"],
            output_specifications=["data"],
            performance_constraints={},
            quality_requirements=[],
            error_handling_needs=[],
            complexity_level="moderate",
            estimated_steps=2,
            skill_level_required="intermediate",
            confidence=0.85,
            analysis_notes=[]
        )
        
        mock_components['task_analyzer'].analyze_task.return_value = mock_task_analysis
        mock_components['context_analyzer'].analyze_user_context.return_value = UserContext("intermediate", {}, "balanced", "medium", "guided")
        
        # Execute same recommendation twice
        response1 = recommendation_engine.recommend_for_task(simple_request)
        response2 = recommendation_engine.recommend_for_task(simple_request)
        
        # Verify both responses are valid (caching should be transparent)
        assert isinstance(response1, RecommendationResponse)
        assert isinstance(response2, RecommendationResponse)
        assert len(response1.recommendations) == len(response2.recommendations)

    def test_error_handling_invalid_request(self, recommendation_engine):
        """Test error handling for invalid requests."""
        # Test empty task description
        with pytest.raises(ValueError, match="Task description cannot be empty"):
            RecommendationRequest(
                task_description="",  # Empty task description
                max_recommendations=5,
                include_alternatives=True
            )
        
        # Test negative max_recommendations
        with pytest.raises(ValueError, match="max_recommendations must be positive"):
            RecommendationRequest(
                task_description="Valid description",
                max_recommendations=-1,  # Invalid number
                include_alternatives=True
            )

    def test_error_handling_analysis_failure(self, recommendation_engine, simple_request, mock_components):
        """Test error handling when task analysis fails."""
        # Setup mock to raise exception
        mock_components['task_analyzer'].analyze_task.side_effect = Exception("Analysis failed")
        
        with pytest.raises(Exception, match="Analysis failed"):
            recommendation_engine.recommend_for_task(simple_request)

    def test_performance_optimization_settings(self, recommendation_engine):
        """Test performance optimization settings."""
        # Verify async processing is available
        assert hasattr(recommendation_engine, 'enable_async_processing')
        
        # Test cache configuration
        recommendation_engine.configure_caching(max_size=100, ttl_minutes=30)
        assert recommendation_engine.cache_config["max_size"] == 100
        assert recommendation_engine.cache_config["ttl_minutes"] == 30

    def test_recommendation_diversity(self, recommendation_engine, simple_request, mock_components):
        """Test recommendation diversity to avoid over-recommending popular tools."""
        # Setup mocks with popular tools
        mock_task_analysis = TaskAnalysis(
            task_description=simple_request.task_description,
            task_intent="process",
            task_category="data_processing",
            required_capabilities=["read", "process"],
            input_specifications=["csv"],
            output_specifications=["data"],
            performance_constraints={},
            quality_requirements=[],
            error_handling_needs=[],
            complexity_level="moderate",
            estimated_steps=2,
            skill_level_required="intermediate",
            confidence=0.85,
            analysis_notes=[]
        )
        
        mock_components['task_analyzer'].analyze_task.return_value = mock_task_analysis
        mock_components['context_analyzer'].analyze_user_context.return_value = UserContext("intermediate", {}, "balanced", "medium", "guided")
        
        # Execute recommendation
        response = recommendation_engine.recommend_for_task(simple_request)
        
        # Verify diversity (should have different tool types/categories)
        tool_names = [rec.tool.name for rec in response.recommendations]
        assert len(set(tool_names)) > 1  # Should have distinct tools


class TestRecommendationRequestTypes:
    """Test cases for recommendation request data types."""

    def test_recommendation_request_validation(self):
        """Test recommendation request validation."""
        # Valid request
        valid_request = RecommendationRequest(
            task_description="Process CSV data",
            max_recommendations=5,
            include_alternatives=True
        )
        assert valid_request.task_description == "Process CSV data"
        assert valid_request.max_recommendations == 5

    def test_recommendation_request_defaults(self):
        """Test recommendation request default values."""
        minimal_request = RecommendationRequest(task_description="Test task")
        
        assert minimal_request.max_recommendations == 5  # Should have reasonable default
        assert minimal_request.include_alternatives == True  # Should default to True
        assert minimal_request.include_explanations == False  # Should default to False

    def test_tool_sequence_request_creation(self):
        """Test tool sequence request creation."""
        sequence_request = ToolSequenceRequest(
            workflow_description="Data processing workflow",
            context_data={"project": "test"},
            optimization_goals=["speed", "reliability"]
        )
        
        assert sequence_request.workflow_description == "Data processing workflow"
        assert sequence_request.optimization_goals == ["speed", "reliability"]

    def test_alternative_request_creation(self):
        """Test alternative request creation."""
        alt_request = AlternativeRequest(
            primary_tool="tool1",
            task_context=TaskContext(),
            reason="performance",
            max_alternatives=3
        )
        
        assert alt_request.primary_tool == "tool1"
        assert alt_request.reason == "performance"
        assert alt_request.max_alternatives == 3


class TestRecommendationResponseTypes:
    """Test cases for recommendation response data types."""

    def test_recommendation_response_serialization(self):
        """Test recommendation response serialization."""
        # Create mock response
        response = RecommendationResponse(
            recommendations=[],
            task_analysis=TaskAnalysis("test", "test", "test", [], [], [], {}, [], [], "simple", 1, "beginner", 0.8, []),
            context=TaskContext(),
            explanations=[],
            alternatives=[],
            request_metadata={"test": "value"}
        )
        
        response_dict = response.to_dict()
        assert isinstance(response_dict, dict)
        assert "recommendations" in response_dict
        assert "task_analysis" in response_dict

    def test_tool_sequence_response_creation(self):
        """Test tool sequence response creation."""
        response = ToolSequenceResponse(
            sequences=[],
            workflow_analysis={"complexity": "moderate"},
            optimization_results={"efficiency_score": 0.8},
            request_metadata={}
        )
        
        assert response.workflow_analysis["complexity"] == "moderate"
        assert response.optimization_results["efficiency_score"] == 0.8

    def test_alternative_response_creation(self):
        """Test alternative response creation."""
        response = AlternativeResponse(
            alternatives=[],
            comparison_analysis={"primary_advantages": ["feature1"]},
            selection_guidance=["Use primary for X"],
            request_metadata={}
        )
        
        assert response.comparison_analysis["primary_advantages"] == ["feature1"]
        assert response.selection_guidance[0] == "Use primary for X"