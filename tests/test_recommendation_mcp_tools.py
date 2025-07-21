#!/usr/bin/env python3
"""
test_recommendation_mcp_tools.py: Comprehensive tests for MCP tool recommendations.

This module tests all the tool recommendation MCP tools to ensure they work correctly,
provide appropriate responses, and handle edge cases gracefully.
"""

import json
import pytest
from unittest.mock import MagicMock, patch, Mock

from context_analyzer import ContextAnalyzer, TaskContext, UserContext, EnvironmentalConstraints
from parameter_search_engine import ParameterSearchEngine
from task_analyzer import TaskAnalyzer, TaskAnalysis
from tool_recommendation_engine import (
    ToolRecommendationEngine, 
    RecommendationRequest,
    RecommendationResponse,
    AlternativeRequest,
    AlternativeResponse,
    ToolSequenceRequest,
    ToolSequenceResponse
)
from tool_recommendation_mcp_tools import (
    recommend_tools_for_task,
    analyze_task_requirements,
    suggest_tool_alternatives,
    recommend_tool_sequence,
    initialize_recommendation_tools,
    create_task_context,
    parse_context_description,
    tool_exists,
    get_tool_availability_status
)


class TestToolRecommendationMCPTools:
    """Test suite for tool recommendation MCP tools."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create mock components
        self.mock_recommendation_engine = MagicMock(spec=ToolRecommendationEngine)
        self.mock_task_analyzer = MagicMock(spec=TaskAnalyzer)
        self.mock_context_analyzer = MagicMock(spec=ContextAnalyzer)
        self.mock_parameter_search_engine = MagicMock(spec=ParameterSearchEngine)
        
        # Initialize tools with mocks
        initialize_recommendation_tools(
            self.mock_recommendation_engine,
            self.mock_task_analyzer
        )

    @pytest.fixture
    def sample_task_analysis(self):
        """Sample task analysis for testing."""
        return TaskAnalysis(
            task_description="read configuration file and parse JSON data",
            task_intent="read",
            task_category="file_operation",
            required_capabilities=["read", "file_handling"],
            input_specifications=["json"],
            output_specifications=["data"],
            performance_constraints={},
            quality_requirements=["validation"],
            error_handling_needs=["error_logging"],
            complexity_level="moderate",
            estimated_steps=3,
            skill_level_required="intermediate",
            confidence=0.8,
            analysis_notes=["Task well-defined"]
        )

    @pytest.fixture
    def sample_recommendation_response(self):
        """Sample recommendation response for testing."""
        from recommendation_algorithms import ToolRecommendation
        
        # Create mock tool
        mock_tool = Mock()
        mock_tool.name = "read"
        mock_tool.tool_id = "read"
        
        # Create mock recommendation
        mock_recommendation = Mock(spec=ToolRecommendation)
        mock_recommendation.tool = mock_tool
        mock_recommendation.confidence_level = 0.9
        mock_recommendation.relevance_score = 0.85
        mock_recommendation.task_alignment = 0.8
        mock_recommendation.recommendation_reasons = ["handles file reading", "supports JSON"]
        mock_recommendation.usage_guidance = ["specify file path", "handle errors"]
        mock_recommendation.complexity_assessment = "moderate"
        
        return RecommendationResponse(
            recommendations=[mock_recommendation],
            task_analysis=Mock(),
            context=Mock(),
            explanations=None,
            alternatives=None,
            request_metadata={}
        )


class TestRecommendToolsForTask(TestToolRecommendationMCPTools):
    """Tests for the recommend_tools_for_task function."""

    def test_basic_recommendation_request(self, sample_task_analysis, sample_recommendation_response):
        """Test basic tool recommendation functionality."""
        # Setup mocks
        self.mock_task_analyzer.analyze_task.return_value = sample_task_analysis
        self.mock_recommendation_engine.recommend_for_task.return_value = sample_recommendation_response
        
        # Make request
        result = recommend_tools_for_task("read configuration file and parse JSON data")
        
        # Verify result is JSON
        assert isinstance(result, str)
        result_dict = json.loads(result)
        
        # Verify response structure
        assert result_dict["success"] is True
        assert "recommendations" in result_dict
        assert "task_analysis" in result_dict
        assert "task_description" in result_dict
        assert result_dict["task_description"] == "read configuration file and parse JSON data"
        
        # Verify engine was called
        self.mock_task_analyzer.analyze_task.assert_called_once_with("read configuration file and parse JSON data")
        self.mock_recommendation_engine.recommend_for_task.assert_called_once()

    def test_recommendation_with_context(self, sample_task_analysis, sample_recommendation_response):
        """Test recommendation with additional context."""
        self.mock_task_analyzer.analyze_task.return_value = sample_task_analysis
        self.mock_recommendation_engine.recommend_for_task.return_value = sample_recommendation_response
        
        result = recommend_tools_for_task(
            "read configuration file",
            context="performance critical",
            complexity_preference="simple"
        )
        
        result_dict = json.loads(result)
        assert result_dict["success"] is True
        assert "context_factors" in result_dict

    def test_recommendation_with_custom_parameters(self, sample_task_analysis, sample_recommendation_response):
        """Test recommendation with custom parameters."""
        self.mock_task_analyzer.analyze_task.return_value = sample_task_analysis
        self.mock_recommendation_engine.recommend_for_task.return_value = sample_recommendation_response
        
        result = recommend_tools_for_task(
            "process data files",
            max_recommendations=3,
            include_alternatives=False,
            complexity_preference="powerful",
            explain_reasoning=False
        )
        
        result_dict = json.loads(result)
        assert result_dict["success"] is True
        
        # Verify parameters were passed correctly
        call_args = self.mock_recommendation_engine.recommend_for_task.call_args[0][0]
        assert call_args.max_recommendations == 3
        assert call_args.include_alternatives is False

    def test_empty_task_description_error(self):
        """Test error handling for empty task description."""
        result = recommend_tools_for_task("")
        result_dict = json.loads(result)
        
        assert result_dict["success"] is False
        assert "error" in result_dict

    def test_task_description_too_long_error(self):
        """Test error handling for task description that is too long."""
        long_description = "x" * 2000  # Exceeds max length
        result = recommend_tools_for_task(long_description)
        result_dict = json.loads(result)
        
        assert result_dict["success"] is False
        assert "error" in result_dict

    def test_recommendation_engine_error_handling(self, sample_task_analysis):
        """Test handling of recommendation engine errors."""
        self.mock_task_analyzer.analyze_task.return_value = sample_task_analysis
        self.mock_recommendation_engine.recommend_for_task.side_effect = Exception("Engine error")
        
        result = recommend_tools_for_task("test task")
        result_dict = json.loads(result)
        
        assert result_dict["success"] is False
        assert "error" in result_dict


class TestAnalyzeTaskRequirements(TestToolRecommendationMCPTools):
    """Tests for the analyze_task_requirements function."""

    def test_basic_task_analysis(self, sample_task_analysis):
        """Test basic task analysis functionality."""
        self.mock_task_analyzer.analyze_task.return_value = sample_task_analysis
        
        from task_analyzer import TaskRequirements
        mock_requirements = TaskRequirements(
            functional_requirements=["file_reading"],
            non_functional_requirements=["performance"],
            input_types=["json"],
            output_types=["data"],
            performance_requirements={"speed": "required"},
            reliability_requirements=["error_handling"],
            usability_requirements=[]
        )
        self.mock_task_analyzer.extract_task_requirements.return_value = mock_requirements
        
        result = analyze_task_requirements("process CSV files")
        result_dict = json.loads(result)
        
        assert result_dict["success"] is True
        assert "analysis" in result_dict
        assert "complexity_assessment" in result_dict
        assert "required_capabilities" in result_dict

    def test_analysis_with_different_detail_levels(self, sample_task_analysis):
        """Test analysis with different detail levels."""
        self.mock_task_analyzer.analyze_task.return_value = sample_task_analysis
        
        from task_analyzer import TaskRequirements
        mock_requirements = TaskRequirements(
            functional_requirements=[],
            non_functional_requirements=[],
            input_types=[],
            output_types=[],
            performance_requirements={},
            reliability_requirements=[],
            usability_requirements=[]
        )
        self.mock_task_analyzer.extract_task_requirements.return_value = mock_requirements
        
        # Test basic level
        result_basic = analyze_task_requirements("test task", detail_level="basic")
        result_dict = json.loads(result_basic)
        assert result_dict["detail_level"] == "basic"
        
        # Test comprehensive level
        result_comprehensive = analyze_task_requirements("test task", detail_level="comprehensive")
        result_dict = json.loads(result_comprehensive)
        assert result_dict["detail_level"] == "comprehensive"

    def test_analysis_with_suggestions(self, sample_task_analysis):
        """Test analysis with task improvement suggestions."""
        # Create low-confidence analysis to trigger suggestions
        low_confidence_analysis = sample_task_analysis
        low_confidence_analysis.confidence = 0.5
        
        self.mock_task_analyzer.analyze_task.return_value = low_confidence_analysis
        
        from task_analyzer import TaskRequirements
        mock_requirements = TaskRequirements(
            functional_requirements=[],
            non_functional_requirements=[],
            input_types=[],
            output_types=[],
            performance_requirements={},
            reliability_requirements=[],
            usability_requirements=[]
        )
        self.mock_task_analyzer.extract_task_requirements.return_value = mock_requirements
        
        result = analyze_task_requirements("vague task", include_suggestions=True)
        result_dict = json.loads(result)
        
        assert result_dict["success"] is True
        assert "suggestions" in result_dict

    def test_empty_task_description_error(self):
        """Test error handling for empty task description."""
        result = analyze_task_requirements("")
        result_dict = json.loads(result)
        
        assert result_dict["success"] is False
        assert "error" in result_dict


class TestSuggestToolAlternatives(TestToolRecommendationMCPTools):
    """Tests for the suggest_tool_alternatives function."""

    def test_basic_alternatives_request(self):
        """Test basic alternative suggestions."""
        mock_response = AlternativeResponse(
            alternatives=[],
            comparison_analysis={"primary_advantages": ["test advantage"]},
            selection_guidance=["test guidance"],
            request_metadata={}
        )
        self.mock_recommendation_engine.get_alternative_recommendations.return_value = mock_response
        
        result = suggest_tool_alternatives("bash")
        result_dict = json.loads(result)
        
        assert result_dict["success"] is True
        assert "primary_tool" in result_dict
        assert result_dict["primary_tool"] == "bash"
        assert "alternatives" in result_dict

    def test_alternatives_with_context(self):
        """Test alternatives with task context."""
        mock_response = AlternativeResponse(
            alternatives=[],
            comparison_analysis={},
            selection_guidance=[],
            request_metadata={}
        )
        self.mock_recommendation_engine.get_alternative_recommendations.return_value = mock_response
        
        result = suggest_tool_alternatives("read", task_context="file processing", max_alternatives=3)
        result_dict = json.loads(result)
        
        assert result_dict["success"] is True
        assert result_dict["task_context"] == "file processing"

    def test_empty_primary_tool_error(self):
        """Test error handling for empty primary tool."""
        result = suggest_tool_alternatives("")
        result_dict = json.loads(result)
        
        assert result_dict["success"] is False
        assert "error" in result_dict


class TestRecommendToolSequence(TestToolRecommendationMCPTools):
    """Tests for the recommend_tool_sequence function."""

    def test_basic_sequence_request(self):
        """Test basic tool sequence recommendation."""
        mock_response = ToolSequenceResponse(
            sequences=[],
            workflow_analysis={"complexity": "moderate"},
            optimization_results={"efficiency": 0.8},
            request_metadata={}
        )
        self.mock_recommendation_engine.recommend_tool_sequence.return_value = mock_response
        
        result = recommend_tool_sequence("read file, process data, save results")
        result_dict = json.loads(result)
        
        assert result_dict["success"] is True
        assert "workflow_description" in result_dict
        assert "sequences" in result_dict

    def test_sequence_with_optimization_goal(self):
        """Test sequence with specific optimization goal."""
        mock_response = ToolSequenceResponse(
            sequences=[],
            workflow_analysis={"complexity": "moderate"},
            optimization_results={"efficiency": 0.9},
            request_metadata={}
        )
        self.mock_recommendation_engine.recommend_tool_sequence.return_value = mock_response
        
        result = recommend_tool_sequence(
            "complex data pipeline",
            optimization_goal="speed",
            max_sequence_length=5,
            allow_parallel_tools=True
        )
        result_dict = json.loads(result)
        
        assert result_dict["success"] is True
        assert result_dict["optimization_goal"] == "speed"

    def test_empty_workflow_description_error(self):
        """Test error handling for empty workflow description."""
        result = recommend_tool_sequence("")
        result_dict = json.loads(result)
        
        assert result_dict["success"] is False
        assert "error" in result_dict


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_create_task_context(self):
        """Test task context creation."""
        context = create_task_context(
            context_description="performance critical",
            complexity_preference="simple",
            user_preferences={"explain": True}
        )
        
        assert isinstance(context, TaskContext)
        assert context.user_context.complexity_preference == "simple"

    def test_parse_context_description(self):
        """Test context description parsing."""
        constraints = parse_context_description("performance critical large dataset")
        
        assert isinstance(constraints, EnvironmentalConstraints)
        assert constraints.resource_limits is not None

    def test_tool_exists(self):
        """Test tool existence checking."""
        assert tool_exists("bash") is True
        assert tool_exists("read") is True
        assert tool_exists("nonexistent_tool") is False

    def test_get_tool_availability_status(self):
        """Test tool availability status."""
        status = get_tool_availability_status()
        
        assert isinstance(status, dict)
        assert "recommendation_engine" in status
        assert "task_analyzer" in status


class TestErrorHandling:
    """Tests for error handling across all tools."""

    def test_uninitialized_engines(self):
        """Test behavior when engines are not initialized."""
        # Reset global state
        import tool_recommendation_mcp_tools
        tool_recommendation_mcp_tools._recommendation_engine = None
        tool_recommendation_mcp_tools._task_analyzer = None
        
        result = recommend_tools_for_task("test task")
        result_dict = json.loads(result)
        
        assert result_dict["success"] is False
        assert "error" in result_dict

    def test_invalid_parameters(self):
        """Test handling of invalid parameters."""
        # Test with None task description
        result = recommend_tools_for_task(None)
        result_dict = json.loads(result)
        assert result_dict["success"] is False

    def test_engine_exceptions(self):
        """Test handling of internal engine exceptions."""
        mock_engine = MagicMock()
        mock_analyzer = MagicMock()
        
        initialize_recommendation_tools(mock_engine, mock_analyzer)
        
        # Make analyzer raise exception
        mock_analyzer.analyze_task.side_effect = Exception("Analysis failed")
        
        result = recommend_tools_for_task("test task")
        result_dict = json.loads(result)
        
        assert result_dict["success"] is False


class TestIntegration:
    """Integration tests for the complete workflow."""

    @patch('tool_recommendation_mcp_tools._recommendation_engine')
    @patch('tool_recommendation_mcp_tools._task_analyzer')
    def test_end_to_end_recommendation_workflow(self, mock_analyzer, mock_engine):
        """Test complete end-to-end recommendation workflow."""
        # Setup realistic mock responses
        mock_task_analysis = TaskAnalysis(
            task_description="read and process JSON files",
            task_intent="read",
            task_category="file_operation",
            required_capabilities=["read", "json_processing"],
            input_specifications=["json"],
            output_specifications=["processed_data"],
            performance_constraints={},
            quality_requirements=["validation"],
            error_handling_needs=["error_logging"],
            complexity_level="moderate",
            estimated_steps=3,
            skill_level_required="intermediate",
            confidence=0.85,
            analysis_notes=["Well-defined task"]
        )
        
        # Create mock tool and recommendation
        mock_tool = Mock()
        mock_tool.name = "read"
        mock_tool.tool_id = "read"
        
        from recommendation_algorithms import ToolRecommendation
        mock_recommendation = Mock(spec=ToolRecommendation)
        mock_recommendation.tool = mock_tool
        mock_recommendation.confidence_level = 0.9
        mock_recommendation.relevance_score = 0.85
        mock_recommendation.task_alignment = 0.8
        mock_recommendation.recommendation_reasons = ["handles JSON files", "reliable file reading"]
        mock_recommendation.usage_guidance = ["specify file path", "handle parse errors"]
        mock_recommendation.complexity_assessment = "moderate"
        
        mock_response = RecommendationResponse(
            recommendations=[mock_recommendation],
            task_analysis=mock_task_analysis,
            context=Mock(),
            explanations=None,
            alternatives=None,
            request_metadata={}
        )
        
        mock_analyzer.analyze_task.return_value = mock_task_analysis
        mock_engine.recommend_for_task.return_value = mock_response
        
        # Test the workflow
        result = recommend_tools_for_task(
            "read and process JSON files",
            context="performance critical",
            explain_reasoning=True
        )
        
        # Verify response
        result_dict = json.loads(result)
        assert result_dict["success"] is True
        assert len(result_dict["recommendations"]) > 0
        assert result_dict["task_description"] == "read and process JSON files"


if __name__ == "__main__":
    pytest.main([__file__])