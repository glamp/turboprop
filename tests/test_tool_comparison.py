#!/usr/bin/env python3
"""
test_tool_comparison.py: Comprehensive tests for tool comparison system.

This module tests the tool comparison and alternative detection capabilities,
including multi-dimensional analysis, decision support, and result formatting.
"""

import pytest
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, Mock, patch

# Import modules that will be implemented
from alternative_detector import AlternativeAnalysis, AlternativeDetector, AlternativeAdvantageAnalysis
from comparison_formatter import ComparisonFormatter
from comparison_metrics import ComparisonMetrics
from decision_support import DecisionSupport, SelectionGuidance, TradeOffAnalysis
from tool_comparison_engine import (
    DetailedComparison,
    TaskComparisonResult,
    ToolComparisonEngine,
    ToolComparisonResult,
)

# Import existing dependencies
from context_analyzer import TaskContext
from mcp_metadata_types import ParameterAnalysis, ToolId
from tool_search_results import ToolSearchResult


class TestToolComparisonEngine:
    """Test the main tool comparison engine."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_alternative_detector = Mock(spec=AlternativeDetector)
        self.mock_comparison_metrics = Mock(spec=ComparisonMetrics)
        self.mock_decision_support = Mock(spec=DecisionSupport)
        
        self.engine = ToolComparisonEngine(
            alternative_detector=self.mock_alternative_detector,
            comparison_metrics=self.mock_comparison_metrics,
            decision_support=self.mock_decision_support
        )

    def test_compare_tools_basic(self):
        """Test basic tool comparison functionality."""
        # Arrange
        tool_ids = ["read", "write", "edit"]
        
        mock_metrics = {
            "read": {"functionality": 0.8, "usability": 0.9, "reliability": 0.7},
            "write": {"functionality": 0.7, "usability": 0.8, "reliability": 0.8},
            "edit": {"functionality": 0.9, "usability": 0.6, "reliability": 0.6}
        }
        self.mock_comparison_metrics.calculate_all_metrics.return_value = mock_metrics
        
        mock_guidance = SelectionGuidance(
            recommended_tool="read",
            confidence=0.85,
            key_factors=["high_usability", "good_reliability"],
            why_recommended=["Most user-friendly option", "Proven reliability"],
            when_to_reconsider=["Need complex editing features"],
            close_alternatives=["write"],
            fallback_options=["edit"],
            beginner_guidance="Start with 'read' for simplicity",
            advanced_user_guidance="Consider 'edit' for complex workflows",
            performance_critical_guidance="All options perform similarly"
        )
        self.mock_decision_support.generate_selection_guidance.return_value = mock_guidance
        
        mock_trade_offs = [
            TradeOffAnalysis(
                trade_off_name="functionality_vs_usability",
                tools_involved=["read", "edit"],
                competing_factors=["feature_richness", "ease_of_use"],
                magnitude=0.7,
                decision_importance="important",
                when_factor_a_matters=["Complex editing needs"],
                when_factor_b_matters=["Simple operations", "New users"],
                recommendation="Choose based on task complexity"
            )
        ]
        self.mock_decision_support.analyze_trade_offs.return_value = mock_trade_offs
        
        # Act
        result = self.engine.compare_tools(tool_ids)
        
        # Assert
        assert isinstance(result, ToolComparisonResult)
        assert result.compared_tools == tool_ids
        assert "read" in result.comparison_matrix
        assert "write" in result.comparison_matrix
        assert "edit" in result.comparison_matrix
        assert result.overall_ranking is not None
        assert result.decision_guidance == mock_guidance
        assert len(result.trade_off_analysis) == 1

    def test_compare_tools_with_criteria(self):
        """Test tool comparison with specific criteria."""
        # Arrange
        tool_ids = ["bash", "grep", "glob"]
        criteria = ["performance", "usability"]
        
        # Mock filtered metrics based on criteria
        mock_metrics = {
            "bash": {"performance": 0.9, "usability": 0.6},
            "grep": {"performance": 0.8, "usability": 0.8},
            "glob": {"performance": 0.7, "usability": 0.9}
        }
        self.mock_comparison_metrics.calculate_all_metrics.return_value = mock_metrics
        
        # Act
        result = self.engine.compare_tools(tool_ids, comparison_criteria=criteria)
        
        # Assert
        assert result.comparison_criteria == criteria
        assert all(tool in result.comparison_matrix for tool in tool_ids)
        for tool_metrics in result.comparison_matrix.values():
            assert set(tool_metrics.keys()) == set(criteria)

    def test_compare_for_task(self):
        """Test task-specific tool comparison."""
        # Arrange
        task_description = "search for code patterns in files"
        candidate_tools = ["grep", "glob"]
        
        # Mock task analysis results
        mock_task_metrics = {
            "grep": {"task_fit": 0.9, "complexity_match": 0.8},
            "glob": {"task_fit": 0.6, "complexity_match": 0.9}
        }
        self.mock_comparison_metrics.calculate_all_metrics.return_value = mock_task_metrics
        
        # Act
        result = self.engine.compare_for_task(task_description, candidate_tools)
        
        # Assert
        assert isinstance(result, TaskComparisonResult)
        assert result.task_description == task_description
        assert result.candidate_tools == candidate_tools
        assert "grep" in result.comparison_matrix
        assert "glob" in result.comparison_matrix

    def test_get_detailed_comparison(self):
        """Test detailed head-to-head comparison."""
        # Arrange
        tool_a = "read"
        tool_b = "edit"
        focus_areas = ["functionality", "complexity"]
        
        # Mock detailed comparison
        mock_detailed = DetailedComparison(
            tool_a="read",
            tool_b="edit",
            similarities=["Both handle file operations", "Similar error handling"],
            differences=["read is read-only", "edit supports modification"],
            tool_a_advantages=["Simpler interface", "Faster for viewing"],
            tool_b_advantages=["Full editing capabilities", "More powerful"],
            use_case_scenarios={
                "read": ["Quick file inspection", "Log analysis"],
                "edit": ["File modification", "Content replacement"]
            },
            switching_guidance=[
                "Use read for viewing only",
                "Use edit when modification needed"
            ],
            confidence=0.9
        )
        
        with patch.object(self.engine, '_perform_detailed_analysis', return_value=mock_detailed):
            # Act
            result = self.engine.get_detailed_comparison(tool_a, tool_b, focus_areas)
            
            # Assert
            assert isinstance(result, DetailedComparison)
            assert result.tool_a == tool_a
            assert result.tool_b == tool_b
            assert len(result.similarities) > 0
            assert len(result.differences) > 0

    def test_empty_tool_list(self):
        """Test behavior with empty tool list."""
        # Act & Assert
        with pytest.raises(ValueError, match="At least two tools are required"):
            self.engine.compare_tools([])

    def test_single_tool_comparison(self):
        """Test behavior with single tool."""
        # Act & Assert
        with pytest.raises(ValueError, match="At least two tools are required"):
            self.engine.compare_tools(["read"])


class TestAlternativeDetector:
    """Test the alternative detection system."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_search_engine = Mock()
        self.mock_similarity_analyzer = Mock()
        
        self.detector = AlternativeDetector(
            tool_search_engine=self.mock_search_engine,
            similarity_analyzer=self.mock_similarity_analyzer
        )

    def test_find_alternatives_basic(self):
        """Test basic alternative detection."""
        # Arrange
        reference_tool = "bash"
        
        # Mock search results for similar tools
        mock_search_results = [
            ToolSearchResult(
                tool_id=ToolId("task"),
                name="task",
                description="Execute tasks with controlled environment",
                category="execution",
                tool_type="function",
                similarity_score=0.8,
                relevance_score=0.7,
                confidence_level="high"
            ),
            ToolSearchResult(
                tool_id=ToolId("subprocess"),
                name="subprocess", 
                description="Run subprocess commands",
                category="execution",
                tool_type="function",
                similarity_score=0.7,
                relevance_score=0.6,
                confidence_level="medium"
            )
        ]
        
        self.mock_search_engine.search_by_functionality.return_value.results = mock_search_results
        
        # Mock similarity analysis
        mock_similarity_scores = {"task": 0.85, "subprocess": 0.75}
        self.mock_similarity_analyzer.calculate_functional_similarity.side_effect = lambda ref, alt: mock_similarity_scores.get(alt, 0.5)
        
        # Act
        alternatives = self.detector.find_alternatives(reference_tool, similarity_threshold=0.7)
        
        # Assert
        assert len(alternatives) == 2
        assert all(isinstance(alt, AlternativeAnalysis) for alt in alternatives)
        assert alternatives[0].tool_id == "task"
        assert alternatives[0].similarity_score >= 0.7
        assert alternatives[1].tool_id == "subprocess"

    def test_find_alternatives_with_threshold(self):
        """Test alternative detection with similarity threshold."""
        # Arrange
        reference_tool = "grep"
        similarity_threshold = 0.8
        
        # Mock search results with varying similarity
        mock_search_results = [
            ToolSearchResult(
                tool_id=ToolId("regex_search"),
                name="regex_search",
                description="Advanced regex pattern matching",
                category="search",
                tool_type="function",
                similarity_score=0.9,
                relevance_score=0.85,
                confidence_level="high"
            ),
            ToolSearchResult(
                tool_id=ToolId("find"),
                name="find",
                description="Find files by various criteria", 
                category="search",
                tool_type="function",
                similarity_score=0.6,  # Below threshold
                relevance_score=0.5,
                confidence_level="low"
            )
        ]
        
        self.mock_search_engine.search_by_functionality.return_value.results = mock_search_results
        self.mock_similarity_analyzer.calculate_functional_similarity.side_effect = lambda ref, alt: {"regex_search": 0.85, "find": 0.6}.get(alt, 0.5)
        
        # Act
        alternatives = self.detector.find_alternatives(reference_tool, similarity_threshold=similarity_threshold)
        
        # Assert
        assert len(alternatives) == 1  # Only regex_search should be included
        assert alternatives[0].tool_id == "regex_search"
        assert alternatives[0].similarity_score >= similarity_threshold

    def test_detect_functional_groups(self):
        """Test functional grouping of tools."""
        # Arrange
        all_tools = ["read", "write", "edit", "grep", "glob", "bash"]
        
        # Mock similarity matrix
        def mock_similarity(tool1, tool2):
            file_ops = {"read", "write", "edit"}
            search_ops = {"grep", "glob"}
            exec_ops = {"bash"}
            
            if tool1 in file_ops and tool2 in file_ops:
                return 0.8
            elif tool1 in search_ops and tool2 in search_ops:
                return 0.8
            elif tool1 in exec_ops and tool2 in exec_ops:
                return 0.8
            else:
                return 0.3
        
        self.mock_similarity_analyzer.calculate_functional_similarity.side_effect = mock_similarity
        
        # Act
        groups = self.detector.detect_functional_groups(all_tools)
        
        # Assert
        assert isinstance(groups, dict)
        assert len(groups) >= 2  # Should identify at least file ops and search ops groups
        
        # Verify file operations group
        file_group_found = any("read" in group and "write" in group for group in groups.values())
        assert file_group_found
        
        # Verify search operations group  
        search_group_found = any("grep" in group and "glob" in group for group in groups.values())
        assert search_group_found

    def test_analyze_alternative_advantages(self):
        """Test alternative advantage analysis."""
        # Arrange
        primary_tool = "read"
        alternative_tool = "edit"
        
        # Mock similarity analyzer to return a float for switching cost analysis
        self.mock_similarity_analyzer.calculate_functional_similarity.return_value = 0.7
        
        # Mock tool metadata for analysis
        with patch.object(self.detector, '_get_tool_capabilities') as mock_capabilities:
            mock_capabilities.side_effect = lambda tool: {
                "read": ["file_reading", "text_output", "simple_interface"],
                "edit": ["file_reading", "file_writing", "text_modification", "complex_interface"]
            }.get(tool, [])
            
            # Act
            analysis = self.detector.analyze_alternative_advantages(primary_tool, alternative_tool)
            
            # Assert
            assert isinstance(analysis, AlternativeAdvantageAnalysis)
            assert analysis.primary_tool == primary_tool
            assert analysis.alternative_tool == alternative_tool
            assert "file_reading" in analysis.shared_capabilities
            assert "file_writing" in analysis.alternative_unique_capabilities
            assert "text_modification" in analysis.alternative_unique_capabilities


class TestComparisonMetrics:
    """Test the comparison metrics system."""

    def setup_method(self):
        """Set up test fixtures."""
        self.metrics = ComparisonMetrics()

    def test_calculate_all_metrics(self):
        """Test comprehensive metrics calculation."""
        # Arrange
        mock_tools = [
            ToolSearchResult(
                tool_id=ToolId("read"),
                name="read",
                description="Read file contents",
                category="file_operations",
                tool_type="function",
                similarity_score=0.8,
                relevance_score=0.8,
                confidence_level="high",
                parameters=[
                    ParameterAnalysis(name="file_path", type="string", required=True, description="Path to file"),
                    ParameterAnalysis(name="limit", type="number", required=False, description="Line limit")
                ],
                parameter_count=2,
                required_parameter_count=1
            ),
            ToolSearchResult(
                tool_id=ToolId("edit"),
                name="edit", 
                description="Edit file contents",
                category="file_operations",
                tool_type="function",
                similarity_score=0.8,
                relevance_score=0.8,
                confidence_level="high",
                parameters=[
                    ParameterAnalysis(name="file_path", type="string", required=True, description="Path to file"),
                    ParameterAnalysis(name="old_string", type="string", required=True, description="Text to replace"),
                    ParameterAnalysis(name="new_string", type="string", required=True, description="Replacement text"),
                    ParameterAnalysis(name="replace_all", type="boolean", required=False, description="Replace all occurrences")
                ],
                parameter_count=4,
                required_parameter_count=3
            )
        ]
        
        # Act
        result = self.metrics.calculate_all_metrics(mock_tools)
        
        # Assert
        assert isinstance(result, dict)
        assert "read" in result
        assert "edit" in result
        
        # Check that all required metrics are present
        for tool_id, metrics in result.items():
            assert "functionality" in metrics
            assert "usability" in metrics
            assert "reliability" in metrics
            assert "performance" in metrics
            assert "compatibility" in metrics
            assert "documentation" in metrics
            
            # Verify metrics are normalized (0-1 range)
            for metric_value in metrics.values():
                assert 0.0 <= metric_value <= 1.0

    def test_calculate_functionality_score(self):
        """Test functionality scoring algorithm."""
        # Arrange
        simple_tool = ToolSearchResult(
            tool_id=ToolId("simple"),
            name="simple",
            description="Simple operation",
            category="basic",
            tool_type="function",
            similarity_score=0.8,
            relevance_score=0.8,
            confidence_level="high",
            parameter_count=1,
            required_parameter_count=1,
            parameters=[ParameterAnalysis(name="input", type="string", required=True, description="Input value")]
        )
        
        complex_tool = ToolSearchResult(
            tool_id=ToolId("complex"),
            name="complex",
            description="Complex multi-step operation with advanced features",
            category="advanced",
            tool_type="function",
            similarity_score=0.8,
            relevance_score=0.8,
            confidence_level="high",
            parameter_count=10,
            required_parameter_count=5,
            parameters=[ParameterAnalysis(name=f"param_{i}", type="string", required=i<5, description=f"Parameter {i}") for i in range(10)]
        )
        
        # Act
        simple_score = self.metrics.calculate_functionality_score(simple_tool)
        complex_score = self.metrics.calculate_functionality_score(complex_tool)
        
        # Assert
        assert 0.0 <= simple_score <= 1.0
        assert 0.0 <= complex_score <= 1.0
        assert complex_score > simple_score  # Complex tool should have higher functionality score

    def test_calculate_usability_score(self):
        """Test usability scoring algorithm."""
        # Arrange
        user_friendly_tool = ToolSearchResult(
            tool_id=ToolId("friendly"),
            name="friendly",
            description="Very user-friendly tool with clear documentation",
            category="user_friendly",
            tool_type="function",
            similarity_score=0.8,
            relevance_score=0.8,
            confidence_level="high",
            parameter_count=2,
            required_parameter_count=1,
            parameters=[
                ParameterAnalysis(name="file", type="string", required=True, description="File to process"),
                ParameterAnalysis(name="verbose", type="boolean", required=False, description="Enable verbose output")
            ]
        )
        
        complex_tool = ToolSearchResult(
            tool_id=ToolId("complex"),
            name="complex", 
            description="Complex tool requiring expertise",
            category="advanced",
            tool_type="function",
            similarity_score=0.8,
            relevance_score=0.8,
            confidence_level="high",
            parameter_count=8,
            required_parameter_count=6,
            parameters=[ParameterAnalysis(name=f"param_{i}", type="object", required=i<6, description="") for i in range(8)]
        )
        
        # Act
        friendly_score = self.metrics.calculate_usability_score(user_friendly_tool)
        complex_score = self.metrics.calculate_usability_score(complex_tool)
        
        # Assert
        assert 0.0 <= friendly_score <= 1.0
        assert 0.0 <= complex_score <= 1.0
        assert friendly_score > complex_score  # User-friendly tool should have higher usability score

    def test_calculate_reliability_score(self):
        """Test reliability scoring algorithm."""
        # Arrange
        reliable_tool = ToolSearchResult(
            tool_id=ToolId("reliable"),
            name="reliable",
            description="Highly reliable tool with extensive error handling",
            category="stable",
            tool_type="function",
            similarity_score=0.8,
            relevance_score=0.8,
            confidence_level="high",
            parameter_count=3,
            required_parameter_count=2
        )
        
        # Act
        score = self.metrics.calculate_reliability_score(reliable_tool)
        
        # Assert
        assert 0.0 <= score <= 1.0


class TestDecisionSupport:
    """Test the decision support system."""

    def setup_method(self):
        """Set up test fixtures."""
        self.decision_support = DecisionSupport()

    def test_generate_selection_guidance(self):
        """Test selection guidance generation."""
        # Arrange
        comparison_result = ToolComparisonResult(
            compared_tools=["read", "edit", "write"],
            comparison_matrix={
                "read": {"functionality": 0.7, "usability": 0.9, "reliability": 0.8},
                "edit": {"functionality": 0.9, "usability": 0.6, "reliability": 0.7},
                "write": {"functionality": 0.8, "usability": 0.8, "reliability": 0.8}
            },
            overall_ranking=["read", "write", "edit"],
            category_rankings={},
            key_differentiators=["usability", "functionality"],
            trade_off_analysis=[],
            decision_guidance=None,  # Will be filled by method
            comparison_criteria=[],
            context_factors=[],
            confidence_scores={"read": 0.9, "edit": 0.7, "write": 0.8}
        )
        
        context = TaskContext()
        
        # Act
        guidance = self.decision_support.generate_selection_guidance(comparison_result, context)
        
        # Assert
        assert isinstance(guidance, SelectionGuidance)
        assert guidance.recommended_tool in comparison_result.compared_tools
        assert 0.0 <= guidance.confidence <= 1.0
        assert len(guidance.key_factors) > 0
        assert len(guidance.why_recommended) > 0

    def test_analyze_trade_offs(self):
        """Test trade-off analysis."""
        # Arrange
        tools = [
            ToolSearchResult(
                tool_id=ToolId("fast"), 
                name="fast",
                description="Fast but limited features",
                category="performance",
                tool_type="function",
                similarity_score=0.8,
                relevance_score=0.8,
                confidence_level="high"
            ),
            ToolSearchResult(
                tool_id=ToolId("feature_rich"),
                name="feature_rich", 
                description="Many features but slower",
                category="comprehensive",
                tool_type="function",
                similarity_score=0.8,
                relevance_score=0.8,
                confidence_level="high"
            )
        ]
        
        metrics = {
            "fast": {"performance": 0.9, "functionality": 0.6},
            "feature_rich": {"performance": 0.6, "functionality": 0.9}
        }
        
        # Act
        trade_offs = self.decision_support.analyze_trade_offs(tools, metrics)
        
        # Assert
        assert isinstance(trade_offs, list)
        assert len(trade_offs) > 0
        
        performance_vs_functionality = next((t for t in trade_offs if "performance" in t.trade_off_name.lower()), None)
        assert performance_vs_functionality is not None
        assert "fast" in performance_vs_functionality.tools_involved
        assert "feature_rich" in performance_vs_functionality.tools_involved

    def test_create_decision_tree(self):
        """Test decision tree creation."""
        # Arrange
        tools = ["read", "edit", "multiedit"]
        context = TaskContext()
        
        # Act  
        decision_tree = self.decision_support.create_decision_tree(tools, context)
        
        # Assert
        assert decision_tree is not None
        # Decision tree should provide a structured way to choose between tools


class TestComparisonFormatter:
    """Test the comparison result formatting."""

    def setup_method(self):
        """Set up test fixtures."""
        self.formatter = ComparisonFormatter()

    def test_format_comparison_table(self):
        """Test comparison table formatting."""
        # Arrange
        comparison_result = ToolComparisonResult(
            compared_tools=["read", "write"],
            comparison_matrix={
                "read": {"functionality": 0.8, "usability": 0.9},
                "write": {"functionality": 0.7, "usability": 0.8}
            },
            overall_ranking=["read", "write"],
            category_rankings={},
            key_differentiators=["usability"],
            trade_off_analysis=[],
            decision_guidance=None,
            comparison_criteria=["functionality", "usability"],
            context_factors=[],
            confidence_scores={"read": 0.9, "write": 0.8}
        )
        
        # Act
        table = self.formatter.format_comparison_table(comparison_result)
        
        # Assert
        assert isinstance(table, str)
        assert "read" in table
        assert "write" in table
        assert "Functionality" in table  # Formatter uses title case
        assert "Usability" in table      # Formatter uses title case

    def test_format_decision_summary(self):
        """Test decision summary formatting."""
        # Arrange
        guidance = SelectionGuidance(
            recommended_tool="read",
            confidence=0.85,
            key_factors=["high_usability", "reliability"],
            why_recommended=["Most user-friendly", "Proven track record"],
            when_to_reconsider=["Need editing capabilities"],
            close_alternatives=["write"],
            fallback_options=["edit"],
            beginner_guidance="Start with read for simplicity",
            advanced_user_guidance="Consider edit for complex tasks",
            performance_critical_guidance="All options perform similarly"
        )
        
        # Act
        summary = self.formatter.format_decision_summary(guidance)
        
        # Assert
        assert isinstance(summary, str)
        assert "read" in summary
        assert str(guidance.confidence) in summary or "85" in summary
        assert "user-friendly" in summary.lower() or "usability" in summary.lower()

    def test_format_trade_off_analysis(self):
        """Test trade-off analysis formatting."""
        # Arrange
        trade_offs = [
            TradeOffAnalysis(
                trade_off_name="performance_vs_features",
                tools_involved=["fast_tool", "feature_rich_tool"],
                competing_factors=["speed", "functionality"],
                magnitude=0.8,
                decision_importance="critical",
                when_factor_a_matters=["Time-sensitive operations"],
                when_factor_b_matters=["Complex data processing"],
                recommendation="Choose based on primary use case"
            )
        ]
        
        # Act
        formatted = self.formatter.format_trade_off_analysis(trade_offs)
        
        # Assert
        assert isinstance(formatted, str)
        assert "Performance Vs Features" in formatted  # Formatter converts underscores to spaces and uses title case
        assert "fast_tool" in formatted
        assert "feature_rich_tool" in formatted


# Integration tests
class TestToolComparisonIntegration:
    """Test integration with existing search and recommendation systems."""

    def setup_method(self):
        """Set up integration test fixtures."""
        # These will test with mock versions of existing systems
        pass

    def test_integration_with_search_engine(self):
        """Test integration with MCPToolSearchEngine."""
        # This will test that comparison engine can work with search results
        pass

    def test_integration_with_recommendation_engine(self):
        """Test integration with ToolRecommendationEngine."""  
        # This will test that comparison results enhance recommendations
        pass


# Test scenarios from the issue specification
class TestComparisonScenarios:
    """Test specific comparison scenarios outlined in the issue."""

    def test_file_operation_tools_scenario(self):
        """Test comparison of file operation tools."""
        # This should test comparing read, write, edit, multiedit
        # and identify complexity, batch_operations, error_handling as differentiators
        pass

    def test_search_tools_scenario(self):
        """Test comparison of search tools."""
        # This should test comparing grep, glob
        # and identify search_type, pattern_support, performance as differentiators
        pass

    def test_execution_tools_scenario(self):
        """Test comparison of execution tools."""
        # This should test comparing bash and alternatives
        # and identify flexibility_vs_safety, power_vs_simplicity trade-offs
        pass


if __name__ == "__main__":
    pytest.main([__file__])