#!/usr/bin/env python3
"""
test_comparison_mcp_tools.py: Tests for MCP comparison and category tools.

This module tests the MCP tools for tool comparison, alternative finding,
and category browsing functionality.
"""

import json
import pytest
from unittest.mock import Mock, patch, MagicMock

from tool_comparison_mcp_tools import (
    compare_mcp_tools,
    find_tool_alternatives,
    analyze_tool_relationships,
    get_tool_recommendations_comparison,
    initialize_comparison_tools,
)
from tool_category_mcp_tools import (
    browse_tools_by_category,
    get_category_overview,
    get_tool_selection_guidance,
    find_tools_by_complexity,
    explore_tool_ecosystem,
    initialize_category_tools,
)
from comparison_response_types import (
    ToolComparisonMCPResponse,
    AlternativesFoundResponse,
    AlternativeAnalysis,
    ToolRelationshipsResponse,
    CategoryBrowseResponse,
    CategoryOverviewResponse,
    SelectionGuidanceResponse,
    create_error_response,
)


class TestComparisonMCPTools:
    """Test cases for tool comparison MCP tools."""

    def setup_method(self):
        """Set up test fixtures."""
        # Mock the global components
        self.mock_comparison_engine = Mock()
        self.mock_decision_support = Mock()
        self.mock_alternative_detector = Mock()
        self.mock_relationship_analyzer = Mock()
        
        # Initialize with mocks
        initialize_comparison_tools(
            self.mock_comparison_engine,
            self.mock_decision_support,
            self.mock_alternative_detector,
            self.mock_relationship_analyzer,
        )

    @patch('tool_comparison_mcp_tools.tool_exists')
    def test_compare_mcp_tools_success(self, mock_tool_exists):
        """Test successful tool comparison."""
        # Setup
        mock_tool_exists.return_value = True
        tool_ids = ['read', 'write', 'edit']
        
        # Mock comparison result
        mock_result = Mock()
        mock_result.to_dict.return_value = {'comparison': 'data'}
        self.mock_comparison_engine.compare_tools.return_value = mock_result
        
        # Mock decision guidance
        mock_guidance = Mock()
        self.mock_decision_support.generate_selection_guidance.return_value = mock_guidance
        
        # Execute
        result = compare_mcp_tools(
            tool_ids=tool_ids,
            comparison_criteria=['functionality', 'usability'],
            include_decision_guidance=True
        )
        
        # Verify
        assert isinstance(result, dict)
        assert result['success'] is True
        assert result['tool_ids'] == tool_ids
        self.mock_comparison_engine.compare_tools.assert_called_once()

    @patch('tool_comparison_mcp_tools.tool_exists')
    def test_compare_mcp_tools_invalid_tool_count(self, mock_tool_exists):
        """Test comparison with invalid number of tools."""
        # Test too few tools
        result = compare_mcp_tools(tool_ids=['single_tool'])
        assert result['success'] is False
        assert 'At least 2 tools required' in result['error']
        
        # Test too many tools
        many_tools = [f'tool_{i}' for i in range(15)]
        result = compare_mcp_tools(tool_ids=many_tools)
        assert result['success'] is False
        assert 'Maximum 10 tools' in result['error']

    @patch('tool_comparison_mcp_tools.tool_exists')
    def test_compare_mcp_tools_missing_tools(self, mock_tool_exists):
        """Test comparison with non-existent tools."""
        # Setup - first tool exists, second doesn't
        mock_tool_exists.side_effect = lambda tool: tool == 'read'
        
        result = compare_mcp_tools(tool_ids=['read', 'nonexistent'])
        
        assert result['success'] is False
        assert 'Tools not found: nonexistent' in result['error']

    @patch('tool_comparison_mcp_tools.tool_exists')
    def test_find_tool_alternatives_success(self, mock_tool_exists):
        """Test successful alternative finding."""
        # Setup
        mock_tool_exists.return_value = True
        
        # Mock alternative analysis
        mock_alternatives = [
            AlternativeAnalysis(
                tool_id='edit',
                similarity_score=0.8,
                complexity_comparison='similar',
                learning_curve='easy',
                unique_capabilities=['syntax_highlighting'],
                performance_comparison='similar'
            ),
            AlternativeAnalysis(
                tool_id='multiedit',
                similarity_score=0.7,
                complexity_comparison='more_complex',
                learning_curve='moderate',
                unique_capabilities=['batch_editing'],
                performance_comparison='faster'
            )
        ]
        self.mock_alternative_detector.find_alternatives.return_value = mock_alternatives
        
        # Execute
        result = find_tool_alternatives(
            reference_tool='read',
            similarity_threshold=0.6,
            max_alternatives=5
        )
        
        # Verify
        assert isinstance(result, dict)
        assert result['success'] is True
        assert result['reference_tool'] == 'read'
        assert len(result['alternatives']) == 2
        self.mock_alternative_detector.find_alternatives.assert_called_once()

    @patch('tool_comparison_mcp_tools.tool_exists')
    def test_find_tool_alternatives_nonexistent_tool(self, mock_tool_exists):
        """Test alternative finding with non-existent reference tool."""
        mock_tool_exists.return_value = False
        
        result = find_tool_alternatives(reference_tool='nonexistent')
        
        assert result['success'] is False
        assert 'Reference tool \'nonexistent\' not found' in result['error']

    @patch('tool_comparison_mcp_tools.tool_exists')
    def test_analyze_tool_relationships_success(self, mock_tool_exists):
        """Test successful relationship analysis."""
        # Setup
        mock_tool_exists.return_value = True
        
        mock_relationships = {
            'alternatives': ['edit', 'multiedit'],
            'complements': ['grep', 'bash'],
            'prerequisites': ['ls']
        }
        self.mock_relationship_analyzer.analyze_tool_relationships.return_value = mock_relationships
        
        # Execute
        result = analyze_tool_relationships(
            tool_id='read',
            relationship_types=['alternatives', 'complements'],
            include_explanations=True
        )
        
        # Verify
        assert isinstance(result, dict)
        assert result['success'] is True
        assert result['tool_id'] == 'read'
        assert 'alternatives' in result['relationships']
        assert 'complements' in result['relationships']
        self.mock_relationship_analyzer.analyze_tool_relationships.assert_called_once()


class TestCategoryMCPTools:
    """Test cases for tool category MCP tools."""

    def setup_method(self):
        """Set up test fixtures."""
        # Mock the global components
        self.mock_tool_catalog = Mock()
        self.mock_context_analyzer = Mock()
        self.mock_task_analyzer = Mock()
        self.mock_decision_support = Mock()
        
        # Initialize with mocks
        initialize_category_tools(
            self.mock_tool_catalog,
            self.mock_context_analyzer,
            self.mock_task_analyzer,
            self.mock_decision_support,
        )

    def test_browse_tools_by_category_success(self):
        """Test successful category browsing."""
        # Setup
        from comparison_response_types import ToolSearchResult
        
        mock_tools = [
            ToolSearchResult(
                tool_id='read',
                name='Read File',
                description='Read file contents',
                category='file_ops',
                complexity='simple',
                score=0.9
            ),
            ToolSearchResult(
                tool_id='write',
                name='Write File', 
                description='Write file contents',
                category='file_ops',
                complexity='simple',
                score=0.85
            )
        ]
        self.mock_tool_catalog.get_tools_by_category.return_value = mock_tools
        
        # Execute
        result = browse_tools_by_category(
            category='file_ops',
            sort_by='popularity',
            max_tools=10
        )
        
        # Verify
        assert isinstance(result, dict)
        assert result['success'] is True
        assert result['category'] == 'file_ops'
        assert len(result['tools']) == 2
        self.mock_tool_catalog.get_tools_by_category.assert_called_once()

    def test_browse_tools_by_category_invalid_category(self):
        """Test browsing with invalid category."""
        result = browse_tools_by_category(category='invalid_category')
        
        assert result['success'] is False
        assert 'Invalid category' in result['error']

    def test_get_category_overview_success(self):
        """Test successful category overview."""
        # Setup
        mock_categories = [
            {'name': 'file_ops', 'tool_count': 5, 'description': 'File operations'},
            {'name': 'web', 'tool_count': 3, 'description': 'Web operations'}
        ]
        self.mock_tool_catalog.get_all_categories.return_value = mock_categories
        
        # Execute
        result = get_category_overview()
        
        # Verify
        assert isinstance(result, dict)
        assert result['success'] is True
        assert len(result['categories']) == 2
        assert 'ecosystem_stats' in result
        self.mock_tool_catalog.get_all_categories.assert_called_once()

    def test_get_tool_selection_guidance_success(self):
        """Test successful tool selection guidance."""
        # Setup
        mock_task_analysis = Mock()
        mock_task_analysis.to_dict.return_value = {'complexity': 'simple'}
        self.mock_task_analyzer.analyze_task.return_value = mock_task_analysis
        
        mock_guidance = Mock()
        self.mock_decision_support.get_tool_selection_guidance.return_value = mock_guidance
        
        # Execute
        result = get_tool_selection_guidance(
            task_description='read a configuration file',
            constraints=['no complex tools'],
            optimization_goal='simplicity'
        )
        
        # Verify
        assert isinstance(result, dict)
        assert result['success'] is True
        assert result['task_description'] == 'read a configuration file'
        self.mock_task_analyzer.analyze_task.assert_called_once()
        self.mock_decision_support.get_tool_selection_guidance.assert_called_once()

    def test_find_tools_by_complexity_success(self):
        """Test successful complexity-based tool finding."""
        # Setup
        mock_tools = [
            Mock(to_dict=lambda: {'tool_id': 'read', 'complexity': 'simple'}),
            Mock(to_dict=lambda: {'tool_id': 'ls', 'complexity': 'simple'})
        ]
        self.mock_tool_catalog.get_tools_by_complexity.return_value = mock_tools
        
        # Execute
        result = find_tools_by_complexity(
            complexity_level='simple',
            category_filter='file_ops'
        )
        
        # Verify
        assert isinstance(result, dict)
        assert result['success'] is True
        assert result['complexity_level'] == 'simple'
        assert len(result['tools']) == 2
        self.mock_tool_catalog.get_tools_by_complexity.assert_called_once()

    def test_find_tools_by_complexity_invalid_level(self):
        """Test complexity finding with invalid level."""
        result = find_tools_by_complexity(complexity_level='invalid')
        
        assert result['success'] is False
        assert 'Invalid complexity' in result['error']


class TestResponseTypes:
    """Test cases for response type classes."""

    def test_tool_comparison_mcp_response_creation(self):
        """Test ToolComparisonMCPResponse creation and serialization."""
        from tool_comparison_engine import ToolComparisonResult
        
        # Create mock comparison result
        mock_comparison_result = Mock(spec=ToolComparisonResult)
        mock_comparison_result.to_dict.return_value = {'test': 'data'}
        
        response = ToolComparisonMCPResponse(
            tool_ids=['read', 'write'],
            comparison_result=mock_comparison_result,
            comparison_criteria=['functionality'],
            detail_level='standard'
        )
        
        result_dict = response.to_dict()
        
        assert result_dict['tool_ids'] == ['read', 'write']
        assert result_dict['comparison_criteria'] == ['functionality']
        assert result_dict['success'] is True
        assert 'timestamp' in result_dict

    def test_alternatives_found_response_creation(self):
        """Test AlternativesFoundResponse creation and serialization."""
        alternatives = [
            AlternativeAnalysis(
                tool_id='edit',
                similarity_score=0.8,
                complexity_comparison='similar',
                learning_curve='easy',
                unique_capabilities=['highlighting'],
                performance_comparison='similar'
            )
        ]
        
        response = AlternativesFoundResponse(
            reference_tool='read',
            alternatives=alternatives,
            similarity_threshold=0.7,
            context_filter='simple'
        )
        
        result_dict = response.to_dict()
        
        assert result_dict['reference_tool'] == 'read'
        assert len(result_dict['alternatives']) == 1
        assert result_dict['similarity_threshold'] == 0.7
        assert result_dict['success'] is True

    def test_create_error_response(self):
        """Test error response creation."""
        error_response = create_error_response(
            'test_tool',
            'Test error message',
            'test_context'
        )
        
        assert error_response['tool_name'] == 'test_tool'
        assert error_response['error'] == 'Test error message'
        assert error_response['context'] == 'test_context'
        assert error_response['success'] is False
        assert 'timestamp' in error_response


if __name__ == '__main__':
    pytest.main([__file__])