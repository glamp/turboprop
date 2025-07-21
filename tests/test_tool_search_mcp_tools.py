#!/usr/bin/env python3
"""
Tests for MCP Tool Search Functionality

Comprehensive test suite for the MCP tool search tools including parameter validation,
response formatting, error handling, and integration with search engines.
"""

import json
from unittest.mock import Mock, patch

import pytest

from mcp_metadata_types import ToolId
from mcp_tool_search_responses import (
    CapabilityMatch,
    CapabilitySearchResponse,
    ToolCategoriesResponse,
    ToolCategory,
    ToolDetailsResponse,
    ToolSearchMCPResponse,
    create_error_response,
)
from mcp_tool_validator import (
    MCPToolValidator,
    ValidatedCapabilityParams,
    ValidatedSearchParams,
    ValidatedToolDetailsParams,
    generate_error_suggestions,
)
from tool_search_mcp_tools import (
    get_tool_details,
    initialize_search_engines,
    list_tool_categories,
    search_mcp_tools,
    search_tools_by_capability,
)
from tool_search_results import ToolSearchResult


class TestMCPToolValidator:
    """Test the MCP tool parameter validator."""

    def setup_method(self):
        """Set up test fixtures."""
        self.validator = MCPToolValidator()

    def test_validate_search_parameters_valid(self):
        """Test validation of valid search parameters."""
        result = self.validator.validate_search_parameters(
            query="file operations",
            category="file_ops",
            tool_type="system",
            max_results=10,
            include_examples=True,
            search_mode="hybrid",
        )

        assert isinstance(result, ValidatedSearchParams)
        assert result.query == "file operations"
        assert result.category == "file_ops"
        assert result.tool_type == "system"
        assert result.max_results == 10
        assert result.include_examples is True
        assert result.search_mode == "hybrid"

    def test_validate_search_parameters_empty_query(self):
        """Test validation fails for empty query."""
        with pytest.raises(ValueError, match="Query cannot be empty"):
            self.validator.validate_search_parameters(
                query="", category=None, tool_type=None, max_results=10, include_examples=True, search_mode="hybrid"
            )

    def test_validate_search_parameters_invalid_category(self):
        """Test validation fails for invalid category."""
        with pytest.raises(ValueError, match="Invalid category"):
            self.validator.validate_search_parameters(
                query="test query",
                category="invalid_category",
                tool_type=None,
                max_results=10,
                include_examples=True,
                search_mode="hybrid",
            )

    def test_validate_search_parameters_invalid_max_results(self):
        """Test validation fails for invalid max_results."""
        with pytest.raises(ValueError, match="max_results must be between"):
            self.validator.validate_search_parameters(
                query="test query",
                category=None,
                tool_type=None,
                max_results=100,  # Too high
                include_examples=True,
                search_mode="hybrid",
            )

    def test_validate_tool_details_parameters_valid(self):
        """Test validation of valid tool details parameters."""
        result = self.validator.validate_tool_details_parameters(
            tool_id="bash",
            include_schema=True,
            include_examples=False,
            include_relationships=True,
            include_usage_guidance=False,
        )

        assert isinstance(result, ValidatedToolDetailsParams)
        assert result.tool_id == "bash"
        assert result.include_schema is True
        assert result.include_examples is False

    def test_validate_tool_details_parameters_empty_tool_id(self):
        """Test validation fails for empty tool ID."""
        with pytest.raises(ValueError, match="Tool ID cannot be empty"):
            self.validator.validate_tool_details_parameters(
                tool_id="",
                include_schema=True,
                include_examples=True,
                include_relationships=True,
                include_usage_guidance=True,
            )

    def test_validate_capability_parameters_valid(self):
        """Test validation of valid capability parameters."""
        result = self.validator.validate_capability_parameters(
            capability_description="timeout support",
            required_parameters=["timeout", "max_time"],
            preferred_complexity="simple",
            max_results=5,
        )

        assert isinstance(result, ValidatedCapabilityParams)
        assert result.capability_description == "timeout support"
        assert result.required_parameters == ["timeout", "max_time"]
        assert result.preferred_complexity == "simple"

    def test_sanitize_query(self):
        """Test query sanitization."""
        # Test normal query
        result = self.validator.sanitize_query("  normal query  ")
        assert result == "normal query"

        # Test query with control characters
        result = self.validator.sanitize_query("query\x00with\x01control\x02chars")
        assert result == "querywithcontrolchars"

        # Test empty query
        result = self.validator.sanitize_query("")
        assert result == ""


class TestMCPToolSearchResponses:
    """Test the MCP tool search response types."""

    def test_tool_search_mcp_response_creation(self):
        """Test creation of ToolSearchMCPResponse."""
        tool_result = ToolSearchResult(
            tool_id=ToolId("test_tool"),
            name="Test Tool",
            description="A test tool",
            category="test",
            tool_type="system",
            similarity_score=0.9,
            relevance_score=0.8,
            confidence_level="high",
        )

        response = ToolSearchMCPResponse(query="test query", results=[tool_result], search_mode="hybrid")

        assert response.query == "test query"
        assert len(response.results) == 1
        assert response.total_results == 1
        assert response.search_mode == "hybrid"
        assert response.timestamp is not None

    def test_tool_search_mcp_response_to_dict(self):
        """Test ToolSearchMCPResponse.to_dict() method."""
        tool_result = ToolSearchResult(
            tool_id=ToolId("test_tool"),
            name="Test Tool",
            description="A test tool",
            category="test",
            tool_type="system",
            similarity_score=0.9,
            relevance_score=0.8,
            confidence_level="high",
        )

        response = ToolSearchMCPResponse(query="test query", results=[tool_result], search_mode="hybrid")

        result_dict = response.to_dict()

        assert result_dict["success"] is True
        assert result_dict["query"] == "test query"
        assert result_dict["total_results"] == 1
        assert len(result_dict["results"]) == 1

    def test_tool_details_response_creation(self):
        """Test creation of ToolDetailsResponse."""
        tool_result = ToolSearchResult(
            tool_id=ToolId("test_tool"),
            name="Test Tool",
            description="A test tool",
            category="test",
            tool_type="system",
            similarity_score=1.0,
            relevance_score=1.0,
            confidence_level="high",
        )

        response = ToolDetailsResponse(
            tool_id="test_tool",
            tool_details=tool_result,
            included_sections={"schema": True, "examples": False, "relationships": True, "usage_guidance": False},
        )

        assert response.tool_id == "test_tool"
        assert response.tool_details.name == "Test Tool"
        assert response.included_sections["schema"] is True

    def test_tool_categories_response_creation(self):
        """Test creation of ToolCategoriesResponse."""
        categories = [
            ToolCategory(name="file_ops", description="File operations", tool_count=10),
            ToolCategory(name="web", description="Web operations", tool_count=5),
        ]

        response = ToolCategoriesResponse(categories=categories)

        assert len(response.categories) == 2
        assert response.total_categories == 2
        assert response.total_tools == 15

    def test_capability_search_response_creation(self):
        """Test creation of CapabilitySearchResponse."""
        tool_result = ToolSearchResult(
            tool_id=ToolId("test_tool"),
            name="Test Tool",
            description="A test tool",
            category="test",
            tool_type="system",
            similarity_score=0.9,
            relevance_score=0.8,
            confidence_level="high",
        )

        capability_match = CapabilityMatch(
            tool=tool_result,
            capability_score=0.9,
            parameter_match_score=0.8,
            complexity_match_score=0.7,
            overall_match_score=0.85,
            match_explanation="High capability match",
        )

        response = CapabilitySearchResponse(capability_description="timeout support", results=[capability_match])

        assert response.capability_description == "timeout support"
        assert len(response.results) == 1
        assert response.total_results == 1

    def test_create_error_response(self):
        """Test creation of error responses."""
        error_response = create_error_response(
            tool_name="test_tool", error_message="Test error", context="test context"
        )

        assert error_response["success"] is False
        assert error_response["error"]["tool"] == "test_tool"
        assert error_response["error"]["message"] == "Test error"
        assert error_response["error"]["context"] == "test context"
        assert "suggestions" in error_response["error"]


class TestMCPToolSearchFunctions:
    """Test the main MCP tool search functions."""

    def setup_method(self):
        """Set up test fixtures."""
        # Mock the global search engines
        self.mock_tool_search_engine = Mock()
        self.mock_parameter_search_engine = Mock()
        self.mock_validator = Mock()
        self.mock_db_manager = Mock()

    @patch("tool_search_mcp_tools._get_search_engine")
    @patch("tool_search_mcp_tools._get_validator")
    def test_search_mcp_tools_success(self, mock_get_validator, mock_get_search_engine):
        """Test successful MCP tool search."""
        # Setup mocks
        mock_validator = Mock()
        mock_search_engine = Mock()
        mock_get_validator.return_value = mock_validator
        mock_get_search_engine.return_value = mock_search_engine

        # Mock validator response
        validated_params = ValidatedSearchParams(
            query="file operations",
            category=None,
            tool_type=None,
            max_results=10,
            include_examples=True,
            search_mode="hybrid",
        )
        mock_validator.validate_search_parameters.return_value = validated_params

        # Mock search engine response
        tool_result = ToolSearchResult(
            tool_id=ToolId("test_tool"),
            name="Test Tool",
            description="A test tool",
            category="test",
            tool_type="system",
            similarity_score=0.9,
            relevance_score=0.8,
            confidence_level="high",
        )

        from tool_search_results import ToolSearchResponse

        mock_search_response = ToolSearchResponse(
            query="file operations", results=[tool_result], search_strategy="hybrid"
        )
        mock_search_engine.search_hybrid.return_value = mock_search_response

        # Execute function
        result = search_mcp_tools("file operations")

        # Verify result is valid JSON
        result_dict = json.loads(result)
        assert result_dict["success"] is True
        assert result_dict["query"] == "file operations"

    @patch("tool_search_mcp_tools._get_validator")
    def test_search_mcp_tools_validation_error(self, mock_get_validator):
        """Test MCP tool search with validation error."""
        # Setup validator to raise error
        mock_validator = Mock()
        mock_get_validator.return_value = mock_validator
        mock_validator.validate_search_parameters.side_effect = ValueError("Invalid query")

        # Execute function
        result = search_mcp_tools("")

        # Verify error response
        result_dict = json.loads(result)
        assert result_dict["success"] is False
        assert "Invalid query" in result_dict["error"]["message"]

    @patch("tool_search_mcp_tools.tool_exists")
    @patch("tool_search_mcp_tools._get_validator")
    def test_get_tool_details_tool_not_found(self, mock_get_validator, mock_tool_exists):
        """Test get_tool_details with non-existent tool."""
        # Setup mocks
        mock_tool_exists.return_value = False

        # Execute function
        result = get_tool_details("nonexistent_tool")

        # Verify error response
        result_dict = json.loads(result)
        assert result_dict["success"] is False
        assert "not found" in result_dict["error"]["message"]

    @patch("tool_search_mcp_tools.get_representative_tools")
    @patch("tool_search_mcp_tools.load_tool_categories")
    def test_list_tool_categories_success(self, mock_load_categories, mock_get_representative_tools):
        """Test successful tool categories listing."""
        # Setup mock categories
        mock_categories = [
            ToolCategory(name="file_ops", description="File operations", tool_count=10),
            ToolCategory(name="web", description="Web operations", tool_count=5),
        ]
        mock_load_categories.return_value = mock_categories

        # Setup mock representative tools
        mock_get_representative_tools.return_value = ["sample_tool1", "sample_tool2", "sample_tool3"]

        # Execute function
        result = list_tool_categories()

        # Verify result
        result_dict = json.loads(result)
        assert result_dict["success"] is True
        assert len(result_dict["categories"]) == 2
        assert result_dict["total_categories"] == 2

    @patch("tool_search_mcp_tools._get_parameter_search_engine")
    @patch("tool_search_mcp_tools._get_validator")
    def test_search_tools_by_capability_success(self, mock_get_validator, mock_get_parameter_engine):
        """Test successful capability search."""
        # Setup mocks
        mock_validator = Mock()
        mock_parameter_engine = Mock()
        mock_get_validator.return_value = mock_validator
        mock_get_parameter_engine.return_value = mock_parameter_engine

        # Mock validator response
        validated_params = ValidatedCapabilityParams(
            capability_description="timeout support",
            required_parameters=["timeout"],
            preferred_complexity="simple",
            max_results=10,
        )
        mock_validator.validate_capability_parameters.return_value = validated_params

        # Mock search engine response
        tool_result = ToolSearchResult(
            tool_id=ToolId("timeout_tool"),
            name="Timeout Tool",
            description="A tool with timeout support",
            category="system",
            tool_type="system",
            similarity_score=0.9,
            relevance_score=0.8,
            confidence_level="high",
        )
        mock_parameter_engine.search_by_parameters.return_value = [tool_result]

        # Execute function
        result = search_tools_by_capability("timeout support", ["timeout"])

        # Verify result
        result_dict = json.loads(result)
        assert result_dict["success"] is True
        assert result_dict["capability_description"] == "timeout support"


class TestErrorSuggestions:
    """Test error suggestion generation."""

    def test_generate_error_suggestions_query_empty(self):
        """Test suggestions for empty query error."""
        suggestions = generate_error_suggestions("search_mcp_tools", "query cannot be empty")

        assert len(suggestions) > 0
        assert any("descriptive search query" in suggestion.lower() for suggestion in suggestions)

    def test_generate_error_suggestions_invalid_category(self):
        """Test suggestions for invalid category error."""
        suggestions = generate_error_suggestions("search_mcp_tools", "invalid category 'bad_category'")

        assert len(suggestions) > 0
        assert any("valid categories are" in suggestion.lower() for suggestion in suggestions)

    def test_generate_error_suggestions_tool_not_found(self):
        """Test suggestions for tool not found error."""
        suggestions = generate_error_suggestions("get_tool_details", "tool 'xyz' not found")

        assert len(suggestions) > 0
        assert any("spelled correctly" in suggestion.lower() for suggestion in suggestions)


class TestInitialization:
    """Test search engine initialization."""

    @patch("tool_search_mcp_tools.MCPToolSearchEngine")
    @patch("tool_search_mcp_tools.ParameterSearchEngine")
    @patch("tool_search_mcp_tools.MCPToolValidator")
    def test_initialize_search_engines(self, mock_validator_class, mock_param_engine_class, mock_search_engine_class):
        """Test search engine initialization."""
        # Setup mocks
        mock_db_manager = Mock()
        mock_embedding_generator = Mock()

        # Execute initialization
        initialize_search_engines(mock_db_manager, mock_embedding_generator)

        # Verify engines were created
        mock_search_engine_class.assert_called_once_with(mock_db_manager, mock_embedding_generator)
        mock_param_engine_class.assert_called_once()
        mock_validator_class.assert_called_once()


class TestIntegration:
    """Integration tests for MCP tool search functionality."""

    def setup_method(self):
        """Set up integration test fixtures."""
        # Create real validator for integration tests
        self.validator = MCPToolValidator()

    def test_full_search_workflow_integration(self):
        """Test complete search workflow with real validator."""
        # Test parameter validation
        validated_params = self.validator.validate_search_parameters(
            query="file operations",
            category="file_ops",
            tool_type="system",
            max_results=5,
            include_examples=True,
            search_mode="hybrid",
        )

        assert validated_params.query == "file operations"
        assert validated_params.max_results == 5

        # Test response creation
        response = ToolSearchMCPResponse(
            query=validated_params.query, results=[], search_mode=validated_params.search_mode
        )

        response_dict = response.to_dict()
        assert response_dict["success"] is True
        assert response_dict["query"] == "file operations"

    def test_error_handling_integration(self):
        """Test error handling integration."""
        # Test validation error
        with pytest.raises(ValueError):
            self.validator.validate_search_parameters(
                query="",  # Invalid empty query
                category=None,
                tool_type=None,
                max_results=10,
                include_examples=True,
                search_mode="hybrid",
            )

        # Test error response creation
        error_response = create_error_response("search_mcp_tools", "Query cannot be empty", "")

        assert error_response["success"] is False
        assert len(error_response["error"]["suggestions"]) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
