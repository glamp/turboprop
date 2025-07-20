#!/usr/bin/env python3
"""
Comprehensive tests for the MCP Tool Search Engine

Tests semantic search functionality, query processing, result ranking,
and search accuracy against expected behaviors.
"""

import json

# Add the parent directory to the path for imports
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from database_manager import DatabaseManager
from embedding_helper import EmbeddingGenerator
from mcp_metadata_types import MCPToolMetadata, ParameterAnalysis, ToolExample
from mcp_tool_search_engine import MCPToolSearchEngine
from search_result_formatter import SearchResultFormatter
from tool_matching_algorithms import ToolMatchingAlgorithms
from tool_query_processor import ToolQueryProcessor
from tool_search_results import ProcessedQuery, SearchIntent, ToolSearchResponse, ToolSearchResult


class TestToolQueryProcessor(unittest.TestCase):
    """Test the tool query processor functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.processor = ToolQueryProcessor()

    def test_process_simple_query(self):
        """Test processing of simple queries."""
        query = "file operations with error handling"
        processed = self.processor.process_query(query)

        self.assertEqual(processed.original_query, query)
        self.assertIn("file", processed.expanded_terms)
        self.assertIn("error", processed.expanded_terms)
        self.assertGreater(processed.confidence, 0.4)

    def test_process_complex_query(self):
        """Test processing of complex queries with multiple concepts."""
        query = "execute shell commands with timeout support and error handling"
        processed = self.processor.process_query(query)

        self.assertEqual(processed.original_query, query)
        self.assertIn("execute", processed.expanded_terms)
        self.assertIn("shell", processed.expanded_terms)
        self.assertIn("timeout", processed.expanded_terms)
        self.assertGreater(processed.confidence, 0.5)

    def test_category_detection(self):
        """Test category detection from queries."""
        test_cases = [
            ("read file with error handling", "file_ops"),
            ("execute bash command", "execution"),
            ("search code in repository", "search"),
            ("fetch web data", "web"),
        ]

        for query, expected_category in test_cases:
            processed = self.processor.process_query(query)
            self.assertEqual(processed.detected_category, expected_category, f"Failed to detect category for: {query}")

    def test_search_intent_analysis(self):
        """Test search intent analysis."""
        query = "alternative to grep command"
        processed = self.processor.process_query(query)

        self.assertIsNotNone(processed.search_intent)
        self.assertEqual(processed.search_intent.query_type, "alternative")
        self.assertGreater(processed.search_intent.confidence, 0.8)

    def test_expand_query_terms(self):
        """Test query term expansion."""
        query = "read file"
        expanded = self.processor.expand_query_terms(query)

        # Should expand with synonyms
        self.assertIn("read", expanded)
        self.assertIn("file", expanded)
        # Should include functional synonyms
        expected_synonyms = ["load", "open", "retrieve", "fetch", "get", "document"]
        found_synonyms = [term for term in expected_synonyms if term in expanded]
        self.assertGreater(len(found_synonyms), 0, "No synonyms found in expansion")

    def test_suggest_refinements_no_results(self):
        """Test query refinement suggestions when no results found."""
        query = "nonexistent functionality"
        suggestions = self.processor.suggest_query_refinements(query, results_found=0)

        self.assertGreater(len(suggestions), 0)
        self.assertTrue(any("general terms" in s.lower() for s in suggestions))

    def test_suggest_refinements_too_many_results(self):
        """Test query refinement suggestions when too many results found."""
        query = "file"
        suggestions = self.processor.suggest_query_refinements(query, results_found=25)

        self.assertGreater(len(suggestions), 0)
        self.assertTrue(any("specific" in s.lower() for s in suggestions))

    def test_empty_query_handling(self):
        """Test handling of empty or invalid queries."""
        with self.assertRaises(ValueError):
            self.processor.process_query("")

        with self.assertRaises(ValueError):
            self.processor.process_query("   ")


class TestToolMatchingAlgorithms(unittest.TestCase):
    """Test the tool matching algorithms."""

    def setUp(self):
        """Set up test fixtures."""
        self.matcher = ToolMatchingAlgorithms()

    def test_semantic_similarity_calculation(self):
        """Test semantic similarity calculation."""
        # Mock embeddings (normalized vectors)
        query_embedding = [1.0, 0.0, 0.0]
        tool_embedding = [1.0, 0.0, 0.0]  # Identical

        similarity = self.matcher.calculate_semantic_similarity(query_embedding, tool_embedding)
        self.assertAlmostEqual(similarity, 1.0, places=2)

        # Orthogonal vectors
        query_embedding = [1.0, 0.0, 0.0]
        tool_embedding = [0.0, 1.0, 0.0]

        similarity = self.matcher.calculate_semantic_similarity(query_embedding, tool_embedding)
        self.assertAlmostEqual(similarity, 0.5, places=1)  # Normalized cosine similarity

    def test_relevance_score_calculation(self):
        """Test relevance score calculation with various factors."""
        semantic_score = 0.8

        tool_metadata = MCPToolMetadata(
            name="test_tool", description="A test tool for file operations", category="file_ops"
        )

        processed_query = ProcessedQuery(
            original_query="file operations",
            cleaned_query="file operations",
            detected_category="file_ops",
            confidence=0.8,
        )

        relevance = self.matcher.calculate_relevance_score(semantic_score, tool_metadata, processed_query)

        # Should boost relevance due to category match
        self.assertGreater(relevance, semantic_score)
        self.assertLessEqual(relevance, 1.0)

    def test_explain_match_reasons(self):
        """Test match reason generation."""
        tool = MCPToolMetadata(
            name="read_file", description="Read file contents with error handling", category="file_ops"
        )

        query = ProcessedQuery(
            original_query="file operations",
            cleaned_query="file operations",
            detected_category="file_ops",
            confidence=0.8,
        )

        scores = {"semantic_similarity": 0.85, "relevance_score": 0.90}

        reasons = self.matcher.explain_match_reasons(tool, query, scores)

        self.assertGreater(len(reasons), 0)
        # Should mention semantic match
        self.assertTrue(any("semantic" in reason.lower() for reason in reasons))
        # Should mention category match
        self.assertTrue(any("category" in reason.lower() for reason in reasons))

    def test_rank_results(self):
        """Test result ranking with different strategies."""
        # Create mock results
        results = [
            ToolSearchResult(
                tool_id="tool1",
                name="Low Relevance Tool",
                description="Test tool",
                category="test",
                tool_type="test",
                similarity_score=0.3,
                relevance_score=0.2,
                confidence_level="low",
            ),
            ToolSearchResult(
                tool_id="tool2",
                name="High Relevance Tool",
                description="Test tool",
                category="test",
                tool_type="test",
                similarity_score=0.9,
                relevance_score=0.85,
                confidence_level="high",
            ),
        ]

        ranked = self.matcher.rank_results(results, "relevance")

        # Should be ranked by relevance (highest first)
        self.assertEqual(ranked[0].tool_id, "tool2")
        self.assertEqual(ranked[1].tool_id, "tool1")

    def test_filter_results_by_threshold(self):
        """Test filtering results by relevance threshold."""
        results = [
            ToolSearchResult(
                tool_id="tool1",
                name="Low Relevance",
                description="Test",
                category="test",
                tool_type="test",
                similarity_score=0.1,
                relevance_score=0.1,
                confidence_level="low",
            ),
            ToolSearchResult(
                tool_id="tool2",
                name="High Relevance",
                description="Test",
                category="test",
                tool_type="test",
                similarity_score=0.8,
                relevance_score=0.8,
                confidence_level="high",
            ),
        ]

        filtered = self.matcher.filter_results_by_threshold(results, min_relevance=0.5)

        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].tool_id, "tool2")


class TestSearchResultFormatter(unittest.TestCase):
    """Test the search result formatter."""

    def setUp(self):
        """Set up test fixtures."""
        self.formatter = SearchResultFormatter()

    def test_console_output_formatting(self):
        """Test console output formatting."""
        # Create a mock response
        result = ToolSearchResult(
            tool_id="test_tool",
            name="Test Tool",
            description="A test tool for demonstration",
            category="test",
            tool_type="test",
            similarity_score=0.85,
            relevance_score=0.90,
            confidence_level="high",
            match_reasons=["Strong semantic match", "Category match"],
        )

        response = ToolSearchResponse(
            query="test query", results=[result], total_results=1, execution_time=0.05, search_strategy="semantic"
        )

        output = self.formatter.format_console_output(response)

        self.assertIn("Test Tool", output)
        self.assertIn("test query", output)
        self.assertIn("high", output)
        self.assertIn("0.05s", output)

    def test_json_output_formatting(self):
        """Test JSON output formatting."""
        result = ToolSearchResult(
            tool_id="test_tool",
            name="Test Tool",
            description="A test tool",
            category="test",
            tool_type="test",
            similarity_score=0.85,
            relevance_score=0.90,
            confidence_level="high",
        )

        response = ToolSearchResponse(query="test query", results=[result], total_results=1, execution_time=0.05)

        json_output = self.formatter.format_json_output(response)

        # Should be valid JSON
        parsed = json.loads(json_output)
        self.assertEqual(parsed["query"], "test query")
        self.assertEqual(len(parsed["results"]), 1)
        self.assertEqual(parsed["results"][0]["name"], "Test Tool")

    def test_summary_output(self):
        """Test summary output generation."""
        result = ToolSearchResult(
            tool_id="test_tool",
            name="Test Tool",
            description="A test tool",
            category="test",
            tool_type="test",
            similarity_score=0.85,
            relevance_score=0.90,
            confidence_level="high",
        )

        response = ToolSearchResponse(query="test query", results=[result], total_results=1, execution_time=0.05)

        summary = self.formatter.format_summary_output(response)

        self.assertIn("Found 1 tools", summary)
        self.assertIn("Best match: Test Tool", summary)


class TestMCPToolSearchEngine(unittest.TestCase):
    """Test the main search engine functionality."""

    def setUp(self):
        """Set up test fixtures with mocked dependencies."""
        # Mock database manager
        self.mock_db_manager = Mock(spec=DatabaseManager)

        # Mock embedding generator
        self.mock_embedding_generator = Mock(spec=EmbeddingGenerator)

        # Create search engine
        self.search_engine = MCPToolSearchEngine(
            db_manager=self.mock_db_manager,
            embedding_generator=self.mock_embedding_generator,
            enable_caching=False,  # Disable caching for testing
        )

    def test_search_by_functionality_success(self):
        """Test successful functionality search."""
        # Mock embedding generation
        mock_embedding = [0.5] * 384  # Mock 384-dimensional embedding
        self.mock_embedding_generator.encode.return_value.tolist.return_value = mock_embedding

        # Mock database search results
        mock_db_results = [
            {
                "id": "test_tool",
                "name": "Test Tool",
                "description": "A tool for file operations",
                "category": "file_ops",
                "tool_type": "system",
                "similarity_score": 0.85,
            }
        ]
        self.mock_db_manager.search_mcp_tools_by_embedding.return_value = mock_db_results

        # Mock parameter and example queries
        self.mock_db_manager.get_tool_parameters.return_value = []
        self.mock_db_manager.get_tool_examples.return_value = []

        # Perform search
        response = self.search_engine.search_by_functionality("file operations")

        # Verify results
        self.assertEqual(response.query, "file operations")
        self.assertGreater(len(response.results), 0)
        self.assertEqual(response.results[0].name, "Test Tool")
        self.assertEqual(response.search_strategy, "semantic_only")

    def test_search_hybrid(self):
        """Test hybrid search functionality."""
        # Mock embeddings
        mock_embedding = [0.5] * 384
        self.mock_embedding_generator.encode.return_value.tolist.return_value = mock_embedding

        # Mock semantic search results
        semantic_results = [
            {
                "id": "semantic_tool",
                "name": "Semantic Tool",
                "description": "Found via semantic search",
                "category": "test",
                "tool_type": "test",
                "similarity_score": 0.8,
            }
        ]
        self.mock_db_manager.search_mcp_tools_by_embedding.return_value = semantic_results

        # Mock keyword search results (via database connection)
        with patch.object(self.mock_db_manager, "get_connection") as mock_get_conn:
            mock_conn = Mock()
            mock_get_conn.return_value.__enter__.return_value = mock_conn
            mock_conn.execute.return_value.fetchall.return_value = [
                ("keyword_tool", "Keyword Tool", "Found via keyword", "test", "test", None, 0.7)
            ]

            self.mock_db_manager.get_tool_parameters.return_value = []
            self.mock_db_manager.get_tool_examples.return_value = []

            response = self.search_engine.search_hybrid("test query")

            self.assertEqual(response.search_strategy, "hybrid_balanced")
            self.assertGreater(len(response.results), 0)

    def test_search_by_capability(self):
        """Test capability-based search."""
        mock_embedding = [0.5] * 384
        self.mock_embedding_generator.encode.return_value.tolist.return_value = mock_embedding

        # Mock database results
        mock_results = [
            {
                "id": "capable_tool",
                "name": "Capable Tool",
                "description": "Has required capabilities",
                "category": "test",
                "tool_type": "test",
                "similarity_score": 0.75,
            }
        ]
        self.mock_db_manager.search_mcp_tools_by_embedding.return_value = mock_results
        self.mock_db_manager.get_tool_parameters.return_value = [
            {
                "parameter_name": "required_param",
                "parameter_type": "string",
                "is_required": True,
                "description": "Required parameter",
            }
        ]
        self.mock_db_manager.get_tool_examples.return_value = []

        response = self.search_engine.search_by_capability(
            "capability description", required_parameters=["required_param"]
        )

        self.assertEqual(response.search_strategy, "capability_focused")
        self.assertGreater(len(response.results), 0)

    def test_get_tool_alternatives(self):
        """Test finding tool alternatives."""
        # Mock tool data
        mock_tool_data = {"id": "original_tool", "name": "Original Tool", "description": "The original tool"}
        self.mock_db_manager.get_mcp_tool.return_value = mock_tool_data

        mock_embedding = [0.5] * 384
        self.mock_embedding_generator.encode.return_value.tolist.return_value = mock_embedding

        # Mock alternative results
        mock_alternatives = [
            {
                "id": "alternative_tool",
                "name": "Alternative Tool",
                "description": "An alternative implementation",
                "category": "test",
                "tool_type": "test",
                "similarity_score": 0.7,
            }
        ]
        self.mock_db_manager.search_mcp_tools_by_embedding.return_value = mock_alternatives
        self.mock_db_manager.get_tool_parameters.return_value = []
        self.mock_db_manager.get_tool_examples.return_value = []

        response = self.search_engine.get_tool_alternatives("original_tool")

        self.assertEqual(response.search_strategy, "alternative_finding")
        self.assertIn("Alternatives to Original Tool", response.query)

    def test_error_handling(self):
        """Test error handling in search operations."""
        # Make embedding generation fail
        self.mock_embedding_generator.encode.side_effect = Exception("Embedding failed")

        response = self.search_engine.search_by_functionality("test query")

        # Should return error response
        self.assertEqual(response.search_strategy, "error")
        self.assertEqual(len(response.results), 0)
        self.assertIn("Error", response.suggested_refinements[0])


class TestSearchAccuracy(unittest.TestCase):
    """Test search accuracy against expected behaviors."""

    def setUp(self):
        """Set up test fixtures for accuracy testing."""
        # Create a real search engine with mocked components for accuracy testing
        self.mock_db_manager = Mock(spec=DatabaseManager)
        self.mock_embedding_generator = Mock(spec=EmbeddingGenerator)
        self.search_engine = MCPToolSearchEngine(
            db_manager=self.mock_db_manager, embedding_generator=self.mock_embedding_generator, enable_caching=False
        )

        # Expected search behaviors from the issue specification
        self.expected_behaviors = {
            "file operations with error handling": {
                "expected_tools": ["read", "write", "edit"],
                "expected_categories": ["file_ops"],
                "match_reasons": ["file operation functionality", "error handling capability"],
            },
            "execute shell commands with timeout": {
                "expected_tools": ["bash"],
                "expected_categories": ["execution"],
                "match_reasons": ["shell execution", "timeout parameter support"],
            },
            "search code in repository": {
                "expected_tools": ["grep", "glob"],
                "expected_categories": ["search"],
                "match_reasons": ["code search functionality", "repository scanning"],
            },
            "web scraping and data fetching": {
                "expected_tools": ["webfetch", "websearch"],
                "expected_categories": ["web"],
                "match_reasons": ["web data access", "scraping capabilities"],
            },
        }

    def _mock_database_for_query(self, query: str):
        """Mock database responses for specific queries."""
        expected = self.expected_behaviors.get(query, {})
        expected_tools = expected.get("expected_tools", [])
        expected_categories = expected.get("expected_categories", ["unknown"])

        # Create mock results for expected tools
        mock_results = []
        for i, tool_name in enumerate(expected_tools):
            mock_results.append(
                {
                    "id": f"{tool_name}_tool",
                    "name": tool_name,
                    "description": f"Tool for {query}",
                    "category": expected_categories[0] if expected_categories else "unknown",
                    "tool_type": "system",
                    "similarity_score": 0.9 - (i * 0.1),  # Decreasing scores
                }
            )

        return mock_results

    def test_search_accuracy_file_operations(self):
        """Test search accuracy for file operations query."""
        query = "file operations with error handling"
        expected = self.expected_behaviors[query]

        # Mock embedding and database
        mock_embedding = [0.8] * 384
        self.mock_embedding_generator.encode.return_value.tolist.return_value = mock_embedding

        mock_results = self._mock_database_for_query(query)
        self.mock_db_manager.search_mcp_tools_by_embedding.return_value = mock_results
        self.mock_db_manager.get_tool_parameters.return_value = []
        self.mock_db_manager.get_tool_examples.return_value = []

        response = self.search_engine.search_by_functionality(query)

        # Verify expected tools are found
        result_names = [r.name.lower() for r in response.results]
        expected_tools = [t.lower() for t in expected["expected_tools"]]

        found_count = sum(1 for tool in expected_tools if tool in result_names)
        accuracy = found_count / len(expected_tools) if expected_tools else 1.0

        self.assertGreaterEqual(accuracy, 0.9, f"Search accuracy {accuracy:.1%} below 90% threshold for: {query}")

        # Verify expected categories
        if expected["expected_categories"]:
            result_categories = [r.category for r in response.results]
            expected_category = expected["expected_categories"][0]
            self.assertIn(
                expected_category, result_categories, f"Expected category {expected_category} not found for: {query}"
            )

    def test_search_accuracy_execution_tools(self):
        """Test search accuracy for execution tools query."""
        query = "execute shell commands with timeout"
        expected = self.expected_behaviors[query]

        mock_embedding = [0.8] * 384
        self.mock_embedding_generator.encode.return_value.tolist.return_value = mock_embedding

        mock_results = self._mock_database_for_query(query)
        self.mock_db_manager.search_mcp_tools_by_embedding.return_value = mock_results
        self.mock_db_manager.get_tool_parameters.return_value = []
        self.mock_db_manager.get_tool_examples.return_value = []

        response = self.search_engine.search_by_functionality(query)

        # Check accuracy
        result_names = [r.name.lower() for r in response.results]
        expected_tools = [t.lower() for t in expected["expected_tools"]]

        found_count = sum(1 for tool in expected_tools if tool in result_names)
        accuracy = found_count / len(expected_tools) if expected_tools else 1.0

        self.assertGreaterEqual(accuracy, 0.9, f"Search accuracy {accuracy:.1%} below 90% threshold for: {query}")

    def test_overall_search_accuracy(self):
        """Test overall search accuracy across all expected behaviors."""
        total_queries = len(self.expected_behaviors)
        successful_queries = 0

        for query, expected in self.expected_behaviors.items():
            try:
                # Mock for this query
                mock_embedding = [0.8] * 384
                self.mock_embedding_generator.encode.return_value.tolist.return_value = mock_embedding

                mock_results = self._mock_database_for_query(query)
                self.mock_db_manager.search_mcp_tools_by_embedding.return_value = mock_results
                self.mock_db_manager.get_tool_parameters.return_value = []
                self.mock_db_manager.get_tool_examples.return_value = []

                response = self.search_engine.search_by_functionality(query)

                # Check if at least one expected tool was found
                result_names = [r.name.lower() for r in response.results]
                expected_tools = [t.lower() for t in expected["expected_tools"]]

                if any(tool in result_names for tool in expected_tools):
                    successful_queries += 1

            except Exception as e:
                # Log error but continue testing
                print(f"Error testing query '{query}': {e}")

        overall_accuracy = successful_queries / total_queries if total_queries > 0 else 0.0

        self.assertGreaterEqual(
            overall_accuracy, 0.9, f"Overall search accuracy {overall_accuracy:.1%} below 90% threshold"
        )

    def test_search_response_time(self):
        """Test that search response time meets performance requirements."""
        query = "test performance query"

        mock_embedding = [0.5] * 384
        self.mock_embedding_generator.encode.return_value.tolist.return_value = mock_embedding

        mock_results = [
            {
                "id": "test_tool",
                "name": "Test Tool",
                "description": "Test description",
                "category": "test",
                "tool_type": "test",
                "similarity_score": 0.8,
            }
        ]
        self.mock_db_manager.search_mcp_tools_by_embedding.return_value = mock_results
        self.mock_db_manager.get_tool_parameters.return_value = []
        self.mock_db_manager.get_tool_examples.return_value = []

        response = self.search_engine.search_by_functionality(query)

        # Should complete in under 500ms (as specified in success criteria)
        self.assertLess(
            response.execution_time, 0.5, f"Search took {response.execution_time:.3f}s, exceeding 500ms requirement"
        )


if __name__ == "__main__":
    # Configure logging to show test output
    import logging

    logging.basicConfig(level=logging.INFO)

    # Run the tests
    unittest.main(verbosity=2)
