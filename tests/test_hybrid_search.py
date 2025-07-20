#!/usr/bin/env python3
"""
test_hybrid_search.py: Comprehensive tests for hybrid search functionality.

This module tests:
- Hybrid search engine with semantic + text fusion
- Query analysis and routing
- Fusion algorithms (RRF, weighted scoring)
- Search mode selection (auto, hybrid, semantic, text)
- Advanced features (regex, file filtering, Boolean operators)
"""

import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from database_manager import DatabaseManager
from embedding_helper import EmbeddingGenerator
from hybrid_search import (
    FusionWeights,
    HybridSearchEngine,
    HybridSearchFormatter,
    HybridSearchResult,
    QueryAnalyzer,
    QueryCharacteristics,
    SearchMode,
    search_hybrid,
    search_hybrid_with_details,
)
from search_result_types import CodeSearchResult, CodeSnippet


class TestQueryAnalyzer:
    """Test the QueryAnalyzer class for query characteristic detection."""

    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = QueryAnalyzer()

    def test_analyze_simple_query(self):
        """Test analysis of simple queries."""
        characteristics = self.analyzer.analyze_query("parse json")

        assert characteristics.word_count == 2
        assert not characteristics.has_quoted_phrases
        assert not characteristics.has_boolean_operators
        assert not characteristics.has_regex_patterns
        assert not characteristics.has_wildcards
        assert characteristics.estimated_intent in ["general", "semantic"]

    def test_analyze_quoted_phrase_query(self):
        """Test analysis of queries with quoted phrases."""
        characteristics = self.analyzer.analyze_query('"exact phrase" search')

        assert characteristics.has_quoted_phrases
        assert characteristics.estimated_intent == "exact"
        assert characteristics.word_count == 2  # Words outside quotes

    def test_analyze_boolean_query(self):
        """Test analysis of Boolean operator queries."""
        characteristics = self.analyzer.analyze_query("authentication AND jwt OR token")

        assert characteristics.has_boolean_operators
        assert characteristics.estimated_intent == "exact"
        assert characteristics.word_count == 5

    def test_analyze_regex_query(self):
        """Test analysis of regex pattern queries."""
        characteristics = self.analyzer.analyze_query("function.*\\{.*return")

        assert characteristics.has_regex_patterns
        assert characteristics.estimated_intent == "technical"

    def test_analyze_technical_query(self):
        """Test analysis of technical term queries."""
        characteristics = self.analyzer.analyze_query("def function class")

        assert characteristics.is_technical_term
        assert characteristics.estimated_intent == "technical"

    def test_analyze_natural_language_query(self):
        """Test analysis of natural language queries."""
        characteristics = self.analyzer.analyze_query("how to implement user authentication system")

        assert characteristics.is_natural_language
        assert characteristics.estimated_intent == "semantic"
        assert characteristics.word_count == 6

    def test_analyze_wildcard_query(self):
        """Test analysis of wildcard queries."""
        characteristics = self.analyzer.analyze_query("user* auth?")

        assert characteristics.has_wildcards
        assert characteristics.word_count == 2


class TestFusionWeights:
    """Test the FusionWeights configuration class."""

    def test_default_weights(self):
        """Test default fusion weights."""
        weights = FusionWeights()

        assert weights.semantic_weight == 0.6
        assert weights.text_weight == 0.4
        assert weights.rrf_k == 60
        assert weights.boost_exact_matches is True
        assert weights.exact_match_boost == 1.5

    def test_custom_weights(self):
        """Test custom fusion weights."""
        weights = FusionWeights(semantic_weight=0.8, text_weight=0.2, rrf_k=100, exact_match_boost=2.0)

        assert weights.semantic_weight == 0.8
        assert weights.text_weight == 0.2
        assert weights.rrf_k == 100
        assert weights.exact_match_boost == 2.0


class TestHybridSearchEngine:
    """Test the HybridSearchEngine class."""

    def setup_method(self):
        """Set up test fixtures with mocked dependencies."""
        self.mock_db_manager = Mock(spec=DatabaseManager)
        self.mock_embedder = Mock(spec=EmbeddingGenerator)

        # Mock database manager methods
        self.mock_db_manager.create_fts_index.return_value = None
        self.mock_db_manager.search_full_text.return_value = [
            ("id1", "/path/file1.py", "def test_function(): pass", 0.9),
            ("id2", "/path/file2.py", "class TestClass: pass", 0.8),
        ]
        # Mock execute_with_retry for semantic search (returns path, content, distance)
        self.mock_db_manager.execute_with_retry.return_value = [
            ("/path/file1.py", "def test_function(): pass", 0.1),
            ("/path/file2.py", "class TestClass: pass", 0.2),
        ]

        # Mock embedder
        self.mock_embedder.encode.return_value = Mock()
        self.mock_embedder.encode.return_value.tolist.return_value = [0.1] * 384

        self.engine = HybridSearchEngine(self.mock_db_manager, self.mock_embedder)

    def test_engine_initialization(self):
        """Test hybrid search engine initialization."""
        assert self.engine.db_manager is self.mock_db_manager
        assert self.engine.embedder is self.mock_embedder
        assert isinstance(self.engine.query_analyzer, QueryAnalyzer)
        assert isinstance(self.engine.default_weights, FusionWeights)

        # Verify FTS index creation was called
        self.mock_db_manager.create_fts_index.assert_called_once()

    def test_determine_search_mode_quoted_phrase(self):
        """Test search mode determination for quoted phrases."""
        characteristics = QueryCharacteristics(has_quoted_phrases=True, word_count=3)

        mode = self.engine._determine_search_mode(characteristics)
        assert mode == SearchMode.TEXT_ONLY

    def test_determine_search_mode_natural_language(self):
        """Test search mode determination for natural language queries."""
        characteristics = QueryCharacteristics(is_natural_language=True, word_count=5, is_technical_term=False)

        mode = self.engine._determine_search_mode(characteristics)
        assert mode == SearchMode.SEMANTIC_ONLY

    def test_determine_search_mode_technical_term(self):
        """Test search mode determination for technical terms."""
        characteristics = QueryCharacteristics(is_technical_term=True, word_count=2)

        mode = self.engine._determine_search_mode(characteristics)
        assert mode == SearchMode.TEXT_ONLY

    def test_determine_search_mode_default_hybrid(self):
        """Test search mode determination defaults to hybrid."""
        characteristics = QueryCharacteristics(word_count=3)

        mode = self.engine._determine_search_mode(characteristics)
        assert mode == SearchMode.HYBRID

    def test_adapt_weights_quoted_phrases(self):
        """Test weight adaptation for quoted phrase queries."""
        characteristics = QueryCharacteristics(has_quoted_phrases=True)

        weights = self.engine._adapt_weights(characteristics)
        assert weights.text_weight > weights.semantic_weight
        assert weights.text_weight == 0.8
        assert weights.exact_match_boost == 2.0

    def test_adapt_weights_natural_language(self):
        """Test weight adaptation for natural language queries."""
        characteristics = QueryCharacteristics(is_natural_language=True, word_count=5)

        weights = self.engine._adapt_weights(characteristics)
        assert weights.semantic_weight > weights.text_weight
        assert weights.semantic_weight == 0.8

    def test_is_exact_match_content(self):
        """Test exact match detection in content."""
        result = Mock(spec=CodeSearchResult)
        result.snippet = Mock()
        result.snippet.text = "def parse_json(data): return json.loads(data)"
        result.file_path = "/path/to/file.py"

        is_match = self.engine._is_exact_match(result, "parse_json")
        assert is_match

    def test_is_exact_match_path(self):
        """Test exact match detection in file path."""
        result = Mock(spec=CodeSearchResult)
        result.snippet = Mock()
        result.snippet.text = "some content"
        result.file_path = "/path/to/auth_utils.py"

        is_match = self.engine._is_exact_match(result, "auth_utils")
        assert is_match

    def test_expand_query_simple_term(self):
        """Test query expansion for common terms."""
        expanded = self.engine._expand_query("auth")
        assert "authentication login signin user" in expanded

        expanded = self.engine._expand_query("db connection")
        assert "database data storage" in expanded

    def test_expand_query_no_expansion(self):
        """Test query expansion when no expansion applies."""
        original = "complex custom query"
        expanded = self.engine._expand_query(original)
        assert expanded == original

    @patch("hybrid_search.search_index_enhanced")
    def test_search_semantic_only(self, mock_search):
        """Test semantic-only search mode."""
        # Setup mock search results
        mock_result = Mock(spec=CodeSearchResult)
        mock_result.similarity_score = 0.9
        mock_search.return_value = [mock_result]

        results = self.engine._search_semantic_only("test query", 5, False)

        assert len(results) == 1
        assert results[0].semantic_score == 0.9
        assert results[0].match_type == "semantic"
        assert results[0].fusion_method == "semantic_only"

    def test_search_text_only(self):
        """Test text-only search mode."""
        results = self.engine._search_text_only("test query", 5)

        assert len(results) == 2  # From mocked search_full_text return
        assert all(r.match_type == "text" for r in results)
        assert all(r.fusion_method == "text_only" for r in results)

    @patch("search_utils.search_index_enhanced")
    def test_search_hybrid_mode(self, mock_search):
        """Test hybrid search mode with result fusion."""
        # Setup mock semantic results
        mock_semantic_result = Mock(spec=CodeSearchResult)
        mock_semantic_result.file_path = "/path/file1.py"
        mock_semantic_result.similarity_score = 0.8
        mock_search.return_value = [mock_semantic_result]

        weights = FusionWeights()
        results = self.engine._search_hybrid("test query", 5, weights, False)

        assert len(results) > 0
        # Should contain both semantic and text results after fusion
        for result in results:
            assert hasattr(result, "fusion_score")
            assert hasattr(result, "fusion_method")


class TestHybridSearchFormatter:
    """Test the HybridSearchFormatter class."""

    def test_format_empty_results(self):
        """Test formatting when no results are found."""
        results = []
        formatted = HybridSearchFormatter.format_results(results, "test query")

        assert "No hybrid search results found" in formatted
        assert "test query" in formatted

    def test_format_results_basic(self):
        """Test basic result formatting."""
        # Create mock results
        mock_snippet = Mock(spec=CodeSnippet)
        mock_snippet.text = "def test(): pass"
        mock_snippet.start_line = 1
        mock_snippet.end_line = 1

        mock_code_result = Mock(spec=CodeSearchResult)
        mock_code_result.file_path = "/path/to/file.py"
        mock_code_result.snippet = mock_snippet

        hybrid_result = HybridSearchResult(
            code_result=mock_code_result, semantic_score=0.8, text_score=0.6, fusion_score=0.75, match_type="hybrid"
        )

        formatted = HybridSearchFormatter.format_results([hybrid_result], "test query", show_scores=True)

        assert "Found 1 hybrid search results" in formatted
        assert "/path/to/file.py" in formatted
        assert "def test(): pass" in formatted
        assert "🔀" in formatted  # Hybrid emoji

    def test_format_results_different_match_types(self):
        """Test formatting results with different match types."""
        mock_snippet = Mock(spec=CodeSnippet)
        mock_snippet.text = "content"
        mock_snippet.start_line = 1
        mock_snippet.end_line = 1

        # Create results with different match types
        results = []
        match_types = ["semantic", "text", "hybrid"]
        emojis = ["🧠", "📝", "🔀"]

        for match_type, emoji in zip(match_types, emojis):
            mock_code_result = Mock(spec=CodeSearchResult)
            mock_code_result.file_path = f"/path/{match_type}.py"
            mock_code_result.snippet = mock_snippet

            hybrid_result = HybridSearchResult(code_result=mock_code_result, match_type=match_type)
            results.append(hybrid_result)

        formatted = HybridSearchFormatter.format_results(results, "test")

        # Check that all match type emojis are present
        for emoji in emojis:
            assert emoji in formatted


class TestConvenienceFunctions:
    """Test the convenience functions for easy integration."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_db_manager = Mock(spec=DatabaseManager)
        self.mock_embedder = Mock(spec=EmbeddingGenerator)

    @patch("hybrid_search.HybridSearchEngine")
    def test_search_hybrid_function(self, mock_engine_class):
        """Test the search_hybrid convenience function."""
        # Setup mock engine
        mock_engine = Mock()
        mock_hybrid_result = Mock(spec=HybridSearchResult)
        mock_code_result = Mock(spec=CodeSearchResult)
        mock_hybrid_result.code_result = mock_code_result
        mock_engine.search.return_value = [mock_hybrid_result]
        mock_engine_class.return_value = mock_engine

        results = search_hybrid(self.mock_db_manager, self.mock_embedder, "test query", 5)

        assert len(results) == 1
        assert results[0] == mock_code_result
        mock_engine_class.assert_called_once()
        mock_engine.search.assert_called_once_with("test query", 5, SearchMode.AUTO)

    @patch("hybrid_search.HybridSearchEngine")
    def test_search_hybrid_with_details_function(self, mock_engine_class):
        """Test the search_hybrid_with_details convenience function."""
        # Setup mock engine
        mock_engine = Mock()
        mock_hybrid_result = Mock(spec=HybridSearchResult)
        mock_engine.search.return_value = [mock_hybrid_result]
        mock_engine_class.return_value = mock_engine

        results = search_hybrid_with_details(self.mock_db_manager, self.mock_embedder, "test query", 5, "hybrid")

        assert len(results) == 1
        assert results[0] == mock_hybrid_result
        mock_engine.search.assert_called_once_with("test query", 5, SearchMode.HYBRID)


class TestIntegration:
    """Integration tests using temporary database."""

    def setup_method(self):
        """Set up integration test fixtures with temporary database."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test.duckdb"
        self.db_manager = DatabaseManager(self.db_path)

        # Create test table
        self.db_manager.execute_with_retry(
            """
            CREATE TABLE IF NOT EXISTS code_files (
                id VARCHAR PRIMARY KEY,
                path VARCHAR,
                content TEXT,
                embedding DOUBLE[384]
            )
        """
        )

        # Insert test data
        test_files = [
            ("id1", "/test/auth.py", "def authenticate(user): return jwt.encode(user)", [0.1] * 384),
            ("id2", "/test/utils.py", "def parse_json(data): return json.loads(data)", [0.2] * 384),
            ("id3", "/test/main.py", "class UserManager: def create_user(self): pass", [0.3] * 384),
        ]

        for file_id, path, content, embedding in test_files:
            self.db_manager.execute_with_retry(
                "INSERT INTO code_files (id, path, content, embedding) VALUES (?, ?, ?, ?)",
                (file_id, path, content, embedding),
            )

        # Setup mock embedder
        self.mock_embedder = Mock(spec=EmbeddingGenerator)
        self.mock_embedder.encode.return_value = Mock()
        self.mock_embedder.encode.return_value.tolist.return_value = [0.15] * 384

        # Mock search_full_text since FTS index creation fails in test environment
        self.db_manager.search_full_text = Mock(
            return_value=[("id1", "/test/auth.py", "def authenticate(user): return jwt.encode(user)", 0.9)]
        )

    def teardown_method(self):
        """Clean up test fixtures."""
        if self.db_manager:
            self.db_manager.cleanup()

        # Remove temp directory
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("hybrid_search.search_index_enhanced")
    def test_integration_auto_mode(self, mock_search):
        """Test integration with auto mode selection."""
        # Mock semantic search results
        mock_result = Mock(spec=CodeSearchResult)
        mock_result.file_path = "/test/auth.py"
        mock_result.similarity_score = 0.8
        mock_search.return_value = [mock_result]

        engine = HybridSearchEngine(self.db_manager, self.mock_embedder)
        results = engine.search("authentication function", k=5, mode=SearchMode.AUTO)

        assert len(results) > 0
        assert all(isinstance(r, HybridSearchResult) for r in results)

    def test_integration_text_search(self):
        """Test integration with text search mode."""
        # Mock the FTS search
        self.db_manager.search_full_text = Mock(
            return_value=[("id1", "/test/auth.py", "def authenticate(user): return jwt.encode(user)", 1.0)]
        )

        engine = HybridSearchEngine(self.db_manager, self.mock_embedder)
        results = engine.search("authenticate", k=5, mode=SearchMode.TEXT_ONLY)

        assert len(results) > 0
        assert results[0].match_type == "text"


class TestPerformance:
    """Performance and stress tests for hybrid search."""

    def setup_method(self):
        """Set up performance test fixtures."""
        self.mock_db_manager = Mock(spec=DatabaseManager)
        self.mock_embedder = Mock(spec=EmbeddingGenerator)

        # Mock fast responses
        self.mock_db_manager.create_fts_index.return_value = None
        self.mock_db_manager.search_full_text.return_value = []
        self.mock_embedder.encode.return_value = Mock()
        self.mock_embedder.encode.return_value.tolist.return_value = [0.1] * 384

    @patch("search_utils.search_index_enhanced")
    def test_search_performance(self, mock_search):
        """Test search performance with timing."""
        mock_search.return_value = []

        engine = HybridSearchEngine(self.mock_db_manager, self.mock_embedder)

        start_time = time.time()
        results = engine.search("test query", k=10)
        elapsed = time.time() - start_time

        # Should complete quickly (under 1 second for mocked calls)
        assert elapsed < 1.0
        assert isinstance(results, list)

    def test_large_result_handling(self):
        """Test handling of large result sets."""
        # Mock large result set
        large_results = [(f"id{i}", f"/path/file{i}.py", f"content {i}", 0.5) for i in range(1000)]
        # Mock search_full_text to respect the limit parameter

        def mock_search_full_text(query, limit=None, **kwargs):
            if limit is not None:
                return large_results[:limit]
            return large_results

        self.mock_db_manager.search_full_text.side_effect = mock_search_full_text

        engine = HybridSearchEngine(self.mock_db_manager, self.mock_embedder)
        results = engine.search("test", k=50, mode=SearchMode.TEXT_ONLY)

        # Should handle large results and limit to k
        assert len(results) <= 50


if __name__ == "__main__":
    pytest.main([__file__])
