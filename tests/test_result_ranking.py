#!/usr/bin/env python3
"""
test_result_ranking.py: Comprehensive tests for advanced result ranking system.

This module tests all aspects of the advanced ranking and confidence scoring
system including multi-factor ranking, match reason generation, deduplication,
and confidence assessment.
"""

import tempfile
from typing import List
from unittest.mock import Mock, patch

import pytest

from turboprop.ranking_exceptions import InvalidRankingWeightsError
from turboprop.ranking_scorers import ConstructTypeScorer, FileSizeScorer, FileTypeScorer, RecencyScorer
from turboprop.ranking_utils import (
    ConfidenceScorer,
    MatchReason,
    MatchReasonGenerator,
    RankingContext,
    ResultDeduplicator,
)
from turboprop.result_ranking import (
    RankingWeights,
    ResultRanker,
    calculate_advanced_confidence,
    generate_match_explanations,
    rank_search_results,
)
from turboprop.search_result_types import CodeSearchResult, CodeSnippet


class TestRankingWeights:
    """Test ranking weights configuration."""

    def test_default_weights(self):
        """Test that default weights are properly initialized."""
        weights = RankingWeights()
        assert weights.embedding_similarity == 0.4
        assert weights.file_type_relevance == 0.2
        assert weights.construct_type_matching == 0.2
        assert weights.file_recency == 0.1
        assert weights.file_size_optimization == 0.1

    def test_weights_validation(self):
        """Test that weights are validated during initialization."""
        # Test that invalid weights (sum > 1.1) raise an exception
        with pytest.raises(
            InvalidRankingWeightsError, match="Ranking weights must sum to approximately 1.0, got 1.500"
        ):
            RankingWeights(
                embedding_similarity=0.5,
                file_type_relevance=0.5,
                construct_type_matching=0.3,  # Total = 1.5 > 1.0
                file_recency=0.1,
                file_size_optimization=0.1,
            )

        # Test that slightly off weights (sum within 0.1 of 1.0) only log a warning
        with patch("turboprop.result_ranking.logger") as mock_logger:
            RankingWeights(
                embedding_similarity=0.36,
                file_type_relevance=0.25,
                construct_type_matching=0.25,  # Total = 1.06, outside tolerance but within 0.1
                file_recency=0.1,
                file_size_optimization=0.1,
            )
            # Should log a warning about weights not summing exactly to 1.0
            mock_logger.warning.assert_called_once()


class TestFileTypeScorer:
    """Test file type relevance scoring."""

    def test_source_code_files_get_high_scores(self):
        """Test that source code files get high relevance scores."""
        score = FileTypeScorer.score_file_type("/path/to/file.py", "search query")
        assert score == 1.0

        score = FileTypeScorer.score_file_type("/path/to/file.js", "search query")
        assert score == 1.0

        score = FileTypeScorer.score_file_type("/path/to/file.java", "search query")
        assert score == 1.0

    def test_test_files_get_reduced_scores(self):
        """Test that test files get slightly reduced scores for non-test queries."""
        score = FileTypeScorer.score_file_type("/path/to/test_example.py", "search query")
        assert score == 0.8  # Reduced score for test file

        # But test files get full score when query is about testing
        score = FileTypeScorer.score_file_type("/path/to/test_example.py", "test function")
        assert score == 1.0

    def test_documentation_files(self):
        """Test scoring for documentation files."""
        # Regular query on doc file
        score = FileTypeScorer.score_file_type("/path/to/readme.md", "search query")
        assert score == 0.6

        # Documentation-related query
        score = FileTypeScorer.score_file_type("/path/to/readme.md", "documentation guide")
        assert score == 0.9

    def test_config_files(self):
        """Test scoring for configuration files."""
        # Regular query on config file
        score = FileTypeScorer.score_file_type("/path/to/config.json", "search query")
        assert score == 0.4

        # Config-related query
        score = FileTypeScorer.score_file_type("/path/to/config.json", "configuration settings")
        assert score == 0.8

    def test_unknown_file_types(self):
        """Test scoring for unknown file types."""
        score = FileTypeScorer.score_file_type("/path/to/file.unknown", "search query")
        assert score == 0.5  # Neutral score


class TestConstructTypeScorer:
    """Test construct type matching scoring."""

    def test_query_type_detection(self):
        """Test detection of construct types from queries."""
        assert ConstructTypeScorer.detect_query_type("find function that") == "function"
        assert ConstructTypeScorer.detect_query_type("show me the class") == "class"
        assert ConstructTypeScorer.detect_query_type("search for variable") == "variable"
        assert ConstructTypeScorer.detect_query_type("import statement") == "import"
        assert ConstructTypeScorer.detect_query_type("error handling") == "exception"
        assert ConstructTypeScorer.detect_query_type("module definition") == "module"
        assert ConstructTypeScorer.detect_query_type("random search") is None

    def test_construct_matching_with_metadata(self):
        """Test construct matching when result has construct metadata."""
        # Create result with construct metadata
        result = CodeSearchResult(
            file_path="/path/to/file.py",
            snippet=CodeSnippet(text="def example():", start_line=1, end_line=1),
            similarity_score=0.8,
            file_metadata={"construct_context": {"construct_types": ["function", "variable"]}},
        )

        # Perfect match
        score = ConstructTypeScorer.score_construct_match(result, "function")
        assert score == 1.0

        # Related match (function/method)
        result.file_metadata["construct_context"]["construct_types"] = ["method"]
        score = ConstructTypeScorer.score_construct_match(result, "function")
        assert score == 0.8

    def test_construct_matching_without_metadata(self):
        """Test construct matching by analyzing snippet content."""
        # Function in snippet
        result = CodeSearchResult(
            file_path="/path/to/file.py",
            snippet=CodeSnippet(text="def example_function():", start_line=1, end_line=1),
            similarity_score=0.8,
        )
        score = ConstructTypeScorer.score_construct_match(result, "function")
        assert score == 0.9

        # Class in snippet
        result.snippet.text = "class ExampleClass:"
        score = ConstructTypeScorer.score_construct_match(result, "class")
        assert score == 0.9

        # Mismatch
        score = ConstructTypeScorer.score_construct_match(result, "variable")
        assert score == 0.3


class TestRecencyScorer:
    """Test file recency scoring."""

    def test_recent_files_get_high_scores(self):
        """Test that recent files get higher scores."""
        from datetime import datetime, timedelta, timezone

        with tempfile.NamedTemporaryFile() as temp_file:
            # Mock recent modification time
            recent_time = datetime.now(tz=timezone.utc) - timedelta(days=1)
            with patch("os.stat") as mock_stat:
                mock_stat.return_value = Mock(st_mtime=recent_time.timestamp())
                score = RecencyScorer.score_recency(temp_file.name)
                assert score == 1.0

    def test_old_files_get_low_scores(self):
        """Test that old files get lower scores."""
        from datetime import datetime, timedelta, timezone

        with tempfile.NamedTemporaryFile() as temp_file:
            # Mock old modification time
            old_time = datetime.now(tz=timezone.utc) - timedelta(days=500)
            with patch("os.stat") as mock_stat:
                mock_stat.return_value = Mock(st_mtime=old_time.timestamp())
                score = RecencyScorer.score_recency(temp_file.name)
                assert score == 0.1

    def test_missing_file_gets_neutral_score(self):
        """Test that missing files get neutral scores."""
        score = RecencyScorer.score_recency("/nonexistent/file.py")
        assert score == 0.5


class TestFileSizeScorer:
    """Test file size optimization scoring."""

    def test_optimal_size_gets_high_score(self):
        """Test that optimally sized files get high scores."""
        result = CodeSearchResult(
            file_path="/path/to/file.py",
            snippet=CodeSnippet(text="code", start_line=1, end_line=1),
            similarity_score=0.8,
            file_metadata={"size": 5000},  # 5KB - in optimal range
        )
        score = FileSizeScorer.score_file_size(result)
        assert score == 1.0

    def test_very_small_files_get_low_scores(self):
        """Test that very small files get lower scores."""
        result = CodeSearchResult(
            file_path="/path/to/file.py",
            snippet=CodeSnippet(text="code", start_line=1, end_line=1),
            similarity_score=0.8,
            file_metadata={"size": 50},  # Very small
        )
        score = FileSizeScorer.score_file_size(result)
        assert score == 0.2

    def test_very_large_files_get_low_scores(self):
        """Test that very large files get lower scores."""
        result = CodeSearchResult(
            file_path="/path/to/file.py",
            snippet=CodeSnippet(text="code", start_line=1, end_line=1),
            similarity_score=0.8,
            file_metadata={"size": 1000000},  # 1MB - very large
        )
        score = FileSizeScorer.score_file_size(result)
        assert score == 0.2

    def test_unknown_size_gets_neutral_score(self):
        """Test that files with unknown size get neutral scores."""
        result = CodeSearchResult(
            file_path="/path/to/file.py",
            snippet=CodeSnippet(text="code", start_line=1, end_line=1),
            similarity_score=0.8,
        )
        score = FileSizeScorer.score_file_size(result)
        assert score == 0.5


class TestMatchReasonGenerator:
    """Test match reason generation."""

    def test_filename_matching(self):
        """Test generation of filename matching reasons."""
        result = CodeSearchResult(
            file_path="/path/to/authentication.py",
            snippet=CodeSnippet(text="def login():", start_line=1, end_line=1),
            similarity_score=0.8,
        )
        context = RankingContext(query="authentication system", query_keywords=["authentication", "system"])

        reasons = MatchReasonGenerator.generate_reasons(result, context)

        # Should find filename match
        filename_reasons = [r for r in reasons if r.category == "name_match"]
        assert len(filename_reasons) > 0
        assert any("authentication" in r.description for r in filename_reasons)

    def test_content_matching(self):
        """Test generation of content matching reasons."""
        result = CodeSearchResult(
            file_path="/path/to/file.py",
            snippet=CodeSnippet(
                text="def authenticate_user():\n    # User authentication logic", start_line=1, end_line=2
            ),
            similarity_score=0.9,
        )
        context = RankingContext(query="authentication function", query_keywords=["authentication", "function"])

        reasons = MatchReasonGenerator.generate_reasons(result, context)

        # Should find content matches
        content_reasons = [r for r in reasons if r.category == "content_match"]
        assert len(content_reasons) > 0

    def test_high_similarity_reason(self):
        """Test generation of high similarity reasons."""
        result = CodeSearchResult(
            file_path="/path/to/file.py",
            snippet=CodeSnippet(text="some code", start_line=1, end_line=1),
            similarity_score=0.9,  # High similarity
        )
        context = RankingContext(query="test query", query_keywords=["test", "query"])

        reasons = MatchReasonGenerator.generate_reasons(result, context)

        # Should include high similarity reason
        similarity_reasons = [r for r in reasons if "semantic similarity" in r.description]
        assert len(similarity_reasons) > 0


class TestConfidenceScorer:
    """Test advanced confidence scoring."""

    def test_high_confidence_calculation(self):
        """Test calculation of high confidence results."""
        result = CodeSearchResult(
            file_path="/path/to/file.py",
            snippet=CodeSnippet(text="def authenticate():", start_line=1, end_line=1),
            similarity_score=0.95,  # Higher similarity
            file_metadata={"construct_context": {"construct_types": ["function"]}},  # Perfect construct match
        )

        # Create similar results for cross-validation
        similar_results = [result, result]  # Mock similar results

        context = RankingContext(
            query="authentication function",
            query_keywords=["authentication", "function"],
            query_type="function",
            all_results=similar_results,
        )

        # Create high-confidence match reasons
        match_reasons = [
            MatchReason(category="content_match", description="Function name matches", confidence=0.95),
            MatchReason(category="structure_match", description="Function construct", confidence=0.9),
        ]

        confidence = ConfidenceScorer.calculate_advanced_confidence(result, context, match_reasons)
        assert confidence == "high"

    def test_low_confidence_calculation(self):
        """Test calculation of low confidence results."""
        result = CodeSearchResult(
            file_path="/path/to/file.py",
            snippet=CodeSnippet(text="some unrelated code", start_line=1, end_line=1),
            similarity_score=0.3,  # Low similarity
        )
        context = RankingContext(
            query="authentication function", query_keywords=["authentication", "function"], query_type="function"
        )

        # Create low-confidence match reasons
        match_reasons = [MatchReason(category="structure_match", description="Generic match", confidence=0.2)]

        confidence = ConfidenceScorer.calculate_advanced_confidence(result, context, match_reasons)
        assert confidence == "low"


class TestResultDeduplicator:
    """Test result deduplication and clustering."""

    def test_duplicate_removal(self):
        """Test removal of duplicate results."""
        # Create similar results
        result1 = CodeSearchResult(
            file_path="/path/to/file.py",
            snippet=CodeSnippet(text="def example(): pass", start_line=1, end_line=1),
            similarity_score=0.8,
        )
        result2 = CodeSearchResult(
            file_path="/path/to/file.py",
            snippet=CodeSnippet(text="def example(): pass", start_line=1, end_line=1),
            similarity_score=0.7,
        )
        result3 = CodeSearchResult(
            file_path="/path/to/other.py",
            snippet=CodeSnippet(text="completely different code", start_line=1, end_line=1),
            similarity_score=0.6,
        )

        results = [result1, result2, result3]
        deduplicated = ResultDeduplicator.deduplicate_results(results, similarity_threshold=0.9)

        # Should remove one of the duplicates
        assert len(deduplicated) == 2

    def test_diversity_enforcement(self):
        """Test enforcement of result diversity."""
        # Create multiple results from same directory
        results = []
        for i in range(5):
            result = CodeSearchResult(
                file_path=f"/same/directory/file{i}.py",
                snippet=CodeSnippet(text=f"code {i}", start_line=1, end_line=1),
                similarity_score=0.8 - i * 0.1,
            )
            results.append(result)

        diverse = ResultDeduplicator.ensure_diversity(results, max_per_directory=2)

        # Should limit results per directory
        assert len(diverse) == 2
        # Should keep the highest scoring results
        assert diverse[0].similarity_score >= diverse[1].similarity_score


class TestResultRanker:
    """Test the main result ranker."""

    def create_test_results(self) -> List[CodeSearchResult]:
        """Create test search results for ranking."""
        return [
            CodeSearchResult(
                file_path="/project/auth/login.py",
                snippet=CodeSnippet(text="def authenticate_user():", start_line=1, end_line=1),
                similarity_score=0.7,
                file_metadata={"size": 3000, "language": "python"},
            ),
            CodeSearchResult(
                file_path="/project/docs/readme.md",
                snippet=CodeSnippet(text="Authentication guide", start_line=1, end_line=1),
                similarity_score=0.8,
                file_metadata={"size": 1500, "language": "markdown"},
            ),
            CodeSearchResult(
                file_path="/project/config/auth.json",
                snippet=CodeSnippet(text='{"auth_method": "oauth"}', start_line=1, end_line=1),
                similarity_score=0.6,
                file_metadata={"size": 500, "language": "json"},
            ),
        ]

    def test_ranking_integration(self):
        """Test the complete ranking process."""
        ranker = ResultRanker()
        results = self.create_test_results()
        context = RankingContext(
            query="authentication function", query_keywords=["authentication", "function"], query_type="function"
        )

        ranked = ranker.rank_results(results, context)

        # Should return ranked results
        assert len(ranked) == len(results)

        # Results should have enhanced metadata
        for result in ranked:
            assert "composite_score" in result.file_metadata
            assert "ranking_factors" in result.file_metadata
            assert "match_reasons" in result.file_metadata
            assert result.confidence_level in ["high", "medium", "low"]

    def test_custom_weights(self):
        """Test ranking with custom weights."""
        custom_weights = RankingWeights(
            embedding_similarity=0.2,
            file_type_relevance=0.6,  # Heavily weight file type
            construct_type_matching=0.1,
            file_recency=0.05,
            file_size_optimization=0.05,
        )

        ranker = ResultRanker(custom_weights)
        results = self.create_test_results()
        context = RankingContext(query="authentication function", query_keywords=["authentication", "function"])

        ranked = ranker.rank_results(results, context)

        # Python source file should rank higher due to file type weight
        python_results = [r for r in ranked if r.file_path.endswith(".py")]
        assert len(python_results) > 0
        assert python_results[0] == ranked[0]  # Should be first


class TestMainAPI:
    """Test the main API functions."""

    def test_rank_search_results(self):
        """Test the main rank_search_results function."""
        results = [
            CodeSearchResult(
                file_path="/path/to/file.py",
                snippet=CodeSnippet(text="def example():", start_line=1, end_line=1),
                similarity_score=0.8,
                file_metadata={"size": 2000},
            )
        ]

        ranked = rank_search_results(results, "search query")

        assert len(ranked) == 1
        assert "composite_score" in ranked[0].file_metadata
        assert "match_reasons" in ranked[0].file_metadata

    def test_generate_match_explanations(self):
        """Test match explanation generation."""
        results = [
            CodeSearchResult(
                file_path="/path/to/authentication.py",
                snippet=CodeSnippet(text="def login():", start_line=1, end_line=1),
                similarity_score=0.8,
            )
        ]

        explanations = generate_match_explanations(results, "authentication system")

        assert "/path/to/authentication.py" in explanations
        assert len(explanations["/path/to/authentication.py"]) > 0

    def test_calculate_advanced_confidence(self):
        """Test advanced confidence calculation."""
        result = CodeSearchResult(
            file_path="/path/to/file.py",
            snippet=CodeSnippet(text="def authenticate():", start_line=1, end_line=1),
            similarity_score=0.9,
        )

        confidence = calculate_advanced_confidence(result, "authentication function")

        assert confidence in ["high", "medium", "low"]


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_results(self):
        """Test ranking with empty results list."""
        ranked = rank_search_results([], "query")
        assert ranked == []

    def test_single_result(self):
        """Test ranking with single result."""
        result = CodeSearchResult(
            file_path="/path/to/file.py",
            snippet=CodeSnippet(text="code", start_line=1, end_line=1),
            similarity_score=0.8,
        )

        ranked = rank_search_results([result], "query")
        assert len(ranked) == 1
        assert ranked[0] == result  # Should be enhanced but same result

    def test_missing_metadata(self):
        """Test ranking with results missing metadata."""
        result = CodeSearchResult(
            file_path="/path/to/file.py",
            snippet=CodeSnippet(text="code", start_line=1, end_line=1),
            similarity_score=0.8,
            file_metadata=None,  # Missing metadata
        )

        ranked = rank_search_results([result], "query")
        assert len(ranked) == 1
        # Should handle missing metadata gracefully
        assert ranked[0].file_metadata is not None  # Should be populated


if __name__ == "__main__":
    pytest.main([__file__])
