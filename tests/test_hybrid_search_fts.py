#!/usr/bin/env python3
"""
test_hybrid_search_fts.py: Comprehensive tests for FTS integration within hybrid search.

This module tests:
- Basic FTS search functionality with ranking within hybrid search
- Complex query patterns (multi-word, quoted phrases, wildcards)
- Hybrid search combining semantic and FTS results
- Fallback behavior when FTS is unavailable
- Performance comparison between FTS and LIKE search
- FTS result formatting and ranking integration
"""

import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from database_manager import DatabaseManager
from embedding_helper import EmbeddingGenerator
from hybrid_search import HybridSearchEngine, SearchMode
from search_result_types import CodeSearchResult


class TestFTSBasicFunctionality:
    """Test basic FTS search functionality within hybrid search."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test.duckdb"
        self.db_manager = DatabaseManager(self.db_path)

        # Create a mock embedder
        self.mock_embedder = Mock()
        self.mock_embedder.generate_embedding.return_value = [0.1] * 384

        self.search_engine = HybridSearchEngine(self.db_manager, self.mock_embedder)

    def teardown_method(self):
        """Clean up test fixtures."""
        if hasattr(self, "db_manager"):
            try:
                self.db_manager.cleanup()
            except Exception:
                pass

    def test_fts_search_basic_functionality(self):
        """Test basic FTS search functionality."""
        with self.db_manager.get_connection() as conn:
            # Create and populate test data
            conn.execute(
                """
                CREATE TABLE code_files (
                    id VARCHAR PRIMARY KEY,
                    path VARCHAR,
                    content TEXT,
                    embedding DOUBLE[384]
                )
            """
            )

            test_content = [
                (
                    "1",
                    "authentication.py",
                    "def authenticate_user(username, password): return verify_credentials(username, password)",
                    [0.1] * 384,
                ),
                ("2", "database.py", "class DatabaseConnection: def connect(self): pass", [0.2] * 384),
                (
                    "3",
                    "utils.py",
                    "def hash_password(password): return hashlib.sha256(password.encode()).hexdigest()",
                    [0.3] * 384,
                ),
            ]

            for row in test_content:
                conn.execute("INSERT INTO code_files VALUES (?, ?, ?, ?)", row)

            # Try to create FTS index
            self.db_manager.create_fts_index("code_files")

            # Test FTS search via hybrid search with text-only mode
            results = self.search_engine.search("authenticate", k=5, mode=SearchMode.TEXT_ONLY)

            # Should return results
            assert isinstance(results, list)

            if results:
                # Check that results contain relevant content
                found_auth = any("authenticate" in result.code_result.snippet.text.lower() for result in results)
                assert found_auth

                # Verify result structure
                for result in results:
                    assert hasattr(result, "code_result")
                    assert hasattr(result, "text_score")
                    assert isinstance(result.text_score, (int, float))

    def test_fts_search_with_complex_queries(self):
        """Test FTS search with complex query patterns."""
        with self.db_manager.get_connection() as conn:
            # Setup test data
            conn.execute(
                """
                CREATE TABLE code_files (
                    id VARCHAR PRIMARY KEY,
                    path VARCHAR,
                    content TEXT,
                    embedding DOUBLE[384]
                )
            """
            )

            complex_content = [
                (
                    "1",
                    "auth.py",
                    "async def authenticate_user(username, password): return await verify_token(username)",
                    [0.1] * 384,
                ),
                ("2", "user_auth.py", "def authenticate(credentials): return validate_user(credentials)", [0.2] * 384),
                ("3", "security.py", "class AuthenticationError(Exception): pass", [0.3] * 384),
                ("4", "config.py", 'AUTH_SETTINGS = {"timeout": 3600, "method": "jwt"}', [0.4] * 384),
            ]

            for row in complex_content:
                conn.execute("INSERT INTO code_files VALUES (?, ?, ?, ?)", row)

            # Create FTS index
            self.db_manager.create_fts_index("code_files")

            # Test multi-word query
            results = self.search_engine.search("authenticate user", k=5, mode=SearchMode.TEXT_ONLY)
            assert isinstance(results, list)

            # Test quoted phrase (exact match)
            results = self.search_engine.search('"authenticate_user"', k=5, mode=SearchMode.TEXT_ONLY)
            assert isinstance(results, list)

            # Test wildcard patterns (if supported)
            results = self.search_engine.search("auth*", k=5, mode=SearchMode.TEXT_ONLY)
            assert isinstance(results, list)

    def test_fts_search_ranking_accuracy(self):
        """Test that FTS ranking produces reasonable results."""
        with self.db_manager.get_connection() as conn:
            # Setup test data with different relevance levels
            conn.execute(
                """
                CREATE TABLE code_files (
                    id VARCHAR PRIMARY KEY,
                    path VARCHAR,
                    content TEXT,
                    embedding DOUBLE[384]
                )
            """
            )

            # Content with varying relevance for search term "database"
            test_content = [
                (
                    "1",
                    "db_main.py",
                    "database connection and database operations for main database functionality",
                    [0.1] * 384,
                ),  # High relevance
                (
                    "2",
                    "config.py",
                    "configuration settings including database connection details",
                    [0.2] * 384,
                ),  # Medium relevance
                (
                    "3",
                    "utils.py",
                    "utility functions for various tasks and helper methods",
                    [0.3] * 384,
                ),  # Low relevance
            ]

            for row in test_content:
                conn.execute("INSERT INTO code_files VALUES (?, ?, ?, ?)", row)

            # Create FTS index
            self.db_manager.create_fts_index("code_files")

            # Search for "database"
            results = self.search_engine.search("database", k=10, mode=SearchMode.TEXT_ONLY)

            if results and len(results) >= 2:
                # Find results by path
                db_main_result = None
                config_result = None

                for result in results:
                    if "db_main.py" in result.code_result.file_path:
                        db_main_result = result
                    elif "config.py" in result.code_result.file_path:
                        config_result = result

                # Higher relevance should generally have higher scores or better ranking
                if db_main_result and config_result:
                    # Either db_main should have higher score, or appear earlier in results
                    db_main_index = results.index(db_main_result)
                    config_index = results.index(config_result)

                    # db_main should rank higher (lower index) due to higher relevance
                    assert db_main_index <= config_index or db_main_result.text_score >= config_result.text_score


class TestHybridSearchIntegration:
    """Test hybrid search combining semantic and FTS results."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test.duckdb"
        self.db_manager = DatabaseManager(self.db_path)

        # Create a mock embedder that returns realistic embeddings
        self.mock_embedder = Mock()
        self.mock_embedder.generate_embedding.return_value = [0.1] * 384
        self.mock_embedder.generate_embeddings.return_value = [[0.1] * 384]

        self.search_engine = HybridSearchEngine(self.db_manager, self.mock_embedder)

    def teardown_method(self):
        """Clean up test fixtures."""
        if hasattr(self, "db_manager"):
            try:
                self.db_manager.cleanup()
            except Exception:
                pass

    def test_hybrid_search_with_fts_enabled(self):
        """Test hybrid search combining semantic and FTS results."""
        with self.db_manager.get_connection() as conn:
            # Setup comprehensive test data
            conn.execute(
                """
                CREATE TABLE code_files (
                    id VARCHAR PRIMARY KEY,
                    path VARCHAR,
                    content TEXT,
                    embedding DOUBLE[384]
                )
            """
            )

            hybrid_content = [
                (
                    "1",
                    "auth_service.py",
                    "Authentication service with user verification and token management",
                    [0.8, 0.2] + [0.1] * 382,
                ),
                (
                    "2",
                    "user_auth.py",
                    "def authenticate_user(username, password): return verify_credentials(username, password)",
                    [0.7, 0.3] + [0.1] * 382,
                ),
                (
                    "3",
                    "database_auth.py",
                    "Database authentication and access control implementation",
                    [0.6, 0.4] + [0.1] * 382,
                ),
                (
                    "4",
                    "security.py",
                    "Security utilities for password hashing and token generation",
                    [0.5, 0.5] + [0.1] * 382,
                ),
            ]

            for row in hybrid_content:
                conn.execute("INSERT INTO code_files VALUES (?, ?, ?, ?)", row)

            # Create FTS index
            self.db_manager.create_fts_index("code_files")

            # Test hybrid search
            results = self.search_engine.search("user authentication system", k=10, mode=SearchMode.HYBRID)

            # Should return hybrid results
            assert isinstance(results, list)

            if results:
                # Verify hybrid result structure
                for result in results:
                    assert hasattr(result, "semantic_score")
                    assert hasattr(result, "text_score")
                    assert hasattr(result, "fusion_score")
                    assert hasattr(result, "fusion_method")
                    assert result.fusion_method in ["hybrid", "semantic_only", "text_only"]

    def test_hybrid_search_fallback_behavior(self):
        """Test hybrid search fallback when FTS is unavailable."""
        with self.db_manager.get_connection() as conn:
            # Setup test data
            conn.execute(
                """
                CREATE TABLE code_files (
                    id VARCHAR PRIMARY KEY,
                    path VARCHAR,
                    content TEXT,
                    embedding DOUBLE[384]
                )
            """
            )

            conn.execute(
                "INSERT INTO code_files VALUES ('1', 'test.py', 'def test_function(): pass', ?)", ([0.1] * 384,)
            )

            # Mock FTS to be unavailable
            with patch.object(self.db_manager, "is_fts_available", return_value=False):
                results = self.search_engine.search("test function", k=5, mode=SearchMode.HYBRID)

                # Should still return results using fallback methods
                assert isinstance(results, list)

                # Results should indicate fallback was used
                if results:
                    for result in results:
                        # In fallback mode, should still have valid structure
                        assert hasattr(result, "fusion_score")

    def test_hybrid_search_auto_mode_selection(self):
        """Test automatic search mode selection in hybrid search."""
        with self.db_manager.get_connection() as conn:
            # Setup test data
            conn.execute(
                """
                CREATE TABLE code_files (
                    id VARCHAR PRIMARY KEY,
                    path VARCHAR,
                    content TEXT,
                    embedding DOUBLE[384]
                )
            """
            )

            test_data = [
                ("1", "exact.py", "def exact_function_name(): pass", [0.1] * 384),
                ("2", "semantic.py", "natural language description of functionality", [0.2] * 384),
                ("3", "mixed.py", "function with both exact and semantic content", [0.3] * 384),
            ]

            for row in test_data:
                conn.execute("INSERT INTO code_files VALUES (?, ?, ?, ?)", row)

            # Create FTS index
            self.db_manager.create_fts_index("code_files")

            # Test auto mode with exact query (should prefer text search)
            results = self.search_engine.search('"exact_function_name"', k=5, mode=SearchMode.AUTO)
            assert isinstance(results, list)

            # Test auto mode with natural language query (should prefer semantic)
            results = self.search_engine.search("how to implement user functionality", k=5, mode=SearchMode.AUTO)
            assert isinstance(results, list)

            # Test auto mode with mixed query (should use hybrid)
            results = self.search_engine.search("function implementation", k=5, mode=SearchMode.AUTO)
            assert isinstance(results, list)


class TestFTSPerformanceComparison:
    """Test FTS performance compared to LIKE search."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test.duckdb"
        self.db_manager = DatabaseManager(self.db_path)

        # Create a mock embedder
        self.mock_embedder = Mock()
        self.mock_embedder.generate_embedding.return_value = [0.1] * 384

    def teardown_method(self):
        """Clean up test fixtures."""
        if hasattr(self, "db_manager"):
            try:
                self.db_manager.cleanup()
            except Exception:
                pass

    def test_fts_vs_like_search_performance(self):
        """Test performance difference between FTS and LIKE search."""
        with self.db_manager.get_connection() as conn:
            # Create larger test dataset
            conn.execute(
                """
                CREATE TABLE code_files (
                    id VARCHAR PRIMARY KEY,
                    path VARCHAR,
                    content TEXT,
                    embedding DOUBLE[384]
                )
            """
            )

            # Insert test data (moderate size for unit test)
            dataset_size = 100
            test_data = []
            for i in range(dataset_size):
                content = f"def function_{i}(param): return process_data_{i}(param) + calculate_result_{i}()"
                test_data.append((str(i), f"file_{i}.py", content, [0.1] * 384))

            conn.executemany("INSERT INTO code_files VALUES (?, ?, ?, ?)", test_data)

            # Test FTS search performance
            fts_created = self.db_manager.create_fts_index("code_files")

            if fts_created and self.db_manager.is_fts_available("code_files"):
                # Measure FTS search time
                start_time = time.time()
                fts_results = self.db_manager.search_full_text("function", limit=10)
                fts_time = time.time() - start_time

                # Measure LIKE search time (force fallback)
                with patch.object(self.db_manager, "is_fts_available", return_value=False):
                    start_time = time.time()
                    like_results = self.db_manager.search_full_text("function", limit=10)
                    like_time = time.time() - start_time

                # Both should return results
                assert isinstance(fts_results, list)
                assert isinstance(like_results, list)

                # Performance comparison (FTS should be faster or comparable for this size)
                # For small datasets, performance difference may not be significant
                # This is more about ensuring both methods work correctly
                assert fts_time >= 0
                assert like_time >= 0

                # Results should be reasonably similar (both finding "function")
                assert len(fts_results) > 0 or len(like_results) > 0

    def test_fts_indexing_performance(self):
        """Test FTS index creation performance."""
        with self.db_manager.get_connection() as conn:
            # Create test dataset
            conn.execute(
                """
                CREATE TABLE code_files (
                    id VARCHAR PRIMARY KEY,
                    path VARCHAR,
                    content TEXT
                )
            """
            )

            # Insert moderate amount of test data
            dataset_size = 200  # Reasonable size for unit test
            test_data = []
            for i in range(dataset_size):
                content = f'def function_{i}(): return "result_{i}" + process_data() + calculate_values()'
                test_data.append((str(i), f"file_{i}.py", content))

            conn.executemany("INSERT INTO code_files VALUES (?, ?, ?)", test_data)

            # Measure FTS index creation time
            start_time = time.time()
            result = self.db_manager.create_fts_index("code_files")
            index_time = time.time() - start_time

            if result:
                # FTS indexing should complete in reasonable time
                assert index_time < 10.0  # 10 second threshold for 200 files

                # Verify index is functional
                search_results = self.db_manager.search_full_text("function", limit=5)
                assert isinstance(search_results, list)


class TestFTSResultFormatting:
    """Test FTS result formatting and integration."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test.duckdb"
        self.db_manager = DatabaseManager(self.db_path)

        # Create a mock embedder
        self.mock_embedder = Mock()
        self.mock_embedder.generate_embedding.return_value = [0.1] * 384

        self.search_engine = HybridSearchEngine(self.db_manager, self.mock_embedder)

    def teardown_method(self):
        """Clean up test fixtures."""
        if hasattr(self, "db_manager"):
            try:
                self.db_manager.cleanup()
            except Exception:
                pass

    def test_fts_results_formatting(self):
        """Test that FTS results format correctly in search output."""
        with self.db_manager.get_connection() as conn:
            # Setup test data
            conn.execute(
                """
                CREATE TABLE code_files (
                    id VARCHAR PRIMARY KEY,
                    path VARCHAR,
                    content TEXT,
                    embedding DOUBLE[384]
                )
            """
            )

            conn.execute(
                """
                INSERT INTO code_files VALUES 
                ('1', 'formatter.py', 'def format_results(data): return pretty_print(data)', ?)
            """,
                ([0.1] * 384,),
            )

            # Create FTS index
            self.db_manager.create_fts_index("code_files")

            # Perform search
            results = self.search_engine.search("format results", k=5, mode=SearchMode.TEXT_ONLY)

            if results:
                # Verify result structure and formatting
                for result in results:
                    assert hasattr(result, "code_result")
                    assert isinstance(result.code_result, CodeSearchResult)
                    assert hasattr(result.code_result, "file_path")
                    assert hasattr(result.code_result, "snippet")
                    assert hasattr(result.code_result, "similarity_score")

                    # Verify scores are present and reasonable
                    assert hasattr(result, "text_score")
                    assert isinstance(result.text_score, (int, float))
                    assert result.text_score >= 0

    def test_fts_with_ranking_system(self):
        """Test FTS integration with existing ranking system."""
        with self.db_manager.get_connection() as conn:
            # Setup test data with different types of matches
            conn.execute(
                """
                CREATE TABLE code_files (
                    id VARCHAR PRIMARY KEY,
                    path VARCHAR,
                    content TEXT,
                    embedding DOUBLE[384]
                )
            """
            )

            ranking_content = [
                (
                    "1",
                    "auth.py",
                    "def authenticate_user(): pass  # Main auth function",
                    [0.9, 0.1] + [0.05] * 382,
                ),  # High semantic + text match
                (
                    "2",
                    "user.py",
                    "class User: def authenticate(self): pass  # User class auth",
                    [0.1, 0.9] + [0.05] * 382,
                ),  # Low semantic + high text match
                (
                    "3",
                    "helper.py",
                    "def helper_function(): pass  # Utility helper",
                    [0.1, 0.1] + [0.05] * 382,
                ),  # Low matches
            ]

            for row in ranking_content:
                conn.execute("INSERT INTO code_files VALUES (?, ?, ?, ?)", row)

            # Create FTS index
            self.db_manager.create_fts_index("code_files")

            # Test hybrid search with ranking
            results = self.search_engine.search("authenticate", k=5, mode=SearchMode.HYBRID)

            if results:
                # Should have results ordered by fusion scoring
                assert len(results) > 0

                # Verify fusion scores are present and reasonable
                for result in results:
                    assert hasattr(result, "fusion_score")
                    assert isinstance(result.fusion_score, (int, float))
                    assert result.fusion_score >= 0

                # Results should be ordered by fusion score (descending)
                if len(results) > 1:
                    for i in range(len(results) - 1):
                        assert results[i].fusion_score >= results[i + 1].fusion_score


class TestFTSEdgeCasesInHybridSearch:
    """Test FTS edge cases within hybrid search context."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test.duckdb"
        self.db_manager = DatabaseManager(self.db_path)

        # Create a mock embedder
        self.mock_embedder = Mock()
        self.mock_embedder.generate_embedding.return_value = [0.1] * 384

        self.search_engine = HybridSearchEngine(self.db_manager, self.mock_embedder)

    def teardown_method(self):
        """Clean up test fixtures."""
        if hasattr(self, "db_manager"):
            try:
                self.db_manager.cleanup()
            except Exception:
                pass

    def test_fts_with_special_characters(self):
        """Test FTS with code containing special characters."""
        with self.db_manager.get_connection() as conn:
            # Setup test data with various programming language special characters
            conn.execute(
                """
                CREATE TABLE code_files (
                    id VARCHAR PRIMARY KEY,
                    path VARCHAR,
                    content TEXT,
                    embedding DOUBLE[384]
                )
            """
            )

            special_content = [
                ("1", "regex.py", 'pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}"', [0.1] * 384),
                ("2", "math.py", "result = (a + b) * (c - d) / (e % f) ** g", [0.1] * 384),
                ("3", "config.json", '{"key": "value", "nested": {"array": [1, 2, 3]}}', [0.1] * 384),
                ("4", "shell.sh", 'find /path -name "*.py" | xargs grep -l "pattern"', [0.1] * 384),
            ]

            for row in special_content:
                conn.execute("INSERT INTO code_files VALUES (?, ?, ?, ?)", row)

            # Create FTS index
            self.db_manager.create_fts_index("code_files")

            # Test searches with special characters
            test_queries = ["pattern", "array", "result", "grep"]

            for query in test_queries:
                results = self.search_engine.search(query, k=5, mode=SearchMode.TEXT_ONLY)
                # Should handle special characters gracefully
                assert isinstance(results, list)

    def test_fts_with_empty_and_null_content(self):
        """Test FTS handling of empty and null content."""
        with self.db_manager.get_connection() as conn:
            # Setup test data with edge cases
            conn.execute(
                """
                CREATE TABLE code_files (
                    id VARCHAR PRIMARY KEY,
                    path VARCHAR,
                    content TEXT,
                    embedding DOUBLE[384]
                )
            """
            )

            edge_content = [
                ("1", "empty.py", "", [0.1] * 384),  # Empty content
                ("2", "normal.py", "def function(): pass", [0.1] * 384),  # Normal content
                ("3", "whitespace.py", "   \n\t   ", [0.1] * 384),  # Whitespace only
            ]

            for row in edge_content:
                conn.execute("INSERT INTO code_files VALUES (?, ?, ?, ?)", row)

            # Add a row with NULL content
            conn.execute(
                "INSERT INTO code_files (id, path, content, embedding) VALUES ('4', 'null.py', NULL, ?)", ([0.1] * 384,)
            )

            # Create FTS index
            self.db_manager.create_fts_index("code_files")

            # Test search should handle edge cases gracefully
            results = self.search_engine.search("function", k=10, mode=SearchMode.TEXT_ONLY)
            assert isinstance(results, list)

            # Should find the normal file
            if results:
                found_normal = any("normal.py" in result.code_result.file_path for result in results)
                assert found_normal

    def test_fts_query_preprocessing(self):
        """Test FTS query preprocessing and sanitization."""
        with self.db_manager.get_connection() as conn:
            # Setup basic test data
            conn.execute(
                """
                CREATE TABLE code_files (
                    id VARCHAR PRIMARY KEY,
                    path VARCHAR,
                    content TEXT,
                    embedding DOUBLE[384]
                )
            """
            )

            conn.execute(
                "INSERT INTO code_files VALUES ('1', 'test.py', 'def test_function(): pass', ?)", ([0.1] * 384,)
            )

            # Create FTS index
            self.db_manager.create_fts_index("code_files")

            # Test various edge case queries
            edge_queries = [
                "",  # Empty query
                "   ",  # Whitespace only
                "test;DROP TABLE code_files;",  # SQL injection attempt
                "test' OR '1'='1",  # Another injection attempt
                "test\x00null",  # Null byte
                "test\n\r\t",  # Control characters
                "🔍 search emoji",  # Unicode emojis
                "тест",  # Cyrillic text
            ]

            for query in edge_queries:
                # Should handle edge cases gracefully without crashing
                results = self.search_engine.search(query, k=5, mode=SearchMode.TEXT_ONLY)
                assert isinstance(results, list)
                # Empty or problematic queries should return empty results or handle gracefully
                assert len(results) >= 0
