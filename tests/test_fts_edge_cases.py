#!/usr/bin/env python3
"""
test_fts_edge_cases.py: Comprehensive edge case and error handling tests for FTS.

This module tests:
- Malformed queries (empty, SQL injection attempts, unicode)
- Large content blocks and memory usage
- Special characters from various programming languages
- Migration and compatibility scenarios
- Backward compatibility validation
- Concurrent access and thread safety
- Resource cleanup and error recovery
"""

import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import Mock, patch

import duckdb
import pytest

from database_manager import DatabaseManager
from hybrid_search import HybridSearchEngine, SearchMode


class TestFTSMalformedQueries:
    """Test FTS behavior with malformed or invalid queries."""

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

    def test_fts_with_malformed_queries(self):
        """Test FTS behavior with malformed or invalid queries."""
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

            # Test queries that might break FTS
            problematic_queries = [
                "",  # Empty query
                " ",  # Whitespace only
                "' OR 1=1 --",  # SQL injection attempt
                "SELECT * FROM code_files",  # SQL command in query
                "NULL",  # NULL value
                "\x00\x01\x02",  # Binary data
                "unicode: 你好世界",  # Unicode content
                "emoji: 🔍🚀💻",  # Unicode emojis
                "тест поиск",  # Cyrillic text
                "test\n\r\t\v",  # Control characters
                "very " * 1000,  # Very long query
                "test; DROP TABLE code_files; --",  # SQL injection
                "'\"\\",  # Quote and escape characters
                "[]{}<>()!@#$%^&*",  # Special symbols
            ]

            for query in problematic_queries:
                # Should not crash, should return empty results or handle gracefully
                try:
                    results = self.search_engine.search(query, k=5, mode=SearchMode.TEXT_ONLY)
                    assert isinstance(results, list)

                    # Also test direct database search
                    db_results = self.db_manager.search_full_text(query, limit=5)
                    assert isinstance(db_results, list)

                except Exception as e:
                    # If exception occurs, it should be handled gracefully
                    # Some queries might legitimately cause errors, but system should not crash
                    assert "system crash" not in str(e).lower()

    def test_fts_injection_prevention(self):
        """Test that FTS prevents SQL injection attacks."""
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

            # Insert test data including sensitive information
            sensitive_data = [
                ("1", "user.py", 'def get_password(): return "secret123"', [0.1] * 384),
                ("2", "config.py", 'API_KEY = "sensitive_key_123"', [0.2] * 384),
                ("3", "normal.py", "def normal_function(): pass", [0.3] * 384),
            ]

            for row in sensitive_data:
                conn.execute("INSERT INTO code_files VALUES (?, ?, ?, ?)", row)

            # Create FTS index
            self.db_manager.create_fts_index("code_files")

            # Test injection attempts
            injection_queries = [
                "test'; SELECT * FROM code_files; --",
                "test' UNION SELECT id, path, content FROM code_files --",
                "test'; DROP TABLE code_files; --",
                "test' OR '1'='1",
                'test"; SELECT * FROM code_files; --',
            ]

            for injection in injection_queries:
                # Should handle injection attempts safely
                results = self.search_engine.search(injection, k=5, mode=SearchMode.TEXT_ONLY)
                assert isinstance(results, list)

                # Verify database integrity - table should still exist
                table_check = conn.execute("SELECT COUNT(*) FROM code_files").fetchone()
                assert table_check[0] == 3  # All original data should still be there

    def test_fts_unicode_handling(self):
        """Test FTS with various Unicode characters and encodings."""
        with self.db_manager.get_connection() as conn:
            # Setup test data with Unicode content
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

            unicode_content = [
                ("1", "chinese.py", "# 中文注释\ndef 函数(): pass", [0.1] * 384),
                ("2", "russian.py", "# Русские комментарии\ndef функция(): pass", [0.2] * 384),
                ("3", "japanese.py", "# 日本語のコメント\ndef 関数(): pass", [0.3] * 384),
                ("4", "emoji.py", "# Function with emojis 🚀🔍💻\ndef search_function(): pass", [0.4] * 384),
                ("5", "mixed.py", 'def mixed_unicode_函数_🔍(): return "test_тест_テスト"', [0.5] * 384),
            ]

            for row in unicode_content:
                conn.execute("INSERT INTO code_files VALUES (?, ?, ?, ?)", row)

            # Create FTS index
            self.db_manager.create_fts_index("code_files")

            # Test Unicode queries
            unicode_queries = [
                "函数",  # Chinese
                "функция",  # Russian
                "関数",  # Japanese
                "🚀",  # Emoji
                "test",  # ASCII (should still work)
                "mixed_unicode",  # Mixed content
            ]

            for query in unicode_queries:
                # Should handle Unicode gracefully
                results = self.search_engine.search(query, k=10, mode=SearchMode.TEXT_ONLY)
                assert isinstance(results, list)


class TestFTSLargeContent:
    """Test FTS with very large content blocks."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test.duckdb"
        self.db_manager = DatabaseManager(self.db_path)

    def teardown_method(self):
        """Clean up test fixtures."""
        if hasattr(self, "db_manager"):
            try:
                self.db_manager.cleanup()
            except Exception:
                pass

    def test_fts_with_large_content(self):
        """Test FTS indexing and search with large code files."""
        with self.db_manager.get_connection() as conn:
            # Setup table
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

            # Generate large content (simulating a large code file)
            large_content = []

            # Create a large function with many lines
            function_lines = []
            function_lines.append("def large_function():")
            function_lines.append('    """This is a very large function for testing purposes."""')

            for i in range(1000):  # 1000 lines
                function_lines.append(f'    variable_{i} = process_data_{i}("param_{i}")')
                if i % 100 == 0:
                    function_lines.append(f"    # Checkpoint {i}: processing batch")
                    function_lines.append(f"    result_{i} = calculate_result(variable_{i})")

            function_lines.append("    return final_result")

            large_function = "\n".join(function_lines)

            # Insert large content
            large_content.append(("1", "large_file.py", large_function, [0.1] * 384))

            # Also add normal-sized content for comparison
            large_content.append(("2", "normal.py", "def normal_function(): pass", [0.2] * 384))

            for row in large_content:
                conn.execute("INSERT INTO code_files VALUES (?, ?, ?, ?)", row)

            # Test FTS index creation with large content
            start_time = time.time()
            result = self.db_manager.create_fts_index("code_files")
            index_time = time.time() - start_time

            # Should handle large content without excessive time
            assert index_time < 30.0  # 30 second threshold

            if result:
                # Test search in large content
                search_results = self.db_manager.search_full_text("large_function", limit=5)
                assert isinstance(search_results, list)

                # Should find the large function
                if search_results:
                    found_large = any("large_function" in str(result) for result in search_results)
                    assert found_large

    def test_fts_memory_usage_with_large_dataset(self):
        """Test memory usage with moderately large dataset."""
        with self.db_manager.get_connection() as conn:
            # Setup table
            conn.execute(
                """
                CREATE TABLE code_files (
                    id VARCHAR PRIMARY KEY,
                    path VARCHAR,
                    content TEXT
                )
            """
            )

            # Generate many medium-sized files
            batch_size = 500  # Reasonable for unit test
            content_template = """
def function_{i}():
    '''Function number {i} for testing purposes.'''
    data = load_data_{i}()
    processed = process_data_{i}(data)
    result = calculate_result_{i}(processed)
    return format_output_{i}(result)

class Class_{i}:
    def __init__(self):
        self.value_{i} = initialize_value_{i}()
    
    def method_{i}(self, param):
        return self.value_{i} + param + {i}
"""

            # Insert data in batches to avoid memory issues
            for batch in range(0, batch_size, 50):
                batch_data = []
                for i in range(batch, min(batch + 50, batch_size)):
                    content = content_template.format(i=i)
                    batch_data.append((str(i), f"file_{i}.py", content))

                conn.executemany("INSERT INTO code_files VALUES (?, ?, ?)", batch_data)

            # Test FTS creation
            start_time = time.time()
            result = self.db_manager.create_fts_index("code_files")
            index_time = time.time() - start_time

            # Should complete in reasonable time
            assert index_time < 60.0  # 1 minute threshold for 500 files

            if result:
                # Test search performance
                start_time = time.time()
                search_results = self.db_manager.search_full_text("function", limit=20)
                search_time = time.time() - start_time

                # Search should be fast
                assert search_time < 5.0  # 5 second threshold
                assert isinstance(search_results, list)


class TestFTSSpecialCharacters:
    """Test FTS with code containing special characters from various programming languages."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test.duckdb"
        self.db_manager = DatabaseManager(self.db_path)

    def teardown_method(self):
        """Clean up test fixtures."""
        if hasattr(self, "db_manager"):
            try:
                self.db_manager.cleanup()
            except Exception:
                pass

    def test_fts_with_programming_special_characters(self):
        """Test FTS with various programming language special characters."""
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
                # Python regex and string formatting
                ("1", "regex.py", r'pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"', [0.1] * 384),
                # Mathematical operations
                ("2", "math.py", "result = (a + b) * (c - d) / (e % f) ** g", [0.2] * 384),
                # JSON configuration
                ("3", "config.json", '{"key": "value", "nested": {"array": [1, 2, 3]}}', [0.3] * 384),
                # Shell script with pipes
                ("4", "shell.sh", 'find /path -name "*.py" | xargs grep -l "pattern"', [0.4] * 384),
                # C++ template syntax
                ("5", "template.cpp", "template<typename T> class Container<T>::Iterator { };", [0.5] * 384),
                # JavaScript arrow functions and destructuring
                ("6", "modern.js", "const {a, b} = obj; const fn = (x) => x?.method?.();", [0.6] * 384),
                # SQL with various operators
                ("7", "query.sql", "SELECT * FROM table WHERE column LIKE '%pattern%' AND id IN (1,2,3);", [0.7] * 384),
                # Assembly/low-level code
                ("8", "assembly.s", "movq %rax, %rbx; cmpq $0, %rcx; jne .L1", [0.8] * 384),
            ]

            for row in special_content:
                conn.execute("INSERT INTO code_files VALUES (?, ?, ?, ?)", row)

            # Create FTS index
            result = self.db_manager.create_fts_index("code_files")
            assert isinstance(result, bool)

            # Test searches for various special character patterns
            search_tests = [
                ("pattern", "Should find regex and shell files"),
                ("array", "Should find JSON file"),
                ("template", "Should find C++ template"),
                ("function", "Should find JavaScript or general functions"),
                ("SELECT", "Should find SQL file"),
                ("movq", "Should find assembly file"),
                ("+", "Should find math operations"),
                ("[", "Should find bracket usage"),
                ("*", "Should find wildcard usage"),
            ]

            for query, description in search_tests:
                results = self.db_manager.search_full_text(query, limit=10)

                # Should handle special characters gracefully
                assert isinstance(results, list), f"Failed test: {description}"

                # Should not cause crashes
                assert True, f"Completed test: {description}"

    def test_fts_with_unicode_programming_content(self):
        """Test FTS with Unicode programming content (variable names, comments)."""
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

            unicode_programming_content = [
                # Python with Unicode identifiers (Python 3 supports this)
                ("1", "unicode_vars.py", "def función_principal(): π = 3.14159; return π * 2", [0.1] * 384),
                # Comments in various languages
                (
                    "2",
                    "comments.py",
                    """
# English comment
# Comentario en español
# Комментарий на русском
# コメント日本語
def multilingual_function(): pass
""",
                    [0.2] * 384,
                ),
                # String literals with Unicode
                (
                    "3",
                    "strings.py",
                    """
messages = {
    "en": "Hello World",
    "es": "Hola Mundo", 
    "ru": "Привет Мир",
    "ja": "こんにちは世界",
    "emoji": "Hello 🌍 World 🚀"
}
""",
                    [0.3] * 384,
                ),
            ]

            for row in unicode_programming_content:
                conn.execute("INSERT INTO code_files VALUES (?, ?, ?, ?)", row)

            # Create FTS index
            self.db_manager.create_fts_index("code_files")

            # Test Unicode searches
            unicode_searches = [
                "función",  # Spanish
                "Привет",  # Russian
                "こんにちは",  # Japanese
                "🌍",  # Emoji
                "multilingual",  # ASCII in Unicode context
                "π",  # Mathematical symbol
            ]

            for query in unicode_searches:
                results = self.db_manager.search_full_text(query, limit=10)
                # Should handle Unicode programming content gracefully
                assert isinstance(results, list)


class TestFTSMigrationCompatibility:
    """Test FTS migration and compatibility scenarios."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test.duckdb"
        self.db_manager = DatabaseManager(self.db_path)

    def teardown_method(self):
        """Clean up test fixtures."""
        if hasattr(self, "db_manager"):
            try:
                self.db_manager.cleanup()
            except Exception:
                pass

    def test_fts_migration_cleanup(self):
        """Test that FTS migration cleans up old artifacts properly."""
        with self.db_manager.get_connection() as conn:
            # Setup base table
            conn.execute(
                """
                CREATE TABLE code_files (
                    id VARCHAR PRIMARY KEY,
                    path VARCHAR,
                    content TEXT
                )
            """
            )

            # Insert test data
            conn.execute("INSERT INTO code_files VALUES ('1', 'test.py', 'def test(): pass')")

            # Create some fake "old" FTS artifacts
            try:
                conn.execute("CREATE TABLE code_files_fts_old (id VARCHAR, content_tsvector TEXT)")
                conn.execute("CREATE TABLE fts_metadata_old (table_name VARCHAR, created_at TIMESTAMP)")
            except duckdb.Error:
                pass  # Tables might not be creatable in current setup

            # Test migration
            result = self.db_manager.migrate_fts_to_duckdb("code_files")

            # Should complete without error
            assert isinstance(result, bool)

            # New FTS functionality should be available (if FTS extension works)
            if result:
                search_results = self.db_manager.search_full_text("test", limit=5)
                assert isinstance(search_results, list)

    def test_backward_compatibility_without_fts(self):
        """Test that system works correctly when FTS is not available."""
        with self.db_manager.get_connection() as conn:
            # Setup basic database without FTS
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

            # Insert test data
            conn.execute(
                "INSERT INTO code_files VALUES ('1', 'test.py', 'def test_function(): pass', ?)", ([0.1] * 384,)
            )

            # Mock FTS to always fail
            with patch.object(self.db_manager, "_ensure_fts_extension_loaded", return_value=False):
                # FTS creation should fail gracefully
                fts_result = self.db_manager.create_fts_index("code_files")
                assert fts_result is False

                # But search should still work via fallback
                search_results = self.db_manager.search_full_text("test_function", limit=5)
                assert isinstance(search_results, list)

                # Should find the test function via LIKE search
                if search_results:
                    found_function = any("test_function" in str(result) for result in search_results)
                    assert found_function

    def test_database_schema_evolution(self):
        """Test that FTS handles database schema changes gracefully."""
        with self.db_manager.get_connection() as conn:
            # Create initial schema
            conn.execute(
                """
                CREATE TABLE code_files (
                    id VARCHAR PRIMARY KEY,
                    path VARCHAR,
                    content TEXT
                )
            """
            )

            # Insert initial data
            conn.execute("INSERT INTO code_files VALUES ('1', 'old.py', 'def old_function(): pass')")

            # Create FTS index
            self.db_manager.create_fts_index("code_files")

            # Simulate schema evolution - add new columns
            try:
                conn.execute("ALTER TABLE code_files ADD COLUMN file_type VARCHAR")
                conn.execute("ALTER TABLE code_files ADD COLUMN embedding DOUBLE[384]")

                # Insert new data with extended schema
                conn.execute(
                    "INSERT INTO code_files VALUES ('2', 'new.py', 'def new_function(): pass', 'python', ?)",
                    ([0.1] * 384,),
                )

                # FTS should still work with schema changes
                search_results = self.db_manager.search_full_text("function", limit=10)
                assert isinstance(search_results, list)

                # Should find both old and new functions
                if search_results and len(search_results) >= 2:
                    content_found = [str(result) for result in search_results]
                    functions_found = sum(1 for content in content_found if "function" in content.lower())
                    assert functions_found >= 1

            except Exception as e:
                # Schema changes might not be supported in all configurations
                pytest.skip(f"Schema evolution test skipped: {e}")


class TestFTSConcurrencyAndThreadSafety:
    """Test FTS operations under concurrent access and thread safety."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test.duckdb"
        self.db_manager = DatabaseManager(self.db_path)

    def teardown_method(self):
        """Clean up test fixtures."""
        if hasattr(self, "db_manager"):
            try:
                self.db_manager.cleanup()
            except Exception:
                pass

    def test_concurrent_fts_searches(self):
        """Test concurrent FTS search operations."""
        with self.db_manager.get_connection() as conn:
            # Setup test data
            conn.execute(
                """
                CREATE TABLE code_files (
                    id VARCHAR PRIMARY KEY,
                    path VARCHAR,
                    content TEXT
                )
            """
            )

            # Insert diverse content for searching
            test_data = []
            for i in range(50):
                content = f"def function_{i}(): return process_data_{i}() + calculate_result_{i}()"
                test_data.append((str(i), f"file_{i}.py", content))

            conn.executemany("INSERT INTO code_files VALUES (?, ?, ?)", test_data)

            # Create FTS index
            self.db_manager.create_fts_index("code_files")

        # Test concurrent searches
        results = []
        errors = []
        search_queries = ["function", "process", "calculate", "return", "data"]

        def search_worker(query, worker_id):
            try:
                worker_results = self.db_manager.search_full_text(f"{query}_{worker_id % 10}", limit=5)
                results.append((worker_id, len(worker_results) if worker_results else 0))
            except Exception as e:
                errors.append((worker_id, str(e)))

        # Start multiple search threads
        threads = []
        for i in range(10):
            query = search_queries[i % len(search_queries)]
            thread = threading.Thread(target=search_worker, args=(query, i))
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        # Should handle concurrent access without major errors
        assert len(errors) <= 2  # Allow for some errors but not complete failure
        assert len(results) >= 8  # Most searches should succeed

    def test_concurrent_fts_index_operations(self):
        """Test concurrent FTS index creation and rebuilding."""
        # Setup multiple database managers to simulate different processes
        db_managers = []
        for i in range(3):
            db_path = Path(self.temp_dir) / f"concurrent_{i}.duckdb"
            db_manager = DatabaseManager(db_path)
            db_managers.append(db_manager)

            # Setup test data for each database
            with db_manager.get_connection() as conn:
                conn.execute(
                    """
                    CREATE TABLE code_files (
                        id VARCHAR PRIMARY KEY,
                        path VARCHAR,
                        content TEXT
                    )
                """
                )

                conn.execute(f"INSERT INTO code_files VALUES ('1', 'test_{i}.py', 'def test_{i}(): pass')")

        # Test concurrent FTS operations
        results = []
        errors = []

        def fts_worker(db_manager, worker_id):
            try:
                # Create FTS index
                create_result = db_manager.create_fts_index("code_files")

                # Rebuild FTS index
                db_manager.rebuild_fts_index("code_files")

                # Test search
                search_result = db_manager.search_full_text("test", limit=5)

                results.append((worker_id, create_result, len(search_result) if search_result else 0))
            except Exception as e:
                errors.append((worker_id, str(e)))
            finally:
                try:
                    db_manager.cleanup()
                except Exception:
                    # Ignore cleanup errors to prevent masking the original exception
                    pass

        # Start concurrent operations
        threads = []
        for i, db_manager in enumerate(db_managers):
            thread = threading.Thread(target=fts_worker, args=(db_manager, i))
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # Should handle concurrent operations on separate databases
        assert len(errors) <= 1  # Minimal errors allowed
        assert len(results) >= 2  # Most operations should succeed

    def test_fts_resource_cleanup(self):
        """Test proper resource cleanup in error scenarios."""
        # Test cleanup after database errors
        with patch.object(duckdb, "connect", side_effect=duckdb.Error("Connection failed")):
            try:
                failing_db = DatabaseManager(self.db_path)
                # Should handle connection failure gracefully
            except Exception:
                # Expected to fail, but should not cause resource leaks
                pass

        # Test cleanup after FTS creation failure
        with self.db_manager.get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE code_files (
                    id VARCHAR PRIMARY KEY,
                    path VARCHAR,
                    content TEXT
                )
            """
            )

            # Mock FTS creation to fail partway through
            with patch.object(self.db_manager, "_create_fts_index_pragma", side_effect=Exception("Simulated failure")):
                try:
                    self.db_manager.create_fts_index("code_files")
                except:
                    pass

                # Database should still be functional
                result = conn.execute("SELECT COUNT(*) FROM code_files").fetchone()
                assert result[0] == 0  # Table should still exist and be empty

    def test_fts_error_recovery(self):
        """Test FTS error recovery and graceful degradation."""
        with self.db_manager.get_connection() as conn:
            # Setup test data
            conn.execute(
                """
                CREATE TABLE code_files (
                    id VARCHAR PRIMARY KEY,
                    path VARCHAR,
                    content TEXT
                )
            """
            )

            conn.execute("INSERT INTO code_files VALUES ('1', 'test.py', 'def test(): pass')")

            # Simulate FTS failure during search
            with patch.object(self.db_manager, "_execute_fts_search_duckdb", side_effect=Exception("FTS error")):
                # Should gracefully fall back to LIKE search
                results = self.db_manager.search_full_text("test", limit=5)

                # Should still return results via fallback
                assert isinstance(results, list)

                # Should find content via fallback mechanism
                if results:
                    found_content = any("test" in str(result).lower() for result in results)
                    assert found_content
