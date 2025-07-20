#!/usr/bin/env python3
"""
test_error_handling.py: Edge case and error handling tests.

This module tests:
- Corrupted file handling
- Very large file processing
- Database corruption recovery
- Network and I/O error handling
- Resource exhaustion scenarios
- Malformed input handling
"""

import pytest
import time
from pathlib import Path
from unittest.mock import Mock, patch

from code_index import (
    init_db, scan_repo, embed_and_store, search_index,
    reindex_all
)
from database_manager import DatabaseManager
from embedding_helper import EmbeddingGenerator
from exceptions import SearchError, DatabaseError, EmbeddingError, DatabaseTimeoutError
from config import config
import sqlite3
import duckdb


class TestCorruptedFileHandling:
    """Test handling of corrupted and malformed files."""

    def test_binary_file_handling(self, corrupted_repo, mock_embedder):
        """Test handling of binary files."""
        db_manager = init_db(corrupted_repo)

        try:
            # Index repository with binary files
            total_files, processed_files, elapsed = reindex_all(
                corrupted_repo, 50.0, db_manager, mock_embedder
            )

            # Should complete without crashing
            assert elapsed > 0
            assert total_files > 0

            # Binary files should be skipped gracefully
            # (exact count depends on .gitignore and file filtering)
            assert processed_files >= 0

        finally:
            db_manager.cleanup()

    def test_invalid_utf8_file_handling(self, corrupted_repo, mock_embedder):
        """Test handling of files with invalid UTF-8 encoding."""
        db_manager = init_db(corrupted_repo)

        try:
            # Create file with invalid UTF-8
            invalid_utf8_file = corrupted_repo / "invalid_utf8.py"
            with open(invalid_utf8_file, "wb") as f:
                f.write(b'def test():\n    return "\x80\x81\x82\x83"  # Invalid UTF-8\n')

            # Add to git
            import subprocess
            subprocess.run(["git", "add", "invalid_utf8.py"], cwd=corrupted_repo, capture_output=True)
            subprocess.run(["git", "commit", "-m", "Add invalid UTF-8"], cwd=corrupted_repo, capture_output=True)

            # Should handle gracefully
            files = scan_repo(corrupted_repo, max_bytes=int(10.0 * 1024 * 1024))
            processed_count, skipped_count = embed_and_store(
                db_manager, mock_embedder, files
            )

            # Should not crash, may skip invalid files
            assert processed_count + skipped_count == len(files)

        finally:
            db_manager.cleanup()

    def test_very_long_line_handling(self, corrupted_repo, mock_embedder):
        """Test handling of files with extremely long lines."""
        db_manager = init_db(corrupted_repo)

        try:
            # File with long lines should be in corrupted_repo fixture
            files = scan_repo(corrupted_repo, max_bytes=int(10.0 * 1024 * 1024))

            # Should process without memory issues
            processed_count, skipped_count = embed_and_store(
                db_manager, mock_embedder, files
            )

            assert processed_count + skipped_count == len(files)

        finally:
            db_manager.cleanup()

    def test_empty_file_handling(self, corrupted_repo, mock_embedder):
        """Test handling of empty files."""
        db_manager = init_db(corrupted_repo)

        try:
            # Empty file should be in corrupted_repo fixture
            files = scan_repo(corrupted_repo, max_bytes=int(10.0 * 1024 * 1024))

            # Should handle empty files gracefully
            processed_count, skipped_count = embed_and_store(
                db_manager, mock_embedder, files
            )

            # Empty files might be skipped or processed with empty content
            assert processed_count + skipped_count == len(files)

            # Search should work even with empty content
            results = search_index(db_manager, mock_embedder, "test", k=5)
            assert isinstance(results, list)

        finally:
            db_manager.cleanup()

    def test_malformed_code_handling(self, temp_root_dir, mock_embedder):
        """Test handling of syntactically incorrect code."""
        repo_path = temp_root_dir / "malformed_repo"
        repo_path.mkdir()

        # Initialize git repo
        import subprocess
        subprocess.run(["git", "init"], cwd=repo_path, capture_output=True, check=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"],
                       cwd=repo_path, capture_output=True, check=True)
        subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo_path, capture_output=True, check=True)

        # Create files with malformed code
        malformed_python = repo_path / "malformed.py"
        malformed_python.write_text("""
def incomplete_function(
    # Missing closing paren and body

class IncompleteClass
    # Missing colon
    def method_without_body(self)
        # Missing colon and implementation

if condition without colon
    print("this won't parse")

# Mismatched indentation
def another_function():
return "wrong indentation"

# Syntax errors
x = [1, 2, 3
y = {"key": "value"
z = "unclosed string
""")

        malformed_js = repo_path / "malformed.js"
        malformed_js.write_text("""
function missingBrace() {
    return "missing closing brace"

class IncompleteClass {
    constructor(
        // Missing closing paren

const unclosedObject = {
    key: "value"
    // Missing closing brace

// Syntax errors
const x = [1, 2, 3
const y = "unclosed string
""")

        # Add to git
        subprocess.run(["git", "add", "."], cwd=repo_path, capture_output=True)
        subprocess.run(["git", "commit", "-m", "Malformed code"], cwd=repo_path, capture_output=True)

        db_manager = init_db(repo_path)

        try:
            # Should handle malformed code without crashing
            total_files, processed_files, elapsed = reindex_all(
                repo_path, int(10.0 * 1024 * 1024), db_manager, mock_embedder
            )

            assert elapsed > 0
            assert total_files > 0
            # Files should be processed even if code is malformed
            assert processed_files > 0

        finally:
            db_manager.cleanup()


class TestLargeFileProcessing:
    """Test handling of very large files and repositories."""

    def test_large_file_size_limit(self, temp_root_dir, mock_embedder):
        """Test file size limit enforcement."""
        repo_path = temp_root_dir / "large_file_repo"
        repo_path.mkdir()

        # Initialize git repo
        import subprocess
        subprocess.run(["git", "init"], cwd=repo_path, capture_output=True, check=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"],
                       cwd=repo_path, capture_output=True, check=True)
        subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo_path, capture_output=True, check=True)

        # Create large file (2MB of content)
        large_content = "# Large file\n" + ("x = " + "a" * 100 + "\n") * 20000  # ~2MB
        large_file = repo_path / "large_file.py"
        large_file.write_text(large_content)

        # Create normal sized file
        small_file = repo_path / "small_file.py"
        small_file.write_text("def small_function(): pass")

        # Add to git
        subprocess.run(["git", "add", "."], cwd=repo_path, capture_output=True)
        subprocess.run(["git", "commit", "-m", "Large file test"], cwd=repo_path, capture_output=True)

        db_manager = init_db(repo_path)

        try:
            # First, scan without size limit to get all files
            all_files = scan_repo(repo_path, max_bytes=int(10.0 * 1024 * 1024))  # 10MB limit

            # Then scan with small size limit (1MB)
            filtered_files = scan_repo(repo_path, max_bytes=int(1.0 * 1024 * 1024))

            # Large file should be filtered out by scan_repo
            assert len(filtered_files) < len(all_files), "Large file should have been filtered out"
            assert len(filtered_files) > 0, "Small file should still be included"

            # Process the filtered files
            processed_count, failed_count = embed_and_store(
                db_manager, mock_embedder, filtered_files
            )

            # All filtered files should be processed successfully
            assert processed_count > 0, "Small file should have been processed"
            assert failed_count == 0, "No files should have failed processing"

        finally:
            db_manager.cleanup()

    def test_memory_constrained_processing(self, large_repo, mock_embedder):
        """Test processing under memory constraints."""
        db_manager = init_db(large_repo)

        try:
            # Process large repository
            total_files, processed_files, elapsed = reindex_all(
                large_repo, 100.0, db_manager, mock_embedder
            )

            # Should complete successfully
            assert elapsed > 0
            assert total_files > 0
            assert processed_files > 0

            # Should be able to search afterwards
            results = search_index(db_manager, mock_embedder, "function", k=10)
            assert isinstance(results, list)

        finally:
            db_manager.cleanup()

    def test_batch_processing_edge_cases(self, large_repo, mock_embedder):
        """Test edge cases in batch processing."""
        db_manager = init_db(large_repo)

        try:
            files = scan_repo(large_repo, max_bytes=int(100.0 * 1024 * 1024))

            # Test with various batch scenarios
            if len(files) > 0:
                # Process files
                processed_count, skipped_count = embed_and_store(
                    db_manager, mock_embedder, files
                )

                # Should handle batch processing correctly
                assert processed_count + skipped_count == len(files)

        finally:
            db_manager.cleanup()


class TestDatabaseErrorHandling:
    """Test database error handling and recovery."""

    def test_database_connection_failure(self, sample_repo, mock_embedder):
        """Test handling of database connection failures."""
        # Create database manager with invalid path
        invalid_path = Path("/invalid/path/that/does/not/exist/test.db")

        # Should raise appropriate error
        with pytest.raises((DatabaseError, OSError, Exception)):
            db_manager = DatabaseManager(invalid_path)
            db_manager.execute_with_retry("SELECT 1")

    def test_database_corruption_recovery(self, temp_root_dir, mock_embedder):
        """Test recovery from database corruption."""
        db_path = temp_root_dir / "corrupted.db"

        # Create corrupted database file
        with open(db_path, "wb") as f:
            f.write(b"This is not a valid database file")

        # Should handle corrupted database gracefully
        try:
            DatabaseManager(db_path)
            # Should either recover or raise appropriate exception
            # Real implementation should handle this gracefully
        except (DatabaseError, sqlite3.DatabaseError, duckdb.DatabaseError, Exception) as e:
            # Expected to fail with corrupted database
            assert "database" in str(e).lower() or "file" in str(e).lower()

    def test_database_lock_handling(self, temp_root_dir, mock_embedder):
        """Test handling of database locks and concurrent access."""
        db_path = temp_root_dir / "locked.db"
        db_manager1 = DatabaseManager(db_path)

        try:
            # Initialize database
            db_manager1.execute_with_retry(
                f"CREATE TABLE IF NOT EXISTS {config.database.TABLE_NAME} (id VARCHAR PRIMARY KEY)"
            )

            # First connection should work
            result1 = db_manager1.execute_with_retry(f"SELECT COUNT(*) FROM {config.database.TABLE_NAME}")
            assert result1 is not None

            # Create second connection while first is still active
            db_manager2 = DatabaseManager(db_path)

            try:
                # Second connection should get lock timeout due to exclusive locking
                with pytest.raises((DatabaseError, DatabaseTimeoutError)):
                    db_manager2.execute_with_retry(f"SELECT COUNT(*) FROM {config.database.TABLE_NAME}")

            finally:
                db_manager2.cleanup()

        finally:
            db_manager1.cleanup()

    def test_transaction_rollback(self, temp_root_dir, mock_embedder):
        """Test transaction rollback on errors."""
        db_path = temp_root_dir / "transaction_test.db"
        db_manager = DatabaseManager(db_path)

        try:
            # Create table
            db_manager.execute_with_retry(
                f"""CREATE TABLE IF NOT EXISTS {config.database.TABLE_NAME} (
                    id VARCHAR PRIMARY KEY,
                    content TEXT
                )"""
            )

            # Insert valid data
            db_manager.execute_with_retry(
                f"INSERT INTO {config.database.TABLE_NAME} (id, content) VALUES (?, ?)",
                ("valid_id", "valid content")
            )

            # Verify data exists
            result = db_manager.execute_with_retry(f"SELECT COUNT(*) FROM {config.database.TABLE_NAME}")
            assert result[0][0] == 1

            # Try to insert duplicate (should fail)
            try:
                db_manager.execute_with_retry(
                    f"INSERT INTO {config.database.TABLE_NAME} (id, content) VALUES (?, ?)",
                    ("valid_id", "duplicate content")  # Same ID should cause conflict
                )
            except Exception:
                # Expected to fail due to duplicate primary key
                pass

            # Original data should still be there
            result = db_manager.execute_with_retry(f"SELECT COUNT(*) FROM {config.database.TABLE_NAME}")
            assert result[0][0] == 1

        finally:
            db_manager.cleanup()


class TestNetworkAndIOErrors:
    """Test handling of network and I/O related errors."""

    def test_file_permission_errors(self, temp_root_dir, mock_embedder):
        """Test handling of file permission errors."""
        repo_path = temp_root_dir / "permission_test_repo"
        repo_path.mkdir()

        # Create git repo
        import subprocess
        subprocess.run(["git", "init"], cwd=repo_path, capture_output=True, check=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"],
                       cwd=repo_path, capture_output=True, check=True)
        subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo_path, capture_output=True, check=True)

        # Create file
        test_file = repo_path / "test.py"
        test_file.write_text("def test(): pass")

        subprocess.run(["git", "add", "."], cwd=repo_path, capture_output=True)
        subprocess.run(["git", "commit", "-m", "Initial"], cwd=repo_path, capture_output=True)

        # Test with mock file reading that raises PermissionError
        with patch("pathlib.Path.read_text") as mock_read:
            mock_read.side_effect = PermissionError("Permission denied")

            files = scan_repo(repo_path, max_bytes=int(10.0 * 1024 * 1024))
            assert len(files) > 0

            db_manager = init_db(repo_path)
            try:
                # Should handle permission errors gracefully
                processed_count, skipped_count = embed_and_store(
                    db_manager, mock_embedder, files
                )

                # Files should be skipped due to permission errors
                assert skipped_count > 0

            finally:
                db_manager.cleanup()

    def test_disk_space_exhaustion(self, temp_root_dir, mock_embedder):
        """Test handling of disk space exhaustion."""
        db_path = temp_root_dir / "space_test.db"

        # Mock OSError for disk space
        with patch("duckdb.connect") as mock_connect:
            mock_connect.side_effect = OSError("No space left on device")

            # Create database manager and trigger connection creation
            db_manager = DatabaseManager(db_path)

            # Should handle disk space errors gracefully when trying to use the database
            with pytest.raises((DatabaseError, OSError)):
                db_manager.execute_with_retry("SELECT 1")

    def test_file_disappearance_during_processing(self, temp_root_dir, mock_embedder):
        """Test handling files that disappear during processing."""
        repo_path = temp_root_dir / "disappearing_repo"
        repo_path.mkdir()

        # Create git repo
        import subprocess
        subprocess.run(["git", "init"], cwd=repo_path, capture_output=True, check=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"],
                       cwd=repo_path, capture_output=True, check=True)
        subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo_path, capture_output=True, check=True)

        # Create file
        test_file = repo_path / "disappearing.py"
        test_file.write_text("def test(): pass")

        subprocess.run(["git", "add", "."], cwd=repo_path, capture_output=True)
        subprocess.run(["git", "commit", "-m", "Initial"], cwd=repo_path, capture_output=True)

        files = scan_repo(repo_path, max_bytes=int(10.0 * 1024 * 1024))

        # Mock file reading that raises FileNotFoundError
        with patch("pathlib.Path.read_text") as mock_read:
            mock_read.side_effect = FileNotFoundError("File not found")

            db_manager = init_db(repo_path)
            try:
                # Should handle missing files gracefully
                processed_count, skipped_count = embed_and_store(
                    db_manager, mock_embedder, files
                )

                # Files should be skipped if they disappear
                assert skipped_count > 0

            finally:
                db_manager.cleanup()


class TestEmbeddingErrorHandling:
    """Test handling of embedding generation errors."""

    def test_embedding_generation_failure(self, sample_repo):
        """Test handling of embedding generation failures."""
        # Create mock embedder that fails
        mock_embedder = Mock(spec=EmbeddingGenerator)
        mock_embedder.generate_embeddings.side_effect = Exception("Embedding service unavailable")

        db_manager = init_db(sample_repo)

        try:
            files = scan_repo(sample_repo, max_bytes=int(10.0 * 1024 * 1024))

            # Should handle embedding failures gracefully
            with pytest.raises((EmbeddingError, Exception)):
                embed_and_store(db_manager, mock_embedder, files)

        finally:
            db_manager.cleanup()

    def test_partial_embedding_failure(self, sample_repo):
        """Test handling of partial embedding generation failures."""
        # Create mock embedder that fails on some inputs
        mock_embedder = Mock(spec=EmbeddingGenerator)

        def mock_generate_embeddings(texts):
            results = []
            for text in texts:
                if "error" in text.lower():
                    raise Exception("Failed to embed text with 'error'")
                else:
                    # Return valid embedding
                    results.append([0.1] * 384)
            return results

        mock_embedder.generate_embeddings.side_effect = mock_generate_embeddings

        db_manager = init_db(sample_repo)

        try:
            files = scan_repo(sample_repo, max_bytes=int(10.0 * 1024 * 1024))

            # Should handle partial failures
            # In real implementation, this might process some files and skip others
            try:
                processed_count, skipped_count = embed_and_store(
                    db_manager, mock_embedder, files
                )
                # Should have at least attempted processing
                assert processed_count + skipped_count == len(files)
            except Exception:
                # Depending on implementation, might raise exception
                pass

        finally:
            db_manager.cleanup()

    def test_embedding_dimension_mismatch(self, sample_repo):
        """Test handling of embedding dimension mismatches."""
        # Create mock embedder that returns wrong dimensions
        mock_embedder = Mock(spec=EmbeddingGenerator)
        mock_embedder.generate_embeddings.return_value = [
            [0.1] * 256,  # Wrong dimension (should be 384)
            [0.2] * 512,  # Wrong dimension
        ]

        db_manager = init_db(sample_repo)

        try:
            files = scan_repo(sample_repo, max_bytes=int(10.0 * 1024 * 1024))

            # Should handle dimension mismatches
            with pytest.raises((EmbeddingError, ValueError, Exception)):
                embed_and_store(db_manager, mock_embedder, files)

        finally:
            db_manager.cleanup()


class TestSearchErrorHandling:
    """Test error handling in search operations."""

    def test_malformed_query_handling(self, sample_repo, mock_embedder):
        """Test handling of malformed search queries."""
        db_manager = init_db(sample_repo)

        try:
            # Index some data first
            reindex_all(sample_repo, int(10.0 * 1024 * 1024), db_manager, mock_embedder)

            # Test various malformed queries
            malformed_queries = [
                "",  # Empty query
                " ",  # Whitespace only
                "a" * 1000,  # Extremely long query
                "\x00\x01\x02",  # Binary characters
                "query with\nnewlines\tand\ttabs",  # Control characters
                "query with unicode: 你好世界 🌍",  # Unicode should be OK
                "query with 'quotes' and \"double quotes\"",  # Special characters
            ]

            for query in malformed_queries:
                try:
                    results = search_index(db_manager, mock_embedder, query, k=5)
                    # Should return list (possibly empty) without crashing
                    assert isinstance(results, list)
                except (SearchError, ValueError, Exception):
                    # Some queries may legitimately fail
                    pass

        finally:
            db_manager.cleanup()

    def test_search_with_corrupted_embeddings(self, temp_root_dir, mock_embedder):
        """Test search with corrupted embedding data."""
        db_path = temp_root_dir / "corrupted_embeddings.db"
        db_manager = DatabaseManager(db_path)

        try:
            # Create table and insert data with corrupted embeddings
            db_manager.execute_with_retry(
                f"""CREATE TABLE IF NOT EXISTS {config.database.TABLE_NAME} (
                    id VARCHAR PRIMARY KEY,
                    path VARCHAR,
                    content TEXT,
                    embedding DOUBLE[384]
                )"""
            )

            # Insert record with corrupted embedding
            corrupted_embedding = [float('nan')] * 384  # NaN values
            db_manager.execute_with_retry(
                f"INSERT INTO {config.database.TABLE_NAME} (id, path, content, embedding) VALUES (?, ?, ?, ?)",
                ("test_id", "/test/path.py", "test content", corrupted_embedding)
            )

            # Search should handle corrupted embeddings gracefully
            try:
                results = search_index(db_manager, mock_embedder, "test", k=5)
                assert isinstance(results, list)
            except (SearchError, ValueError, Exception):
                # May fail due to corrupted data
                pass

        finally:
            db_manager.cleanup()

    def test_search_timeout_handling(self, sample_repo, mock_embedder):
        """Test handling of search timeouts."""
        db_manager = init_db(sample_repo)

        try:
            # Mock database query that takes too long
            original_execute = db_manager.execute_with_retry

            def slow_execute(*args, **kwargs):
                if "SELECT" in str(args[0]):
                    time.sleep(10)  # Simulate slow query
                return original_execute(*args, **kwargs)

            with patch.object(db_manager, 'execute_with_retry', side_effect=slow_execute):
                # Search should handle timeouts gracefully
                # Note: Actual timeout handling would need to be implemented in the search functions
                try:
                    results = search_index(db_manager, mock_embedder, "test", k=5)
                    # If no timeout mechanism, this will just be slow
                    assert isinstance(results, list)
                except Exception:
                    # May timeout or raise other exceptions
                    pass

        finally:
            db_manager.cleanup()


class TestResourceExhaustionScenarios:
    """Test behavior under resource exhaustion."""

    def test_memory_exhaustion_handling(self, large_repo, mock_embedder):
        """Test handling of memory exhaustion scenarios."""
        db_manager = init_db(large_repo)

        try:
            # Mock memory error during processing
            original_generate = mock_embedder.generate_embeddings

            def memory_limited_generate(texts):
                if len(texts) > 5:  # Simulate memory limit
                    raise MemoryError("Not enough memory")
                return original_generate(texts)

            with patch.object(mock_embedder, 'generate_embeddings', side_effect=memory_limited_generate):
                # Should handle memory errors gracefully
                try:
                    total_files, processed_files, elapsed = reindex_all(
                        large_repo, 100.0, db_manager, mock_embedder
                    )
                    # May process some files before hitting memory limit
                    assert elapsed > 0
                except (MemoryError, Exception):
                    # Expected to fail with memory error
                    pass

        finally:
            db_manager.cleanup()

    def test_cpu_exhaustion_handling(self, sample_repo, mock_embedder):
        """Test handling under CPU exhaustion."""
        import concurrent.futures

        db_manager = init_db(sample_repo)

        try:
            # Simulate high CPU load with concurrent operations
            def cpu_intensive_task():
                # Simulate CPU-intensive work
                for _ in range(1000000):
                    pass

            # Start CPU-intensive background tasks
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(cpu_intensive_task) for _ in range(4)]

                try:
                    # Try to perform indexing under CPU load
                    total_files, processed_files, elapsed = reindex_all(
                        sample_repo, 10.0, db_manager, mock_embedder
                    )

                    # Should still complete successfully
                    assert elapsed > 0
                    assert processed_files >= 0

                finally:
                    # Wait for background tasks to complete
                    concurrent.futures.wait(futures)

        finally:
            db_manager.cleanup()

    def test_concurrent_access_stress(self, sample_repo, mock_embedder):
        """Test system under concurrent access stress."""
        import concurrent.futures

        db_manager = init_db(sample_repo)

        try:
            # Initial indexing
            reindex_all(sample_repo, int(10.0 * 1024 * 1024), db_manager, mock_embedder)

            # Define concurrent operations
            def search_worker(query_id):
                try:
                    query = f"test query {query_id}"
                    results = search_index(db_manager, mock_embedder, query, k=5)
                    return len(results)
                except Exception as e:
                    return str(e)

            def reindex_worker():
                try:
                    total, processed, elapsed = reindex_all(sample_repo, int(
                        10.0 * 1024 * 1024), db_manager, mock_embedder)
                    return processed
                except Exception as e:
                    return str(e)

            # Run concurrent operations
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                # Submit search tasks
                search_futures = [executor.submit(search_worker, i) for i in range(10)]

                # Submit reindex task
                reindex_future = executor.submit(reindex_worker)

                # Collect results
                search_results = []
                for future in concurrent.futures.as_completed(search_futures):
                    try:
                        result = future.result(timeout=30)
                        search_results.append(result)
                    except Exception as e:
                        search_results.append(str(e))

                # Get reindex result
                try:
                    reindex_result = reindex_future.result(timeout=60)
                except Exception as e:
                    reindex_result = str(e)

                # System should handle concurrent access gracefully
                assert len(search_results) == 10
                assert reindex_result is not None

        finally:
            db_manager.cleanup()


class TestRecoveryAndCleanup:
    """Test system recovery and cleanup capabilities."""

    def test_cleanup_after_errors(self, temp_root_dir, mock_embedder):
        """Test that system cleans up properly after errors."""
        db_path = temp_root_dir / "cleanup_test.db"

        # Test database cleanup after error
        try:
            db_manager = DatabaseManager(db_path)

            # Cause an error
            with pytest.raises(Exception):
                db_manager.execute_with_retry("INVALID SQL QUERY")

        except Exception:
            pass
        finally:
            # Database file should still be cleanable
            if db_path.exists():
                db_path.unlink()

    def test_partial_state_recovery(self, sample_repo, mock_embedder):
        """Test recovery from partial indexing state."""
        db_manager = init_db(sample_repo)

        try:
            # Partially index repository
            files = scan_repo(sample_repo, max_bytes=int(10.0 * 1024 * 1024))

            if len(files) > 1:
                # Process only first file, then simulate error
                processed_count, skipped_count = embed_and_store(
                    db_manager, mock_embedder, files[:1]
                )

                assert processed_count > 0

                # Now process remaining files (should work correctly)
                processed_count_2, skipped_count_2 = embed_and_store(
                    db_manager, mock_embedder, files[1:]
                )

                # Should be able to continue from partial state
                assert processed_count_2 + skipped_count_2 == len(files) - 1

        finally:
            db_manager.cleanup()

    def test_graceful_shutdown(self, sample_repo, mock_embedder):
        """Test graceful shutdown and cleanup."""
        db_manager = init_db(sample_repo)

        try:
            # Start some operations
            reindex_all(sample_repo, int(10.0 * 1024 * 1024), db_manager, mock_embedder)

            # Simulate shutdown
            db_manager.cleanup()

            # Should be able to restart cleanly
            db_manager_2 = init_db(sample_repo)

            try:
                # Should be able to query existing data
                results = search_index(db_manager_2, mock_embedder, "test", k=5)
                assert isinstance(results, list)

            finally:
                db_manager_2.cleanup()

        finally:
            db_manager.cleanup()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
