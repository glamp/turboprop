#!/usr/bin/env python3
"""
Unit tests for the code_index module.

This test suite covers the core functionality of the turboprop code indexing
and search system, including:
- Database initialization and schema
- File scanning and filtering
- Embedding generation and storage
- Semantic search operations
- Watch mode file handling

The tests use a temporary database and mock files to ensure
isolation and reproducible results.
"""

import os
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pytest
import numpy as np

# Import the modules to test
from code_index import (
    compute_id, init_db, scan_repo, embed_and_store, build_full_index,
    search_index, embed_and_store_single, TABLE_NAME, CODE_EXTENSIONS
)


class TestComputeId:
    """Test the compute_id function for generating unique file identifiers."""
    
    def test_compute_id_deterministic(self):
        """Test that compute_id produces the same hash for identical input."""
        text = "def hello():\n    return 'world'"
        id1 = compute_id(text)
        id2 = compute_id(text)
        assert id1 == id2
        
    def test_compute_id_different_inputs(self):
        """Test that different inputs produce different hashes."""
        text1 = "def hello():\n    return 'world'"
        text2 = "def goodbye():\n    return 'cruel world'"
        id1 = compute_id(text1)
        id2 = compute_id(text2)
        assert id1 != id2
        
    def test_compute_id_unicode_handling(self):
        """Test that unicode characters are handled properly."""
        text = "def func():\n    return 'ñoño 🚀'"
        id = compute_id(text)
        assert isinstance(id, str)
        assert len(id) == 64  # SHA-256 produces 64 character hex string


class TestInitDb:
    """Test database initialization and schema creation."""
    
    def test_init_db_creates_table(self):
        """Test that init_db creates the code_files table with correct schema."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir) / "test.duckdb"
            
            # Mock the DB_PATH to use our temporary file
            with patch('code_index.DB_PATH', str(tmp_path)):
                con = init_db()
                
                # Check that table exists with correct schema (DuckDB syntax)
                result = con.execute("SELECT table_name FROM information_schema.tables WHERE table_name = ?", 
                                   (TABLE_NAME,)).fetchone()
                assert result is not None
                
                # Check column structure (DuckDB syntax)
                columns = con.execute(f"SELECT column_name FROM information_schema.columns WHERE table_name = '{TABLE_NAME}'").fetchall()
                column_names = [col[0] for col in columns]
                expected_columns = ['id', 'path', 'content', 'embedding']
                assert all(col in column_names for col in expected_columns)
                
                con.close()


class TestScanRepo:
    """Test repository scanning and file filtering."""
    
    def setup_method(self):
        """Set up a temporary repository for testing."""
        self.temp_dir = tempfile.mkdtemp()
        self.repo_path = Path(self.temp_dir)
        
        # Create a mock git repository structure
        (self.repo_path / ".git").mkdir()
        
        # Create some test files
        (self.repo_path / "main.py").write_text("print('hello')")
        (self.repo_path / "config.json").write_text('{"key": "value"}')
        (self.repo_path / "ignored.py").write_text("# This should be ignored")
        (self.repo_path / "README.md").write_text("# Test Project")
        (self.repo_path / "binary.exe").write_bytes(b"\\x00\\x01\\x02")
        
        # Create a subdirectory with files
        subdir = self.repo_path / "src"
        subdir.mkdir()
        (subdir / "utils.py").write_text("def util_func(): pass")
        (subdir / "large_file.py").write_text("x = '" + "a" * 1000000 + "'")  # Large file (>1MB)
        
    def teardown_method(self):
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir)
    
    def test_scan_repo_finds_code_files(self):
        """Test that scan_repo finds files with code extensions."""
        with patch('subprocess.run') as mock_run:
            # Mock git ls-files to return the files we created
            expected_files = [
                "main.py",
                "config.json", 
                "ignored.py",
                "src/utils.py",
                "src/large_file.py"
            ]
            mock_run.return_value = Mock(stdout="\n".join(expected_files), returncode=0)
            
            files = scan_repo(self.repo_path, max_bytes=1024*1024)
            
            # Should find .py and .json files but not .exe
            file_names = [f.name for f in files]
            assert "main.py" in file_names
            assert "config.json" in file_names
            assert "utils.py" in file_names
            assert "binary.exe" not in file_names
            
    def test_scan_repo_respects_size_limit(self):
        """Test that scan_repo filters out files exceeding size limit."""
        with patch('subprocess.run') as mock_run:
            expected_files = [
                "main.py",
                "config.json", 
                "ignored.py",
                "src/utils.py",
                "src/large_file.py"
            ]
            mock_run.return_value = Mock(stdout="\n".join(expected_files), returncode=0)
            
            files = scan_repo(self.repo_path, max_bytes=1000)  # Small size limit
            
            # Should not include the large file
            file_names = [f.name for f in files]
            assert "main.py" in file_names
            assert "large_file.py" not in file_names
            
    def test_scan_repo_handles_git_ignore(self):
        """Test that scan_repo respects .gitignore files."""
        with patch('subprocess.run') as mock_run:
            # Mock git ls-files to return only non-ignored files
            # First call is for tracked files, second is for untracked
            mock_run.side_effect = [
                Mock(stdout="main.py\nconfig.json\nsrc/utils.py\n", returncode=0),  # tracked files
                Mock(stdout="", returncode=0)  # untracked files (ignored.py is ignored)
            ]
            
            files = scan_repo(self.repo_path, max_bytes=1024*1024)
            
            # Should not include ignored files
            file_names = [f.name for f in files]
            assert "main.py" in file_names
            assert "ignored.py" not in file_names


class TestEmbedAndStore:
    """Test embedding generation and database storage."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test.duckdb"
        
        # Create mock embedder
        self.mock_embedder = Mock()
        self.mock_embedder.encode.return_value = np.random.rand(384).astype(np.float32)
        
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
        
    def test_embed_and_store_processes_files(self):
        """Test that embed_and_store processes files and stores embeddings."""
        with patch('code_index.DB_PATH', str(self.db_path)):
            con = init_db()
            
            # Create test files
            test_files = []
            for i in range(3):
                file_path = Path(self.temp_dir) / f"test_{i}.py"
                file_path.write_text(f"def func_{i}(): pass")
                test_files.append(file_path)
            
            # Mock SentenceTransformer creation in process_single_file
            with patch('code_index.SentenceTransformer') as mock_st_class:
                mock_st_instance = Mock()
                mock_st_instance.encode.return_value = np.random.rand(384).astype(np.float32)
                mock_st_class.return_value = mock_st_instance
                
                # Set up mock embedder with model name
                self.mock_embedder.model_name = "test-model"
                
                embed_and_store(con, self.mock_embedder, test_files)
                
                # Check that files were stored in database
                count = con.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}").fetchone()[0]
                assert count == 3
                
                # Check that embeddings were generated
                assert mock_st_instance.encode.call_count == 3
            
            con.close()
            
    def test_embed_and_store_handles_unreadable_files(self):
        """Test that embed_and_store skips files that can't be read."""
        with patch('code_index.DB_PATH', str(self.db_path)):
            con = init_db()
            
            # Create a file that will cause a read error
            test_file = Path(self.temp_dir) / "test.py"
            test_file.write_text("def func(): pass")
            
            # Mock Path.read_text to raise an exception
            with patch.object(Path, 'read_text', side_effect=UnicodeDecodeError("utf-8", b"", 0, 1, "invalid")):
                embed_and_store(con, self.mock_embedder, [test_file])
            
            # Should not have stored anything
            count = con.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}").fetchone()[0]
            assert count == 0
            
            con.close()
            
    def test_embed_and_store_file_count_validation(self):
        """Test that all readable files get indexed - regression test for silent failures."""
        with patch('code_index.DB_PATH', str(self.db_path)):
            con = init_db()
            
            # Create test files with various content types
            test_files = []
            
            # Valid Python files
            for i in range(5):
                file_path = Path(self.temp_dir) / f"valid_{i}.py"
                file_path.write_text(f"def func_{i}():\n    return {i}")
                test_files.append(file_path)
            
            # Valid JSON file
            json_file = Path(self.temp_dir) / "config.json"
            json_file.write_text('{"key": "value", "number": 42}')
            test_files.append(json_file)
            
            # Valid JavaScript file
            js_file = Path(self.temp_dir) / "script.js"
            js_file.write_text("function hello() { return 'world'; }")
            test_files.append(js_file)
            
            # Count input files
            input_file_count = len(test_files)
            
            # Mock SentenceTransformer creation in process_single_file
            with patch('code_index.SentenceTransformer') as mock_st_class:
                mock_st_instance = Mock()
                mock_st_instance.encode.return_value = np.random.rand(384).astype(np.float32)
                mock_st_class.return_value = mock_st_instance
                
                # Set up mock embedder with model name
                self.mock_embedder.model_name = "test-model"
                
                # Process files
                embed_and_store(con, self.mock_embedder, test_files)
                
                # Verify that ALL files were processed and stored
                stored_count = con.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}").fetchone()[0]
                assert stored_count == input_file_count, (
                    f"Expected {input_file_count} files to be indexed, but only {stored_count} were stored. "
                    f"This suggests files are being silently skipped during processing."
                )
                
                # Verify embeddings were generated for all files
                assert mock_st_instance.encode.call_count == input_file_count
                
                # Verify all files have non-null embeddings
                null_embedding_count = con.execute(
                    f"SELECT COUNT(*) FROM {TABLE_NAME} WHERE embedding IS NULL"
                ).fetchone()[0]
                assert null_embedding_count == 0, "Some files were stored without embeddings"
            
            con.close()
            
    def test_embed_and_store_mixed_success_and_failure(self):
        """Test that successful files are indexed even when some files fail."""
        with patch('code_index.DB_PATH', str(self.db_path)):
            con = init_db()
            
            # Create a mix of valid and invalid files
            test_files = []
            
            # Valid files
            for i in range(3):
                file_path = Path(self.temp_dir) / f"valid_{i}.py"
                file_path.write_text(f"def func_{i}(): pass")
                test_files.append(file_path)
            
            # Create a file with invalid UTF-8 encoding
            invalid_file = Path(self.temp_dir) / "invalid.py"
            with open(invalid_file, 'wb') as f:
                f.write(b'print("hello")\n')
                f.write(b'\xff\xfe\xfd')  # Invalid UTF-8 sequence
                f.write(b'\nprint("world")')
            test_files.append(invalid_file)
            
            # Mock SentenceTransformer creation in process_single_file
            with patch('code_index.SentenceTransformer') as mock_st_class:
                mock_st_instance = Mock()
                mock_st_instance.encode.return_value = np.random.rand(384).astype(np.float32)
                mock_st_class.return_value = mock_st_instance
                
                # Set up mock embedder with model name
                self.mock_embedder.model_name = "test-model"
                
                # Capture stderr to check error reporting
                import io
                import contextlib
                
                stderr_capture = io.StringIO()
                with contextlib.redirect_stderr(stderr_capture):
                    embed_and_store(con, self.mock_embedder, test_files)
                
                # Check that valid files were stored
                stored_count = con.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}").fetchone()[0]
                assert stored_count == 3, f"Expected 3 valid files to be stored, got {stored_count}"
                
                # Check that error was reported
                stderr_output = stderr_capture.getvalue()
                assert "Failed to process" in stderr_output
                assert "invalid.py" in stderr_output
                assert "Failed to process 1 files out of 4 total" in stderr_output
                assert "Successfully processed 3 files" in stderr_output
            
            con.close()


class TestSearchIndex:
    """Test semantic search functionality."""
    
    def setup_method(self):
        """Set up test environment with sample data."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test.duckdb"
        
        # Create mock embedder
        self.mock_embedder = Mock()
        
        # Set up database with sample data
        with patch('code_index.DB_PATH', str(self.db_path)):
            self.con = init_db()
            
            # Insert sample data with mock embeddings
            sample_data = [
                ("id1", "/path/to/file1.py", "def hello(): return 'world'", 
                 np.random.rand(384).astype(np.float32).tolist()),
                ("id2", "/path/to/file2.py", "def goodbye(): return 'farewell'", 
                 np.random.rand(384).astype(np.float32).tolist()),
                ("id3", "/path/to/file3.py", "class Parser: pass", 
                 np.random.rand(384).astype(np.float32).tolist()),
            ]
            
            self.con.executemany(
                f"INSERT INTO {TABLE_NAME} VALUES (?, ?, ?, ?)",
                sample_data
            )
    
    def teardown_method(self):
        """Clean up test environment."""
        if hasattr(self, 'con'):
            self.con.close()
        shutil.rmtree(self.temp_dir)
    
    def test_search_index_returns_results(self):
        """Test that search_index returns similarity-ranked results."""
        # Mock the embedder to return a specific query embedding
        query_embedding = np.random.rand(384).astype(np.float32)
        self.mock_embedder.encode.return_value = query_embedding
        
        results = search_index(self.con, self.mock_embedder, "hello function", k=2)
        
        # Should return up to k results
        assert len(results) <= 2
        
        # Each result should be a tuple of (path, snippet, distance)
        for path, snippet, distance in results:
            assert isinstance(path, str)
            assert isinstance(snippet, str)
            assert isinstance(distance, float)
            assert distance >= 0.0  # Distance should be non-negative
            
    def test_search_index_handles_no_results(self):
        """Test search behavior when no embeddings exist."""
        # Clear the database
        self.con.execute(f"DELETE FROM {TABLE_NAME}")
        
        query_embedding = np.random.rand(384).astype(np.float32)
        self.mock_embedder.encode.return_value = query_embedding
        
        results = search_index(self.con, self.mock_embedder, "test query", k=5)
        
        # Should return empty results
        assert len(results) == 0


class TestEmbedAndStoreSingle:
    """Test single file embedding and storage."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test.duckdb"
        
        # Create mock embedder
        self.mock_embedder = Mock()
        self.mock_embedder.encode.return_value = np.random.rand(384).astype(np.float32)
        
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
        
    def test_embed_and_store_single_success(self):
        """Test successful single file processing."""
        with patch('code_index.DB_PATH', str(self.db_path)):
            con = init_db()
            
            # Create a test file
            test_file = Path(self.temp_dir) / "test.py"
            test_file.write_text("def test_func(): pass")
            
            result = embed_and_store_single(con, self.mock_embedder, test_file)
            
            # Should return (uid, embedding) tuple
            assert result is not None
            uid, embedding = result
            assert isinstance(uid, str)
            assert isinstance(embedding, np.ndarray)
            
            # Check that file was stored in database
            count = con.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}").fetchone()[0]
            assert count == 1
            
            con.close()
            
    def test_embed_and_store_single_handles_read_error(self):
        """Test handling of file read errors."""
        with patch('code_index.DB_PATH', str(self.db_path)):
            con = init_db()
            
            # Create a file that will cause a read error
            test_file = Path(self.temp_dir) / "test.py"
            test_file.write_text("def test_func(): pass")
            
            # Mock Path.read_text to raise an exception
            with patch.object(Path, 'read_text', side_effect=Exception("Read error")):
                result = embed_and_store_single(con, self.mock_embedder, test_file)
            
            # Should return None on error
            assert result is None
            
            con.close()


class TestBuildFullIndex:
    """Test index building functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test.duckdb"
        
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
        
    def test_build_full_index_counts_embeddings(self):
        """Test that build_full_index returns correct embedding count."""
        with patch('code_index.DB_PATH', str(self.db_path)):
            con = init_db()
            
            # Insert sample data
            sample_data = [
                ("id1", "/path/to/file1.py", "content1", 
                 np.random.rand(384).astype(np.float32).tolist()),
                ("id2", "/path/to/file2.py", "content2", 
                 np.random.rand(384).astype(np.float32).tolist()),
            ]
            
            con.executemany(
                f"INSERT INTO {TABLE_NAME} VALUES (?, ?, ?, ?)",
                sample_data
            )
            
            count = build_full_index(con)
            assert count == 2
            
            con.close()
            
    def test_build_full_index_empty_database(self):
        """Test build_full_index with empty database."""
        with patch('code_index.DB_PATH', str(self.db_path)):
            con = init_db()
            
            count = build_full_index(con)
            assert count == 0
            
            con.close()


class TestCodeExtensions:
    """Test code file extension filtering."""
    
    def test_code_extensions_coverage(self):
        """Test that CODE_EXTENSIONS includes common programming languages."""
        expected_extensions = {
            '.py', '.js', '.ts', '.java', '.cpp', '.c', '.h', '.cs',
            '.go', '.rs', '.php', '.rb', '.swift', '.kt', '.html', '.css',
            '.json', '.yaml', '.yml', '.xml'
        }
        
        # Check that all expected extensions are present
        for ext in expected_extensions:
            assert ext in CODE_EXTENSIONS
            
    def test_code_extensions_case_insensitive(self):
        """Test that extension matching works case-insensitively."""
        # This test ensures that our scanning logic handles case properly
        # (the actual case-insensitive logic is in scan_repo)
        test_extensions = ['.PY', '.JS', '.JSON']
        
        for ext in test_extensions:
            assert ext.lower() in CODE_EXTENSIONS


# Integration tests
class TestIntegration:
    """Integration tests for the complete workflow."""
    
    def setup_method(self):
        """Set up integration test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.repo_path = Path(self.temp_dir)
        self.db_path = Path(self.temp_dir) / "test.duckdb"
        
        # Create a mock git repository
        (self.repo_path / ".git").mkdir()
        
        # Create sample code files
        (self.repo_path / "main.py").write_text("""
def main():
    print("Hello, world!")
    return 0
""")
        
        (self.repo_path / "utils.py").write_text("""
def helper_function():
    return "This is a helper"
    
class UtilityClass:
    def method(self):
        pass
""")
        
        (self.repo_path / "config.json").write_text("""
{
    "database": {
        "host": "localhost",
        "port": 5432
    }
}
""")
        
    def teardown_method(self):
        """Clean up integration test environment."""
        shutil.rmtree(self.temp_dir)
        
    def test_complete_indexing_and_search_workflow(self):
        """Test the complete workflow from scanning to searching."""
        with patch('code_index.DB_PATH', str(self.db_path)):
            # Mock git ls-files
            with patch('subprocess.run') as mock_run:
                mock_run.return_value = Mock(stdout="main.py\nutils.py\nconfig.json\n", returncode=0)
                
                # Mock embedder
                mock_embedder = Mock()
                mock_embedder.encode.return_value = np.random.rand(384).astype(np.float32)
                mock_embedder.model_name = "test-model"
                
                # Initialize database
                con = init_db()
                
                # Scan repository
                files = scan_repo(self.repo_path, max_bytes=1024*1024)
                assert len(files) == 3  # Should find 3 files
                
                # Mock SentenceTransformer creation in process_single_file
                with patch('code_index.SentenceTransformer') as mock_st_class:
                    mock_st_instance = Mock()
                    mock_st_instance.encode.return_value = np.random.rand(384).astype(np.float32)
                    mock_st_class.return_value = mock_st_instance
                    
                    # Embed and store files
                    embed_and_store(con, mock_embedder, files)
                    
                    # Verify embeddings were created
                    embedding_count = build_full_index(con)
                    assert embedding_count == 3
                    
                    # Test search
                    results = search_index(con, mock_embedder, "hello world", k=2)
                    assert len(results) <= 2
                    
                    # Verify result structure
                    for path, snippet, distance in results:
                        assert isinstance(path, str)
                        assert isinstance(snippet, str)
                        assert isinstance(distance, float)
                
                con.close()
                
    def test_incremental_update_workflow(self):
        """Test incremental updates with single file processing."""
        with patch('code_index.DB_PATH', str(self.db_path)):
            # Mock embedder
            mock_embedder = Mock()
            mock_embedder.encode.return_value = np.random.rand(384).astype(np.float32)
            
            # Initialize database
            con = init_db()
            
            # Add a single file
            test_file = self.repo_path / "new_file.py"
            test_file.write_text("def new_function(): pass")
            
            result = embed_and_store_single(con, mock_embedder, test_file)
            assert result is not None
            
            # Verify file was stored
            count = con.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}").fetchone()[0]
            assert count == 1
            
            # Update the same file
            test_file.write_text("def updated_function(): return 'updated'")
            result = embed_and_store_single(con, mock_embedder, test_file)
            assert result is not None
            
            # Should still have only one record (INSERT OR REPLACE)
            count = con.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}").fetchone()[0]
            assert count == 1
            
            # But content should be updated
            content = con.execute(f"SELECT content FROM {TABLE_NAME} WHERE path = ?", 
                                (str(test_file),)).fetchone()[0]
            assert "updated_function" in content
            
            con.close()
            
    def test_example_codebase_file_count_validation(self):
        """Test that example codebase indexing processes all discoverable files."""
        # Use the actual example codebase
        example_path = Path(__file__).parent.parent / "example-codebases" / "bashplotlib"
        
        if not example_path.exists():
            pytest.skip("Example codebase not found")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.duckdb"
            
            with patch('code_index.DB_PATH', str(db_path)):
                # Mock embedder
                mock_embedder = Mock()
                mock_embedder.encode.return_value = np.random.rand(384).astype(np.float32)
                mock_embedder.model_name = "test-model"
                
                # Initialize database
                con = init_db()
                
                # Scan repository to get expected file count (without mocking git)
                expected_files = scan_repo(example_path, max_bytes=1024*1024)
                
                expected_count = len(expected_files)
                
                # Skip if no files found
                if expected_count == 0:
                    pytest.skip("No code files found in example codebase")
                
                # Mock SentenceTransformer creation in process_single_file
                with patch('code_index.SentenceTransformer') as mock_st_class:
                    mock_st_instance = Mock()
                    mock_st_instance.encode.return_value = np.random.rand(384).astype(np.float32)
                    mock_st_class.return_value = mock_st_instance
                    
                    # Process all files
                    embed_and_store(con, mock_embedder, expected_files)
                    
                    # Verify all files were indexed
                    actual_count = con.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}").fetchone()[0]
                    
                    assert actual_count == expected_count, (
                        f"Expected {expected_count} files from example codebase to be indexed, "
                        f"but only {actual_count} were stored. Files found: {[f.name for f in expected_files]}"
                    )
                    
                    # Verify all files have embeddings
                    null_embedding_count = con.execute(
                        f"SELECT COUNT(*) FROM {TABLE_NAME} WHERE embedding IS NULL"
                    ).fetchone()[0]
                    assert null_embedding_count == 0, "Some files were stored without embeddings"
                
                con.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])