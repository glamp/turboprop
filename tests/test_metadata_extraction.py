"""Tests for file metadata extraction during indexing."""
import datetime
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from database_manager import DatabaseManager
from embedding_helper import EmbeddingGenerator
from indexing_operations import (
    embed_and_store,
    embed_and_store_single,
    extract_file_metadata,
)
from language_detection import LanguageDetectionResult


class TestMetadataExtraction:
    """Test cases for file metadata extraction functionality."""

    def test_extract_file_metadata_python_file(self):
        """Test extracting metadata from a Python file."""
        content = 'print("hello")\n# comment\n\ndef main():\n    pass\n'
        path = Path("test.py")
        
        metadata = extract_file_metadata(path, content)
        
        assert metadata["file_type"] == ".py"
        assert metadata["language"] == "Python"
        assert metadata["size_bytes"] == len(content)
        assert metadata["line_count"] == 5
        assert metadata["category"] == "source"

    def test_extract_file_metadata_javascript_file(self):
        """Test extracting metadata from a JavaScript file."""
        content = 'console.log("hello");\nfunction test() {\n  return 42;\n}\n'
        path = Path("script.js")
        
        metadata = extract_file_metadata(path, content)
        
        assert metadata["file_type"] == ".js"
        assert metadata["language"] == "JavaScript"
        assert metadata["size_bytes"] == len(content)
        assert metadata["line_count"] == 4
        assert metadata["category"] == "source"

    def test_extract_file_metadata_json_config(self):
        """Test extracting metadata from a JSON configuration file."""
        content = '{\n  "name": "test",\n  "version": "1.0.0"\n}'
        path = Path("package.json")
        
        metadata = extract_file_metadata(path, content)
        
        assert metadata["file_type"] == ".json"
        assert metadata["language"] == "JSON"
        assert metadata["size_bytes"] == len(content)
        assert metadata["line_count"] == 4
        assert metadata["category"] == "configuration"

    def test_extract_file_metadata_markdown_doc(self):
        """Test extracting metadata from a Markdown documentation file."""
        content = "# Title\n\nContent here\nMore content"
        path = Path("README.md")
        
        metadata = extract_file_metadata(path, content)
        
        assert metadata["file_type"] == ".md"
        assert metadata["language"] == "Markdown"
        assert metadata["size_bytes"] == len(content)
        assert metadata["line_count"] == 4
        assert metadata["category"] == "documentation"

    def test_extract_file_metadata_empty_file(self):
        """Test extracting metadata from an empty file."""
        content = ""
        path = Path("empty.py")
        
        metadata = extract_file_metadata(path, content)
        
        assert metadata["file_type"] == ".py"
        assert metadata["language"] == "Python"
        assert metadata["size_bytes"] == 0
        assert metadata["line_count"] == 0
        assert metadata["category"] == "source"

    def test_extract_file_metadata_single_line(self):
        """Test extracting metadata from a single-line file."""
        content = "print('hello')"  # No newline
        path = Path("single.py")
        
        metadata = extract_file_metadata(path, content)
        
        assert metadata["file_type"] == ".py"
        assert metadata["language"] == "Python"
        assert metadata["size_bytes"] == len(content)
        assert metadata["line_count"] == 1
        assert metadata["category"] == "source"

    def test_extract_file_metadata_dockerfile(self):
        """Test extracting metadata from a Dockerfile."""
        content = "FROM python:3.9\nRUN pip install requirements\nCMD ['python', 'app.py']"
        path = Path("Dockerfile")
        
        metadata = extract_file_metadata(path, content)
        
        assert metadata["file_type"] == ""
        assert metadata["language"] == "Dockerfile"
        assert metadata["size_bytes"] == len(content)
        assert metadata["line_count"] == 3
        assert metadata["category"] == "build"

    @patch('indexing_operations.LanguageDetector')
    def test_embed_and_store_with_metadata(self, mock_detector_class):
        """Test that embed_and_store includes metadata in database operations."""
        # Setup mocks
        mock_detector = Mock()
        mock_detector_class.return_value = mock_detector
        mock_detector.detect_language.return_value = LanguageDetectionResult(
            language="Python",
            file_type=".py",
            confidence=1.0,
            category="source"
        )
        
        mock_db_manager = Mock(spec=DatabaseManager)
        mock_embedder = Mock(spec=EmbeddingGenerator)
        mock_embedder.encode.return_value = Mock()
        mock_embedder.encode.return_value.tolist.return_value = [0.1, 0.2, 0.3]
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write('print("hello")\n')
            temp_path = Path(f.name)
        
        try:
            # Call the function
            embed_and_store(mock_db_manager, mock_embedder, [temp_path])
            
            # Verify database transaction was called with metadata
            mock_db_manager.execute_transaction.assert_called_once()
            operations = mock_db_manager.execute_transaction.call_args[0][0]
            
            # Check that the operation includes all metadata columns
            query, params = operations[0]
            assert "file_type" in query
            assert "language" in query
            assert "size_bytes" in query
            assert "line_count" in query
            assert "category" in query
            
            # Check that the parameters include the metadata values
            assert ".py" in params
            assert "Python" in params
            assert "source" in params
            
        finally:
            # Clean up
            temp_path.unlink(missing_ok=True)

    @patch('indexing_operations.LanguageDetector')
    def test_embed_and_store_single_with_metadata(self, mock_detector_class):
        """Test that embed_and_store_single includes metadata."""
        # Setup mocks
        mock_detector = Mock()
        mock_detector_class.return_value = mock_detector
        mock_detector.detect_language.return_value = LanguageDetectionResult(
            language="JavaScript",
            file_type=".js",
            confidence=1.0,
            category="source"
        )
        
        mock_db_manager = Mock(spec=DatabaseManager)
        mock_embedder = Mock(spec=EmbeddingGenerator)
        mock_embedder.encode.return_value = Mock()
        mock_embedder.encode.return_value.tolist.return_value = [0.1, 0.2, 0.3]
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False) as f:
            f.write('console.log("hello");\n')
            temp_path = Path(f.name)
        
        try:
            # Call the function
            result = embed_and_store_single(mock_db_manager, mock_embedder, temp_path)
            
            # Should return True for success
            assert result is True
            
            # Verify database operation was called with metadata
            mock_db_manager.execute_with_retry.assert_called_once()
            args, kwargs = mock_db_manager.execute_with_retry.call_args
            
            # Check that the query includes all metadata columns
            query = args[0]
            params = args[1]
            assert "file_type" in query
            assert "language" in query
            assert "size_bytes" in query
            assert "line_count" in query
            assert "category" in query
            
            # Check that the parameters include the metadata values
            assert ".js" in params
            assert "JavaScript" in params
            assert "source" in params
            
        finally:
            # Clean up
            temp_path.unlink(missing_ok=True)

    def test_extract_file_metadata_line_count_edge_cases(self):
        """Test line counting edge cases."""
        # File ending with newline
        content_with_newline = "line1\nline2\n"
        metadata = extract_file_metadata(Path("test1.txt"), content_with_newline)
        assert metadata["line_count"] == 2
        
        # File not ending with newline
        content_no_newline = "line1\nline2"
        metadata = extract_file_metadata(Path("test2.txt"), content_no_newline)
        assert metadata["line_count"] == 2
        
        # File with only newlines
        content_only_newlines = "\n\n\n"
        metadata = extract_file_metadata(Path("test3.txt"), content_only_newlines)
        assert metadata["line_count"] == 3