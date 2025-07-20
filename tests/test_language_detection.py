"""Tests for language detection functionality."""
from language_detection import LanguageDetector


class TestLanguageDetector:
    """Test cases for the LanguageDetector class."""

    def test_detect_language_by_extension_python(self):
        """Test detecting Python files by extension."""
        detector = LanguageDetector()
        result = detector.detect_language("test.py", "print('hello')")
        assert result.language == "Python"
        assert result.file_type == ".py"
        assert result.confidence == 1.0

    def test_detect_language_by_extension_javascript(self):
        """Test detecting JavaScript files by extension."""
        detector = LanguageDetector()
        result = detector.detect_language("script.js", "console.log('hello');")
        assert result.language == "JavaScript"
        assert result.file_type == ".js"
        assert result.confidence == 1.0

    def test_detect_language_by_extension_typescript(self):
        """Test detecting TypeScript files by extension."""
        detector = LanguageDetector()
        result = detector.detect_language("app.tsx", "const x: string = 'hello';")
        assert result.language == "TypeScript"
        assert result.file_type == ".tsx"
        assert result.confidence == 1.0

    def test_detect_language_unknown_extension(self):
        """Test handling unknown file extensions."""
        detector = LanguageDetector()
        result = detector.detect_language("unknown.xyz", "some content")
        assert result.language == "Unknown"
        assert result.file_type == ".xyz"
        assert result.confidence == 0.0

    def test_detect_language_no_extension(self):
        """Test handling files without extensions using content analysis."""
        detector = LanguageDetector()
        # Python-like content without extension
        python_content = "#!/usr/bin/env python3\nimport sys\nprint('hello')"
        result = detector.detect_language("script", python_content)
        # Should fall back to content-based detection
        assert result.file_type == ""
        # Language might be detected from content or be Unknown

    def test_detect_language_configuration_files(self):
        """Test detecting configuration file types."""
        detector = LanguageDetector()

        # JSON
        json_result = detector.detect_language("config.json", '{"key": "value"}')
        assert json_result.language == "JSON"
        assert json_result.file_type == ".json"

        # YAML
        yaml_result = detector.detect_language("config.yaml", "key: value")
        assert yaml_result.language == "YAML"
        assert yaml_result.file_type == ".yaml"

    def test_detect_language_documentation_files(self):
        """Test detecting documentation file types."""
        detector = LanguageDetector()

        # Markdown
        md_result = detector.detect_language("README.md", "# Title\nContent here")
        assert md_result.language == "Markdown"
        assert md_result.file_type == ".md"

    def test_detect_language_build_files(self):
        """Test detecting build and deployment files."""
        detector = LanguageDetector()

        # Dockerfile
        dockerfile_result = detector.detect_language("Dockerfile", "FROM python:3.9")
        assert dockerfile_result.language == "Dockerfile"
        assert dockerfile_result.file_type == ""

    def test_language_categorization(self):
        """Test file categorization into source/config/docs/build."""
        detector = LanguageDetector()

        # Source code
        py_result = detector.detect_language("app.py", "print('hello')")
        assert py_result.category == "source"

        # Configuration
        json_result = detector.detect_language("config.json", '{"key": "value"}')
        assert json_result.category == "configuration"

        # Documentation
        md_result = detector.detect_language("README.md", "# Title")
        assert md_result.category == "documentation"

    def test_empty_content(self):
        """Test handling empty file content."""
        detector = LanguageDetector()
        result = detector.detect_language("empty.py", "")
        assert result.language == "Python"  # Should still detect by extension
        assert result.file_type == ".py"

    def test_binary_file_detection(self):
        """Test handling binary files."""
        detector = LanguageDetector()
        # Simulate binary content
        binary_content = b'\x00\x01\x02\x03\x04\xff'.decode('latin1')
        result = detector.detect_language("image.png", binary_content)
        assert result.language == "Binary"
        assert result.file_type == ".png"
