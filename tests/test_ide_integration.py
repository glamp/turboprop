#!/usr/bin/env python3
"""
Tests for IDE integration functionality.

Tests URL generation, cross-platform path handling, and syntax highlighting
for various IDEs and development environments.
"""
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock
import subprocess

from ide_integration import (
    IDEIntegration,
    IDEType,
    IDENavigationUrl,
    SyntaxHighlightingHint,
    get_ide_navigation_urls,
    get_mcp_navigation_actions
)


class TestIDEIntegration(unittest.TestCase):
    """Test cases for IDE integration functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.ide_integration = IDEIntegration()
        self.test_file_path = "/home/user/project/test.py"
        self.test_line = 42
        self.test_column = 10

    def test_generate_navigation_urls_basic(self):
        """Test basic navigation URL generation."""
        urls = self.ide_integration.generate_navigation_urls(
            self.test_file_path, self.test_line, self.test_column
        )
        
        # Should always return at least the generic file:// URL
        self.assertGreater(len(urls), 0)
        
        # Check that generic URL is always present
        generic_urls = [url for url in urls if url.ide_type == IDEType.GENERIC]
        self.assertEqual(len(generic_urls), 1)
        self.assertTrue(generic_urls[0].url.startswith("file://"))

    def test_vscode_url_generation(self):
        """Test VS Code URL generation."""
        with patch.object(self.ide_integration, '_is_ide_available', return_value=True):
            urls = self.ide_integration.generate_navigation_urls(
                self.test_file_path, self.test_line, self.test_column
            )
            
            vscode_urls = [url for url in urls if url.ide_type == IDEType.VSCODE]
            self.assertGreater(len(vscode_urls), 0)
            
            vscode_url = vscode_urls[0]
            expected = f"vscode://file/{Path(self.test_file_path).resolve()}:{self.test_line}:{self.test_column}"
            self.assertEqual(vscode_url.url, expected)
            self.assertEqual(vscode_url.display_name, "VS Code")

    def test_jetbrains_url_generation(self):
        """Test JetBrains IDE URL generation."""
        with patch.object(self.ide_integration, '_is_ide_available', return_value=True):
            urls = self.ide_integration.generate_navigation_urls(
                self.test_file_path, self.test_line, self.test_column
            )
            
            jetbrains_urls = [url for url in urls if url.ide_type == IDEType.JETBRAINS]
            if jetbrains_urls:  # Only test if JetBrains URLs are generated
                jetbrains_url = jetbrains_urls[0]
                self.assertIn("://open?file=", jetbrains_url.url)
                self.assertIn(str(Path(self.test_file_path).resolve()), jetbrains_url.url)
                self.assertIn(f"line={self.test_line}", jetbrains_url.url)
                self.assertIn(f"column={self.test_column}", jetbrains_url.url)

    def test_vim_url_generation(self):
        """Test Vim/Neovim URL generation."""
        with patch.object(self.ide_integration, '_is_ide_available', return_value=True):
            urls = self.ide_integration.generate_navigation_urls(
                self.test_file_path, self.test_line
            )
            
            vim_urls = [url for url in urls if url.ide_type in [IDEType.VIM, IDEType.NEOVIM]]
            if vim_urls:  # Only test if Vim URLs are generated
                vim_url = vim_urls[0]
                expected_path = str(Path(self.test_file_path).resolve())
                self.assertIn(expected_path, vim_url.url)
                self.assertIn(str(self.test_line), vim_url.url)

    @patch('subprocess.run')
    def test_ide_availability_detection_success(self, mock_subprocess):
        """Test successful IDE availability detection."""
        # Mock successful command execution
        mock_subprocess.return_value = MagicMock(returncode=0)
        
        result = self.ide_integration._check_command_exists(['code'])
        self.assertTrue(result)
        mock_subprocess.assert_called()

    @patch('subprocess.run')
    def test_ide_availability_detection_failure(self, mock_subprocess):
        """Test failed IDE availability detection."""
        # Mock failed command execution
        mock_subprocess.side_effect = subprocess.CalledProcessError(1, 'which')
        
        result = self.ide_integration._check_command_exists(['nonexistent-editor'])
        self.assertFalse(result)

    def test_normalize_path_basic(self):
        """Test basic path normalization."""
        test_path = "./relative/path/file.py"
        normalized = self.ide_integration.normalize_path(test_path)
        
        # Should return absolute path
        self.assertTrue(Path(normalized).is_absolute())
        self.assertIn("file.py", normalized)

    @patch('platform.system')
    def test_normalize_path_wsl(self, mock_platform):
        """Test WSL path normalization on Windows."""
        mock_platform.return_value = "Windows"
        self.ide_integration.platform = "windows"
        
        wsl_path = "/mnt/c/Users/test/project/file.py"
        normalized = self.ide_integration.normalize_path(wsl_path)
        
        # Should convert WSL path to Windows path
        self.assertTrue(normalized.startswith("C:"))
        self.assertIn("file.py", normalized)

    def test_get_language_from_extension(self):
        """Test language detection from file extensions."""
        test_cases = [
            ("/path/file.py", "python"),
            ("/path/script.js", "javascript"),
            ("/path/component.tsx", "typescriptreact"),
            ("/path/Main.java", "java"),
            ("/path/main.go", "go"),
            ("/path/lib.rs", "rust"),
            ("/path/config.json", "json"),
            ("/path/README.md", "markdown"),
            ("/path/unknown.xyz", "plaintext"),
        ]
        
        for file_path, expected_language in test_cases:
            with self.subTest(file_path=file_path):
                language = self.ide_integration.get_language_from_extension(file_path)
                self.assertEqual(language, expected_language)

    def test_generate_syntax_hints_python(self):
        """Test syntax highlighting hints for Python code."""
        python_code = '''def test_function():
    """A test function."""
    # This is a comment
    return "hello world"

class TestClass:
    pass'''
        
        hints = self.ide_integration.generate_syntax_hints(
            "test.py", python_code, target_line=1
        )
        
        # Should detect at least some keywords
        self.assertGreater(len(hints), 0)
        
        # Check for function keyword detection
        keyword_hints = [h for h in hints if h.token_type == "keyword"]
        self.assertGreater(len(keyword_hints), 0)

    def test_generate_syntax_hints_javascript(self):
        """Test syntax highlighting hints for JavaScript code."""
        js_code = '''function testFunction() {
    // This is a comment
    const message = "hello world";
    return message;
}

class TestClass {
    constructor() {}
}'''
        
        hints = self.ide_integration.generate_syntax_hints(
            "test.js", js_code, target_line=1
        )
        
        # Should detect keywords and comments
        self.assertGreater(len(hints), 0)
        
        # Check for keyword detection
        keyword_hints = [h for h in hints if h.token_type == "keyword"]
        comment_hints = [h for h in hints if h.token_type == "comment"]
        
        # Should have at least some syntax elements
        self.assertGreater(len(keyword_hints) + len(comment_hints), 0)

    def test_create_mcp_navigation_actions(self):
        """Test MCP navigation actions creation."""
        with patch.object(self.ide_integration, 'generate_navigation_urls') as mock_urls:
            # Mock some navigation URLs
            mock_urls.return_value = [
                IDENavigationUrl(
                    ide_type=IDEType.VSCODE,
                    url="vscode://file/test.py:42",
                    display_name="VS Code",
                    is_available=True
                ),
                IDENavigationUrl(
                    ide_type=IDEType.GENERIC,
                    url="file:///test.py",
                    display_name="System Default",
                    is_available=True
                )
            ]
            
            actions = self.ide_integration.create_mcp_navigation_actions(
                self.test_file_path, self.test_line
            )
            
            # Check structure
            self.assertIn("navigation_urls", actions)
            self.assertIn("file_info", actions)
            
            # Check navigation URLs
            nav_urls = actions["navigation_urls"]
            self.assertEqual(len(nav_urls), 2)
            self.assertEqual(nav_urls[0]["ide"], "VS Code")
            self.assertTrue(nav_urls[0]["available"])
            
            # Check file info
            file_info = actions["file_info"]
            self.assertIn("path", file_info)
            self.assertIn("line", file_info)
            self.assertIn("language", file_info)
            self.assertEqual(file_info["line"], self.test_line)

    def test_convenience_functions(self):
        """Test convenience functions."""
        # Test get_ide_navigation_urls
        with patch.object(IDEIntegration, 'generate_navigation_urls') as mock_method:
            mock_method.return_value = []
            get_ide_navigation_urls(self.test_file_path, self.test_line)
            mock_method.assert_called_once_with(self.test_file_path, self.test_line)
        
        # Test get_mcp_navigation_actions
        with patch.object(IDEIntegration, 'create_mcp_navigation_actions') as mock_method:
            mock_method.return_value = {}
            get_mcp_navigation_actions(self.test_file_path, self.test_line)
            mock_method.assert_called_once_with(self.test_file_path, self.test_line)


class TestSyntaxHighlightingHint(unittest.TestCase):
    """Test cases for SyntaxHighlightingHint data class."""

    def test_syntax_highlighting_hint_creation(self):
        """Test SyntaxHighlightingHint creation and attributes."""
        hint = SyntaxHighlightingHint(
            language="python",
            token_type="keyword",
            start_line=1,
            end_line=1,
            start_column=0,
            end_column=3
        )
        
        self.assertEqual(hint.language, "python")
        self.assertEqual(hint.token_type, "keyword")
        self.assertEqual(hint.start_line, 1)
        self.assertEqual(hint.end_line, 1)
        self.assertEqual(hint.start_column, 0)
        self.assertEqual(hint.end_column, 3)


class TestIDENavigationUrl(unittest.TestCase):
    """Test cases for IDENavigationUrl data class."""

    def test_ide_navigation_url_creation(self):
        """Test IDENavigationUrl creation and attributes."""
        url = IDENavigationUrl(
            ide_type=IDEType.VSCODE,
            url="vscode://file/test.py:42",
            display_name="VS Code",
            is_available=True
        )
        
        self.assertEqual(url.ide_type, IDEType.VSCODE)
        self.assertEqual(url.url, "vscode://file/test.py:42")
        self.assertEqual(url.display_name, "VS Code")
        self.assertTrue(url.is_available)


class TestCrossPlatformCompatibility(unittest.TestCase):
    """Test cross-platform compatibility features."""

    def setUp(self):
        """Set up test fixtures."""
        self.ide_integration = IDEIntegration()

    @patch('platform.system')
    def test_platform_detection(self, mock_platform):
        """Test platform detection."""
        mock_platform.return_value = "Darwin"
        integration = IDEIntegration()
        self.assertEqual(integration.platform, "darwin")
        
        mock_platform.return_value = "Windows"
        integration = IDEIntegration()
        self.assertEqual(integration.platform, "windows")
        
        mock_platform.return_value = "Linux"
        integration = IDEIntegration()
        self.assertEqual(integration.platform, "linux")

    def test_path_resolution(self):
        """Test that paths are properly resolved to absolute paths."""
        relative_path = "./test_file.py"
        urls = self.ide_integration.generate_navigation_urls(relative_path, 1)
        
        # All URLs should contain absolute paths
        for url in urls:
            if url.ide_type != IDEType.GENERIC:  # Generic uses file:// which handles this differently
                # Path in URL should be absolute
                self.assertTrue("/" in url.url or "\\" in url.url)  # Contains path separators


if __name__ == "__main__":
    unittest.main()