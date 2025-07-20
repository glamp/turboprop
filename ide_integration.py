"""
IDE navigation and integration features for Turboprop.

Provides functionality to generate IDE-specific navigation URLs and metadata
for seamless integration with various development environments.
"""
import os
import platform
import subprocess
from pathlib import Path
from typing import Dict, List, Optional
from enum import Enum
from dataclasses import dataclass


class IDEType(Enum):
    """Supported IDE types for navigation URL generation."""
    VSCODE = "vscode"
    JETBRAINS = "jetbrains"
    VIM = "vim"
    NEOVIM = "neovim"
    SUBLIME = "sublime"
    GENERIC = "generic"


@dataclass
class IDENavigationUrl:
    """Container for IDE navigation URLs and metadata."""
    ide_type: IDEType
    url: str
    display_name: str
    is_available: bool = True


@dataclass
class SyntaxHighlightingHint:
    """Syntax highlighting metadata for code snippets."""
    language: str
    token_type: str
    start_line: int
    end_line: int
    start_column: int
    end_column: int


class IDEIntegration:
    """Main class for IDE integration functionality."""

    def __init__(self):
        self.platform = platform.system().lower()
        self._detected_ides = None
        self._ide_cache = {}

    def generate_navigation_urls(
        self,
        file_path: str,
        line_number: int = 1,
        column: int = 1
    ) -> List[IDENavigationUrl]:
        """
        Generate navigation URLs for different IDEs.

        Args:
            file_path: Absolute or relative path to the file
            line_number: Line number to navigate to (1-indexed)
            column: Column number to navigate to (1-indexed)

        Returns:
            List of IDENavigationUrl objects for available IDEs
        """
        # Input validation
        if not file_path or not isinstance(file_path, str):
            raise ValueError("file_path must be a non-empty string")
        
        if not isinstance(line_number, int) or line_number < 1:
            raise ValueError("line_number must be a positive integer (1-indexed)")
        
        if not isinstance(column, int) or column < 1:
            raise ValueError("column must be a positive integer (1-indexed)")
        
        # Validate file path can be resolved
        try:
            abs_path = Path(file_path).resolve()
        except (OSError, ValueError) as e:
            raise ValueError(f"Invalid file path '{file_path}': {e}")
        
        urls = []

        # VS Code
        if self._is_ide_available(IDEType.VSCODE):
            vscode_url = f"vscode://file/{abs_path}:{line_number}:{column}"
            urls.append(IDENavigationUrl(
                ide_type=IDEType.VSCODE,
                url=vscode_url,
                display_name="VS Code",
                is_available=True
            ))

        # JetBrains IDEs (IntelliJ, PyCharm, WebStorm, etc.)
        if self._is_ide_available(IDEType.JETBRAINS):
            # Try different JetBrains IDE schemes
            for scheme in ["idea", "pycharm", "webstorm", "phpstorm"]:
                jetbrains_url = f"{scheme}://open?file={abs_path}&line={line_number}&column={column}"
                urls.append(IDENavigationUrl(
                    ide_type=IDEType.JETBRAINS,
                    url=jetbrains_url,
                    display_name=f"{scheme.title()}",
                    is_available=True
                ))
                break  # Only add one JetBrains URL for now

        # Vim/Neovim
        if self._is_ide_available(IDEType.VIM):
            vim_url = f"vim://{abs_path}:{line_number}"
            urls.append(IDENavigationUrl(
                ide_type=IDEType.VIM,
                url=vim_url,
                display_name="Vim",
                is_available=True
            ))

        if self._is_ide_available(IDEType.NEOVIM):
            nvim_url = f"nvim://{abs_path}:{line_number}"
            urls.append(IDENavigationUrl(
                ide_type=IDEType.NEOVIM,
                url=nvim_url,
                display_name="Neovim",
                is_available=True
            ))

        # Sublime Text
        if self._is_ide_available(IDEType.SUBLIME):
            sublime_url = f"subl://{abs_path}:{line_number}:{column}"
            urls.append(IDENavigationUrl(
                ide_type=IDEType.SUBLIME,
                url=sublime_url,
                display_name="Sublime Text",
                is_available=True
            ))

        # Generic file:// URL as fallback
        generic_url = f"file://{abs_path}"
        urls.append(IDENavigationUrl(
            ide_type=IDEType.GENERIC,
            url=generic_url,
            display_name="System Default",
            is_available=True
        ))

        return urls

    def _is_ide_available(self, ide_type: IDEType) -> bool:
        """
        Check if a specific IDE is available on the system.

        Args:
            ide_type: The IDE type to check for

        Returns:
            True if the IDE is likely available, False otherwise
        """
        if ide_type in self._ide_cache:
            return self._ide_cache[ide_type]

        available = False

        try:
            if ide_type == IDEType.VSCODE:
                available = self._check_command_exists(["code", "code-insiders"])
            elif ide_type == IDEType.JETBRAINS:
                # Check for common JetBrains IDE executables
                jetbrains_commands = [
                    "idea", "pycharm", "webstorm", "phpstorm",
                    "goland", "clion", "rider", "rubymine"
                ]
                available = self._check_command_exists(jetbrains_commands)
            elif ide_type == IDEType.VIM:
                available = self._check_command_exists(["vim", "gvim"])
            elif ide_type == IDEType.NEOVIM:
                available = self._check_command_exists(["nvim"])
            elif ide_type == IDEType.SUBLIME:
                available = self._check_command_exists(["subl", "sublime_text"])
        except (OSError, ValueError, TypeError) as e:
            # Handle specific exceptions that might occur during IDE detection
            # OSError: System-level errors (command not found, permission issues)
            # ValueError: Invalid IDE type or command parameters
            # TypeError: Type mismatches in command processing
            available = False
            # Log the specific error for debugging
            import sys
            print(f"IDE detection failed for {ide_type.value}: {type(e).__name__}: {e}", 
                  file=sys.stderr)

        self._ide_cache[ide_type] = available
        return available

    def _check_command_exists(self, commands: List[str]) -> bool:
        """
        Check if any of the given commands exist in PATH.

        Args:
            commands: List of command names to check

        Returns:
            True if any command exists, False otherwise
        """
        for cmd in commands:
            try:
                if self.platform == "windows":
                    subprocess.run(
                        ["where", cmd],
                        check=True,
                        capture_output=True,
                        timeout=5
                    )
                else:
                    subprocess.run(
                        ["which", cmd],
                        check=True,
                        capture_output=True,
                        timeout=5
                    )
                return True
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                continue
        return False

    def normalize_path(self, file_path: str) -> str:
        """
        Normalize file path for cross-platform compatibility.

        Args:
            file_path: Input file path (absolute or relative)

        Returns:
            Normalized absolute path suitable for the current platform
        """
        # Input validation
        if not file_path or not isinstance(file_path, str):
            raise ValueError("file_path must be a non-empty string")
        
        try:
            path = Path(file_path)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid file path '{file_path}': {e}")

        # Handle WSL path translation on Windows
        if self.platform == "windows" and str(path).startswith("/mnt/"):
            try:
                # Convert WSL paths like /mnt/c/... to C:\...
                parts = path.parts
                
                # Validate WSL path structure: /mnt/<drive>/<path...>
                if len(parts) < 3:
                    raise ValueError("WSL path must have at least drive letter: /mnt/<drive>/")
                
                if parts[0] != "/" or parts[1] != "mnt":
                    raise ValueError("Invalid WSL path format: must start with /mnt/")
                
                drive_letter = parts[2].lower()
                if len(drive_letter) != 1 or not drive_letter.isalpha():
                    raise ValueError(f"Invalid WSL drive letter: '{drive_letter}' (must be single letter)")
                
                # Build Windows path
                remaining_parts = parts[3:] if len(parts) > 3 else []
                windows_drive = drive_letter.upper() + ":"
                
                if remaining_parts:
                    return str(Path(windows_drive).joinpath(*remaining_parts))
                else:
                    return windows_drive + "\\"
                    
            except (ValueError, IndexError) as e:
                # If WSL path conversion fails, log warning and fall back to original path
                import sys
                print(f"WSL path conversion failed for '{path}': {e}. Using original path.", 
                      file=sys.stderr)
                # Continue to normal path resolution

        return str(path.resolve())

    def get_language_from_extension(self, file_path: str) -> str:
        """
        Determine programming language from file extension.

        Args:
            file_path: File path to analyze

        Returns:
            Language identifier string
        """
        extension_map = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".jsx": "javascriptreact",
            ".tsx": "typescriptreact",
            ".java": "java",
            ".cpp": "cpp",
            ".c": "c",
            ".h": "c",
            ".hpp": "cpp",
            ".cs": "csharp",
            ".go": "go",
            ".rs": "rust",
            ".rb": "ruby",
            ".php": "php",
            ".swift": "swift",
            ".kt": "kotlin",
            ".scala": "scala",
            ".r": "r",
            ".sql": "sql",
            ".sh": "shellscript",
            ".bash": "shellscript",
            ".zsh": "shellscript",
            ".fish": "shellscript",
            ".ps1": "powershell",
            ".html": "html",
            ".css": "css",
            ".scss": "scss",
            ".sass": "sass",
            ".less": "less",
            ".xml": "xml",
            ".json": "json",
            ".yaml": "yaml",
            ".yml": "yaml",
            ".toml": "toml",
            ".ini": "ini",
            ".cfg": "ini",
            ".conf": "ini",
            ".md": "markdown",
            ".rst": "restructuredtext",
            ".tex": "latex",
            ".dockerfile": "dockerfile",
            ".makefile": "makefile",
            ".cmake": "cmake",
        }

        suffix = Path(file_path).suffix.lower()
        return extension_map.get(suffix, "plaintext")

    def generate_syntax_hints(
        self,
        file_path: str,
        content: str,
        target_line: Optional[int] = None
    ) -> List[SyntaxHighlightingHint]:
        """
        Generate basic syntax highlighting hints for code content.

        Args:
            file_path: File path for language detection
            content: File content to analyze
            target_line: Specific line to focus on (optional)

        Returns:
            List of syntax highlighting hints
        """
        # Input validation
        if not file_path or not isinstance(file_path, str):
            raise ValueError("file_path must be a non-empty string")
        
        if not isinstance(content, str):
            raise ValueError("content must be a string")
        
        if target_line is not None and (not isinstance(target_line, int) or target_line < 1):
            raise ValueError("target_line must be a positive integer (1-indexed) or None")
        
        language = self.get_language_from_extension(file_path)
        hints = []

        lines = content.split('\n')
        start_line = target_line - 2 if target_line and target_line > 2 else 1
        end_line = target_line + 2 if target_line and target_line < len(lines) - 2 else len(lines)

        # Basic syntax highlighting based on language
        for line_num, line in enumerate(lines[start_line - 1:end_line], start_line):
            line_content = line.strip()

            if not line_content:
                continue

            # Simple keyword detection based on language
            if language == "python":
                if any(keyword in line_content for keyword in ["def ", "class ", "import ", "from "]):
                    hints.append(SyntaxHighlightingHint(
                        language=language,
                        token_type="keyword",
                        start_line=line_num,
                        end_line=line_num,
                        start_column=1,
                        end_column=len(line)
                    ))
            elif language in ["javascript", "typescript"]:
                if any(keyword in line_content for keyword in ["function", "class", "const", "let", "var"]):
                    hints.append(SyntaxHighlightingHint(
                        language=language,
                        token_type="keyword",
                        start_line=line_num,
                        end_line=line_num,
                        start_column=1,
                        end_column=len(line)
                    ))

            # Comment detection
            if line_content.startswith(("//", "#", "/*", "<!--")):
                hints.append(SyntaxHighlightingHint(
                    language=language,
                    token_type="comment",
                    start_line=line_num,
                    end_line=line_num,
                    start_column=1,
                    end_column=len(line)
                ))

        return hints

    def create_mcp_navigation_actions(
        self,
        file_path: str,
        line_number: int = 1
    ) -> Dict:
        """
        Create MCP-compatible navigation actions for IDE integration.

        Args:
            file_path: File path for navigation
            line_number: Line number to navigate to

        Returns:
            Dictionary containing MCP action metadata
        """
        # Input validation - delegate to generate_navigation_urls which already validates
        # This method calls generate_navigation_urls which has comprehensive validation
        urls = self.generate_navigation_urls(file_path, line_number)
        normalized_path = self.normalize_path(file_path)

        actions = {
            "navigation_urls": [
                {
                    "ide": url.display_name,
                    "url": url.url,
                    "available": url.is_available
                }
                for url in urls
            ],
            "file_info": {
                "path": normalized_path,
                "relative_path": os.path.relpath(normalized_path),
                "line": line_number,
                "language": self.get_language_from_extension(file_path)
            }
        }

        return actions


# Global instance for easy access
ide_integration = IDEIntegration()


def get_ide_navigation_urls(file_path: str, line_number: int = 1) -> List[IDENavigationUrl]:
    """Convenience function to get IDE navigation URLs."""
    return ide_integration.generate_navigation_urls(file_path, line_number)


def get_mcp_navigation_actions(file_path: str, line_number: int = 1) -> Dict:
    """Convenience function to get MCP navigation actions."""
    return ide_integration.create_mcp_navigation_actions(file_path, line_number)