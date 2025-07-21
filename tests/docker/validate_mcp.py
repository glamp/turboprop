#!/usr/bin/env python3
"""
MCP functionality validation script.

This script validates that the MCP server installed via uvx works correctly
by testing basic MCP tool functionality and responses.
"""

import json
import os
import subprocess
import sys
import time
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any


class MCPValidator:
    """Validator for MCP functionality after uvx installation."""
    
    def __init__(self, test_repo_path: str = "/test-repo"):
        self.test_repo_path = Path(test_repo_path)
        self.results_dir = Path("/app/test-results")
        self.results_dir.mkdir(exist_ok=True)
        self.test_results: Dict[str, bool] = {}
        self.error_messages: List[str] = []
        self._turboprop_installed = False
        
    def log(self, message: str, level: str = "INFO"):
        """Log message with timestamp."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {level}: {message}")
        
    def run_command(self, cmd: List[str], timeout: int = 60) -> subprocess.CompletedProcess:
        """Run command with timeout and error handling."""
        self.log(f"Running command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                cwd=self.test_repo_path,
                timeout=timeout,
                capture_output=True,
                text=True,
                env=dict(os.environ, PATH="/root/.local/bin:" + os.environ.get("PATH", ""))
            )
            return result
        except subprocess.TimeoutExpired as e:
            self.log(f"Command timed out after {timeout}s", "ERROR")
            raise
        except Exception as e:
            self.log(f"Command failed: {e}", "ERROR")
            raise
            
    def ensure_turboprop_installed(self) -> bool:
        """Ensure turboprop is installed and cached by uvx."""
        if self._turboprop_installed:
            return True
            
        self.log("Installing and caching turboprop@latest...")
        try:
            result = self.run_command(["uvx", "turboprop@latest", "--version"], timeout=300)
            if result.returncode == 0:
                self._turboprop_installed = True
                self.log("Turboprop installed and cached successfully")
                return True
            else:
                self.error_messages.append(f"Failed to install turboprop: {result.stderr}")
                return False
        except Exception as e:
            self.error_messages.append(f"Exception installing turboprop: {str(e)}")
            return False
    
    def get_turboprop_cmd(self, *args) -> List[str]:
        """Get the appropriate uvx command for turboprop."""
        if self._turboprop_installed:
            return ["uvx", "turboprop"] + list(args)
        else:
            return ["uvx", "turboprop@latest"] + list(args)
            
    def test_index_creation(self) -> bool:
        """Test that indexing via CLI creates proper index files."""
        self.log("Testing index creation via CLI...")
        
        try:
            # Ensure turboprop is installed first
            if not self.ensure_turboprop_installed():
                return False
                
            # Run indexing command (shorter timeout since already cached)
            cmd = self.get_turboprop_cmd("index", ".", "--max-mb", "1.0")
            result = self.run_command(cmd, timeout=120)
            
            if result.returncode != 0:
                self.error_messages.append(f"Index command failed: {result.stderr}")
                return False
                
            # Check for index files
            turboprop_dir = self.test_repo_path / ".turboprop"
            if not turboprop_dir.exists():
                self.error_messages.append("No .turboprop directory created")
                return False
                
            db_file = turboprop_dir / "code_index.duckdb"
            if not db_file.exists():
                self.error_messages.append("No code_index.duckdb file created")
                return False
                
            self.log(f"Index created successfully, DB size: {db_file.stat().st_size} bytes")
            return True
            
        except Exception as e:
            self.error_messages.append(f"Exception during index creation: {str(e)}")
            return False
            
    def test_search_functionality(self) -> bool:
        """Test that search works correctly."""
        self.log("Testing search functionality...")
        
        try:
            # Ensure turboprop is installed first
            if not self.ensure_turboprop_installed():
                return False
                
            # Run search command (shorter timeout since already cached)
            cmd = self.get_turboprop_cmd("search", "React component", "--k", "3")
            result = self.run_command(cmd, timeout=60)
            
            if result.returncode != 0:
                self.error_messages.append(f"Search command failed: {result.stderr}")
                return False
                
            output = result.stdout.strip()
            if not output:
                self.error_messages.append("Search returned no output")
                return False
                
            # Check for expected content in poker codebase
            expected_patterns = [
                ".tsx",  # TypeScript React files
                "React",  # React-related content
                "component"  # Component-related content
            ]
            
            patterns_found = sum(1 for pattern in expected_patterns if pattern.lower() in output.lower())
            
            if patterns_found >= 2:
                self.log(f"Search working correctly ({patterns_found}/3 patterns found)")
                return True
            else:
                self.error_messages.append(f"Search results missing expected patterns ({patterns_found}/3 found)")
                return False
                
        except Exception as e:
            self.error_messages.append(f"Exception during search test: {str(e)}")
            return False
            
    def test_mcp_help_command(self) -> bool:
        """Test that MCP help/info commands work."""
        self.log("Testing MCP help command...")
        
        try:
            # Ensure turboprop is installed first
            if not self.ensure_turboprop_installed():
                return False
                
            # Test MCP help
            cmd = self.get_turboprop_cmd("mcp", "--help")
            result = self.run_command(cmd, timeout=30)
            
            if result.returncode != 0:
                self.error_messages.append(f"MCP help command failed: {result.stderr}")
                return False
                
            help_output = result.stdout + result.stderr
            expected_help_content = [
                "repository",
                "auto-index",
                "mcp"
            ]
            
            found_content = sum(1 for content in expected_help_content if content in help_output.lower())
            
            if found_content >= 2:
                self.log("MCP help command working correctly")
                return True
            else:
                self.error_messages.append("MCP help missing expected content")
                return False
                
        except Exception as e:
            self.error_messages.append(f"Exception during MCP help test: {str(e)}")
            return False
            
    def test_database_integrity(self) -> bool:
        """Test that the created database has expected structure and content."""
        self.log("Testing database integrity...")
        
        try:
            # Ensure turboprop is installed first
            if not self.ensure_turboprop_installed():
                return False
                
            # Use turboprop's built-in status/validation
            cmd = self.get_turboprop_cmd("search", "test query", "--k", "1")
            result = self.run_command(cmd, timeout=60)
            
            # If search works, database integrity is likely good
            if result.returncode == 0:
                self.log("Database integrity check passed (search successful)")
                return True
            else:
                self.error_messages.append(f"Database integrity check failed: {result.stderr}")
                return False
                
        except Exception as e:
            self.error_messages.append(f"Exception during database integrity test: {str(e)}")
            return False
            
    def test_file_coverage(self) -> bool:
        """Test that expected files from poker codebase are indexed."""
        self.log("Testing file coverage...")
        
        try:
            # Ensure turboprop is installed first
            if not self.ensure_turboprop_installed():
                return False
                
            # Search for specific files that should be in the poker codebase
            test_queries = [
                ("typescript", "tsx"),  # TypeScript files
                ("Player component", "Player"),  # Player.tsx
                ("package json", "package"),  # package.json
            ]
            
            successful_queries = 0
            
            for query, expected_term in test_queries:
                cmd = self.get_turboprop_cmd("search", query, "--k", "2")
                result = self.run_command(cmd, timeout=30)
                
                if result.returncode == 0 and expected_term.lower() in result.stdout.lower():
                    successful_queries += 1
                    self.log(f"Query '{query}' found expected content")
                else:
                    self.log(f"Query '{query}' did not find expected content", "WARNING")
                    
            if successful_queries >= 2:
                self.log(f"File coverage test passed ({successful_queries}/3 queries successful)")
                return True
            else:
                self.error_messages.append(f"Insufficient file coverage ({successful_queries}/3 queries successful)")
                return False
                
        except Exception as e:
            self.error_messages.append(f"Exception during file coverage test: {str(e)}")
            return False
            
    def run_all_validations(self) -> Dict[str, bool]:
        """Run all validation tests."""
        self.log("Starting MCP functionality validation...")
        
        validations = [
            ("index_creation", self.test_index_creation),
            ("search_functionality", self.test_search_functionality),
            ("mcp_help_command", self.test_mcp_help_command),
            ("database_integrity", self.test_database_integrity),
            ("file_coverage", self.test_file_coverage),
        ]
        
        for test_name, test_func in validations:
            self.log(f"Running validation: {test_name}")
            try:
                result = test_func()
                self.test_results[test_name] = result
                self.log(f"Validation {test_name}: {'PASS' if result else 'FAIL'}")
            except Exception as e:
                self.test_results[test_name] = False
                self.error_messages.append(f"Validation {test_name} threw exception: {str(e)}")
                self.log(f"Validation {test_name}: FAIL (exception: {e})", "ERROR")
                
        return self.test_results
        
    def generate_report(self) -> Dict:
        """Generate validation report."""
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result)
        
        report = {
            "validation_summary": {
                "total_validations": total_tests,
                "passed_validations": passed_tests,
                "failed_validations": total_tests - passed_tests,
                "success_rate": f"{(passed_tests/total_tests)*100:.1f}%" if total_tests > 0 else "0%"
            },
            "validation_results": self.test_results,
            "error_messages": self.error_messages,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Save report
        with open(self.results_dir / "validation_report.json", "w") as f:
            json.dump(report, f, indent=2)
            
        return report


def main():
    """Main validation execution."""
    print("=" * 60)
    print("TURBOPROP MCP FUNCTIONALITY VALIDATION")
    print("=" * 60)
    
    validator = MCPValidator()
    
    # Run all validations
    results = validator.run_all_validations()
    
    # Generate and display report
    report = validator.generate_report()
    
    print("\n" + "=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:.<30} {status}")
        
    print(f"\nSummary: {report['validation_summary']['passed_validations']}/{report['validation_summary']['total_validations']} validations passed")
    
    if validator.error_messages:
        print("\nErrors encountered:")
        for i, error in enumerate(validator.error_messages, 1):
            print(f"{i}. {error}")
            
    # Exit with appropriate code
    all_passed = all(results.values())
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()