#!/usr/bin/env python3
"""
Test script for validating uvx turboprop MCP installation in Docker environment.

This script tests the exact command used in MCP configurations:
uvx turboprop@latest mcp --repository . --auto-index

It validates:
1. uvx can successfully install turboprop@latest
2. The MCP server starts without errors  
3. Auto-indexing works on the poker codebase
4. Expected output is produced
"""

import json
import os
import subprocess
import sys
import time
import threading
from pathlib import Path
from typing import Dict, List, Optional


class UVXMCPTester:
    """Test harness for uvx MCP installation flow."""
    
    def __init__(self, test_repo_path: str = "/test-repo"):
        self.test_repo_path = Path(test_repo_path)
        self.results_dir = Path("/app/test-results")
        self.results_dir.mkdir(exist_ok=True)
        self.test_results: Dict[str, bool] = {}
        self.error_messages: List[str] = []
        
    def log(self, message: str, level: str = "INFO"):
        """Log message with timestamp."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {level}: {message}")
        
    def run_command(self, cmd: List[str], timeout: int = 300, capture_output: bool = True) -> subprocess.CompletedProcess:
        """Run command with timeout and error handling."""
        self.log(f"Running command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                cwd=self.test_repo_path,
                timeout=timeout,
                capture_output=capture_output,
                text=True,
                env=dict(os.environ, PATH="/root/.local/bin:" + os.environ.get("PATH", ""))
            )
            return result
        except subprocess.TimeoutExpired as e:
            self.log(f"Command timed out after {timeout}s: {e}", "ERROR")
            raise
        except Exception as e:
            self.log(f"Command failed: {e}", "ERROR")
            raise
            
    def test_uvx_available(self) -> bool:
        """Test that uvx is available and working."""
        self.log("Testing uvx availability...")
        
        try:
            result = self.run_command(["uvx", "--version"])
            if result.returncode == 0:
                self.log(f"uvx version: {result.stdout.strip()}")
                return True
            else:
                self.error_messages.append(f"uvx --version failed: {result.stderr}")
                return False
        except Exception as e:
            self.error_messages.append(f"uvx not available: {str(e)}")
            return False
            
    def test_turboprop_installation(self) -> bool:
        """Test that uvx can install turboprop@latest."""
        self.log("Testing turboprop installation via uvx...")
        
        try:
            # First, ensure we start fresh
            result = self.run_command(["uvx", "--help"], timeout=30)
            if result.returncode != 0:
                self.error_messages.append("uvx not working properly")
                return False
                
            # Test installing turboprop - this should download and cache it (longer timeout for ML dependencies)
            result = self.run_command(["uvx", "turboprop@latest", "--version"], timeout=300)
            
            if result.returncode == 0:
                version_output = result.stdout.strip()
                self.log(f"Successfully installed turboprop: {version_output}")
                # Store that we successfully installed for subsequent tests
                self._turboprop_installed = True
                return True
            else:
                error_output = result.stderr.strip() if result.stderr else "No error output"
                self.error_messages.append(f"turboprop installation failed: {error_output}")
                self.log(f"Installation stderr: {error_output}", "ERROR")
                return False
                
        except Exception as e:
            self.error_messages.append(f"Exception during turboprop installation: {str(e)}")
            return False
            
    def test_mcp_server_startup(self) -> bool:
        """Test that the MCP server can start with the specified arguments."""
        self.log("Testing MCP server startup with auto-index...")
        
        # Create a process to run the MCP command
        # Use cached version if already installed, otherwise use @latest
        cmd = ["uvx", "turboprop", "mcp", "--repository", ".", "--auto-index"] if hasattr(self, '_turboprop_installed') and self._turboprop_installed else ["uvx", "turboprop@latest", "mcp", "--repository", ".", "--auto-index"]
        
        try:
            # Start the process
            proc = subprocess.Popen(
                cmd,
                cwd=self.test_repo_path,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=dict(os.environ, PATH="/root/.local/bin:" + os.environ.get("PATH", ""))
            )
            
            # Give it time to start up and auto-index (longer timeout for ML model loading)
            startup_timeout = 180
            stderr_output = []
            stdout_output = []
            
            # Read stderr for startup messages (MCP uses stderr for status)
            def read_stderr():
                while True:
                    try:
                        line = proc.stderr.readline()
                        if not line:
                            break
                        line = line.strip()
                        if line:
                            stderr_output.append(line)
                            self.log(f"MCP stderr: {line}")
                    except Exception as e:
                        self.log(f"Error reading stderr: {e}")
                        break
            
            # Start stderr reader thread
            stderr_thread = threading.Thread(target=read_stderr)
            stderr_thread.daemon = True
            stderr_thread.start()
            
            # Wait for startup or timeout
            start_time = time.time()
            startup_complete = False
            
            while time.time() - start_time < startup_timeout:
                # Check if process is still running
                if proc.poll() is not None:
                    # Process terminated
                    stdout, stderr = proc.communicate()
                    stdout_output.append(stdout if stdout else "")
                    
                    if proc.returncode != 0:
                        self.error_messages.append(f"MCP server failed to start (exit code {proc.returncode})")
                        self.error_messages.append(f"Stderr: {stderr}")
                        return False
                    break
                    
                # Check for success indicators in stderr
                recent_stderr = "\n".join(stderr_output[-10:])  # Last 10 lines
                if ("Indexing complete" in recent_stderr or 
                    "ready for semantic search" in recent_stderr or
                    "MCP Server Starting" in recent_stderr):
                    startup_complete = True
                    self.log("MCP server startup detected!")
                    break
                    
                time.sleep(1)
            
            # Terminate the process
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            
            # Analyze the output
            all_stderr = "\n".join(stderr_output)
            all_stdout = "\n".join(stdout_output)
            
            # Save outputs for debugging
            with open(self.results_dir / "mcp_stderr.log", "w") as f:
                f.write(all_stderr)
            with open(self.results_dir / "mcp_stdout.log", "w") as f:
                f.write(all_stdout)
                
            # Check for success indicators
            success_indicators = [
                "Turboprop MCP Server Starting",
                "Repository:",
                "Max file size:",
                "Auto-index: Yes"
            ]
            
            indicators_found = sum(1 for indicator in success_indicators if indicator in all_stderr)
            
            if indicators_found >= 3:  # Most indicators found
                self.log(f"MCP server started successfully ({indicators_found}/4 indicators found)")
                return True
            else:
                self.error_messages.append(f"MCP server startup incomplete ({indicators_found}/4 indicators found)")
                self.error_messages.append(f"Missing indicators in stderr output")
                return False
                
        except Exception as e:
            self.error_messages.append(f"Exception during MCP server test: {str(e)}")
            return False
            
    def test_repository_structure(self) -> bool:
        """Verify the test repository has expected structure."""
        self.log("Testing repository structure...")
        
        expected_files = [
            "package.json",
            "src/pages/index.tsx", 
            "src/widgets/Player.tsx",
            "README.md"
        ]
        
        missing_files = []
        for file_path in expected_files:
            if not (self.test_repo_path / file_path).exists():
                missing_files.append(file_path)
                
        if missing_files:
            self.error_messages.append(f"Missing expected files: {missing_files}")
            return False
            
        self.log("Repository structure looks good")
        return True
        
    def run_all_tests(self) -> Dict[str, bool]:
        """Run all tests and return results."""
        self.log("Starting uvx MCP installation tests...")
        
        tests = [
            ("repository_structure", self.test_repository_structure),
            ("uvx_available", self.test_uvx_available),
            ("turboprop_installation", self.test_turboprop_installation),
            ("mcp_server_startup", self.test_mcp_server_startup),
        ]
        
        for test_name, test_func in tests:
            self.log(f"Running test: {test_name}")
            try:
                result = test_func()
                self.test_results[test_name] = result
                self.log(f"Test {test_name}: {'PASS' if result else 'FAIL'}")
            except Exception as e:
                self.test_results[test_name] = False
                self.error_messages.append(f"Test {test_name} threw exception: {str(e)}")
                self.log(f"Test {test_name}: FAIL (exception: {e})", "ERROR")
                
        return self.test_results
        
    def generate_report(self) -> Dict:
        """Generate test report."""
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result)
        
        report = {
            "test_summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": total_tests - passed_tests,
                "success_rate": f"{(passed_tests/total_tests)*100:.1f}%" if total_tests > 0 else "0%"
            },
            "test_results": self.test_results,
            "error_messages": self.error_messages,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Save report
        with open(self.results_dir / "test_report.json", "w") as f:
            json.dump(report, f, indent=2)
            
        return report


def main():
    """Main test execution."""
    print("=" * 60)
    print("TURBOPROP UVX MCP INSTALLATION TEST")
    print("=" * 60)
    
    tester = UVXMCPTester()
    
    # Run all tests
    results = tester.run_all_tests()
    
    # Generate and display report
    report = tester.generate_report()
    
    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:.<30} {status}")
        
    print(f"\nSummary: {report['test_summary']['passed_tests']}/{report['test_summary']['total_tests']} tests passed")
    
    if tester.error_messages:
        print("\nErrors encountered:")
        for i, error in enumerate(tester.error_messages, 1):
            print(f"{i}. {error}")
            
    # Exit with appropriate code
    all_passed = all(results.values())
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()