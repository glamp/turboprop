"""
Configuration constants for MCP response standardization system.

This module centralizes all magic numbers and configuration values used
throughout the MCP response processing system for better maintainability
and consistency.
"""

# Response Size Limits (bytes)
RESPONSE_SIZE_SMALL_THRESHOLD = 5_000  # 5KB - considered small response
RESPONSE_SIZE_LARGE_THRESHOLD = 10_000  # 10KB - considered large response
RESPONSE_SIZE_VERY_LARGE_THRESHOLD = 20_000  # 20KB - very large response threshold
RESPONSE_SIZE_WARNING_THRESHOLD = 50_000  # 50KB - generates performance warning
RESPONSE_SIZE_CRITICAL_THRESHOLD = 100_000  # 100KB - critical size limit

# Performance Thresholds (seconds)
EXECUTION_TIME_OPTIMAL = 2.0  # 2s - optimal execution time
EXECUTION_TIME_WARNING = 5.0  # 5s - warning threshold for slow execution

# Result Count Limits
RESULT_COUNT_SUMMARY_THRESHOLD = 10  # Results count that triggers summary generation
RESULT_COUNT_PERFORMANCE_WARNING = 100  # Large result count that impacts performance
RESULT_COUNT_COMPLEX_RESPONSE = 10  # Results count for complex response classification

# Data Structure Size Limits
MAX_LIST_SIZE = 10_000  # Maximum items in a list to prevent DoS
MAX_DICT_SIZE = 1_000  # Maximum keys in a dict to prevent DoS
MAX_STRING_LENGTH = 100_000  # Maximum string length (100KB)
MAX_KEY_LENGTH = 50  # Maximum length for dictionary keys
MAX_TOOL_NAME_LENGTH = 100  # Maximum tool name length

# Cache Configuration
DEFAULT_CACHE_SIZE = 1_000  # Default cache capacity (entries)
DEFAULT_CACHE_TTL = 3_600  # Default cache TTL (1 hour)
CACHE_EVICTION_PERCENTAGE = 20  # Percentage of cache to evict when full (20%)

# Validation Rules
MAX_NESTING_LEVEL = 10  # Maximum nesting depth for data structures
JAVASCRIPT_SAFE_INTEGER = 2**53  # JavaScript safe integer limit

# Debug and Metadata Limits
DEBUG_INFO_SIZE_THRESHOLD = 50_000  # Size threshold for debug info summarization
TOP_CATEGORIES_LIMIT = 3  # Number of top categories to display
TOP_ENTRIES_DEFAULT_LIMIT = 10  # Default limit for top accessed entries

# Tool-Specific Estimated Response Times (seconds)
TOOL_RESPONSE_TIME_ESTIMATES = {
    "search_mcp_tools": 1.0,
    "get_tool_details": 0.5,
    "recommend_tools_for_task": 1.5,
    "compare_mcp_tools": 2.0,
    "analyze_task_requirements": 1.0,
}

# Default response time for unknown tools
DEFAULT_TOOL_RESPONSE_TIME = 0.8

# Cache Health Thresholds
CACHE_CAPACITY_WARNING_THRESHOLD = 0.9  # 90% capacity usage warning
CACHE_EXPIRED_ENTRIES_WARNING_RATIO = 0.3  # 30% expired entries warning
CACHE_HIT_RATE_WARNING_THRESHOLD = 0.3  # 30% hit rate warning

# Content Quality Thresholds
LARGE_OBJECT_SUMMARIZATION_THRESHOLD = 1_000  # Size threshold for object summarization
ESSENTIAL_RESULT_FIELDS_LIMIT = 5  # Number of essential fields to keep in summaries

# Response Format Constants
RESPONSE_FORMAT_VERSION = "1.0"
RESPONSE_ID_LENGTH = 8  # Length of generated response IDs

# Regex Patterns for Sanitization
SAFE_KEY_PATTERN = r"[^a-zA-Z0-9_.-]"  # Pattern for safe dictionary keys
SAFE_TOOL_NAME_PATTERN = r"[^a-zA-Z0-9_-]"  # Pattern for safe tool names

# HTML Sanitization
DANGEROUS_HTML_TAGS = ["iframe", "object", "embed", "applet", "meta", "link"]

# Control Characters Pattern (excluding common whitespace)
CONTROL_CHARS_PATTERN = r"[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F-\x9F]"

# Performance Multipliers
RESULT_COMPLEXITY_MULTIPLIER_MEDIUM = 1.2  # >10 results
RESULT_COMPLEXITY_MULTIPLIER_HIGH = 1.5  # >20 results

# Truncation Indicators
TRUNCATION_SUFFIX = "..."
CACHE_KEY_DISPLAY_LENGTH = 16  # Length to show in logs before truncation
