import hashlib
import json
import logging
import time
from typing import Any, Dict, List, Optional

from .mcp_response_config import (
    JAVASCRIPT_SAFE_INTEGER,
    MAX_DICT_SIZE,
    MAX_KEY_LENGTH,
    MAX_LIST_SIZE,
    MAX_STRING_LENGTH,
    MAX_TOOL_NAME_LENGTH,
    RESPONSE_FORMAT_VERSION,
    SAFE_KEY_PATTERN,
)

logger = logging.getLogger(__name__)


def generate_response_id() -> str:
    """Generate unique response ID"""
    return hashlib.md5(f"{time.time()}".encode()).hexdigest()[:8]


def classify_error(error_message: str) -> str:
    """Classify error type for better handling"""
    error_lower = error_message.lower()

    if "not found" in error_lower or "missing" in error_lower:
        return "NOT_FOUND"
    elif "invalid" in error_lower or "malformed" in error_lower:
        return "INVALID_INPUT"
    elif "permission" in error_lower or "access" in error_lower:
        return "ACCESS_DENIED"
    elif "timeout" in error_lower or "connection" in error_lower:
        return "NETWORK_ERROR"
    else:
        return "UNKNOWN_ERROR"


def generate_error_suggestions(error_message: str, tool_name: str) -> List[str]:
    """Generate helpful suggestions for error recovery"""
    error_type = classify_error(error_message)

    suggestions = []

    if error_type == "NOT_FOUND":
        suggestions.extend(
            [
                "Check that the requested resource exists",
                "Verify the spelling and format of identifiers",
                "Try using search functions to find similar items",
            ]
        )
    elif error_type == "INVALID_INPUT":
        suggestions.extend(
            [
                "Review input parameters for correct format",
                "Check documentation for required fields",
                "Validate input data types and values",
            ]
        )
    elif error_type == "NETWORK_ERROR":
        suggestions.extend(
            ["Check network connectivity", "Retry the request after a brief delay", "Verify service availability"]
        )

    # Tool-specific suggestions
    if tool_name == "search_mcp_tools":
        suggestions.append("Try different search terms or broader queries")
    elif tool_name == "get_tool_details":
        suggestions.append("Verify the tool ID exists using search_mcp_tools first")

    return suggestions


def generate_recovery_options(error_message: str, tool_name: str) -> List[str]:
    """Generate recovery action options"""
    return [
        "Retry the operation with corrected parameters",
        "Use alternative tools or approaches",
        "Contact support if the issue persists",
    ]


def get_relevant_documentation(tool_name: str, error_message: str) -> List[str]:
    """Get relevant documentation links"""
    return [f"/docs/mcp-tools/{tool_name}", "/docs/troubleshooting/common-errors", "/docs/api-reference"]


def get_tool_version(tool_name: str) -> str:
    """Get version of the tool"""
    return "1.0.0"  # This would be retrieved from actual tool metadata


def get_system_state_summary() -> Dict[str, Any]:
    """Get summary of current system state for debugging"""
    return {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC"),
        "python_version": "3.9+",
        "available_memory": "OK",
        "disk_space": "OK",
    }


class MCPResponseStandardizer:
    """Standardize response formats across all tool search MCP tools"""

    def __init__(self):
        from mcp_response_optimizer import MCPResponseOptimizer
        from mcp_response_validator import MCPResponseValidator
        from tool_search_response_cache import ToolSearchResponseCache

        self.validator = MCPResponseValidator()
        self.optimizer = MCPResponseOptimizer()
        self.cache = ToolSearchResponseCache()

    def standardize_tool_search_response(
        self, response_data: Dict[str, Any], tool_name: str, query_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Standardize tool search response format"""

        # Sanitize and validate inputs
        sanitized_response_data = self._sanitize_response_data(response_data)
        sanitized_tool_name = self._sanitize_tool_name(tool_name)
        sanitized_query_context = self._sanitize_query_context(query_context)

        # Add standard metadata
        standardized = {
            "success": True,
            "tool": sanitized_tool_name,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC"),
            "version": RESPONSE_FORMAT_VERSION,
            "response_id": generate_response_id(),
            **sanitized_response_data,
        }

        # Add query context if available
        if sanitized_query_context:
            standardized["query_context"] = sanitized_query_context

        # Add performance metadata
        standardized["performance"] = {
            "cached": False,
            "execution_time": response_data.get("execution_time"),
            "result_count": self._count_results(response_data),
        }

        # Add navigation hints
        standardized["navigation"] = {
            "follow_up_suggestions": self._generate_follow_up_suggestions(response_data, tool_name),
            "related_queries": self._generate_related_queries(response_data),
            "improvement_hints": self._generate_improvement_hints(response_data),
        }

        # Validate and optimize
        validated_response = self.validator.validate_response(standardized, tool_name)
        optimized_response = self.optimizer.optimize_response(validated_response)

        return optimized_response

    def standardize_error_response(
        self, error_message: str, tool_name: str, context: Optional[str] = None, suggestions: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Create standardized error response"""

        return {
            "success": False,
            "tool": tool_name,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC"),
            "version": RESPONSE_FORMAT_VERSION,
            "response_id": generate_response_id(),
            "error": {
                "message": error_message,
                "context": context,
                "error_type": classify_error(error_message),
                "suggestions": suggestions or generate_error_suggestions(error_message, tool_name),
                "recovery_options": generate_recovery_options(error_message, tool_name),
                "documentation_links": get_relevant_documentation(tool_name, error_message),
            },
            "debug_info": {"tool_version": get_tool_version(tool_name), "system_state": get_system_state_summary()},
        }

    def _generate_follow_up_suggestions(self, response_data: Dict[str, Any], tool_name: str) -> List[str]:
        """Generate follow-up action suggestions"""
        suggestions = []

        if tool_name == "search_mcp_tools":
            if response_data.get("total_results", 0) > 10:
                suggestions.append("Use more specific terms to narrow results")
            elif response_data.get("total_results", 0) < 3:
                suggestions.append("Try broader search terms or different synonyms")

            suggestions.append("Use get_tool_details() for comprehensive information about specific tools")
            suggestions.append("Use compare_mcp_tools() to compare similar tools")

        elif tool_name == "recommend_tools_for_task":
            suggestions.append("Use get_tool_details() to learn more about recommended tools")
            suggestions.append("Use compare_mcp_tools() to compare top recommendations")
            suggestions.append("Use analyze_task_requirements() for deeper task analysis")

        elif tool_name == "get_tool_details":
            suggestions.append("Use search_mcp_tools() to find similar tools")
            suggestions.append("Use compare_mcp_tools() to compare with alternatives")

        elif tool_name == "compare_mcp_tools":
            suggestions.append("Use get_tool_details() for deeper analysis of preferred tools")
            suggestions.append("Use recommend_tools_for_task() to explore additional options")

        return suggestions

    def _generate_related_queries(self, response_data: Dict[str, Any]) -> List[str]:
        """Generate related query suggestions"""
        related_queries = []

        # Extract keywords from results to suggest related searches
        if "results" in response_data:
            # This would analyze result content to suggest related terms
            related_queries.extend(
                [
                    "Similar tools in same category",
                    "Alternative approaches to same task",
                    "Tools with complementary functionality",
                ]
            )

        return related_queries[:3]  # Limit to top 3

    def _generate_improvement_hints(self, response_data: Dict[str, Any]) -> List[str]:
        """Generate hints for improving search results"""
        hints = []

        result_count = self._count_results(response_data)

        if result_count == 0:
            hints.extend(
                [
                    "Try broader search terms",
                    "Check spelling and terminology",
                    "Use synonyms or alternative descriptions",
                ]
            )
        elif result_count > 20:
            hints.extend(
                ["Add more specific criteria", "Filter by category or functionality", "Use exact phrases in quotes"]
            )
        else:
            hints.append("Results look good - use follow-up tools for deeper analysis")

        return hints

    def _count_results(self, response_data: Dict[str, Any]) -> int:
        """Count results in response for performance metadata"""
        if "results" in response_data:
            return len(response_data["results"])
        elif "recommendations" in response_data:
            return len(response_data["recommendations"])
        elif "alternatives" in response_data:
            return len(response_data["alternatives"])
        elif "comparison_result" in response_data:
            comparison = response_data["comparison_result"]
            if isinstance(comparison, dict) and "tools" in comparison:
                return len(comparison["tools"])
        return 0

    def _sanitize_response_data(self, response_data: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize response data to prevent injection attacks"""
        if not isinstance(response_data, dict):
            logger.warning(f"Invalid response_data type: {type(response_data)}, using empty dict")
            return {}

        sanitized = {}
        for key, value in response_data.items():
            sanitized_key = self._sanitize_string_key(key)
            sanitized_value = self._sanitize_value(value)
            if sanitized_key and sanitized_value is not None:
                sanitized[sanitized_key] = sanitized_value

        return sanitized

    def _sanitize_tool_name(self, tool_name: str) -> str:
        """Sanitize tool name to prevent injection"""
        if not isinstance(tool_name, str):
            logger.warning(f"Invalid tool_name type: {type(tool_name)}, using 'unknown'")
            return "unknown"

        # Allow only alphanumeric characters, underscores, and hyphens
        import re

        sanitized = re.sub(r"[^a-zA-Z0-9_-]", "", tool_name)

        if not sanitized:
            logger.warning(f"Tool name '{tool_name}' sanitized to empty string, using 'unknown'")
            return "unknown"

        # Limit length to prevent buffer overflow attacks
        return sanitized[:MAX_TOOL_NAME_LENGTH]

    def _sanitize_query_context(self, query_context: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Sanitize query context to prevent injection"""
        if query_context is None:
            return None

        if not isinstance(query_context, dict):
            logger.warning(f"Invalid query_context type: {type(query_context)}, ignoring")
            return None

        sanitized = {}
        for key, value in query_context.items():
            sanitized_key = self._sanitize_string_key(key)
            sanitized_value = self._sanitize_value(value)
            if sanitized_key and sanitized_value is not None:
                sanitized[sanitized_key] = sanitized_value

        return sanitized if sanitized else None

    def _sanitize_string_key(self, key: str) -> str:
        """Sanitize dictionary keys"""
        if not isinstance(key, str):
            logger.warning(f"Non-string key detected: {type(key)}, converting to string")
            key = str(key)

        # Remove potentially dangerous characters
        import re

        sanitized = re.sub(SAFE_KEY_PATTERN, "", key)

        # Limit key length
        return sanitized[:MAX_KEY_LENGTH] if sanitized else ""

    def _sanitize_value(self, value: Any) -> Any:
        """Sanitize values recursively"""
        if value is None:
            return None
        elif isinstance(value, bool):
            return value
        elif isinstance(value, (int, float)):
            # Validate numeric ranges to prevent overflow
            if isinstance(value, int) and abs(value) > JAVASCRIPT_SAFE_INTEGER:  # JavaScript safe integer
                logger.warning(f"Integer value too large: {value}, capping to safe range")
                return JAVASCRIPT_SAFE_INTEGER if value > 0 else -JAVASCRIPT_SAFE_INTEGER
            elif isinstance(value, float) and (value != value or abs(value) == float("inf")):  # NaN or Inf
                logger.warning(f"Invalid float value: {value}, replacing with 0")
                return 0.0
            return value
        elif isinstance(value, str):
            return self._sanitize_string_value(value)
        elif isinstance(value, (list, tuple)):
            # Limit list/tuple size to prevent DoS
            if len(value) > MAX_LIST_SIZE:
                logger.warning(f"Sequence too large ({len(value)} items), truncating to {MAX_LIST_SIZE}")
                value = value[:MAX_LIST_SIZE]
            sanitized_items = [self._sanitize_value(item) for item in value]
            # Preserve original type (list or tuple)
            return type(value)(sanitized_items)
        elif isinstance(value, dict):
            # Limit dict size to prevent DoS
            if len(value) > MAX_DICT_SIZE:
                logger.warning(f"Dict too large ({len(value)} keys), truncating")
                value = dict(list(value.items())[:MAX_DICT_SIZE])
            sanitized_dict = {}
            for k, v in value.items():
                sanitized_key = self._sanitize_string_key(str(k))
                sanitized_val = self._sanitize_value(v)
                if sanitized_key and sanitized_val is not None:
                    sanitized_dict[sanitized_key] = sanitized_val
            return sanitized_dict
        else:
            # Convert unknown types to string and sanitize
            logger.warning(f"Unknown value type: {type(value)}, converting to string")
            return self._sanitize_string_value(str(value))

    def _sanitize_string_value(self, value: str) -> str:
        """Sanitize string values to prevent injection"""
        if not isinstance(value, str):
            value = str(value)

        # Limit string length to prevent buffer overflow
        if len(value) > MAX_STRING_LENGTH:  # 100KB limit
            logger.warning(f"String too long ({len(value)} chars), truncating to {MAX_STRING_LENGTH}")
            value = value[:MAX_STRING_LENGTH]

        # Remove or escape potentially dangerous patterns
        import re

        # Remove null bytes and other control characters except common ones
        value = re.sub(r"[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F-\x9F]", "", value)

        # Remove script tags and javascript: URLs (case insensitive)
        value = re.sub(r"<\s*script[^>]*>.*?</\s*script\s*>", "", value, flags=re.IGNORECASE | re.DOTALL)
        value = re.sub(r"javascript\s*:", "", value, flags=re.IGNORECASE)

        # Remove other potentially dangerous HTML tags
        dangerous_tags = ["iframe", "object", "embed", "applet", "meta", "link"]
        for tag in dangerous_tags:
            value = re.sub(rf"<\s*{tag}[^>]*>.*?</\s*{tag}\s*>", "", value, flags=re.IGNORECASE | re.DOTALL)
            value = re.sub(rf"<\s*{tag}[^>]*/?>", "", value, flags=re.IGNORECASE)

        # Remove event handlers
        value = re.sub(r'\bon\w+\s*=\s*["\'][^"\']*["\']', "", value, flags=re.IGNORECASE)

        return value


def standardize_mcp_tool_response(func):
    """Decorator to automatically standardize MCP tool responses"""

    def wrapper(*args, **kwargs):
        start_time = time.time()

        try:
            # Execute original function
            response_data = func(*args, **kwargs)

            # Add execution time to response data
            execution_time = time.time() - start_time
            if isinstance(response_data, dict):
                response_data["execution_time"] = execution_time

            # Standardize response
            standardizer = MCPResponseStandardizer()
            standardized_response = standardizer.standardize_tool_search_response(
                response_data=response_data, tool_name=func.__name__, query_context={"args": args, "kwargs": kwargs}
            )
            return json.dumps(standardized_response)

        except Exception as e:
            # Create standardized error response
            standardizer = MCPResponseStandardizer()
            error_response = standardizer.standardize_error_response(
                error_message=str(e), tool_name=func.__name__, context=f"Error in {func.__name__} with args: {args}"
            )
            return json.dumps(error_response)

    return wrapper
