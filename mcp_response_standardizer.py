import hashlib
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

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

        # Add standard metadata
        standardized = {
            "success": True,
            "tool": tool_name,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC"),
            "version": "1.0",
            "response_id": generate_response_id(),
            **response_data,
        }

        # Add query context if available
        if query_context:
            standardized["query_context"] = query_context

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
            "version": "1.0",
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
