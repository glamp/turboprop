import hashlib
import logging
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def generate_cache_key(response: Dict[str, Any]) -> str:
    """Generate cache key for response"""
    # Create key from tool name, query context, and relevant params
    key_components = [
        response.get("tool", ""),
        str(response.get("query_context", {})),
        str(response.get("timestamp", "")),
    ]

    key_string = "|".join(key_components)
    return hashlib.md5(key_string.encode()).hexdigest()


class ResponseCompressor:
    """Compress response data for storage efficiency"""

    def compress_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Apply compression strategies to response"""
        compressed = response.copy()

        # Remove redundant metadata for large responses
        if self._is_large_response(response):
            compressed = self._remove_redundant_metadata(compressed)

        # Compress large text fields
        compressed = self._compress_text_fields(compressed)

        return compressed

    def _is_large_response(self, response: Dict[str, Any]) -> bool:
        """Check if response is considered large"""
        import json

        try:
            return len(json.dumps(response)) > 10000  # 10KB threshold
        except:
            return False

    def _remove_redundant_metadata(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Remove redundant metadata from large responses"""
        # Keep essential fields, remove verbose debugging info
        if "debug_info" in response and len(str(response)) > 50000:
            response["debug_info"] = {"summarized": True}

        return response

    def _compress_text_fields(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Compress long text fields"""
        # This would implement text compression if needed
        return response


class MCPResponseOptimizer:
    """Optimize MCP responses for performance and usability"""

    def __init__(self):
        from tool_search_response_cache import ToolSearchResponseCache

        self.cache = ToolSearchResponseCache()
        self.compressor = ResponseCompressor()

    def optimize_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Apply all optimization strategies to response"""

        # Check for cached version
        cache_key = generate_cache_key(response)
        if cached_response := self.cache.get(cache_key):
            cached_response["performance"]["cached"] = True
            return cached_response

        # Apply optimizations
        optimized = self._optimize_result_structure(response)
        optimized = self._optimize_metadata(optimized)
        optimized = self._add_progressive_disclosure(optimized)

        # Cache optimized response
        self.cache.set(cache_key, optimized)

        return optimized

    def _optimize_result_structure(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize result structure for efficient processing"""

        # Limit deep nesting for better JSON parsing
        if "results" in response:
            response["results"] = [self._flatten_result_structure(result) for result in response["results"]]

        # Create result summaries for large result sets
        if len(response.get("results", [])) > 10:
            response["result_summary"] = {
                "total_count": len(response["results"]),
                "top_categories": self._extract_top_categories(response["results"]),
                "confidence_summary": self._extract_confidence_summary(response["results"]),
            }

        return response

    def _flatten_result_structure(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Flatten deeply nested result structures"""
        flattened = {}

        for key, value in result.items():
            if isinstance(value, dict) and len(str(value)) > 1000:
                # Flatten large nested objects
                flattened[key] = self._summarize_large_object(value)
            else:
                flattened[key] = value

        return flattened

    def _summarize_large_object(self, obj: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize large nested objects"""
        summary = {
            "summarized": True,
            "original_keys": list(obj.keys())[:5],  # First 5 keys
            "total_keys": len(obj.keys()),
        }

        # Keep essential fields if they exist
        essential_fields = ["id", "name", "title", "description", "type", "category"]
        for field in essential_fields:
            if field in obj:
                summary[field] = obj[field]

        return summary

    def _extract_top_categories(self, results: List[Dict[str, Any]]) -> List[str]:
        """Extract top categories from results"""
        category_counts = defaultdict(int)

        for result in results:
            category = result.get("category") or result.get("type") or "uncategorized"
            category_counts[category] += 1

        # Return top 3 categories
        sorted_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
        return [cat for cat, count in sorted_categories[:3]]

    def _extract_confidence_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract confidence score summary"""
        scores = []

        for result in results:
            score = result.get("confidence_score") or result.get("relevance_score") or result.get("score")
            if score is not None:
                try:
                    scores.append(float(score))
                except (ValueError, TypeError):
                    continue

        if not scores:
            return {"available": False}

        return {
            "available": True,
            "average": sum(scores) / len(scores),
            "highest": max(scores),
            "lowest": min(scores),
            "total_scored_results": len(scores),
        }

    def _optimize_metadata(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize metadata for better processing"""

        # Ensure metadata is JSON serializable
        if "performance" in response:
            performance = response["performance"]
            for key, value in performance.items():
                if value is None or str(value) == "nan":
                    performance[key] = 0

        # Add processing hints for Claude Code
        response["processing_hints"] = {
            "format_version": "1.0",
            "recommended_parsing": "json",
            "large_response": len(str(response)) > 5000,
            "has_results": "results" in response or "recommendations" in response,
        }

        return response

    def _add_progressive_disclosure(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Add progressive disclosure structure for complex responses"""

        # Create summary view for complex responses
        if self._is_complex_response(response):
            response["summary_view"] = {
                "key_points": self._extract_key_points(response),
                "quick_recommendations": self._extract_quick_recommendations(response),
                "next_steps": self._extract_next_steps(response),
            }

            # Mark detailed sections
            response["detailed_sections"] = {
                "available": True,
                "sections": list(response.keys()),
                "usage_hint": "Access detailed sections as needed for deeper analysis",
            }

        return response

    def _is_complex_response(self, response: Dict[str, Any]) -> bool:
        """Check if response is complex enough to need progressive disclosure"""
        # Consider complex if:
        # - Has more than 10 results
        # - Response size is large
        # - Has multiple data sections

        result_count = len(response.get("results", []))
        if result_count > 10:
            return True

        response_size = len(str(response))
        if response_size > 20000:  # 20KB
            return True

        data_sections = sum(
            1 for key in response.keys() if key in ["results", "recommendations", "comparison_result", "details"]
        )
        if data_sections > 1:
            return True

        return False

    def _extract_key_points(self, response: Dict[str, Any]) -> List[str]:
        """Extract key points from response"""
        points = []

        # Add summary based on response type
        if "results" in response:
            result_count = len(response["results"])
            points.append(f"Found {result_count} matching tools")

            if result_count > 0:
                # Add category summary
                categories = self._extract_top_categories(response["results"])
                if categories:
                    points.append(f"Top categories: {', '.join(categories)}")

        elif "recommendations" in response:
            rec_count = len(response["recommendations"])
            points.append(f"Generated {rec_count} tool recommendations")

        elif "comparison_result" in response:
            points.append("Tool comparison analysis completed")

        # Add performance insight
        if response.get("performance", {}).get("cached"):
            points.append("Results retrieved from cache for faster response")

        return points[:3]  # Limit to top 3 key points

    def _extract_quick_recommendations(self, response: Dict[str, Any]) -> List[str]:
        """Extract quick recommendations from response"""
        recommendations = []

        # Get from navigation hints if available
        nav_suggestions = response.get("navigation", {}).get("follow_up_suggestions", [])
        recommendations.extend(nav_suggestions[:2])

        # Add general recommendations based on response type
        if "results" in response and len(response["results"]) > 5:
            recommendations.append("Consider using comparison tools for detailed analysis")

        return recommendations[:3]  # Limit to top 3

    def _extract_next_steps(self, response: Dict[str, Any]) -> List[str]:
        """Extract suggested next steps"""
        next_steps = []

        tool_name = response.get("tool", "")

        if tool_name == "search_mcp_tools":
            next_steps.extend(
                [
                    "Use get_tool_details() for specific tool information",
                    "Use compare_mcp_tools() to compare similar options",
                ]
            )
        elif tool_name == "recommend_tools_for_task":
            next_steps.extend(
                [
                    "Review detailed information for top recommendations",
                    "Compare recommended tools using comparison functions",
                ]
            )
        elif tool_name == "get_tool_details":
            next_steps.extend(["Search for similar tools if needed", "Compare with alternative tools"])

        # Add improvement suggestions
        improvement_hints = response.get("navigation", {}).get("improvement_hints", [])
        next_steps.extend(improvement_hints[:1])

        return next_steps[:3]  # Limit to top 3 next steps
