#!/usr/bin/env python3
"""
Tool Registry for MCP Tool Management

This module provides a registry for managing discovered MCP tools,
including registration, health checking, and lifecycle management.
"""

import time
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from logging_config import get_logger

if TYPE_CHECKING:
    from mcp_tool_discovery import MCPTool

logger = get_logger(__name__)


class ToolRegistry:
    """
    Registry for managing discovered MCP tools.

    Provides tool registration, retrieval, health checking, and lifecycle
    management for all discovered tools.
    """

    def __init__(self):
        """Initialize the tool registry."""
        self._tools: Dict[str, "MCPTool"] = {}
        self._health_cache: Dict[str, Dict[str, Any]] = {}
        self._registration_times: Dict[str, float] = {}

        logger.info("Initialized Tool Registry")

    def register_tool(self, tool: "MCPTool") -> None:
        """
        Register a tool in the registry.

        Args:
            tool: MCPTool instance to register
        """
        self._tools[tool.id] = tool
        self._registration_times[tool.id] = time.time()

        # Clear any cached health info for this tool
        if tool.id in self._health_cache:
            del self._health_cache[tool.id]

        logger.debug("Registered tool: %s (%s)", tool.name, tool.id)

    def get_tool(self, tool_id: str) -> Optional["MCPTool"]:
        """
        Retrieve a tool by ID.

        Args:
            tool_id: ID of the tool to retrieve

        Returns:
            MCPTool instance or None if not found
        """
        return self._tools.get(tool_id)

    def get_all_tools(self) -> List["MCPTool"]:
        """
        Get all registered tools.

        Returns:
            List of all registered MCPTool instances
        """
        return list(self._tools.values())

    def get_tools_by_category(self, category: str) -> List["MCPTool"]:
        """
        Get tools filtered by category.

        Args:
            category: Category to filter by

        Returns:
            List of tools in the specified category
        """
        return [tool for tool in self._tools.values() if tool.category == category]

    def get_tools_by_type(self, tool_type: str) -> List["MCPTool"]:
        """
        Get tools filtered by type.

        Args:
            tool_type: Type to filter by ('system', 'custom', 'third_party')

        Returns:
            List of tools of the specified type
        """
        return [tool for tool in self._tools.values() if tool.tool_type == tool_type]

    def remove_tool(self, tool_id: str) -> bool:
        """
        Remove a tool from the registry.

        Args:
            tool_id: ID of the tool to remove

        Returns:
            True if tool was removed, False if not found
        """
        if tool_id in self._tools:
            tool_name = self._tools[tool_id].name
            del self._tools[tool_id]

            # Clean up related data
            if tool_id in self._health_cache:
                del self._health_cache[tool_id]
            if tool_id in self._registration_times:
                del self._registration_times[tool_id]

            logger.debug("Removed tool: %s (%s)", tool_name, tool_id)
            return True

        return False

    def check_tool_health(self, tool_id: str) -> Dict[str, Any]:
        """
        Check the health status of a tool.

        Args:
            tool_id: ID of the tool to check

        Returns:
            Dictionary with health status information
        """
        tool = self._tools.get(tool_id)
        if not tool:
            return {
                "status": "not_found",
                "is_healthy": False,
                "message": "Tool not found in registry",
                "checked_at": time.time(),
            }

        # Check if we have cached health info that's still valid
        if tool_id in self._health_cache:
            cached_health = self._health_cache[tool_id]
            cache_age = time.time() - cached_health.get("checked_at", 0)
            if cache_age < 300:  # Cache valid for 5 minutes
                return cached_health

        # Perform health check
        health_status = self._perform_health_check(tool)

        # Cache the result
        self._health_cache[tool_id] = health_status

        return health_status

    def _perform_health_check(self, tool: "MCPTool") -> Dict[str, Any]:
        """
        Perform actual health check on a tool.

        Args:
            tool: MCPTool instance to check

        Returns:
            Dictionary with health status
        """
        health_status = {
            "status": "healthy",
            "is_healthy": True,
            "message": "Tool is properly registered",
            "checked_at": time.time(),
            "tool_name": tool.name,
            "tool_type": tool.tool_type,
        }

        # Basic validation checks
        issues = []

        # Check required fields
        if not tool.name or not tool.description:
            issues.append("Missing required name or description")

        if not tool.category:
            issues.append("Missing tool category")

        # Check parameters
        for param in tool.parameters:
            if not param.name or not param.description:
                issues.append(f"Parameter {param.name} missing name or description")

        # Update health status based on issues
        if issues:
            health_status.update(
                {
                    "status": "degraded",
                    "is_healthy": False,
                    "message": f"Tool has validation issues: {'; '.join(issues)}",
                    "issues": issues,
                }
            )

        return health_status

    def get_registry_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the tool registry.

        Returns:
            Dictionary with registry statistics
        """
        stats = {
            "total_tools": len(self._tools),
            "registration_time": time.time(),
            "tools_by_type": {},
            "tools_by_category": {},
            "health_summary": {"healthy": 0, "degraded": 0, "unhealthy": 0},
        }

        # Count tools by type and category
        for tool in self._tools.values():
            # By type
            tool_type = tool.tool_type
            stats["tools_by_type"][tool_type] = stats["tools_by_type"].get(tool_type, 0) + 1

            # By category
            category = tool.category
            stats["tools_by_category"][category] = stats["tools_by_category"].get(category, 0) + 1

        # Health summary
        for tool_id in self._tools:
            health = self.check_tool_health(tool_id)
            if health["is_healthy"]:
                stats["health_summary"]["healthy"] += 1
            elif health["status"] == "degraded":
                stats["health_summary"]["degraded"] += 1
            else:
                stats["health_summary"]["unhealthy"] += 1

        return stats

    def cleanup_unhealthy_tools(self) -> int:
        """
        Remove tools that are no longer healthy.

        Returns:
            Number of tools removed
        """
        tools_to_remove = []

        for tool_id in self._tools:
            health = self.check_tool_health(tool_id)
            if health["status"] == "not_found" or not health["is_healthy"]:
                # Only remove if critically unhealthy, not just degraded
                if health["status"] in ["not_found", "error"]:
                    tools_to_remove.append(tool_id)

        removed_count = 0
        for tool_id in tools_to_remove:
            if self.remove_tool(tool_id):
                removed_count += 1

        if removed_count > 0:
            logger.info("Cleaned up %d unhealthy tools from registry", removed_count)

        return removed_count

    def validate_all_tools(self) -> Dict[str, Any]:
        """
        Validate all tools in the registry.

        Returns:
            Dictionary with validation results
        """
        validation_results = {"valid_tools": 0, "invalid_tools": 0, "issues": [], "validated_at": time.time()}

        for tool_id, tool in self._tools.items():
            health = self.check_tool_health(tool_id)

            if health["is_healthy"]:
                validation_results["valid_tools"] += 1
            else:
                validation_results["invalid_tools"] += 1
                validation_results["issues"].append(
                    {
                        "tool_id": tool_id,
                        "tool_name": tool.name,
                        "status": health["status"],
                        "message": health["message"],
                        "issues": health.get("issues", []),
                    }
                )

        return validation_results

    def clear_health_cache(self) -> None:
        """Clear all cached health information."""
        self._health_cache.clear()
        logger.debug("Cleared tool health cache")

    def refresh_all_health_checks(self) -> Dict[str, Any]:
        """
        Refresh health checks for all tools.

        Returns:
            Dictionary with updated health information
        """
        self.clear_health_cache()

        health_results = {}
        for tool_id in self._tools:
            health_results[tool_id] = self.check_tool_health(tool_id)

        logger.info("Refreshed health checks for %d tools", len(self._tools))
        return health_results
