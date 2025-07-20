#!/usr/bin/env python3
"""
MCP Tool Search Response Types

This module provides structured response types specifically for MCP tool search operations,
enabling comprehensive JSON responses for tool discovery, details, categories, and capabilities.
"""

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import response_config
from mcp_metadata_types import ParameterAnalysis, ToolExample, ToolId
from tool_search_results import ToolSearchResult


@dataclass
class ToolSearchMCPResponse:
    """Response for search_mcp_tools MCP tool."""
    
    # Core search information
    query: str
    results: List[ToolSearchResult] = field(default_factory=list)
    search_mode: str = "hybrid"
    total_results: int = 0
    
    # Search metadata
    execution_time: Optional[float] = None
    search_strategy: str = "hybrid"
    confidence_distribution: Dict[str, int] = field(default_factory=dict)
    
    # User guidance
    query_suggestions: List[str] = field(default_factory=list)
    navigation_hints: List[str] = field(default_factory=list)
    category_breakdown: Dict[str, int] = field(default_factory=dict)
    
    # Response metadata
    timestamp: Optional[str] = None
    version: str = response_config.RESPONSE_VERSION
    
    def __post_init__(self):
        """Initialize computed fields after construction."""
        if self.timestamp is None:
            self.timestamp = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
        
        if self.total_results == 0:
            self.total_results = len(self.results)
            
        # Auto-compute confidence distribution if not provided
        if not self.confidence_distribution and self.results:
            self._compute_confidence_distribution()
            
        # Auto-compute category breakdown if not provided
        if not self.category_breakdown and self.results:
            self._compute_category_breakdown()
    
    def _compute_confidence_distribution(self):
        """Compute confidence level distribution from results."""
        self.confidence_distribution = {"high": 0, "medium": 0, "low": 0}
        for result in self.results:
            if result.confidence_level:
                self.confidence_distribution[result.confidence_level] += 1
                
    def _compute_category_breakdown(self):
        """Compute category distribution from results."""
        self.category_breakdown = {}
        for result in self.results:
            category = result.category or "unknown"
            self.category_breakdown[category] = self.category_breakdown.get(category, 0) + 1
    
    def add_suggestions(self, suggestions: List[str]) -> None:
        """Add query refinement suggestions."""
        self.query_suggestions.extend(suggestions)
        
    def add_navigation_hints(self, hints: List[str]) -> None:
        """Add navigation and usage hints."""
        self.navigation_hints.extend(hints)
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "query": self.query,
            "results": [result.to_dict() for result in self.results],
            "total_results": self.total_results,
            "search_mode": self.search_mode,
            "execution_time": self.execution_time,
            "search_strategy": self.search_strategy,
            "confidence_distribution": self.confidence_distribution,
            "category_breakdown": self.category_breakdown,
            "query_suggestions": self.query_suggestions,
            "navigation_hints": self.navigation_hints,
            "timestamp": self.timestamp,
            "version": self.version,
            "success": True
        }


@dataclass
class ToolDetailsResponse:
    """Response for get_tool_details MCP tool."""
    
    tool_id: str
    tool_details: ToolSearchResult
    included_sections: Dict[str, bool]
    
    # Additional detail sections
    parameter_schema: Optional[Dict[str, Any]] = None
    usage_examples: List[Dict[str, Any]] = field(default_factory=list)
    relationships: Dict[str, List[str]] = field(default_factory=dict)
    usage_guidance: Dict[str, List[str]] = field(default_factory=dict)
    
    # Metadata
    timestamp: Optional[str] = None
    version: str = response_config.RESPONSE_VERSION
    
    def __post_init__(self):
        """Initialize computed fields after construction."""
        if self.timestamp is None:
            self.timestamp = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "tool_id": self.tool_id,
            "tool_details": self.tool_details.to_dict(),
            "included_sections": self.included_sections,
            "timestamp": self.timestamp,
            "version": self.version,
            "success": True
        }
        
        # Add optional sections based on inclusion flags
        if self.included_sections.get("schema") and self.parameter_schema:
            result["parameter_schema"] = self.parameter_schema
            
        if self.included_sections.get("examples") and self.usage_examples:
            result["usage_examples"] = self.usage_examples
            
        if self.included_sections.get("relationships") and self.relationships:
            result["relationships"] = self.relationships
            
        if self.included_sections.get("usage_guidance") and self.usage_guidance:
            result["usage_guidance"] = self.usage_guidance
            
        return result


@dataclass
class ToolCategory:
    """Information about a tool category."""
    
    name: str
    description: str
    tool_count: int
    representative_tools: List[str] = field(default_factory=list)
    
    def add_description(self, description: str) -> None:
        """Add or update category description."""
        self.description = description
        
    def add_representative_tools(self, tools: List[str]) -> None:
        """Add representative tools for this category."""
        self.representative_tools = tools
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "tool_count": self.tool_count,
            "representative_tools": self.representative_tools
        }


@dataclass
class ToolCategoriesResponse:
    """Response for list_tool_categories MCP tool."""
    
    categories: List[ToolCategory] = field(default_factory=list)
    total_categories: int = 0
    total_tools: int = 0
    
    # Metadata
    timestamp: Optional[str] = None
    version: str = response_config.RESPONSE_VERSION
    
    def __post_init__(self):
        """Initialize computed fields after construction."""
        if self.timestamp is None:
            self.timestamp = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
            
        if self.total_categories == 0:
            self.total_categories = len(self.categories)
            
        if self.total_tools == 0:
            self.total_tools = sum(cat.tool_count for cat in self.categories)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "categories": [cat.to_dict() for cat in self.categories],
            "total_categories": self.total_categories,
            "total_tools": self.total_tools,
            "timestamp": self.timestamp,
            "version": self.version,
            "success": True
        }


@dataclass
class CapabilityMatch:
    """Information about how a tool matches capability requirements."""
    
    tool: ToolSearchResult
    capability_score: float
    parameter_match_score: float
    complexity_match_score: float
    overall_match_score: float
    match_explanation: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "tool": self.tool.to_dict(),
            "capability_score": self.capability_score,
            "parameter_match_score": self.parameter_match_score,
            "complexity_match_score": self.complexity_match_score,
            "overall_match_score": self.overall_match_score,
            "match_explanation": self.match_explanation
        }


@dataclass
class CapabilitySearchResponse:
    """Response for search_tools_by_capability MCP tool."""
    
    capability_description: str
    results: List[CapabilityMatch] = field(default_factory=list)
    total_results: int = 0
    
    # Search criteria
    search_criteria: Dict[str, Any] = field(default_factory=dict)
    
    # Execution metadata
    execution_time: Optional[float] = None
    
    # Metadata
    timestamp: Optional[str] = None
    version: str = response_config.RESPONSE_VERSION
    
    def __post_init__(self):
        """Initialize computed fields after construction."""
        if self.timestamp is None:
            self.timestamp = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
            
        if self.total_results == 0:
            self.total_results = len(self.results)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "capability_description": self.capability_description,
            "results": [result.to_dict() for result in self.results],
            "total_results": self.total_results,
            "search_criteria": self.search_criteria,
            "execution_time": self.execution_time,
            "timestamp": self.timestamp,
            "version": self.version,
            "success": True
        }


def create_error_response(tool_name: str, error_message: str, context: Optional[str] = None) -> Dict[str, Any]:
    """Create standardized error response for MCP tools."""
    return {
        "success": False,
        "error": {
            "tool": tool_name,
            "message": error_message,
            "context": context,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "suggestions": _generate_error_suggestions(tool_name, error_message)
        }
    }


def _generate_error_suggestions(tool_name: str, error_message: str) -> List[str]:
    """Generate helpful suggestions based on error message."""
    suggestions = []
    
    if "not found" in error_message.lower():
        suggestions.extend([
            "Check that the tool ID or name is spelled correctly",
            "Use search_mcp_tools to find available tools",
            "Verify that the tool catalog is up to date"
        ])
    elif "parameter" in error_message.lower():
        suggestions.extend([
            "Check parameter names and types",
            "Refer to tool documentation for valid parameter values",
            "Use get_tool_details to see parameter requirements"
        ])
    elif "query" in error_message.lower():
        suggestions.extend([
            "Try using different search terms",
            "Make your query more specific or more general",
            "Use natural language to describe what you're looking for"
        ])
    else:
        suggestions.append("Check the tool documentation for usage guidelines")
        
    return suggestions