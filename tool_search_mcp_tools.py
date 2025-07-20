#!/usr/bin/env python3
"""
Tool Search MCP Tools

This module implements the core MCP tools for tool search functionality,
providing natural language tool discovery, detailed tool inspection,
category browsing, and capability matching.
"""

import json
import time
from typing import Any, Dict, List, Optional

from database_manager import DatabaseManager
from embedding_helper import EmbeddingGenerator
from logging_config import get_logger
from mcp_metadata_types import ParameterAnalysis, ToolExample, ToolId
from mcp_tool_search_engine import MCPToolSearchEngine
from mcp_tool_search_responses import (
    CapabilityMatch,
    CapabilitySearchResponse,
    ToolCategory,
    ToolCategoriesResponse,
    ToolDetailsResponse,
    ToolSearchMCPResponse,
    create_error_response,
)
from mcp_tool_validator import (
    MCPToolValidator,
    ValidatedCapabilityParams,
    ValidatedSearchParams,
    ValidatedToolDetailsParams,
    tool_exists,
)
from parameter_search_engine import ParameterSearchEngine
from tool_search_results import ToolSearchResult

logger = get_logger(__name__)

# Global instances that will be initialized
_tool_search_engine: Optional[MCPToolSearchEngine] = None
_parameter_search_engine: Optional[ParameterSearchEngine] = None
_validator: Optional[MCPToolValidator] = None
_db_manager: Optional[DatabaseManager] = None


def initialize_search_engines(
    db_manager: DatabaseManager,
    embedding_generator: EmbeddingGenerator
) -> None:
    """
    Initialize the search engines with database and embedding components.
    
    This function should be called once during server startup to initialize
    the global search engine instances.
    
    Args:
        db_manager: Database manager instance
        embedding_generator: Embedding generator instance
    """
    global _tool_search_engine, _parameter_search_engine, _validator, _db_manager
    
    _db_manager = db_manager
    _tool_search_engine = MCPToolSearchEngine(db_manager, embedding_generator)
    _parameter_search_engine = ParameterSearchEngine(_tool_search_engine)
    _validator = MCPToolValidator()
    
    logger.info("Initialized tool search MCP components")


def _get_search_engine() -> MCPToolSearchEngine:
    """Get the initialized tool search engine."""
    if _tool_search_engine is None:
        raise RuntimeError("Tool search engines not initialized. Call initialize_search_engines() first.")
    return _tool_search_engine


def _get_parameter_search_engine() -> ParameterSearchEngine:
    """Get the initialized parameter search engine."""
    if _parameter_search_engine is None:
        raise RuntimeError("Parameter search engine not initialized. Call initialize_search_engines() first.")
    return _parameter_search_engine


def _get_validator() -> MCPToolValidator:
    """Get the initialized validator."""
    if _validator is None:
        raise RuntimeError("Validator not initialized. Call initialize_search_engines() first.")
    return _validator


def _get_db_manager() -> DatabaseManager:
    """Get the initialized database manager."""
    if _db_manager is None:
        raise RuntimeError("Database manager not initialized. Call initialize_search_engines() first.")
    return _db_manager


def search_mcp_tools(
    query: str,
    category: Optional[str] = None,
    tool_type: Optional[str] = None,
    max_results: int = 10,
    include_examples: bool = True,
    search_mode: str = "hybrid"
) -> dict:
    """
    Search for MCP tools by functionality or description.
    
    This tool enables Claude to discover available MCP tools using natural language
    descriptions. It combines semantic search with keyword matching to find tools
    that match functional requirements.
    
    Args:
        query: Natural language description of desired functionality
               Examples: "file operations with error handling", "execute shell commands"
        category: Optional filter by tool category (file_ops, web, analysis, etc.)
        tool_type: Optional filter by tool type (system, custom, third_party)  
        max_results: Maximum number of tools to return (1-50)
        include_examples: Whether to include usage examples in results
        search_mode: Search strategy ('semantic', 'hybrid', 'keyword')
        
    Returns:
        Structured JSON with tool search results, metadata, and suggestions
        
    Examples:
        search_mcp_tools("find tools for reading files")
        search_mcp_tools("web scraping tools", category="web")
        search_mcp_tools("command execution", tool_type="system", max_results=5)
    """
    try:
        logger.info("Starting MCP tool search for query: '%s'", query)
        start_time = time.time()
        
        # Validate and process query parameters
        validator = _get_validator()
        validated_params = validator.validate_search_parameters(
            query, category, tool_type, max_results, include_examples, search_mode
        )
        
        # Execute search using the search engine
        search_engine = _get_search_engine()
        
        if search_mode == "hybrid":
            search_results = search_engine.search_hybrid(
                query=validated_params.query,
                k=validated_params.max_results,
                semantic_weight=0.7,
                keyword_weight=0.3
            )
        else:
            search_results = search_engine.search_by_functionality(
                query=validated_params.query,
                k=validated_params.max_results,
                category_filter=validated_params.category,
                tool_type_filter=validated_params.tool_type
            )
        
        # Create structured response
        response = ToolSearchMCPResponse(
            query=query,
            results=search_results.results,
            search_mode=search_mode,
            execution_time=time.time() - start_time,
            search_strategy=search_results.search_strategy
        )
        
        # Add suggestions and metadata
        response.add_suggestions(generate_query_suggestions(query, search_results.results))
        response.add_navigation_hints(generate_navigation_hints(search_results.results))
        
        logger.info("MCP tool search completed: %d results in %.3fs", len(response.results), response.execution_time)
        
        return json.dumps(response.to_dict())
        
    except Exception as e:
        logger.error("Error in search_mcp_tools: %s", e)
        return json.dumps(create_error_response("search_mcp_tools", str(e), query))


def get_tool_details(
    tool_id: str,
    include_schema: bool = True,
    include_examples: bool = True,
    include_relationships: bool = True,
    include_usage_guidance: bool = True
) -> dict:
    """
    Get comprehensive information about a specific MCP tool.
    
    This tool provides detailed information about a specific MCP tool including
    its parameters, usage examples, relationships with other tools, and best practices.
    
    Args:
        tool_id: Identifier of the tool to inspect (e.g., 'bash', 'read', 'search_code')
        include_schema: Include full parameter schema and type information
        include_examples: Include usage examples and code snippets
        include_relationships: Include alternative and complementary tools
        include_usage_guidance: Include best practices and common pitfalls
        
    Returns:
        Comprehensive tool documentation and metadata
        
    Examples:
        get_tool_details("bash")
        get_tool_details("read", include_schema=False)
        get_tool_details("search_code", include_relationships=False)
    """
    try:
        logger.info("Getting tool details for: %s", tool_id)
        start_time = time.time()
        
        # Validate tool existence
        if not tool_exists(tool_id):
            return json.dumps(create_error_response("get_tool_details", f"Tool '{tool_id}' not found", tool_id))
        
        # Validate parameters
        validator = _get_validator()
        validated_params = validator.validate_tool_details_parameters(
            tool_id, include_schema, include_examples, include_relationships, include_usage_guidance
        )
        
        # Load comprehensive tool information
        tool_details = load_tool_details(
            validated_params.tool_id, 
            validated_params.include_schema, 
            validated_params.include_examples, 
            validated_params.include_relationships,
            validated_params.include_usage_guidance
        )
        
        # Create detailed response
        response = ToolDetailsResponse(
            tool_id=tool_id,
            tool_details=tool_details,
            included_sections={
                "schema": include_schema,
                "examples": include_examples, 
                "relationships": include_relationships,
                "usage_guidance": include_usage_guidance
            }
        )
        
        # Add optional sections based on request
        if include_schema:
            response.parameter_schema = _generate_parameter_schema(tool_details)
            
        if include_examples:
            response.usage_examples = _format_usage_examples(tool_details.examples)
            
        if include_relationships:
            response.relationships = _get_tool_relationships(ToolId(tool_id))
            
        if include_usage_guidance:
            response.usage_guidance = _generate_usage_guidance(tool_details)
        
        logger.info("Tool details retrieved for %s in %.3fs", tool_id, time.time() - start_time)
        
        return json.dumps(response.to_dict())
        
    except Exception as e:
        logger.error("Error in get_tool_details: %s", e)
        return json.dumps(create_error_response("get_tool_details", str(e), tool_id))


def list_tool_categories() -> dict:
    """
    Get overview of available tool categories and their contents.
    
    This tool provides a structured overview of all available tool categories,
    helping Claude understand the organization of available tools and browse
    by functional area.
    
    Returns:
        Dictionary of categories with tool counts, descriptions, and representative tools
        
    Examples:
        list_tool_categories()
    """
    try:
        logger.info("Loading tool categories overview")
        start_time = time.time()
        
        # Load category information from catalog
        categories = load_tool_categories()
        
        # Create category overview response
        response = ToolCategoriesResponse(categories=categories)
        
        # Add category descriptions and representative tools
        for category in response.categories:
            category.add_description(get_category_description(category.name))
            category.add_representative_tools(get_representative_tools(category.name, limit=3))
        
        logger.info("Tool categories loaded: %d categories in %.3fs", len(categories), time.time() - start_time)
        
        return json.dumps(response.to_dict())
        
    except Exception as e:
        logger.error("Error in list_tool_categories: %s", e)
        return json.dumps(create_error_response("list_tool_categories", str(e)))


def search_tools_by_capability(
    capability_description: str,
    required_parameters: Optional[List[str]] = None,
    preferred_complexity: str = "any",
    max_results: int = 10
) -> dict:
    """
    Search for tools by specific capability requirements.
    
    This tool finds tools that have specific capabilities or parameter requirements,
    enabling precise tool matching for technical requirements.
    
    Args:
        capability_description: Description of required capability
        required_parameters: List of parameter names that must be supported  
        preferred_complexity: Complexity preference ('simple', 'moderate', 'complex', 'any')
        max_results: Maximum number of tools to return
        
    Returns:
        Tools matching capability requirements with explanations
        
    Examples:
        search_tools_by_capability("timeout support")
        search_tools_by_capability("file path handling", required_parameters=["file_path"])
        search_tools_by_capability("error handling", preferred_complexity="simple")
    """
    try:
        logger.info("Starting capability search: %s", capability_description)
        start_time = time.time()
        
        # Validate parameters
        validator = _get_validator()
        validated_params = validator.validate_capability_parameters(
            capability_description, required_parameters, preferred_complexity, max_results
        )
        
        # Search by capability using parameter-aware search
        parameter_search_engine = _get_parameter_search_engine()
        
        # Convert complexity preference
        from parameter_ranking import ComplexityPreference
        complexity_map = {
            "simple": ComplexityPreference.SIMPLE,
            "moderate": ComplexityPreference.MODERATE,
            "complex": ComplexityPreference.COMPLEX,
            "any": ComplexityPreference.ANY
        }
        complexity_pref = complexity_map.get(validated_params.preferred_complexity, ComplexityPreference.ANY)
        
        capability_results = parameter_search_engine.search_by_parameters(
            required_parameters=validated_params.required_parameters,
            complexity_preference=complexity_pref,
            k=validated_params.max_results
        )
        
        # Convert results to capability matches
        capability_matches = []
        for tool_result in capability_results:
            capability_match = CapabilityMatch(
                tool=tool_result,
                capability_score=tool_result.relevance_score,
                parameter_match_score=_calculate_parameter_match_score(tool_result, validated_params.required_parameters),
                complexity_match_score=_calculate_complexity_match_score(tool_result, complexity_pref),
                overall_match_score=tool_result.relevance_score,
                match_explanation=_generate_capability_match_explanation(tool_result, validated_params)
            )
            capability_matches.append(capability_match)
        
        # Create capability search response
        response = CapabilitySearchResponse(
            capability_description=capability_description,
            results=capability_matches,
            execution_time=time.time() - start_time,
            search_criteria={
                "required_parameters": validated_params.required_parameters,
                "preferred_complexity": validated_params.preferred_complexity,
                "max_results": validated_params.max_results
            }
        )
        
        logger.info("Capability search completed: %d results in %.3fs", 
                   len(capability_matches), response.execution_time)
        
        return json.dumps(response.to_dict())
        
    except Exception as e:
        logger.error("Error in search_tools_by_capability: %s", e)
        return json.dumps(create_error_response("search_tools_by_capability", str(e), capability_description))


# Helper functions

def load_tool_details(
    tool_id: str,
    include_schema: bool,
    include_examples: bool,
    include_relationships: bool,
    include_usage_guidance: bool
) -> ToolSearchResult:
    """Load comprehensive tool details from the database."""
    # This is a simplified implementation - would load from actual database
    # For now, create a basic ToolSearchResult with available information
    
    db_manager = _get_db_manager()
    
    # Get basic tool information
    try:
        tool_data = db_manager.get_mcp_tool(tool_id)
        if not tool_data:
            raise ValueError(f"Tool {tool_id} not found in database")
        
        # Create ToolSearchResult with basic information
        tool_result = ToolSearchResult(
            tool_id=ToolId(tool_id),
            name=tool_data.get("name", tool_id),
            description=tool_data.get("description", ""),
            category=tool_data.get("category", "unknown"),
            tool_type=tool_data.get("tool_type", "unknown"),
            similarity_score=1.0,  # Perfect match for direct lookup
            relevance_score=1.0,
            confidence_level="high"
        )
        
        # Add parameters if available
        if include_schema:
            parameters = db_manager.get_tool_parameters(ToolId(tool_id))
            tool_result.parameters = [
                ParameterAnalysis(
                    name=p["parameter_name"],
                    type=p["parameter_type"] or "string",
                    required=p["is_required"],
                    description=p["description"] or ""
                ) for p in parameters
            ]
        
        # Add examples if available
        if include_examples:
            examples = db_manager.get_tool_examples(ToolId(tool_id))
            tool_result.examples = [
                ToolExample(
                    use_case=ex["use_case"] or "",
                    example_call=ex["example_call"] or "",
                    expected_output=ex["expected_output"] or "",
                    context=ex["context"] or "",
                    effectiveness_score=ex["effectiveness_score"] or 0.8
                ) for ex in examples
            ]
        
        return tool_result
        
    except Exception as e:
        logger.error("Error loading tool details for %s: %s", tool_id, e)
        # Return a basic result with error information
        return ToolSearchResult(
            tool_id=ToolId(tool_id),
            name=tool_id,
            description="Tool details not available",
            category="unknown",
            tool_type="unknown",
            similarity_score=0.0,
            relevance_score=0.0,
            confidence_level="low",
            match_reasons=[f"Error loading details: {e}"]
        )


def load_tool_categories() -> List[ToolCategory]:
    """Load available tool categories from the database."""
    try:
        db_manager = _get_db_manager()
        
        with db_manager.get_connection() as conn:
            # Get category statistics from database
            category_stats = conn.execute("""
                SELECT category, COUNT(*) as tool_count
                FROM mcp_tools
                GROUP BY category
                ORDER BY tool_count DESC
            """).fetchall()
            
            categories = []
            for category_name, tool_count in category_stats:
                category = ToolCategory(
                    name=category_name or "unknown",
                    description="",  # Will be populated by get_category_description
                    tool_count=tool_count
                )
                categories.append(category)
            
            return categories
            
    except Exception as e:
        logger.error("Error loading tool categories: %s", e)
        # Return default categories if database query fails
        return [
            ToolCategory(name="file_ops", description="File operations and management", tool_count=0),
            ToolCategory(name="execution", description="Command and script execution", tool_count=0),
            ToolCategory(name="search", description="Search and discovery tools", tool_count=0),
            ToolCategory(name="web", description="Web and network operations", tool_count=0),
            ToolCategory(name="analysis", description="Data analysis and processing", tool_count=0),
        ]


def get_category_description(category_name: str) -> str:
    """Get description for a tool category."""
    descriptions = {
        "file_ops": "Tools for reading, writing, and manipulating files and directories",
        "execution": "Tools for executing commands, scripts, and system operations",
        "search": "Tools for searching and discovering content across various sources",
        "web": "Tools for web scraping, HTTP requests, and internet operations",
        "analysis": "Tools for analyzing data, code, and system information",
        "development": "Tools for software development and debugging",
        "notebook": "Tools for working with Jupyter notebooks and interactive documents",
        "workflow": "Tools for automating and managing complex workflows",
        "data": "Tools for data processing, transformation, and manipulation",
        "system": "Tools for system administration and monitoring",
        "unknown": "Uncategorized tools with various functionalities"
    }
    return descriptions.get(category_name, f"Tools in the {category_name} category")


def get_representative_tools(category_name: str, limit: int = 3) -> List[str]:
    """Get representative tools for a category."""
    try:
        db_manager = _get_db_manager()
        
        with db_manager.get_connection() as conn:
            # Get most common tools in this category
            tools = conn.execute("""
                SELECT name
                FROM mcp_tools
                WHERE category = ?
                ORDER BY name
                LIMIT ?
            """, (category_name, limit)).fetchall()
            
            return [tool[0] for tool in tools]
            
    except Exception as e:
        logger.error("Error getting representative tools for %s: %s", category_name, e)
        return []


def generate_query_suggestions(query: str, results: List[ToolSearchResult]) -> List[str]:
    """Generate query refinement suggestions based on results."""
    suggestions = []
    
    if not results:
        suggestions.extend([
            f"Try broader terms related to '{query}'",
            "Check if the tool catalog is up to date",
            "Consider using synonyms or related technical terms"
        ])
    elif len(results) < 3:
        suggestions.extend([
            f"Try more specific terms to refine '{query}'",
            "Add programming language or framework context",
            "Include related function or parameter names"
        ])
    elif len(results) > 15:
        suggestions.extend([
            f"Add more specific context to '{query}' to narrow results",
            "Include tool type or category constraints",
            "Focus on particular implementation patterns"
        ])
    
    return suggestions


def generate_navigation_hints(results: List[ToolSearchResult]) -> List[str]:
    """Generate navigation hints for tool usage."""
    hints = []
    
    if results:
        hints.append(f"Found {len(results)} matching tools")
        
        # Check for different complexity levels
        simple_tools = sum(1 for r in results if r.is_simple_tool())
        complex_tools = sum(1 for r in results if r.is_complex_tool())
        
        if simple_tools > 0:
            hints.append(f"{simple_tools} tools are simple to use with few parameters")
        if complex_tools > 0:
            hints.append(f"{complex_tools} tools offer advanced functionality with more parameters")
            
        # Check for tools with good examples
        tools_with_examples = sum(1 for r in results if r.has_good_examples())
        if tools_with_examples > 0:
            hints.append(f"{tools_with_examples} tools have detailed usage examples")
    
    return hints


def _generate_parameter_schema(tool_details: ToolSearchResult) -> Dict[str, Any]:
    """Generate JSON schema for tool parameters."""
    schema = {
        "type": "object",
        "properties": {},
        "required": []
    }
    
    for param in tool_details.parameters:
        param_schema = {
            "type": param.type,
            "description": param.description
        }
        
        if param.default_value is not None:
            param_schema["default"] = param.default_value
            
        if param.examples:
            param_schema["examples"] = param.examples
            
        if param.constraints:
            param_schema.update(param.constraints)
            
        schema["properties"][param.name] = param_schema
        
        if param.required:
            schema["required"].append(param.name)
    
    return schema


def _format_usage_examples(examples: List[ToolExample]) -> List[Dict[str, Any]]:
    """Format tool examples for response."""
    return [example.to_dict() for example in examples]


def _get_tool_relationships(tool_id: ToolId) -> Dict[str, List[str]]:
    """Get relationships for a tool."""
    try:
        db_manager = _get_db_manager()
        
        with db_manager.get_connection() as conn:
            # Get relationships from database
            relationships = conn.execute("""
                SELECT relationship_type, related_tool_id
                FROM tool_relationships
                WHERE tool_id = ?
            """, (str(tool_id),)).fetchall()
            
            result = {}
            for rel_type, related_id in relationships:
                if rel_type not in result:
                    result[rel_type] = []
                result[rel_type].append(related_id)
                
            return result
            
    except Exception as e:
        logger.error("Error getting relationships for %s: %s", tool_id, e)
        return {}


def _generate_usage_guidance(tool_details: ToolSearchResult) -> Dict[str, List[str]]:
    """Generate usage guidance for a tool."""
    guidance = {
        "best_practices": [],
        "common_pitfalls": [],
        "tips": []
    }
    
    # Add guidance based on tool characteristics
    if tool_details.is_simple_tool():
        guidance["tips"].append("This is a simple tool with minimal parameters")
        guidance["best_practices"].append("Ideal for quick operations and scripting")
    elif tool_details.is_complex_tool():
        guidance["tips"].append("This is a complex tool with many configuration options")
        guidance["best_practices"].append("Review all parameters before use")
        guidance["common_pitfalls"].append("Don't forget to set required parameters")
    
    if tool_details.has_good_examples():
        guidance["tips"].append("Check the examples for common usage patterns")
    
    return guidance


def _calculate_parameter_match_score(tool_result: ToolSearchResult, required_params: List[str]) -> float:
    """Calculate how well tool parameters match requirements."""
    if not required_params:
        return 1.0
        
    tool_param_names = {p.name.lower() for p in tool_result.parameters}
    matches = sum(1 for param in required_params if param.lower() in tool_param_names)
    
    return matches / len(required_params) if required_params else 1.0


def _calculate_complexity_match_score(tool_result: ToolSearchResult, complexity_pref) -> float:
    """Calculate how well tool complexity matches preference."""
    tool_complexity = tool_result.complexity_score
    
    if complexity_pref.value == "any":
        return 1.0
    elif complexity_pref.value == "simple":
        return 1.0 - tool_complexity  # Lower complexity is better
    elif complexity_pref.value == "moderate":
        return 1.0 - abs(tool_complexity - 0.5) * 2  # Closer to 0.5 is better
    elif complexity_pref.value == "complex":
        return tool_complexity  # Higher complexity is better
    else:
        return 0.5


def _generate_capability_match_explanation(tool_result: ToolSearchResult, params: ValidatedCapabilityParams) -> str:
    """Generate explanation for why tool matches capability requirements."""
    explanations = []
    
    # Parameter match explanation
    if params.required_parameters:
        param_score = _calculate_parameter_match_score(tool_result, params.required_parameters)
        if param_score > 0.8:
            explanations.append("Strong parameter match")
        elif param_score > 0.5:
            explanations.append("Partial parameter match")
        else:
            explanations.append("Limited parameter match")
    
    # Complexity explanation
    if params.preferred_complexity != "any":
        if tool_result.is_simple_tool() and params.preferred_complexity == "simple":
            explanations.append("Matches simple complexity preference")
        elif tool_result.is_complex_tool() and params.preferred_complexity == "complex":
            explanations.append("Matches complex tool preference")
        elif params.preferred_complexity == "moderate" and not tool_result.is_simple_tool() and not tool_result.is_complex_tool():
            explanations.append("Matches moderate complexity preference")
    
    return "; ".join(explanations) if explanations else "General capability match"