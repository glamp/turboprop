# MCP Tools Reference - Complete Tool Catalog

This document provides a comprehensive reference for all MCP tools available through the MCP Tool Search System, including their capabilities, parameters, and usage patterns.

## Table of Contents

### MCP Tools
- [Core Search Tools](#core-search-tools)
  - [search_mcp_tools](#search_mcp_tools)
  - [recommend_tools_for_task](#recommend_tools_for_task)
  - [compare_mcp_tools](#compare_mcp_tools)
- [System Tools](#system-tools)
  - [get_tool_details](#get_tool_details)
  - [list_tool_categories](#list_tool_categories)
  - [analyze_task_requirements](#analyze_task_requirements)
  - [find_tool_alternatives](#find_tool_alternatives)

### Tool Categories
- [Tool Categories](#tool-categories)
  - [file_ops (File Operations)](#file_ops-file-operations)
  - [execution (Command Execution)](#execution-command-execution)
  - [web (Web Operations)](#web-web-operations)
  - [data (Data Processing)](#data-data-processing)
  - [analysis (Code Analysis)](#analysis-code-analysis)
  - [system (System Operations)](#system-system-operations)

### Performance and Operations
- [Performance Profiles](#performance-profiles)
  - [Response Time Expectations](#response-time-expectations)
  - [Scalability Characteristics](#scalability-characteristics)
  - [Optimization Recommendations](#optimization-recommendations)
- [Error Codes and Troubleshooting](#error-codes-and-troubleshooting)
  - [Common Error Codes](#common-error-codes)
  - [Performance Troubleshooting](#performance-troubleshooting)
- [Integration Patterns](#integration-patterns)
  - [Basic Integration](#basic-integration)
  - [Advanced Integration](#advanced-integration)
  - [Error Handling Pattern](#error-handling-pattern)

---

## Core Search Tools

### search_mcp_tools

**Description**: Search for MCP tools using natural language queries with multiple search strategies.

**Category**: Core Search  
**Complexity**: Low  
**Performance**: Fast (< 500ms typical)

**Parameters**:
```json
{
  "query": {
    "type": "string",
    "required": true,
    "description": "Natural language description of desired functionality",
    "examples": ["read configuration files", "execute shell commands", "web scraping tools"]
  },
  "category": {
    "type": "string",
    "required": false,
    "description": "Filter by tool category",
    "enum": ["file_ops", "web", "execution", "analysis", "data", "system"],
    "default": null
  },
  "tool_type": {
    "type": "string", 
    "required": false,
    "description": "Filter by tool type",
    "enum": ["system", "custom", "third_party"],
    "default": null
  },
  "max_results": {
    "type": "integer",
    "required": false,
    "description": "Maximum number of tools to return",
    "minimum": 1,
    "maximum": 50,
    "default": 10
  },
  "include_examples": {
    "type": "boolean",
    "required": false,
    "description": "Whether to include usage examples",
    "default": true
  },
  "search_mode": {
    "type": "string",
    "required": false,
    "description": "Search strategy to use",
    "enum": ["semantic", "hybrid", "keyword"],
    "default": "hybrid"
  }
}
```

**Response Schema**:
```json
{
  "success": true,
  "query": "string",
  "results": [
    {
      "tool_id": "string",
      "name": "string",
      "description": "string",
      "similarity_score": "number (0.0-1.0)",
      "confidence_level": "string (low|medium|high)",
      "match_reasons": ["string"],
      "parameters": [{"name": "string", "type": "string", "required": "boolean"}],
      "examples": [{"title": "string", "code": "string"}],
      "alternatives": ["string"],
      "complexity_score": "number (0.0-1.0)",
      "when_to_use": "string"
    }
  ],
  "total_results": "integer",
  "execution_time": "number",
  "query_suggestions": ["string"],
  "category_breakdown": {"category": "count"},
  "timestamp": "string (ISO 8601)"
}
```

**Usage Examples**:
```python
# Basic search
search_mcp_tools("file operations")

# Filtered search  
search_mcp_tools("web scraping", category="web", max_results=5)

# Semantic-only search
search_mcp_tools("data transformation", search_mode="semantic")

# Comprehensive search
search_mcp_tools(
    query="secure file transfer", 
    category="file_ops",
    include_examples=True,
    max_results=15
)
```

**Performance Characteristics**:
- Typical response time: 200-500ms
- Cache hit rate: ~75% for common queries
- Memory usage: ~50MB per 1000 tools indexed
- Concurrent capacity: 100+ simultaneous searches

---

### recommend_tools_for_task

**Description**: Get intelligent tool recommendations tailored to specific development tasks with contextual reasoning.

**Category**: Recommendation  
**Complexity**: Medium  
**Performance**: Medium (< 1s typical)

**Parameters**:
```json
{
  "task_description": {
    "type": "string",
    "required": true,
    "description": "Natural language description of the task to accomplish",
    "examples": [
      "read configuration files safely", 
      "process CSV data and generate reports",
      "deploy application with monitoring"
    ]
  },
  "context": {
    "type": "string",
    "required": false,
    "description": "Additional context about environment or constraints",
    "examples": [
      "Python environment, performance critical",
      "beginner user, safety first",
      "enterprise environment, audit logging required"
    ],
    "default": null
  },
  "max_recommendations": {
    "type": "integer",
    "required": false,
    "description": "Maximum number of recommendations to return",
    "minimum": 1,
    "maximum": 10,
    "default": 5
  },
  "include_alternatives": {
    "type": "boolean",
    "required": false,
    "description": "Include alternative tool options",
    "default": true
  },
  "complexity_preference": {
    "type": "string",
    "required": false,
    "description": "Preference for tool complexity level",
    "enum": ["simple", "balanced", "powerful"],
    "default": "balanced"
  },
  "explain_reasoning": {
    "type": "boolean",
    "required": false,
    "description": "Include detailed explanations for recommendations",
    "default": true
  }
}
```

**Response Schema**:
```json
{
  "success": true,
  "task_description": "string",
  "recommendations": [
    {
      "tool": {
        "tool_id": "string",
        "name": "string", 
        "description": "string"
      },
      "recommendation_score": "number (0.0-1.0)",
      "confidence_level": "string (low|medium|high)",
      "task_alignment": "number (0.0-1.0)",
      "capability_match": "number (0.0-1.0)",
      "complexity_alignment": "number (0.0-1.0)",
      "recommendation_reasons": ["string"],
      "usage_guidance": ["string"],
      "when_to_use": "string",
      "alternative_tools": ["string"]
    }
  ],
  "task_analysis": {
    "task_category": "string",
    "complexity_level": "string (simple|medium|complex)",
    "required_capabilities": ["string"],
    "confidence": "number (0.0-1.0)"
  },
  "explanations": ["string"]
}
```

---

### compare_mcp_tools

**Description**: Compare multiple MCP tools across various dimensions to support decision-making.

**Category**: Analysis  
**Complexity**: Medium  
**Performance**: Medium (< 2s typical)

**Parameters**:
```json
{
  "tool_ids": {
    "type": "array",
    "items": {"type": "string"},
    "required": true,
    "description": "List of tool IDs to compare",
    "minItems": 2,
    "maxItems": 10,
    "examples": [["read", "write", "edit"], ["bash", "task"]]
  },
  "comparison_criteria": {
    "type": "array", 
    "items": {"type": "string"},
    "required": false,
    "description": "Specific aspects to compare",
    "enum": [
      "functionality", "usability", "performance", 
      "reliability", "complexity", "documentation"
    ],
    "default": ["functionality", "usability", "reliability"]
  },
  "include_decision_guidance": {
    "type": "boolean",
    "required": false,
    "description": "Include selection recommendations",
    "default": true
  },
  "comparison_context": {
    "type": "string",
    "required": false,
    "description": "Context for comparison (affects scoring)",
    "examples": ["automation scripts", "beginner users", "production environment"],
    "default": null
  },
  "detail_level": {
    "type": "string",
    "required": false,
    "description": "Level of detail in comparison",
    "enum": ["basic", "standard", "comprehensive"],
    "default": "standard"
  }
}
```

---

## System Tools

### get_tool_details

**Description**: Get comprehensive information about a specific MCP tool including documentation and examples.

**Category**: System  
**Complexity**: Low  
**Performance**: Fast (< 200ms typical)

**Parameters**:
```json
{
  "tool_id": {
    "type": "string",
    "required": true,
    "description": "ID of the tool to get details for",
    "examples": ["read", "bash", "search_code"]
  },
  "include_examples": {
    "type": "boolean", 
    "required": false,
    "description": "Include usage examples",
    "default": true
  },
  "include_alternatives": {
    "type": "boolean",
    "required": false,
    "description": "Include alternative tools",
    "default": true
  },
  "include_usage_stats": {
    "type": "boolean",
    "required": false,
    "description": "Include usage statistics (if available)",
    "default": false
  }
}
```

---

### list_tool_categories

**Description**: List all available tool categories with descriptions and tool counts.

**Category**: System  
**Complexity**: Low  
**Performance**: Very Fast (< 100ms)

**Parameters**:
```json
{
  "include_counts": {
    "type": "boolean",
    "required": false,
    "description": "Include tool counts for each category",
    "default": true
  },
  "include_descriptions": {
    "type": "boolean",
    "required": false,
    "description": "Include category descriptions",
    "default": true
  }
}
```

---

### analyze_task_requirements

**Description**: Analyze a task description to understand requirements and suggest appropriate capabilities.

**Category**: Analysis  
**Complexity**: Medium  
**Performance**: Medium (< 1s typical)

**Parameters**:
```json
{
  "task_description": {
    "type": "string",
    "required": true,
    "description": "Natural language description of the task",
    "examples": [
      "build and deploy microservice application",
      "process large CSV files with error handling",
      "secure user authentication system"
    ]
  },
  "detail_level": {
    "type": "string",
    "required": false,
    "description": "Analysis depth and detail",
    "enum": ["basic", "standard", "comprehensive"],
    "default": "standard"
  }
}
```

---

### find_tool_alternatives

**Description**: Find alternative tools for a given tool or functionality with similarity analysis.

**Category**: Discovery  
**Complexity**: Medium  
**Performance**: Medium (< 1s typical)

**Parameters**:
```json
{
  "reference_tool": {
    "type": "string",
    "required": true,
    "description": "Tool ID to find alternatives for",
    "examples": ["bash", "read", "search_code"]
  },
  "similarity_threshold": {
    "type": "number",
    "required": false,
    "description": "Minimum similarity score for alternatives",
    "minimum": 0.0,
    "maximum": 1.0,
    "default": 0.6
  },
  "context_filter": {
    "type": "string",
    "required": false,
    "description": "Context-based filtering for alternatives",
    "examples": ["beginner-friendly", "performance-critical", "enterprise-grade"],
    "default": null
  },
  "max_alternatives": {
    "type": "integer",
    "required": false,
    "description": "Maximum number of alternatives to return",
    "minimum": 1,
    "maximum": 20,
    "default": 5
  }
}
```

## Tool Categories

### file_ops (File Operations)
Tools for file system operations including reading, writing, and manipulation.

**Common Tools**: read, write, edit, glob, ls  
**Typical Use Cases**: Configuration management, log analysis, file processing  
**Performance Profile**: Generally fast, I/O bound

### execution (Command Execution)
Tools for executing system commands, scripts, and processes.

**Common Tools**: bash, task, python_exec  
**Typical Use Cases**: Automation, deployment, system administration  
**Performance Profile**: Variable, depends on command complexity

### web (Web Operations)  
Tools for web-related operations including HTTP requests and web scraping.

**Common Tools**: web_fetch, http_client, web_scraper  
**Typical Use Cases**: API integration, web data extraction, monitoring  
**Performance Profile**: Network bound, variable latency

### data (Data Processing)
Tools for data manipulation, transformation, and analysis.

**Common Tools**: csv_processor, json_parser, data_transformer  
**Typical Use Cases**: Data analysis, ETL pipelines, report generation  
**Performance Profile**: CPU and memory intensive for large datasets

### analysis (Code Analysis)
Tools for analyzing code structure, quality, and patterns.

**Common Tools**: search_code, code_analyzer, dependency_finder  
**Typical Use Cases**: Code review, refactoring, documentation  
**Performance Profile**: CPU intensive, caching beneficial

### system (System Operations)
Tools for system-level operations and management.

**Common Tools**: process_manager, resource_monitor, log_analyzer  
**Typical Use Cases**: Monitoring, debugging, system administration  
**Performance Profile**: System resource dependent

## Performance Profiles

### Response Time Expectations

| Category | Tool Count | Typical Response | 95th Percentile |
|----------|------------|------------------|-----------------|
| search_mcp_tools | 1-10 results | 200-500ms | 800ms |
| recommend_tools_for_task | 3-5 recommendations | 500ms-1s | 1.5s |
| compare_mcp_tools | 2-5 tools | 1-2s | 3s |
| get_tool_details | Single tool | 100-200ms | 300ms |
| analyze_task_requirements | Single task | 500ms-1s | 1.2s |

### Scalability Characteristics

**Tool Catalog Size**: 
- Small (< 100 tools): All operations < 100ms
- Medium (100-1000 tools): Search < 500ms, Recommendations < 1s  
- Large (1000-10000 tools): Search < 800ms, Recommendations < 2s
- Enterprise (> 10000 tools): May require sharding and optimization

**Concurrent Users**:
- 1-10 users: No performance impact
- 10-50 users: Minimal queuing, < 10% slowdown
- 50-100 users: Moderate queuing, < 25% slowdown  
- 100+ users: Requires load balancing and caching optimization

### Optimization Recommendations

**For High-Frequency Usage**:
- Enable result caching (TOOL_SEARCH_CACHE_SIZE=2000+)
- Use hybrid search mode for best speed/accuracy balance
- Limit max_results to actual needs
- Pre-warm cache with common queries

**For Large Catalogs**:
- Implement category filtering to reduce search space
- Use keyword search for exact tool name lookups
- Consider semantic search only for exploratory queries
- Monitor and optimize database indexes

**For Enterprise Deployments**:  
- Deploy dedicated search instances
- Implement query result caching at application level
- Use analytics to identify and pre-cache common patterns
- Consider async processing for complex analysis tasks

## Error Codes and Troubleshooting

### Common Error Codes

**VALIDATION_ERROR**: Invalid parameters provided
- Check parameter types and required fields
- Validate enum values and ranges
- Ensure query strings are not empty

**NOT_FOUND_ERROR**: Tool or resource not found
- Verify tool IDs exist in catalog
- Check category names are valid
- Ensure database is properly indexed

**TIMEOUT_ERROR**: Operation timed out
- Reduce max_results or complexity
- Check system load and resources
- Consider breaking large requests into smaller ones

**SYSTEM_ERROR**: Internal system error
- Check logs for detailed error information
- Verify database connectivity
- Ensure sufficient system resources

### Performance Troubleshooting

**Slow Search Performance**:
1. Check query complexity and result limits
2. Verify cache configuration and hit rates
3. Monitor database query performance
4. Consider index optimization

**High Memory Usage**:
1. Reduce cache sizes if memory constrained
2. Limit concurrent search operations
3. Monitor embedding model memory usage
4. Consider pagination for large result sets

**Inconsistent Results**:
1. Verify tool catalog is up to date
2. Check embedding model consistency
3. Review search algorithm configuration
4. Validate query preprocessing logic

## Integration Patterns

### Basic Integration
```python
from tool_search_mcp_tools import search_mcp_tools

# Simple search
results = await search_mcp_tools("file processing tools")
if results['success']:
    for tool in results['results']:
        print(f"{tool['name']}: {tool['description']}")
```

### Advanced Integration  
```python
from tool_search_mcp_tools import (
    search_mcp_tools, 
    recommend_tools_for_task,
    compare_mcp_tools
)

# Comprehensive tool selection workflow
async def select_optimal_tool(task: str, context: str = None):
    # Get recommendations
    recommendations = await recommend_tools_for_task(task, context)
    
    if not recommendations['success']:
        return None
    
    # Get top candidates
    top_tools = [r['tool']['tool_id'] for r in recommendations['recommendations'][:3]]
    
    # Compare options
    comparison = await compare_mcp_tools(top_tools, include_decision_guidance=True)
    
    if comparison['success']:
        return comparison['comparison']['summary']['recommended_choice']
    
    # Fallback to top recommendation
    return recommendations['recommendations'][0]['tool']['tool_id']
```

### Error Handling Pattern
```python
from tool_search_mcp_tools import search_mcp_tools

async def robust_search(query: str):
    try:
        results = await search_mcp_tools(query, max_results=10)
        
        if results['success']:
            return results['results']
        else:
            # Handle specific errors
            error = results.get('error', {})
            error_type = error.get('error_type', 'unknown')
            
            if error_type == 'VALIDATION_ERROR':
                # Fix query and retry
                fixed_query = fix_query(query)
                return await search_mcp_tools(fixed_query)
            
            # Return empty results for other errors
            return []
            
    except Exception as e:
        # Log error and return fallback
        logger.error(f"Search failed: {e}")
        return get_fallback_tools(query)
```

This reference provides comprehensive documentation for all MCP tools available through the search system. For implementation details and advanced usage patterns, refer to the [API Reference](../api_reference.md) and [Integration Guide](../integration_guide.md).