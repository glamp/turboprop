# MCP Tool Search System - User Guide

## Overview

The MCP Tool Search System transforms how Claude Code and other AI agents discover and select MCP tools. Instead of manually knowing what tools are available, you can now use natural language to find, compare, and get recommendations for the optimal tools for any development task.

## Key Features

### 1. Natural Language Tool Discovery
Search for tools using plain English descriptions of what you want to accomplish:

```python
# Find tools for file operations
search_mcp_tools("read configuration files safely")

# Find tools for web operations  
search_mcp_tools("scrape web data with error handling")

# Find tools by category
search_mcp_tools("command execution tools", category="execution")
```

### 2. Intelligent Tool Recommendations
Get personalized recommendations based on your specific task requirements:

```python
# Get recommendations for a development task
recommend_tools_for_task(
    task_description="process CSV files and generate reports",
    context="performance critical, large files",
    complexity_preference="balanced"
)

# Analyze task requirements first
analyze_task_requirements("deploy application with monitoring")
```

### 3. Tool Comparison and Alternatives
Compare multiple tools to understand trade-offs and choose the best option:

```python
# Compare similar tools
compare_mcp_tools(["read", "write", "edit"])

# Find alternatives to a specific tool
find_tool_alternatives("bash", context_filter="beginner-friendly")

# Get detailed tool information
get_tool_details("search_code", include_examples=True)
```

## Search Strategies

The system supports multiple search strategies optimized for different scenarios:

### semantic search
Best for: Conceptual queries, finding tools by purpose
```python
search_mcp_tools("tools for data transformation", search_mode="semantic")
```

### hybrid search (Recommended)
Best for: Most queries, combines semantic understanding with keyword matching
```python
search_mcp_tools("bash shell scripting", search_mode="hybrid")
```

### Keyword Search
Best for: Exact term matching, finding tools by specific names
```python
search_mcp_tools("file read write", search_mode="keyword")
```

## Writing Effective Search Queries

### Good Query Examples
- ✅ "read configuration files with error handling"
- ✅ "execute shell commands with timeout support"
- ✅ "search code using natural language queries"
- ✅ "web scraping tools for JSON APIs"

### Query Improvement Tips
- **Be specific**: Include context about your use case
- **Mention constraints**: Include requirements like "with error handling" or "for large files"
- **Use domain terms**: Include relevant technical terminology
- **Specify complexity**: Mention if you need "simple" or "powerful" tools

### Less Effective Queries
- ❌ "file tools" (too vague)
- ❌ "good tools" (no specific functionality)
- ❌ "fast" (unclear what operation needs to be fast)

## Understanding Search Results

### Result Structure
Each search result includes:

```json
{
  "tool_id": "bash",
  "name": "Bash",
  "description": "Execute shell commands with optional timeout",
  "similarity_score": 0.89,
  "confidence_level": "high",
  "match_reasons": [
    "Shell command execution functionality",
    "Timeout parameter support",
    "Error handling capabilities"
  ],
  "parameters": [...],
  "examples": [...],
  "alternatives": ["task"],
  "when_to_use": "For system commands, file operations, and script execution"
}
```

### Confidence Levels
- **High (0.8-1.0)**: Excellent match, highly recommended
- **Medium (0.6-0.8)**: Good match, worth considering
- **Low (0.4-0.6)**: Partial match, review carefully

### Match Reasons
Each result explains why it matched your query:
- Functional capability alignment
- Parameter compatibility
- Use case relevance
- Domain expertise match

## Advanced Features

### Context-Aware Recommendations
Provide context for better recommendations:

```python
recommend_tools_for_task(
    task_description="read and process data files",
    context="beginner user, small files, safety first"
)
```

### Parameter-Specific Search
Find tools with specific parameter requirements:

```python
search_tools_by_capability(
    capability_description="timeout support",
    required_parameters=["timeout"],
    preferred_complexity="simple"
)
```

### Tool Relationship Analysis
Understand how tools relate to each other:

```python
analyze_tool_relationships(
    tool_id="bash",
    relationship_types=["alternatives", "complements"]
)
```

## Integration with Claude Code

### Automatic Suggestions
Claude Code can now proactively suggest optimal tools:

```
🔍 Tool Search Suggestion: For this file processing task, consider using 'read' 
instead of 'bash cat' for better error handling and parameter validation.

Confidence: High (0.91)
Reasoning: Direct file reading, built-in error handling, type safety
```

### Follow-up Recommendations
After using a tool, get suggestions for next steps:

```json
{
  "navigation_hints": [
    "Use get_tool_details() for comprehensive information about recommended tools",
    "Use compare_mcp_tools() to compare similar options",
    "Consider analyze_task_requirements() for complex tasks"
  ]
}
```

## Best Practices

### 1. Start with Task Analysis
For complex tasks, analyze requirements first:
```python
analyze_task_requirements("build and deploy microservice application")
```

### 2. Use Recommendations for Guidance
Get intelligent suggestions rather than guessing:
```python
recommend_tools_for_task("secure file transfer between servers")
```

### 3. Compare Options for Important Decisions
Understand trade-offs before choosing:
```python
compare_mcp_tools(["bash", "task"], comparison_context="automation scripts")
```

### 4. Leverage Tool Relationships
Explore alternatives and complements:
```python
find_tool_alternatives("current_tool", similarity_threshold=0.7)
```

### 5. Provide Context for Better Results
Include relevant constraints and preferences:
```python
search_mcp_tools(
    "data processing tools", 
    context="performance critical, large datasets, Python environment"
)
```

## Troubleshooting

### Common Issues

#### "No relevant tools found"
**Possible causes:**
- Query too specific or using uncommon terminology
- Tool catalog not fully updated
- Searching for non-existent capabilities

**Solutions:**
- Try broader, more general terms
- Use synonyms or alternative descriptions
- Check available categories with `list_tool_categories()`

#### "Too many results, hard to choose"
**Possible causes:**
- Query too broad or generic
- Not enough context provided

**Solutions:**
- Add specific requirements and constraints
- Use `recommend_tools_for_task()` instead of search
- Apply filters like category or complexity

#### "Recommendations don't seem relevant"
**Possible causes:**
- Task description unclear or ambiguous
- Missing important context
- Edge case not covered by training data

**Solutions:**
- Use `analyze_task_requirements()` first
- Provide more detailed task description
- Add context about environment and constraints

### Performance Tips

1. **Use caching**: Repeated similar queries are cached for faster response
2. **Limit results**: Use `max_results` parameter for faster processing
3. **Be specific**: More specific queries are processed more efficiently
4. **Use appropriate search mode**: Choose semantic, hybrid, or keyword based on need

## Advanced Configuration

### Environment Variables
```bash
# Tool search performance settings
TOOL_SEARCH_CACHE_SIZE=1000
TOOL_SEARCH_CACHE_TTL=3600

# Search behavior settings
TOOL_SEARCH_DEFAULT_MODE=hybrid
TOOL_SEARCH_MAX_RESULTS=20
TOOL_SEARCH_SIMILARITY_THRESHOLD=0.3

# Learning and adaptation settings
TOOL_SEARCH_ENABLE_LEARNING=true
TOOL_SEARCH_LEARNING_RATE=0.1
```

### Customization Options
```python
# Customize search behavior
search_mcp_tools(
    query="file processing",
    search_mode="hybrid",
    semantic_weight=0.8,  # Emphasize semantic matching
    keyword_weight=0.2,   # De-emphasize keyword matching
    include_examples=True,
    max_results=15
)
```