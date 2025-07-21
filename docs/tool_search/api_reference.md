# MCP MCP Tool Search System - API Reference

## Core MCP Tools

### search_mcp_tools()

Search for MCP tools by functionality or description.

**Signature:**
```python
def search_mcp_tools(
    query: str,
    category: Optional[str] = None,
    tool_type: Optional[str] = None,
    max_results: int = 10,
    include_examples: bool = True,
    search_mode: str = 'hybrid'
) -> dict
```

**Parameters:**
- `query` (str): Natural language description of desired functionality
- `category` (str, optional): Filter by tool category ('file_ops', 'web', 'analysis', etc.)
- `tool_type` (str, optional): Filter by tool type ('system', 'custom', 'third_party')
- `max_results` (int): Maximum number of tools to return (1-50, default: 10)
- `include_examples` (bool): Whether to include usage examples (default: True)
- `search_mode` (str): Search strategy ('semantic', 'hybrid', 'keyword', default: 'hybrid')

**Returns:**
```json
{
  "success": true,
  "query": "file operations",
  "results": [
    {
      "tool_id": "read",
      "name": "Read",
      "description": "Read file contents from filesystem",
      "similarity_score": 0.89,
      "confidence_level": "high",
      "match_reasons": ["file reading functionality", "filesystem access"],
      "parameters": [...],
      "examples": [...],
      "alternatives": ["write", "edit"],
      "complexity_score": 0.3
    }
  ],
  "total_results": 5,
  "execution_time": 0.45,
  "query_suggestions": [
    "Try 'file reading with error handling' for more specific results"
  ],
  "category_breakdown": {
    "file_ops": 3,
    "execution": 2
  },
  "timestamp": "2024-01-15 10:30:45 UTC"
}
```

**Examples:**
```python
# Basic search
search_mcp_tools("file operations")

# Filtered search
search_mcp_tools("web scraping", category="web", max_results=5)

# Semantic search only
search_mcp_tools("data transformation", search_mode="semantic")
```

### recommend_tools_for_task()

Get intelligent tool recommendations for a specific development task.

**Signature:**
```python
def recommend_tools_for_task(
    task_description: str,
    context: Optional[str] = None,
    max_recommendations: int = 5,
    include_alternatives: bool = True,
    complexity_preference: str = 'balanced',
    explain_reasoning: bool = True
) -> dict
```

**Parameters:**
- `task_description` (str): Natural language description of the task
- `context` (str, optional): Additional context about environment or constraints
- `max_recommendations` (int): Maximum recommendations to return (1-10, default: 5)
- `include_alternatives` (bool): Include alternative options (default: True)
- `complexity_preference` (str): Preference for tool complexity ('simple', 'balanced', 'powerful')
- `explain_reasoning` (bool): Include detailed explanations (default: True)

**Returns:**
```json
{
  "success": true,
  "task_description": "read configuration files safely",
  "recommendations": [
    {
      "tool": {
        "tool_id": "read",
        "name": "Read",
        "description": "Read file contents with error handling"
      },
      "recommendation_score": 0.92,
      "confidence_level": "high",
      "task_alignment": 0.89,
      "capability_match": 0.95,
      "complexity_alignment": 0.88,
      "recommendation_reasons": [
        "Excellent for safe file reading operations",
        "Built-in error handling for missing files",
        "Parameter validation prevents common errors"
      ],
      "usage_guidance": [
        "Use file_path parameter for target file",
        "Consider offset/limit for large files",
        "Handle FileNotFoundError exceptions"
      ],
      "when_to_use": "For reading configuration files, logs, or structured data",
      "alternative_tools": ["bash", "edit"]
    }
  ],
  "task_analysis": {
    "task_category": "file_operation",
    "complexity_level": "simple",
    "required_capabilities": ["file_reading", "error_handling"],
    "confidence": 0.87
  },
  "explanations": [
    "#1 read: Recommended because it provides direct file reading with built-in safety features..."
  ]
}
```

### compare_mcp_tools()

Compare multiple MCP tools across various dimensions.

**Signature:**
```python
def compare_mcp_tools(
    tool_ids: List[str],
    comparison_criteria: Optional[List[str]] = None,
    include_decision_guidance: bool = True,
    comparison_context: Optional[str] = None,
    detail_level: str = 'standard'
) -> dict
```

**Parameters:**
- `tool_ids` (List[str]): List of tool IDs to compare (2-10 tools)
- `comparison_criteria` (List[str], optional): Specific aspects to compare
- `include_decision_guidance` (bool): Include selection recommendations (default: True)
- `comparison_context` (str, optional): Context for comparison
- `detail_level` (str): Level of detail ('basic', 'standard', 'comprehensive')

**Comparison Criteria Options:**
- 'functionality': Feature richness and capability breadth
- 'usability': Ease of use and learning curve
- 'performance': Speed and resource efficiency  
- 'reliability': Stability and error handling
- 'complexity': Tool complexity and parameter requirements
- 'documentation': Documentation quality and examples

**Returns:**
```json
{
  "success": true,
  "comparison": {
    "tools": [
      {
        "tool_id": "read",
        "overall_score": 0.89,
        "dimension_scores": {
          "functionality": 0.85,
          "usability": 0.95,
          "performance": 0.88,
          "reliability": 0.92
        },
        "strengths": ["Simple interface", "Excellent error handling"],
        "weaknesses": ["Limited to read operations"],
        "best_for": ["Configuration files", "Log analysis"]
      }
    ],
    "summary": {
      "recommended_choice": "read",
      "reasoning": "Best balance of simplicity and reliability",
      "use_case_recommendations": {
        "simple_file_reading": "read",
        "complex_file_operations": "edit",
        "automation_scripts": "bash"
      }
    }
  }
}
```

### get_tool_details()

Get comprehensive information about a specific MCP tool.

**Signature:**
```python
def get_tool_details(
    tool_id: str,
    include_examples: bool = True,
    include_alternatives: bool = True,
    include_usage_stats: bool = False
) -> dict
```

**Parameters:**
- `tool_id` (str): ID of the tool to get details for
- `include_examples` (bool): Include usage examples (default: True)
- `include_alternatives` (bool): Include alternative tools (default: True)
- `include_usage_stats` (bool): Include usage statistics (default: False)

### analyze_task_requirements()

Analyze a task description to understand requirements and suggest appropriate tools.

**Signature:**
```python
def analyze_task_requirements(
    task_description: str,
    detail_level: str = 'standard'
) -> dict
```

**Parameters:**
- `task_description` (str): Natural language description of the task
- `detail_level` (str): Analysis depth ('basic', 'standard', 'comprehensive')

### find_tool_alternatives()

Find alternative tools for a given tool or functionality.

**Signature:**
```python
def find_tool_alternatives(
    reference_tool: str,
    similarity_threshold: float = 0.6,
    context_filter: Optional[str] = None,
    max_alternatives: int = 5
) -> dict
```

## Component Classes

### MCPToolSearchEngine

Core search engine for tool discovery.

```python
class MCPToolSearchEngine:
    def __init__(self, db_manager, embedding_generator, query_processor):
        # Initialize search engine components
        
    def search_by_functionality(self, query: str, k: int = 10, 
                              category_filter: Optional[str] = None) -> List[ToolSearchResult]:
        # Search tools by functional description
        
    def search_hybrid(self, query: str, k: int = 10, 
                     semantic_weight: float = 0.7) -> List[ToolSearchResult]:
        # Hybrid search combining semantic and keyword matching
```

### ToolRecommendationEngine

Intelligent recommendation system for task-based tool selection.

```python
class ToolRecommendationEngine:
    def recommend_for_task(self, task_description: str, 
                          context: Optional[TaskContext] = None) -> List[ToolRecommendation]:
        # Get recommendations for specific task
        
    def get_alternative_recommendations(self, primary_tool: str, 
                                      task_context: TaskContext) -> List[AlternativeRecommendation]:
        # Get alternative tool options
```

## Response Data Structures

### ToolSearchResult
```python
@dataclass
class ToolSearchResult:
    tool_id: str
    name: str  
    description: str
    similarity_score: float
    confidence_level: str  # 'high', 'medium', 'low'
    match_reasons: List[str]
    parameters: List[ParameterInfo]
    examples: List[ToolExample]
    alternatives: List[str]
    complexity_score: float
```

### ToolRecommendation
```python
@dataclass
class ToolRecommendation:
    tool: ToolSearchResult
    recommendation_score: float
    confidence_level: str
    task_alignment: float
    recommendation_reasons: List[str]
    usage_guidance: List[str]
    when_to_use: str
    alternative_tools: List[str]
```

### ParameterInfo
```python
@dataclass
class ParameterInfo:
    name: str
    type: str
    description: str
    required: bool
    default_value: Optional[Any]
    validation_rules: List[str]
    examples: List[str]
```

### ToolExample
```python
@dataclass
class ToolExample:
    title: str
    description: str
    code: str
    expected_output: Optional[str]
    use_case: str
    complexity_level: str
```

## Error Handling

All MCP tools return standardized error responses:

```json
{
  "success": false,
  "tool": "search_mcp_tools",
  "error": {
    "message": "Query cannot be empty",
    "context": "search_mcp_tools called with empty query parameter",
    "error_type": "validation_error",
    "suggestions": [
      "Provide a descriptive query about the functionality you need",
      "Try queries like 'file operations' or 'web scraping tools'"
    ],
    "recovery_options": [
      "Retry with a valid query string",
      "Use list_tool_categories() to explore available tools"
    ]
  }
}
```

### Error Types
- `validation_error`: Invalid parameters or input
- `not_found_error`: Tool or resource not found
- `system_error`: Internal system error
- `timeout_error`: Operation timed out
- `permission_error`: Insufficient permissions

## Performance Considerations

### Caching
- Search results cached for 1 hour by default
- Tool details cached until tool catalog updates
- Recommendation results cached based on context hash

### Rate Limits
- No explicit rate limits for normal usage
- Concurrent requests automatically queued
- Large result sets (>50 tools) may have longer response times

### Optimization Tips
1. Use specific queries for faster results
2. Limit result counts for better performance
3. Cache frequently used tool details
4. Use hybrid search mode for best balance of accuracy and speed

## Configuration Options

### Environment Variables
```bash
# Core search settings
TOOL_SEARCH_ENABLED=true
TOOL_SEARCH_DEFAULT_MODE=hybrid
TOOL_SEARCH_MAX_RESULTS=20

# Performance settings
TOOL_SEARCH_CACHE_SIZE=1000
TOOL_SEARCH_CACHE_TTL=3600
TOOL_SEARCH_TIMEOUT=30

# Search behavior
TOOL_SEARCH_SIMILARITY_THRESHOLD=0.3
TOOL_SEARCH_SEMANTIC_WEIGHT=0.7
TOOL_SEARCH_KEYWORD_WEIGHT=0.3

# Learning and adaptation
TOOL_SEARCH_ENABLE_LEARNING=true
TOOL_SEARCH_LEARNING_RATE=0.1
TOOL_SEARCH_FEEDBACK_COLLECTION=true
```

### Configuration File Format
```json
{
  "tool_search": {
    "enabled": true,
    "search_mode": "hybrid",
    "cache_settings": {
      "size": 1000,
      "ttl": 3600
    },
    "performance": {
      "max_results": 20,
      "timeout": 30,
      "concurrent_searches": 5
    },
    "learning": {
      "enabled": true,
      "feedback_collection": true,
      "learning_rate": 0.1
    }
  }
}
```

## Usage Examples

### Basic Tool Discovery
```python
# Simple search
results = search_mcp_tools("file operations")
print(f"Found {len(results['results'])} tools")

# With filtering
results = search_mcp_tools(
    "web scraping", 
    category="web",
    max_results=5
)
```

### Task-Based Recommendations
```python
# Get recommendations for a specific task
recommendations = recommend_tools_for_task(
    "process CSV data and generate charts",
    context="Python environment, data analysis workflow"
)

# Use the top recommendation
best_tool = recommendations['recommendations'][0]
print(f"Recommended: {best_tool['tool']['name']}")
```

### Tool Comparison Workflow
```python
# Compare multiple options
comparison = compare_mcp_tools(
    ["read", "bash", "edit"],
    comparison_context="configuration file processing"
)

# Get decision guidance
recommendation = comparison['comparison']['summary']['recommended_choice']
print(f"Best choice: {recommendation}")
```

### Advanced Integration
```python
# Analyze requirements first
analysis = analyze_task_requirements(
    "build and deploy microservice with monitoring"
)

# Get targeted recommendations
recommendations = recommend_tools_for_task(
    analysis['task_description'],
    context=analysis['inferred_context']
)

# Compare top options
top_tools = [rec['tool']['tool_id'] for rec in recommendations['recommendations'][:3]]
comparison = compare_mcp_tools(top_tools)
```