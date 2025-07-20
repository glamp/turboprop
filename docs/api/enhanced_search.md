# Enhanced Search API Documentation

Turboprop's enhanced search system provides sophisticated code search capabilities that go far beyond simple semantic similarity. This document covers all the enhanced APIs, data structures, and configuration options.

## Table of Contents

1. [Search Result Types](#search-result-types)
2. [Hybrid Search Engine](#hybrid-search-engine) 
3. [Result Ranking System](#result-ranking-system)
4. [MCP Integration](#mcp-integration)
5. [Configuration Options](#configuration-options)
6. [Search Operations](#search-operations)
7. [Code Construct Extraction](#code-construct-extraction)
8. [Language Detection](#language-detection)

## Search Result Types

### CodeSnippet

Represents a code fragment with precise line information and context.

```python
from search_result_types import CodeSnippet

snippet = CodeSnippet(
    text="def authenticate(token: str) -> bool:",
    start_line=42,
    end_line=42,
    context_before="# JWT token validation",
    context_after="    return jwt.decode(token, SECRET_KEY)"
)

print(snippet)  # Line 42: def authenticate(token: str) -> bool:
```

**Fields:**
- `text: str` - The actual code text
- `start_line: int` - Starting line number
- `end_line: int` - Ending line number  
- `context_before: Optional[str]` - Code context before the snippet
- `context_after: Optional[str]` - Code context after the snippet

### CodeSearchResult

Comprehensive search result with rich metadata and IDE integration.

```python
from search_result_types import CodeSearchResult

result = CodeSearchResult(
    file_path="/path/to/auth.py",
    snippet=snippet,
    similarity_score=0.92,
    file_language="python",
    construct_type="function",
    match_reasons=["Contains JWT authentication logic"],
    confidence_score=0.89,
    ide_navigation_url="vscode://file/path/to/auth.py:42",
    syntax_highlighting_hint="python"
)
```

**Key Features:**
- Rich metadata including language, construct type, and confidence scores
- IDE navigation support for VS Code, PyCharm, and other editors
- Explainable match reasons for transparency
- MCP integration for Claude Code compatibility

## Hybrid Search Engine

The hybrid search engine combines semantic search with exact text matching using Reciprocal Rank Fusion (RRF).

### SearchMode Enum

```python
from hybrid_search import SearchMode

# Available search modes
SearchMode.SEMANTIC_ONLY  # Pure semantic search
SearchMode.TEXT_ONLY      # Exact text matching only  
SearchMode.HYBRID         # Combined semantic + text
SearchMode.AUTO           # Automatically choose best mode
```

### HybridSearchEngine

```python
from hybrid_search import HybridSearchEngine, FusionWeights

# Configure fusion weights
weights = FusionWeights(
    semantic_weight=0.6,      # Weight for semantic results
    text_weight=0.4,          # Weight for text results
    rrf_k=60,                 # RRF parameter (higher = more fusion)
    boost_exact_matches=True, # Boost exact keyword matches
    exact_match_boost=1.5     # Boost factor for exact matches
)

# Initialize search engine
engine = HybridSearchEngine(db_path=".turboprop/code_index.duckdb")

# Perform hybrid search
results = engine.search(
    query="JWT authentication middleware",
    mode=SearchMode.HYBRID,
    k=10,
    weights=weights
)
```

### Query Analysis

The system automatically analyzes queries to optimize search strategy:

```python
from hybrid_search import QueryAnalyzer

analyzer = QueryAnalyzer()
characteristics = analyzer.analyze_query("JWT authentication")

print(characteristics.has_code_keywords)    # True
print(characteristics.semantic_complexity) # 0.7 
print(characteristics.suggested_mode)      # SearchMode.HYBRID
```

## Result Ranking System

Advanced multi-factor ranking that considers file type, construct type, recency, and other factors.

### RankingWeights

```python
from result_ranking import RankingWeights

weights = RankingWeights(
    semantic_similarity=0.4,    # Base semantic similarity
    file_type_relevance=0.2,    # File type matching
    construct_type_match=0.15,  # Code construct relevance
    recency_score=0.15,         # File recency from Git
    file_size_optimization=0.1  # Prefer appropriately-sized files
)
```

### Ranking Scorers

Individual scoring components for different relevance factors:

```python
from ranking_scorers import (
    FileTypeScorer, 
    ConstructTypeScorer, 
    RecencyScorer, 
    FileSizeScorer
)

# File type scoring based on query context
file_scorer = FileTypeScorer()
score = file_scorer.score_result(result, "authentication middleware")

# Construct type scoring (functions, classes, etc.)  
construct_scorer = ConstructTypeScorer()
score = construct_scorer.score_result(result, "function definition")
```

### Match Explanations

Get human-readable explanations for why results were matched:

```python
from result_ranking import generate_match_explanations

explanations = generate_match_explanations(results, "JWT auth")
print(explanations[0])  # "Strong semantic match for authentication concepts"
```

## MCP Integration

Structured response types for Claude Code integration with rich metadata.

### SearchResponse

```python
from mcp_response_types import SearchResponse

response = SearchResponse(
    query="authentication middleware",
    results=search_results,
    total_count=45,
    execution_time=0.234,
    query_analysis=QueryAnalysis(...),
    result_clusters=[...],
    suggestions=["Try searching for 'JWT validation'", "Consider 'session management'"]
)

# Convert to JSON for MCP
json_response = response.to_json()
```

### MCP Tools

Available MCP tools for Claude Code integration:

```python
# Tool: index_repository
{
    "name": "index_repository", 
    "description": "Build searchable index from code repository",
    "inputSchema": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Repository path"},
            "max_mb": {"type": "number", "description": "Max file size in MB"},
            "force_reindex": {"type": "boolean", "description": "Force full reindex"}
        },
        "required": ["path"]
    }
}

# Tool: search_code
{
    "name": "search_code",
    "description": "Search code using natural language",
    "inputSchema": {
        "type": "object", 
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "k": {"type": "number", "description": "Number of results"},
            "mode": {"type": "string", "enum": ["semantic", "text", "hybrid", "auto"]}
        },
        "required": ["query"]
    }
}
```

## Configuration Options

Comprehensive configuration system with environment variable support.

### Core Configuration

```python
from config import config

# Database configuration
db_path = config.database.get_db_path()
table_name = config.database.TABLE_NAME  # "code_files"

# Search configuration
max_file_size = config.search.MAX_FILE_SIZE_MB      # 1.0
snippet_context = config.search.SNIPPET_CONTEXT     # 3 lines
max_results = config.search.DEFAULT_MAX_RESULTS     # 10

# Embedding configuration
model_name = config.embedding.MODEL_NAME            # "all-MiniLM-L6-v2" 
dimensions = config.embedding.DIMENSIONS            # 384
batch_size = config.embedding.BATCH_SIZE            # 32
```

### Environment Variables

Override defaults with environment variables:

```bash
export TURBOPROP_MAX_FILE_SIZE_MB=2.0
export TURBOPROP_SNIPPET_CONTEXT_LINES=5
export TURBOPROP_DEFAULT_SEARCH_MODE=hybrid
export TURBOPROP_RRF_K=80
```

### Response Configuration  

Configure response formats and detail levels:

```python
from response_config import ResponseDetailLevel

# Configure response detail
config.set_response_detail(ResponseDetailLevel.COMPREHENSIVE)

# Available levels:
# - MINIMAL: Basic results only
# - STANDARD: Standard metadata
# - COMPREHENSIVE: Full metadata with explanations
# - DEBUG: All available information
```

## Search Operations

High-level search operations that coordinate all enhanced features.

### Basic Search

```python
from search_operations import perform_enhanced_search

results = perform_enhanced_search(
    query="JWT authentication middleware",
    db_path=".turboprop/code_index.duckdb",
    k=10,
    search_mode=SearchMode.AUTO,
    include_ranking=True,
    include_explanations=True
)
```

### Construct-Specific Search

```python
from construct_search import ConstructSearchOperations

construct_ops = ConstructSearchOperations(db_path)

# Search for specific construct types
function_results = construct_ops.search_functions("authentication")
class_results = construct_ops.search_classes("middleware") 
method_results = construct_ops.search_methods("validate")
```

## Code Construct Extraction

Extract and search specific programming constructs from code.

### Supported Constructs

- Functions/methods
- Classes  
- Constants
- Imports
- Decorators
- Comments with documentation

### Usage Example

```python
from code_construct_extractor import CodeConstructExtractor

extractor = CodeConstructExtractor()

# Extract constructs from Python code
constructs = extractor.extract_constructs(
    code_content=file_content,
    file_path="/path/to/module.py",
    language="python"
)

for construct in constructs:
    print(f"{construct.type}: {construct.name} at line {construct.line_number}")
```

## Language Detection

Automatic language detection for improved search relevance.

```python
from language_detection import detect_language_from_path, detect_language_from_content

# Detect from file path
lang = detect_language_from_path("/path/to/auth.py")  # "python"

# Detect from content
lang = detect_language_from_content(code_content)     # "javascript"

# Language-specific search improvements
from search_utils import get_language_specific_keywords
keywords = get_language_specific_keywords("python", "authentication")
# Returns: ["auth", "login", "token", "jwt", "session"]
```

## Error Handling

The enhanced search system includes comprehensive error handling:

```python
from ranking_exceptions import (
    RankingError,
    InvalidRankingWeightsError, 
    InvalidSearchResultError
)

from search_result_types import SearchError

try:
    results = perform_enhanced_search(query)
except SearchError as e:
    print(f"Search failed: {e.message}")
    print(f"Suggestions: {', '.join(e.suggestions)}")
except RankingError as e:
    print(f"Ranking failed: {e}")
```

## Performance Considerations

### Search Performance

- **Hybrid search**: ~50-200ms for typical queries
- **Semantic-only**: ~30-100ms  
- **Text-only**: ~10-50ms
- **Large codebases**: Linear scaling with indexed file count

### Memory Usage

- **Base memory**: ~50-100MB for embedding model
- **Per file**: ~1-2KB indexed content + ~1.5KB embedding data
- **Search memory**: Scales with result count (k parameter)

### Optimization Tips

```python
# For large codebases, use smaller k values
results = search(query, k=5)  # Instead of k=20

# Use semantic-only for concept searches
results = search(query, mode=SearchMode.SEMANTIC_ONLY)

# Use text-only for exact matches  
results = search(query, mode=SearchMode.TEXT_ONLY)

# Configure appropriate file size limits
config.search.MAX_FILE_SIZE_MB = 0.5  # For very large repos
```