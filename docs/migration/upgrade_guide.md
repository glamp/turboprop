# Migration Guide: From Basic to Enhanced Search

This guide will help you migrate from Turboprop's basic tuple-based search system to the new enhanced search system with structured results, hybrid search, and advanced ranking.

## Table of Contents

1. [Migration Overview](#migration-overview)
2. [Breaking Changes](#breaking-changes)
3. [Database Migration](#database-migration)
4. [API Changes](#api-changes)
5. [Configuration Updates](#configuration-updates)
6. [Code Migration Examples](#code-migration-examples)
7. [Testing Your Migration](#testing-your-migration)
8. [Rollback Procedures](#rollback-procedures)

## Migration Overview

The enhanced search system introduces significant improvements while maintaining backward compatibility where possible. The main changes include:

### What's New ✨
- **Structured Results**: Rich `CodeSearchResult` objects instead of simple tuples
- **Hybrid Search**: Combines semantic search with exact text matching
- **Advanced Ranking**: Multi-factor ranking beyond simple similarity scores
- **MCP Integration**: Enhanced Claude Code integration with structured responses
- **Language Detection**: Automatic language-specific search improvements
- **Construct Extraction**: Search specific code constructs (functions, classes, etc.)
- **Explainable Results**: Match explanations and confidence scores

### Compatibility Promise 🛡️
- Existing database indexes remain compatible
- Basic CLI commands continue to work unchanged  
- Simple search queries produce equivalent results
- MCP tools maintain backward compatibility

## Breaking Changes

### 1. Search Result Format

**Before (Basic System):**
```python
# Results were simple tuples
results = search_index(query, k=5)
for path, content, score in results:
    print(f"{path}: {score:.3f}")
```

**After (Enhanced System):**
```python
# Results are now CodeSearchResult objects
results = search_index_enhanced(query, k=5)
for result in results:
    print(f"{result.file_path}: {result.similarity_score:.3f}")
    print(f"Language: {result.file_language}")
    print(f"Confidence: {result.confidence_score:.3f}")
```

### 2. Import Changes

**Before:**
```python
from code_index import search_index, build_full_index
```

**After:**  
```python
from search_operations import perform_enhanced_search
from search_result_types import CodeSearchResult
from hybrid_search import HybridSearchEngine, SearchMode
```

### 3. Configuration System

**Before:**
```python
# Configuration was scattered across modules
MAX_FILE_SIZE = 1.0
SNIPPET_LINES = 3
```

**After:**
```python
# Centralized configuration
from config import config
config.search.MAX_FILE_SIZE_MB = 1.0
config.search.SNIPPET_CONTEXT = 3
```

## Database Migration

### Automatic Migration

The enhanced system automatically migrates your existing database:

```bash
# Your existing database will be automatically upgraded
turboprop search "your query"  # Migration happens transparently
```

### Manual Migration

For more control, you can manually migrate:

```bash
# Check current index status
turboprop status

# Force database schema upgrade  
turboprop index . --force-reindex

# Verify migration
turboprop status
```

### What Gets Migrated

- **Schema Changes**: New columns added for metadata (language, construct_type, etc.)
- **Embeddings Preserved**: Existing embeddings are kept to avoid recomputation
- **New Metadata**: Added automatically during next search/index operation
- **Performance Indexes**: New indexes added for improved search performance

## API Changes

### Search Functions

**Old API:**
```python
def search_index(query: str, k: int = 5) -> List[Tuple[str, str, float]]:
    # Returns list of (path, content, score) tuples
```

**New API:**  
```python
def search_index_enhanced(query: str, k: int = 5) -> List[CodeSearchResult]:
    # Returns list of CodeSearchResult objects

def perform_enhanced_search(
    query: str,
    db_path: str,
    k: int = 5,
    search_mode: SearchMode = SearchMode.AUTO,
    include_ranking: bool = True,
    include_explanations: bool = True
) -> List[CodeSearchResult]:
    # Full-featured search with all enhancements
```

### MCP Tool Changes

**Old MCP Response:**
```json
{
    "results": [
        ["path/to/file.py", "code content", 0.85],
        ["other/file.js", "other content", 0.72]
    ]
}
```

**New MCP Response:**
```json
{
    "query": "authentication middleware", 
    "results": [
        {
            "file_path": "path/to/file.py",
            "snippet": {
                "text": "def authenticate(request):",
                "start_line": 42,
                "end_line": 42
            },
            "similarity_score": 0.85,
            "confidence_score": 0.89,
            "file_language": "python",
            "construct_type": "function",
            "match_reasons": ["Contains authentication logic"],
            "ide_navigation_url": "vscode://file/path/to/file.py:42"
        }
    ],
    "total_count": 15,
    "execution_time": 0.045,
    "suggestions": ["Try 'JWT validation'", "Consider 'session management'"]
}
```

## Configuration Updates

### Environment Variables

**Before:**
```bash
# Limited configuration options
export TURBOPROP_DB_PATH=".turboprop"
```

**After:**
```bash
# Comprehensive configuration
export TURBOPROP_DB_PATH=".turboprop"
export TURBOPROP_MAX_FILE_SIZE_MB=1.0
export TURBOPROP_SEARCH_MODE=hybrid
export TURBOPROP_SNIPPET_CONTEXT_LINES=3
export TURBOPROP_RRF_K=60
export TURBOPROP_RESPONSE_DETAIL=standard
```

### Configuration File Support

Create `.turboprop/config.json` for project-specific settings:

```json
{
    "search": {
        "max_file_size_mb": 2.0,
        "default_search_mode": "hybrid",
        "snippet_context_lines": 5
    },
    "ranking": {
        "semantic_weight": 0.4,
        "file_type_weight": 0.2,
        "recency_weight": 0.15
    },
    "response": {
        "detail_level": "comprehensive",
        "include_explanations": true,
        "include_ide_links": true
    }
}
```

## Code Migration Examples

### 1. Basic Search Usage

**Before:**
```python
from code_index import search_index

def find_auth_code():
    results = search_index("authentication", k=10)
    for path, content, score in results:
        if score > 0.8:
            print(f"High confidence match in {path}")
            # Extract relevant snippet manually
            lines = content.split('\n')
            snippet = '\n'.join(lines[:5])
            print(snippet)
```

**After:**
```python
from search_operations import perform_enhanced_search
from search_result_types import CodeSearchResult

def find_auth_code():
    results = perform_enhanced_search("authentication", k=10)
    for result in results:
        if result.confidence_score > 0.8:
            print(f"High confidence match in {result.file_path}")
            print(f"Language: {result.file_language}")
            print(f"Construct: {result.construct_type}")
            print(result.snippet.text)
            print(f"Reasons: {', '.join(result.match_reasons)}")
```

### 2. Custom Search Logic

**Before:**
```python
def search_with_filters(query: str, min_score: float = 0.7):
    all_results = search_index(query, k=50)
    filtered = [(p, c, s) for p, c, s in all_results if s >= min_score]
    return filtered
```

**After:**
```python
from hybrid_search import HybridSearchEngine, SearchMode

def search_with_filters(query: str, min_confidence: float = 0.7):
    engine = HybridSearchEngine()
    all_results = engine.search(query, k=50, mode=SearchMode.HYBRID)
    filtered = [r for r in all_results if r.confidence_score >= min_confidence]
    return filtered
```

### 3. MCP Tool Integration

**Before:**
```python
def mcp_search_tool(query: str, k: int = 5):
    results = search_index(query, k)
    return {
        "results": [[path, content[:200], score] for path, content, score in results]
    }
```

**After:**
```python
from mcp_response_types import create_search_response_from_results
from search_operations import perform_enhanced_search

def mcp_search_tool(query: str, k: int = 5):
    results = perform_enhanced_search(query, k=k)
    return create_search_response_from_results(
        query=query,
        results=results,
        include_suggestions=True,
        include_clusters=True
    ).to_json()
```

## Testing Your Migration

### 1. Verify Basic Functionality

```bash
# Test basic search still works
turboprop search "authentication" --k 5

# Test enhanced features  
turboprop search "JWT validation" --mode hybrid --explain

# Test MCP integration
turboprop mcp --repository . --test-search "error handling"
```

### 2. Compare Results

Run side-by-side comparison to verify result quality:

```python
# test_migration.py
from search_operations import perform_enhanced_search
from code_index import search_index  # Legacy function (if available)

query = "authentication middleware"

# Legacy results
try:
    legacy_results = search_index(query, k=5)
    print("Legacy results:", len(legacy_results))
except:
    print("Legacy search not available")

# Enhanced results  
enhanced_results = perform_enhanced_search(query, k=5)
print("Enhanced results:", len(enhanced_results))

# Compare top result
if enhanced_results:
    result = enhanced_results[0]
    print(f"Top result: {result.file_path}")
    print(f"Confidence: {result.confidence_score}")
    print(f"Language: {result.file_language}")
```

### 3. Performance Testing

```bash
# Time both search modes
time turboprop search "authentication" --mode semantic
time turboprop search "authentication" --mode text  
time turboprop search "authentication" --mode hybrid

# Test large result sets
turboprop search "function" --k 50 --explain
```

## Rollback Procedures

If you need to rollback to the basic system:

### 1. Backup Current State

```bash
# Backup enhanced database
cp .turboprop/code_index.duckdb .turboprop/code_index.duckdb.enhanced.backup

# Backup configuration
cp .turboprop/config.json .turboprop/config.json.backup
```

### 2. Downgrade Options

**Option A: Use Legacy Mode**
```bash
# Set environment to disable enhanced features
export TURBOPROP_LEGACY_MODE=true
turboprop search "your query"
```

**Option B: Rebuild with Basic Mode**
```bash
# Remove enhanced database
rm .turboprop/code_index.duckdb

# Reinstall basic version (if needed)
pip install turboprop==1.0.0  # Replace with basic version

# Rebuild index
turboprop index . --basic-mode
```

### 3. Restore Backup

```bash
# Restore from backup if needed
cp .turboprop/code_index.duckdb.basic.backup .turboprop/code_index.duckdb
```

## Common Migration Issues

### Issue: Import Errors

**Problem:**
```python
ImportError: No module named 'search_result_types'
```

**Solution:**
```python
# Update imports
from search_result_types import CodeSearchResult
from search_operations import perform_enhanced_search
```

### Issue: Configuration Not Found

**Problem:**
```
ConfigValidationError: Configuration file not found
```

**Solution:**
```bash
# Create default configuration
turboprop init-config

# Or set environment variables
export TURBOPROP_MAX_FILE_SIZE_MB=1.0
```

### Issue: Performance Regression

**Problem:**
Search is slower than before.

**Solution:**
```python
# Use faster search modes for performance-critical code
from hybrid_search import SearchMode

results = perform_enhanced_search(
    query, 
    mode=SearchMode.SEMANTIC_ONLY,  # Fastest option
    k=5  # Limit results
)
```

### Issue: Different Results

**Problem:**
Results differ from basic system.

**Explanation:**
This is expected! The enhanced system provides:
- Better ranking through multi-factor scoring
- More relevant results through hybrid search
- Language-specific improvements

**Validation:**
```bash
# Compare result quality manually
turboprop search "your query" --explain --verbose
```

## Getting Help

If you encounter issues during migration:

1. **Check logs**: `turboprop search "test" --verbose`
2. **Validate config**: `turboprop config --validate`
3. **Test basic functionality**: `turboprop status`
4. **Run diagnostics**: `turboprop diagnose --full`

For additional support, refer to the [troubleshooting guide](../developer/troubleshooting.md) or open an issue on GitHub.

## Next Steps

After successful migration:

1. **Explore new features**: Try hybrid search modes and result explanations
2. **Optimize configuration**: Tune ranking weights for your use case  
3. **Update integrations**: Leverage structured results in your tools
4. **Provide feedback**: Share your migration experience to help improve the process

The enhanced search system opens up many new possibilities. Take time to explore the [user guide](../user/search_guide.md) to get the most out of your upgraded system!