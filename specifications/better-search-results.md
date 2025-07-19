# Turboprop Search Results Optimization for MCP and Claude Code

## Executive Summary

This document outlines a comprehensive plan to optimize Turboprop's search functionality for better integration with MCP (Model Context Protocol) and Claude Code. The current implementation returns basic tuple data that, while functional, lacks the rich metadata and structured information needed for optimal AI agent interaction and IDE integration.

## Current State Analysis

### Existing Data Format

- **Return Type**: Simple tuples `(file_path, snippet, distance_score)`
- **Database Schema**: `id, path, content, embedding, last_modified, file_mtime`
- **Snippet Format**: First 300 characters of file content
- **Scoring**: Cosine distance (0.0 = identical, 1.0 = completely different)

### Current Strengths ✅

1. **Clean, simple format** - Easy for AI agents to parse
2. **Semantic similarity scoring** - Perfect for AI understanding relevance
3. **Full content storage** - Allows for deeper analysis when needed
4. **File modification tracking** - Good for incremental updates
5. **DuckDB vector operations** - Fast, efficient similarity search

### Current Limitations 🚧

1. **Minimal metadata** - No file type, language, or structural information
2. **Basic snippets** - No context-aware content extraction
3. **No code structure awareness** - Missing functions, classes, imports
4. **Limited IDE integration** - No line number references or navigation aids
5. **String-based MCP responses** - Hard to parse programmatically
6. **No result clustering** - Potential for redundant results

## Optimization Opportunities for MCP/Claude Code

### 1. Enhanced Metadata Structure

**Problem**: Current schema lacks contextual information that Claude needs to understand code relationships and structure.

**Solution**: Extend database schema with rich metadata:

```sql
-- Enhanced code_files table
ALTER TABLE code_files ADD COLUMN file_type VARCHAR;
ALTER TABLE code_files ADD COLUMN language VARCHAR;
ALTER TABLE code_files ADD COLUMN size_bytes INTEGER;
ALTER TABLE code_files ADD COLUMN line_count INTEGER;

-- New code_constructs table for extracted programming constructs
CREATE TABLE code_constructs (
    id VARCHAR PRIMARY KEY,
    file_id VARCHAR REFERENCES code_files(id),
    construct_type VARCHAR,  -- 'function', 'class', 'variable', 'import'
    name VARCHAR,
    start_line INTEGER,
    end_line INTEGER,
    signature TEXT,
    docstring TEXT,
    embedding DOUBLE[384]
);

-- New repository_context table for git and project information
CREATE TABLE repository_context (
    id VARCHAR PRIMARY KEY,
    repo_path VARCHAR,
    git_branch VARCHAR,
    git_commit_hash VARCHAR,
    project_type VARCHAR,  -- 'python', 'javascript', 'mixed', etc.
    dependencies TEXT,     -- JSON array of dependencies
    created_at TIMESTAMP
);
```

### 2. Structured Content Snippets

**Problem**: Current snippets are simple string truncations that don't provide meaningful code context.

**Solution**: Implement intelligent snippet extraction:

- **Context-aware snippets**: Show the complete function/class containing the match
- **Multiple snippets per file**: Return several relevant code sections
- **Syntax-aware truncation**: Break at logical boundaries (end of functions, etc.)
- **Line number annotations**: Include precise file:line references for IDE navigation

### 3. Enhanced Search Result Objects

**Problem**: Simple tuples don't provide enough structure for complex AI reasoning.

**Solution**: Create rich result objects:

```python
@dataclass
class CodeSearchResult:
    # File information
    file_path: str
    relative_path: str
    file_type: str
    language: str

    # Content and context
    primary_snippet: CodeSnippet
    additional_snippets: List[CodeSnippet]
    full_content: Optional[str]

    # Relevance and scoring
    similarity_score: float
    confidence_level: str  # 'high', 'medium', 'low'
    match_reasons: List[str]

    # Code structure
    containing_function: Optional[str]
    containing_class: Optional[str]
    imports: List[str]
    dependencies: List[str]

    # Navigation aids
    line_references: List[int]
    ide_navigation_url: str  # file:///path/to/file:line

    # Relationships
    related_files: List[str]
    cross_references: List[CrossReference]

@dataclass
class CodeSnippet:
    content: str
    start_line: int
    end_line: int
    context_type: str  # 'function', 'class', 'module', 'fragment'
    highlighted_lines: List[int]  # Lines that specifically match the query
```

### 4. MCP Tool Response Optimization

**Problem**: Current MCP tools return formatted strings that are hard for Claude to process programmatically.

**Solution**: Restructure MCP responses:

```python
# Instead of returning formatted strings, return structured data
@mcp.tool()
def search_code_structured(query: str, max_results: int = 5) -> dict:
    """Enhanced semantic search with structured results"""
    results = perform_enhanced_search(query, max_results)

    return {
        "query": query,
        "total_results": len(results),
        "results": [result.to_dict() for result in results],
        "search_metadata": {
            "execution_time": 0.123,
            "confidence_distribution": {"high": 2, "medium": 2, "low": 1},
            "languages_found": ["python", "javascript"],
            "search_type": "semantic"
        },
        "suggested_refinements": [
            "function definition for parse_json",
            "JSON parsing in Python specifically",
            "error handling for JSON parsing"
        ]
    }
```

### 5. Claude Code Integration Features

**Problem**: Results don't integrate well with IDE workflows and code understanding tasks.

**Solution**: Add IDE-centric features:

- **File:line navigation**: `file:///path/to/file:123` format for direct IDE opening
- **Code relationship mapping**: Track imports, function calls, inheritance hierarchies
- **Smart context extraction**: Show relevant surrounding code for better understanding
- **Multi-language support**: Language-specific parsing and context extraction
- **Git integration**: Branch context, change history, blame information

## Implementation Plan

### Phase 1: Enhanced Data Schema (Week 1-2)

- [ ] Extend `code_files` table with metadata columns
- [ ] Create `code_constructs` table for extracted code elements
- [ ] Create `repository_context` table for project information
- [ ] Implement schema migration logic
- [ ] Add indexing for performance

### Phase 2: Intelligent Content Processing (Week 3-4)

- [ ] Implement language detection and classification
- [ ] Create AST-based code construct extraction
- [ ] Build context-aware snippet generation
- [ ] Add function/class boundary detection
- [ ] Implement import statement extraction

### Phase 3: Enhanced Search Results (Week 5-6)

- [ ] Create `CodeSearchResult` and related data classes
- [ ] Implement confidence scoring algorithms
- [ ] Add relevance explanation generation
- [ ] Create cross-reference relationship mapping
- [ ] Build result clustering and deduplication

### Phase 4: MCP Tool Restructuring (Week 7-8)

- [ ] Replace string-based responses with structured data
- [ ] Add pagination and filtering capabilities
- [ ] Implement search refinement suggestions
- [ ] Create specialized search tools (functions, classes, patterns)
- [ ] Add result caching and performance optimization

### Phase 5: Claude Code Integration (Week 9-10)

- [ ] Add IDE navigation URL generation
- [ ] Implement syntax highlighting hints
- [ ] Create git integration features
- [ ] Add project-wide relationship mapping
- [ ] Build search analytics and query suggestions

### Phase 6: Performance & UX (Week 11-12)

- [ ] Implement incremental search and result caching
- [ ] Add search result clustering algorithms
- [ ] Create hybrid search modes (semantic + exact)
- [ ] Optimize database queries and indexing
- [ ] Add comprehensive testing and benchmarking

## Expected Benefits

### For AI Agents (Claude)

- **Richer context**: Better understanding of code relationships and structure
- **Structured data**: Easier programmatic processing of search results
- **Confidence indicators**: Better decision-making about result relevance
- **Code relationships**: Understanding of imports, dependencies, and cross-references

### For Developers

- **IDE integration**: Direct navigation to specific code locations
- **Better search relevance**: More accurate results through enhanced metadata
- **Context awareness**: See complete functions/classes, not just fragments
- **Multi-language support**: Consistent experience across programming languages

### For MCP Integration

- **Standardized responses**: Consistent structured data format
- **Extensible architecture**: Easy to add new search capabilities
- **Performance optimization**: Cached results and intelligent pagination
- **Rich metadata**: Support for complex AI reasoning tasks

## Technical Considerations

### Backward Compatibility

- Maintain existing tuple-based API for legacy compatibility
- Add feature flags for gradual rollout of new functionality
- Provide migration path for existing indexes

### Performance Impact

- New metadata extraction will increase indexing time
- Enhanced search results may impact query performance
- Implement caching strategies and database optimization

### Storage Requirements

- Additional metadata will increase database size
- Code construct extraction requires more processing power
- Consider compression for large codebases

### Language Support

- Start with Python, JavaScript, TypeScript
- Add language-specific parsers incrementally
- Design extensible architecture for new languages

## Metrics and Success Criteria

### Search Quality

- **Relevance score**: Improve average relevance by 25%
- **Context completeness**: 90% of results include complete function/class context
- **Cross-reference accuracy**: 95% accurate relationship mapping

### Developer Experience

- **Navigation efficiency**: 50% reduction in time to find relevant code
- **IDE integration**: 90% of results navigable directly to IDE
- **Search satisfaction**: 85% developer satisfaction score

### AI Agent Performance

- **Structured data adoption**: 100% MCP tools use structured responses
- **Context utilization**: Claude uses enhanced metadata in 80% of interactions
- **Query refinement**: 60% improvement in follow-up query success

## Conclusion

The proposed enhancements will transform Turboprop from a basic semantic search tool into a comprehensive code intelligence platform optimized for AI agents and modern development workflows. The structured approach ensures backward compatibility while providing the rich metadata and context that Claude Code and other AI tools need to provide exceptional developer assistance.

The implementation plan spreads the work across 12 weeks, allowing for iterative development, testing, and refinement. Each phase builds upon the previous one, ensuring a stable and well-tested progression toward the enhanced functionality.

This investment in search result optimization will significantly improve the developer experience and unlock new possibilities for AI-assisted code understanding, navigation, and development workflows.
