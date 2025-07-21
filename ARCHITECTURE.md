# Turboprop Architecture

This document provides detailed technical information about Turboprop's architecture, implementation details, and advanced features.

## System Overview

Turboprop is a semantic code search system built on three core technologies:
- **DuckDB** for fast analytical queries and native vector operations
- **SentenceTransformers** for semantic embeddings generation
- **Git integration** for intelligent file discovery and change monitoring

## Core Components

### 1. Code Indexer (`code_index.py`)

**Primary Functions:**
- `scan_repo()` - Git-aware file discovery using `git ls-files`
- `embed_and_store()` - Generates semantic embeddings and stores in DuckDB
- `build_full_index()` - Validates embeddings and builds search indices
- `search_index()` - Performs semantic search with ranking algorithms
- `watch_mode()` - Real-time file monitoring with intelligent debouncing

**Key Features:**
- Parallel processing with configurable worker pools
- Incremental updates (only processes changed files)
- Memory-efficient batch processing
- Advanced file locking for concurrent access safety

### 2. HTTP API Server (`server.py`)

FastAPI-based REST API providing:
- `POST /index` - Trigger repository reindexing
- `GET /search` - Query the semantic index
- `GET /status` - Index health and statistics
- Background file watcher for automatic updates

### 3. MCP Server Integration

Model Context Protocol (MCP) server providing tools for AI assistants:
- `index_repository` - Build searchable index
- `search_code` - Semantic search with natural language
- `get_index_status` - Index health monitoring
- `watch_repository` - File change monitoring
- `list_indexed_files` - Index content inspection

## Database Schema

### Primary Table: `code_files`

```sql
CREATE TABLE code_files (
    id VARCHAR PRIMARY KEY,        -- SHA-256 hash of path + content
    path VARCHAR NOT NULL,         -- Absolute file path
    content TEXT NOT NULL,         -- Full file content
    embedding DOUBLE[384] NOT NULL -- 384-dimension vector embeddings
);

-- Indices for performance
CREATE INDEX idx_code_files_path ON code_files(path);
CREATE INDEX idx_code_files_embedding ON code_files USING HNSW(embedding);
```

### Metadata Tables

```sql
-- Index metadata and configuration
CREATE TABLE index_metadata (
    key VARCHAR PRIMARY KEY,
    value TEXT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- File change tracking
CREATE TABLE file_hashes (
    path VARCHAR PRIMARY KEY,
    content_hash VARCHAR NOT NULL,
    last_modified TIMESTAMP,
    file_size INTEGER
);
```

## Search Architecture

### Enhanced Search System

Turboprop implements a sophisticated multi-mode search system:

#### 1. Search Modes

**AUTO Mode (Recommended)**
- Automatically chooses optimal strategy based on query analysis
- Uses hybrid approach for complex queries
- Falls back to text search for code-specific patterns
- Applies query preprocessing and optimization

**HYBRID Mode**
- Combines semantic similarity with exact text matching
- Uses Reciprocal Rank Fusion (RRF) for result merging
- Configurable fusion parameters (`RRF_K` constant)
- Balances conceptual understanding with keyword precision

**SEMANTIC Mode**
- Pure vector similarity search using cosine distance
- Ideal for finding conceptually similar code
- Language and framework agnostic matching
- Best for discovering alternative implementations

**TEXT Mode**
- Fast exact text matching with LIKE queries
- Optimized for finding specific syntax or identifiers
- Uses database full-text search capabilities
- Minimal overhead for known patterns

#### 2. Ranking Algorithm

Multi-factor ranking system with weighted components:

```python
# Ranking factors (configurable weights)
SEMANTIC_WEIGHT = 0.40      # Cosine similarity score
FILE_TYPE_WEIGHT = 0.20     # Language/extension relevance
CONSTRUCT_WEIGHT = 0.15     # Code structure matching
RECENCY_WEIGHT = 0.15       # Git modification timestamp
FILE_SIZE_WEIGHT = 0.10     # Optimal file size preference

final_score = (
    semantic_score * SEMANTIC_WEIGHT +
    file_type_score * FILE_TYPE_WEIGHT +
    construct_score * CONSTRUCT_WEIGHT +
    recency_score * RECENCY_WEIGHT +
    size_score * FILE_SIZE_WEIGHT
)
```

#### 3. Result Enhancement

**Confidence Scoring:**
- Normalized similarity scores (0.0 - 1.0 range)
- Multi-factor confidence calculation
- Threshold-based result filtering

**Match Explanation:**
- Semantic similarity reasoning
- Keyword match highlighting
- Context relevance analysis

**IDE Integration:**
- VS Code navigation links (`vscode://file/path:line`)
- PyCharm/IntelliJ support (`idea://open?file=path&line=N`)
- Vim/Neovim jump commands

### Vector Operations

**Embedding Generation:**
```python
# Using SentenceTransformers
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embeddings = model.encode(texts, batch_size=64, show_progress_bar=True)
```

**Similarity Search:**
```sql
-- DuckDB native vector operations
SELECT path, content, 
       list_dot_product(embedding, $query_embedding) AS similarity
FROM code_files 
ORDER BY similarity DESC 
LIMIT $k;
```

## File Processing Pipeline

### 1. Discovery Phase
- Use `git ls-files` for Git-tracked file enumeration
- Respect `.gitignore` rules automatically
- Apply configurable file size filters
- Skip binary and generated files

### 2. Content Processing
- Read file contents with encoding detection
- Apply language-specific preprocessing
- Generate content fingerprints (SHA-256)
- Extract metadata (language, constructs, etc.)

### 3. Embedding Generation
- Batch processing for efficiency
- GPU acceleration when available (CUDA/MPS)
- Fallback to CPU processing for compatibility
- Progress tracking and error handling

### 4. Storage Optimization
- Atomic database transactions
- Duplicate detection and deduplication  
- Compression for large content
- Index optimization after bulk inserts

## Concurrent Access & Safety

### File Locking Strategy

**Lock Types:**
- **Shared locks** for read operations (searches)
- **Exclusive locks** for write operations (indexing)
- **Process locks** for cross-process coordination

**Lock Implementation:**
```python
# Advanced file locking with timeout and retry
class AdvancedFileLock:
    def __init__(self, lock_file, timeout=30.0):
        self.lock_file = lock_file
        self.timeout = timeout
        self.retry_interval = 0.1
        
    def acquire(self, exclusive=True):
        # Platform-specific locking (fcntl on Unix, msvcrt on Windows)
        # Automatic stale lock detection and cleanup
        # Deadlock prevention with ordered lock acquisition
```

**Safety Guarantees:**
- No database corruption during concurrent access
- Atomic index updates (complete or rollback)
- Graceful handling of process interruption
- Automatic stale lock cleanup on restart

### Process Coordination

**Multi-process scenarios:**
- Multiple developers indexing same repository
- CI/CD systems with concurrent builds
- Background watchers with manual indexing
- MCP server with CLI tool usage

**Coordination mechanisms:**
- File-based process locks
- Database-level transaction isolation
- Cooperative lock yielding
- Process-safe temporary file handling

## Configuration System

### Configuration Sources (Priority Order)

1. **Environment Variables** (highest priority)
2. **YAML Configuration Files** (`.turboprop.yml`)
3. **Command Line Arguments**
4. **Built-in Defaults** (lowest priority)

### Environment Variables

```bash
# Database configuration
export TURBOPROP_DB_MEMORY_LIMIT="2GB"
export TURBOPROP_DB_THREADS="8"
export TURBOPROP_DB_MAX_RETRIES="3"

# File processing
export TURBOPROP_MAX_FILE_SIZE_MB="2.0"
export TURBOPROP_DEBOUNCE_SECONDS="3.0"
export TURBOPROP_BATCH_SIZE="200"

# Search configuration
export TURBOPROP_DEFAULT_MAX_RESULTS="10"
export TURBOPROP_MIN_SIMILARITY="0.2"
export TURBOPROP_SEARCH_MODE="hybrid"

# Embedding model
export TURBOPROP_EMBED_MODEL="all-MiniLM-L6-v2"
export TURBOPROP_DEVICE="cpu"  # or "cuda", "mps"

# Server settings
export TURBOPROP_HOST="127.0.0.1"
export TURBOPROP_PORT="8080"

# Logging
export TURBOPROP_LOG_LEVEL="INFO"
export TURBOPROP_LOG_FILE="turboprop.log"
```

### YAML Configuration

```yaml
# .turboprop.yml example
database:
  memory_limit: "2GB"
  threads: 8
  max_retries: 3
  connection_timeout: 60.0
  auto_vacuum: true

file_processing:
  max_file_size_mb: 2.0
  debounce_seconds: 3.0
  batch_size: 200
  max_workers: 8

search:
  default_max_results: 10
  max_results_limit: 50
  min_similarity: 0.2
  mode: "hybrid"

embedding:
  model: "all-MiniLM-L6-v2"
  device: "cpu"
  batch_size: 64

server:
  host: "127.0.0.1"
  port: 8080
  request_timeout: 60.0

logging:
  level: "INFO"
  file: "turboprop.log"
```

## Performance Characteristics

### Indexing Performance

**Typical throughput:**
- **Small repositories** (< 1K files): 2-5 seconds
- **Medium repositories** (1K-10K files): 30-120 seconds
- **Large repositories** (10K+ files): 5-30 minutes

**Factors affecting performance:**
- File size distribution
- Available CPU/GPU resources
- Disk I/O characteristics
- Network storage latency

**Optimization techniques:**
- Parallel processing with worker pools
- Incremental updates (changed files only)
- Batch embedding generation
- Memory-mapped file reading

### Search Performance

**Query response times:**
- **Semantic search**: 50-200ms (typical)
- **Hybrid search**: 100-500ms (typical)
- **Text search**: 10-50ms (typical)

**Scalability characteristics:**
- Linear scaling with repository size
- Constant time for individual queries
- Memory usage proportional to index size
- Efficient caching of frequent queries

## MCP Tool Search System

### Architecture Overview

The MCP Tool Search System is an intelligent tool discovery and recommendation engine that helps AI agents (like Claude Code) automatically find and select the most appropriate tools for development tasks.

### Core Components

#### 1. Tool Knowledge Base
- **Comprehensive tool database** with descriptions, capabilities, and usage patterns
- **Semantic embeddings** for each tool's functionality and purpose
- **Relationship mapping** between tools and common development tasks
- **Context-aware metadata** including complexity levels and prerequisites

#### 2. Search Engine
- **Multi-modal search** supporting semantic, keyword, and hybrid queries
- **Intent recognition** to understand what type of tool assistance is needed
- **Contextual filtering** based on project type, user experience, and task complexity
- **Relevance scoring** using multiple factors (functionality match, ease of use, reliability)

#### 3. Recommendation System
- **Task-based recommendations** using machine learning models
- **Usage pattern analysis** to suggest optimal tool combinations
- **Contextual suggestions** based on current development environment
- **Learning from feedback** to improve future recommendations

### Search Modes

#### Natural Language Tool Discovery
```python
# Example: Finding tools by functionality description
search_mcp_tools("read configuration files safely")
# Returns: read, config_parser, yaml_loader, etc.

search_mcp_tools("execute shell commands with timeout support") 
# Returns: bash, subprocess_manager, shell_runner, etc.
```

#### Intelligent Tool Recommendations
```python
# Example: Getting task-specific tool suggestions
recommend_tools_for_task(
    task="process CSV files and generate reports",
    context="performance critical, large files", 
    complexity_preference="balanced"
)
# Returns ranked list with explanations
```

#### Tool Comparison and Analysis
```python
# Example: Comparing similar tools
compare_mcp_tools(["read", "write", "edit"])
# Returns detailed comparison matrix

find_tool_alternatives("bash", context_filter="beginner-friendly")
# Returns safer/simpler alternatives
```

### Integration with Claude Code

The tool search system provides several enhancements to Claude Code's capabilities:

#### Proactive Tool Suggestions
- **Real-time analysis** of conversation context to suggest optimal tools
- **Confidence scoring** for tool recommendations
- **Explanation generation** for why specific tools are suggested
- **Alternative options** when primary tools aren't optimal

#### Context-Aware Selection
- **Project type detection** (web app, CLI tool, data science, etc.)
- **Technology stack recognition** (Python, JavaScript, Go, etc.)
- **User skill level adaptation** (beginner, intermediate, advanced)
- **Task complexity assessment** for appropriate tool selection

#### Learning and Adaptation
- **Usage pattern tracking** to improve future recommendations
- **Feedback integration** from successful/unsuccessful tool usage
- **Personalization** based on user preferences and project patterns
- **Continuous improvement** through machine learning models

### Performance and Caching

#### Search Optimization
- **In-memory tool database** for fast lookups
- **Precomputed embeddings** for semantic search
- **Query caching** for frequently requested tool information
- **Response streaming** for large result sets

#### Scalability Design
- **Modular architecture** supporting additional tool sources
- **Lazy loading** of detailed tool documentation
- **Distributed caching** for multi-user environments
- **API rate limiting** and resource management

## Troubleshooting

### Common Issues

**Apple Silicon MPS Compatibility:**
- Error: "Cannot copy out of meta tensor; no data!"
- **Solution:** Automatic fallback to CPU processing
- **Cause:** PyTorch MPS backend incompatibility with certain models

**File Locking Issues:**
- Error: "Database is locked" or "Cannot acquire lock"
- **Solution:** Check for stale processes, restart with `--force-unlock`
- **Prevention:** Use proper shutdown procedures

**Memory Issues:**
- Error: "Out of memory" during indexing
- **Solution:** Reduce batch size, increase file size limits
- **Configuration:** Adjust `TURBOPROP_BATCH_SIZE` and `max_workers`

### Debugging Features

**Verbose logging:**
```bash
export TURBOPROP_LOG_LEVEL="DEBUG"
turboprop index . --verbose
```

**Database inspection:**
```sql
-- Check index health
SELECT COUNT(*) as total_files, 
       AVG(LENGTH(content)) as avg_file_size,
       MAX(LENGTH(content)) as max_file_size
FROM code_files;

-- Check embedding distribution
SELECT path, LENGTH(content), 
       list_dot_product(embedding, embedding) as embedding_norm
FROM code_files 
LIMIT 10;
```

## Extension Points

### Custom Embedding Models

```python
# Using different SentenceTransformer models
EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"  # Higher quality
EMBED_MODEL = "sentence-transformers/all-distilroberta-v1"  # Faster processing
```

### Custom File Filters

```python
def custom_file_filter(filepath, content):
    """Custom logic for including/excluding files"""
    if filepath.endswith('.generated'):
        return False
    if len(content) > 1_000_000:  # 1MB limit
        return False
    return True
```

### Search Result Post-processing

```python
def enhance_search_results(results, query, context):
    """Custom result enhancement and ranking"""
    for result in results:
        result['custom_score'] = calculate_custom_relevance(result, query)
        result['ide_links'] = generate_ide_links(result['path'])
    return sorted(results, key=lambda x: x['custom_score'], reverse=True)
```

## Contributing

### Development Setup

```bash
# Clone and setup development environment
git clone https://github.com/glamp/turboprop
cd turboprop
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Run tests
pytest tests/ --cov=.

# Code quality
python -m black .
python -m isort .
python -m flake8 .
```

### Architecture Guidelines

**Core Principles:**
- Maintain backwards compatibility in APIs
- Prioritize performance for large repositories
- Ensure thread and process safety
- Provide comprehensive error handling
- Include extensive logging and debugging

**Code Organization:**
- Keep CLI and library code separate
- Use dependency injection for configurability
- Implement proper abstraction layers
- Include comprehensive type hints
- Maintain high test coverage

### Future Enhancements

**Planned Features:**
- Language-specific code parsing and analysis
- Advanced query syntax with filters and operators
- Integration with additional IDEs and editors
- Cloud-based sharing and collaboration features
- Machine learning improvements for ranking

**Performance Improvements:**
- GPU-accelerated vector operations
- Distributed indexing for enormous repositories
- Advanced caching and memoization
- Optimized storage formats and compression
- Streaming search results for large queries