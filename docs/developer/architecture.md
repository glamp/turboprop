# Developer Guide: Architecture and Extension

This guide provides an in-depth look at Turboprop's enhanced search architecture, design decisions, and how to extend the system with new features.

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Core Components](#core-components)
3. [Search Pipeline](#search-pipeline)
4. [Extension Guide](#extension-guide)
5. [Performance Tuning](#performance-tuning)
6. [Contributing Guidelines](#contributing-guidelines)
7. [Testing Framework](#testing-framework)

## System Architecture

Turboprop follows a modular architecture that separates concerns while maintaining high performance and extensibility.

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Client Layer                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │     CLI     │  │ MCP Server  │  │  HTTP API   │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  Search Operations Layer                    │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │         Hybrid Search Engine                            │ │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐      │ │
│  │  │  Semantic   │ │    Text     │ │   Fusion    │      │ │
│  │  │   Search    │ │   Search    │ │  Algorithm  │      │ │
│  │  └─────────────┘ └─────────────┘ └─────────────┘      │ │
│  └─────────────────────────────────────────────────────────┘ │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │              Result Ranking System                      │ │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐      │ │
│  │  │  File Type  │ │ Construct   │ │   Recency   │      │ │
│  │  │   Scorer    │ │   Scorer    │ │   Scorer    │      │ │
│  │  └─────────────┘ └─────────────┘ └─────────────┘      │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Data Layer                               │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │              Database Manager                           │ │
│  │    ┌─────────────┐    ┌─────────────┐                   │ │
│  │    │   DuckDB    │    │ Embeddings  │                   │ │
│  │    │   Vector    │    │ Generator   │                   │ │
│  │    │   Store     │    │ (SentTrans) │                   │ │
│  │    └─────────────┘    └─────────────┘                   │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### Design Principles

1. **Modular Design**: Each component has clear responsibilities and interfaces
2. **Performance First**: Optimized for sub-second search across large codebases  
3. **Extensibility**: Plugin architecture for new languages, scorers, and search modes
4. **Type Safety**: Comprehensive type hints and structured data classes
5. **Testability**: Comprehensive test coverage with isolated components
6. **Configuration**: Everything configurable via environment variables or config files

## Core Components

### 1. Hybrid Search Engine (`hybrid_search.py`)

The heart of the enhanced search system, combining multiple search strategies.

```python
class HybridSearchEngine:
    """
    Core search engine that coordinates semantic and text search.
    
    Key Responsibilities:
    - Query analysis and routing
    - Semantic embedding search via DuckDB
    - Text-based search with exact matching
    - Reciprocal Rank Fusion for result combination
    - Search mode selection (auto, hybrid, semantic, text)
    """
    
    def __init__(self, db_path: str):
        self.db_manager = DatabaseManager(db_path)
        self.embedding_gen = EmbeddingGenerator()
        self.query_analyzer = QueryAnalyzer()
    
    def search(self, query: str, mode: SearchMode, **kwargs) -> List[CodeSearchResult]:
        # 1. Analyze query characteristics
        # 2. Generate embeddings (if semantic search)
        # 3. Execute search strategies in parallel
        # 4. Apply fusion algorithm
        # 5. Return ranked results
```

**Key Algorithms:**

- **Reciprocal Rank Fusion (RRF)**: Combines rankings from different search strategies
- **Query Analysis**: Determines optimal search strategy based on query characteristics
- **Adaptive Weighting**: Adjusts semantic vs. text weights based on query type

### 2. Result Ranking System (`result_ranking.py`)

Advanced multi-factor ranking that goes beyond simple similarity scores.

```python
class ResultRanker:
    """
    Multi-factor result ranking system.
    
    Ranking Factors:
    - Base semantic similarity (0.4 weight)
    - File type relevance (0.2 weight)
    - Code construct matching (0.15 weight)
    - File recency from Git (0.15 weight)
    - File size optimization (0.1 weight)
    """
    
    def rank_results(self, results: List[CodeSearchResult], 
                    query: str, weights: RankingWeights) -> List[CodeSearchResult]:
        # 1. Apply individual scorers
        # 2. Combine weighted scores
        # 3. Generate match explanations
        # 4. Calculate confidence scores
        # 5. Sort by final ranking score
```

**Individual Scorers:**

```python
class FileTypeScorer:
    """Scores based on file type relevance to query context."""
    
    def score_result(self, result: CodeSearchResult, query: str) -> float:
        # Language-specific relevance scoring
        # Example: "authentication middleware" -> boost Python/JS files
        
class ConstructTypeScorer:
    """Scores based on code construct type matching."""
    
    def score_result(self, result: CodeSearchResult, query: str) -> float:
        # Match construct types to query intent
        # Example: "class definition" -> boost class constructs

class RecencyScorer:
    """Scores based on Git file modification recency."""
    
    def score_result(self, result: CodeSearchResult, query: str) -> float:
        # Recent files more likely to be relevant
```

### 3. Search Result Types (`search_result_types.py`)

Structured data classes that replace simple tuples with rich metadata.

```python
@dataclass
class CodeSearchResult:
    """
    Comprehensive search result with rich metadata.
    
    Replaces simple (path, content, score) tuples with:
    - Structured snippets with line numbers
    - Language detection and syntax highlighting
    - IDE navigation support
    - Confidence scoring and match explanations
    - MCP integration metadata
    """
    file_path: str
    snippet: CodeSnippet
    similarity_score: float
    confidence_score: float
    file_language: Optional[str]
    construct_type: Optional[str] 
    match_reasons: List[str]
    ide_navigation_url: Optional[str]
    
    def to_legacy_tuple(self) -> Tuple[str, str, float]:
        """Backward compatibility with legacy code."""
        return (self.file_path, self.snippet.text, self.similarity_score)
```

### 4. Database Manager (`database_manager.py`)

Handles all database operations with concurrent access protection.

```python
class DatabaseManager:
    """
    Thread-safe database operations with advanced locking.
    
    Features:
    - Process-safe file locking to prevent corruption
    - Automatic schema migrations
    - Vector similarity search via DuckDB
    - Batch operations for performance
    - Connection pooling and cleanup
    """
    
    def search_similar(self, query_embedding: List[float], 
                      k: int = 10) -> List[Tuple]:
        # Native DuckDB vector operations for speed
        # Returns: [(path, content, similarity_score), ...]
```

### 5. MCP Server (`mcp_server.py`)

Model Context Protocol integration for Claude Code.

```python
class TurbopropMCPServer:
    """
    MCP server exposing turboprop functionality to Claude.
    
    Tools Provided:
    - index_repository: Build searchable index
    - search_code: Semantic search with structured results  
    - get_index_status: Index health and statistics
    - watch_repository: Real-time monitoring
    - list_indexed_files: File inventory
    """
```

## Search Pipeline

The complete search pipeline from query to results:

```python
def search_pipeline_example(query: str) -> List[CodeSearchResult]:
    """Complete search pipeline demonstration."""
    
    # 1. Query Analysis
    analyzer = QueryAnalyzer()
    characteristics = analyzer.analyze_query(query)
    
    # 2. Search Mode Selection
    if characteristics.suggested_mode == SearchMode.AUTO:
        mode = SearchMode.HYBRID  # Default for complex queries
    else:
        mode = characteristics.suggested_mode
    
    # 3. Hybrid Search Execution  
    engine = HybridSearchEngine(db_path)
    raw_results = engine.search(query, mode=mode, k=50)
    
    # 4. Result Ranking
    ranker = ResultRanker()
    weights = RankingWeights()  # Use configured weights
    ranked_results = ranker.rank_results(raw_results, query, weights)
    
    # 5. Post-processing
    # - Deduplicate similar results
    # - Add match explanations  
    # - Calculate confidence scores
    # - Limit to requested count
    
    return ranked_results[:10]  # Top 10 results
```

### Performance Optimizations

The pipeline includes several performance optimizations:

1. **Parallel Execution**: Semantic and text searches run concurrently
2. **Early Termination**: Stop processing when confidence thresholds are met
3. **Result Caching**: Cache embeddings and intermediate results
4. **Batch Operations**: Process multiple queries efficiently

## Extension Guide

### Adding New Search Modes

```python
# 1. Extend SearchMode enum
class SearchMode(Enum):
    SEMANTIC_ONLY = "semantic"
    TEXT_ONLY = "text" 
    HYBRID = "hybrid"
    AUTO = "auto"
    FUZZY = "fuzzy"  # New mode!

# 2. Implement search strategy
class FuzzySearchStrategy:
    def search(self, query: str, k: int) -> List[CodeSearchResult]:
        # Implement fuzzy matching logic
        pass

# 3. Integrate with HybridSearchEngine
class HybridSearchEngine:
    def search(self, query: str, mode: SearchMode, **kwargs):
        if mode == SearchMode.FUZZY:
            strategy = FuzzySearchStrategy()
            return strategy.search(query, **kwargs)
        # ... existing modes
```

### Adding New Ranking Scorers

```python
# 1. Create scorer class
class ComplexityScorer:
    """Scores based on code complexity metrics."""
    
    def score_result(self, result: CodeSearchResult, query: str) -> float:
        complexity = self.calculate_complexity(result.snippet.text)
        # Prefer moderately complex code over too simple/complex
        if 10 <= complexity <= 50:
            return 1.0
        elif complexity < 10:
            return 0.7  # Too simple
        else:
            return 0.5  # Too complex

# 2. Register with RankingWeights
@dataclass
class RankingWeights:
    semantic_similarity: float = 0.35    # Reduced to make room
    file_type_relevance: float = 0.2
    construct_type_match: float = 0.15
    recency_score: float = 0.15
    file_size_optimization: float = 0.05  # Reduced
    complexity_score: float = 0.1         # New scorer!

# 3. Integrate with ResultRanker
class ResultRanker:
    def __init__(self):
        self.scorers = {
            'complexity': ComplexityScorer(),
            # ... existing scorers
        }
```

### Adding Language Support

```python
# 1. Extend language detection
LANGUAGE_EXTENSIONS = {
    '.py': 'python',
    '.js': 'javascript', 
    '.ts': 'typescript',
    '.rs': 'rust',        # New language!
    '.go': 'go',          # New language!
}

# 2. Add language-specific keywords
LANGUAGE_KEYWORDS = {
    'rust': {
        'function': ['fn', 'function'],
        'error': ['Result', 'Error', 'panic'],
        'async': ['async', 'await', 'tokio'],
    },
    'go': {
        'function': ['func', 'function'],
        'error': ['error', 'err', 'panic'],
        'concurrency': ['goroutine', 'channel', 'go func'],
    }
}

# 3. Add construct extraction patterns
CONSTRUCT_PATTERNS = {
    'rust': {
        'function': r'fn\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(',
        'struct': r'struct\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*[{<]',
        'enum': r'enum\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*[{<]',
    }
}
```

### Adding New Response Formats

```python
# 1. Create response type
@dataclass
class DetailedSearchResponse:
    """Enhanced response with additional metadata."""
    query: str
    results: List[CodeSearchResult]
    execution_timeline: Dict[str, float]  # Performance metrics
    query_suggestions: List[str]
    related_searches: List[str]
    
    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

# 2. Add response configuration
class ResponseDetailLevel(Enum):
    MINIMAL = "minimal"
    STANDARD = "standard" 
    COMPREHENSIVE = "comprehensive"
    DEBUG = "debug"
    DETAILED = "detailed"  # New level!

# 3. Integrate with MCP tools
def search_code_mcp_tool(query: str, detail_level: str = "standard"):
    if detail_level == "detailed":
        return create_detailed_response(query, results)
    # ... existing levels
```

## Performance Tuning

### Indexing Performance

**Batch Size Tuning:**
```python
# config.py
EMBEDDING_BATCH_SIZE = 32  # Default
# For high-memory systems: 64 or 128
# For low-memory systems: 16 or 8

# Measure optimal batch size for your hardware
def benchmark_batch_sizes():
    for batch_size in [8, 16, 32, 64, 128]:
        start = time.time()
        # Process embeddings with batch_size
        elapsed = time.time() - start
        print(f"Batch size {batch_size}: {elapsed:.2f}s")
```

**Parallel Processing:**
```python
# Enable parallel file processing
import concurrent.futures

def process_files_parallel(files: List[str], max_workers: int = 4):
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_file, f) for f in files]
        return [f.result() for f in futures]
```

### Search Performance

**Query Optimization:**
```python
# Cache embeddings for repeated queries
class EmbeddingCache:
    def __init__(self, max_size: int = 1000):
        self.cache = {}
        self.max_size = max_size
    
    def get_embedding(self, query: str) -> Optional[List[float]]:
        return self.cache.get(query)
    
    def set_embedding(self, query: str, embedding: List[float]):
        if len(self.cache) >= self.max_size:
            # LRU eviction
            oldest = next(iter(self.cache))
            del self.cache[oldest]
        self.cache[query] = embedding
```

**Database Optimization:**
```sql
-- Add indexes for common query patterns
CREATE INDEX idx_file_language ON code_files(file_language);
CREATE INDEX idx_construct_type ON code_files(construct_type);
CREATE INDEX idx_file_size ON code_files(LENGTH(content));

-- Analyze query performance
EXPLAIN ANALYZE SELECT * FROM code_files 
WHERE array_dot_product(embedding, $1) > 0.7 
ORDER BY array_dot_product(embedding, $1) DESC 
LIMIT 10;
```

### Memory Optimization

**Embedding Model Management:**
```python
class LazyEmbeddingGenerator:
    """Load embedding model only when needed."""
    
    def __init__(self):
        self._model = None
    
    @property
    def model(self):
        if self._model is None:
            self._model = SentenceTransformer('all-MiniLM-L6-v2')
        return self._model
    
    def __del__(self):
        # Cleanup model when not needed
        if hasattr(self, '_model') and self._model:
            del self._model
```

## Contributing Guidelines

### Code Standards

1. **Type Hints**: All functions must have complete type annotations
2. **Docstrings**: Google-style docstrings for all public functions
3. **Error Handling**: Specific exception types with helpful messages
4. **Testing**: Minimum 90% test coverage for new code
5. **Performance**: No regressions in search performance benchmarks

### Development Setup

```bash
# 1. Clone and setup environment
git clone https://github.com/your-org/turboprop.git
cd turboprop
python -m venv .venv
source .venv/bin/activate

# 2. Install development dependencies
pip install -e ".[dev]"

# 3. Install pre-commit hooks
pre-commit install

# 4. Run tests to verify setup
pytest tests/ --cov=. --cov-report=html

# 5. Run performance benchmarks
python -m pytest tests/performance/ -v
```

### Pull Request Process

1. **Feature Branch**: Create branch from `main`: `git checkout -b feature/your-feature`
2. **Implementation**: Follow TDD approach - write tests first
3. **Documentation**: Update relevant documentation files
4. **Performance**: Run benchmarks to ensure no regressions
5. **Testing**: Ensure all tests pass and coverage > 90%
6. **Code Review**: Submit PR with clear description and examples

### Testing Framework

Turboprop has comprehensive test coverage across multiple categories:

```python
# Unit Tests - Test individual components
tests/test_hybrid_search.py
tests/test_result_ranking.py
tests/test_search_result_types.py

# Integration Tests - Test complete workflows  
tests/integration/test_full_workflow.py

# Performance Tests - Benchmark search performance
tests/performance/test_search_benchmarks.py

# Quality Tests - Verify result accuracy
tests/quality/test_data_accuracy.py

# Edge Case Tests - Handle error conditions
tests/edge_cases/test_error_handling.py
```

### Performance Benchmarks

All PRs must pass performance benchmarks:

```python
# Benchmark search performance
def test_search_performance():
    # Index sample codebase
    # Measure search times across different modes
    # Assert: hybrid search < 200ms for 95th percentile
    # Assert: semantic search < 100ms for 95th percentile
    # Assert: text search < 50ms for 95th percentile

# Benchmark indexing performance  
def test_indexing_performance():
    # Measure indexing speed
    # Assert: > 100 files/second indexing rate
    # Assert: Memory usage < 500MB for 10k files
```

### Architecture Decision Records (ADRs)

Major architectural decisions are documented in `docs/adr/`:

- `ADR-001: Hybrid Search Architecture`
- `ADR-002: Result Ranking Algorithm Selection`
- `ADR-003: MCP Integration Design`  
- `ADR-004: Database Schema Evolution`

### Debugging and Profiling

```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('turboprop')

# Profile search performance
import cProfile
import pstats

def profile_search(query: str):
    profiler = cProfile.Profile()
    profiler.enable()
    
    results = perform_enhanced_search(query)
    
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('tottime')
    stats.print_stats(20)  # Top 20 functions by time
```

This architecture enables Turboprop to provide fast, accurate, and extensible semantic code search while maintaining clean separation of concerns and high testability.