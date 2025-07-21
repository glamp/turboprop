# Search Algorithms - Technical Reference

This document provides detailed technical documentation of the search algorithms used in the MCP Tool Search System, including implementation details, performance characteristics, and optimization strategies.

## Algorithm Overview

The MCP Tool Search System implements multiple search algorithms that can be used independently or in combination to provide optimal tool discovery capabilities.

### Search Algorithm Hierarchy

```
Search Algorithms
├── Semantic Search
│   ├── Dense Vector Search
│   ├── Sentence Transformer Embeddings
│   └── Cosine Similarity Ranking
├── Keyword Search
│   ├── TF-IDF Scoring
│   ├── BM25 Relevance
│   └── Fuzzy String Matching
├── Hybrid Search
│   ├── Reciprocal Rank Fusion (RRF)
│   ├── Weighted Score Combination
│   └── Adaptive Weight Adjustment
└── Specialized Algorithms
    ├── Domain-Specific Search
    ├── Context-Aware Ranking
    └── Learning-Enhanced Search
```

## Semantic Search Algorithm

### Implementation Details

The semantic search algorithm uses dense vector representations to find tools based on conceptual similarity rather than exact keyword matching.

#### Embedding Generation
```python
class SemanticSearchAlgorithm:
    def __init__(self):
        self.model_name = "all-MiniLM-L6-v2"
        self.embedding_dim = 384
        self.similarity_threshold = 0.3
    
    async def generate_query_embedding(self, query: str) -> List[float]:
        """Generate embedding for search query."""
        # Preprocess query
        processed_query = self.preprocess_query(query)
        
        # Generate embedding using SentenceTransformer
        embedding = self.model.encode(processed_query, normalize_embeddings=True)
        
        return embedding.tolist()
    
    def preprocess_query(self, query: str) -> str:
        """Preprocess query for optimal embedding generation."""
        # Remove stop words for technical queries
        # Expand abbreviations (e.g., "auth" -> "authentication")
        # Apply domain-specific transformations
        return processed_query
```

#### Vector Similarity Search
```sql
-- DuckDB vector similarity query
SELECT 
    tool_id,
    name,
    description,
    list_dot_product(embedding, $1) as similarity_score,
    CASE 
        WHEN list_dot_product(embedding, $1) > 0.8 THEN 'high'
        WHEN list_dot_product(embedding, $1) > 0.6 THEN 'medium'
        ELSE 'low'
    END as confidence_level
FROM mcp_tools 
WHERE list_dot_product(embedding, $1) > $2
ORDER BY similarity_score DESC 
LIMIT $3;
```

#### Performance Characteristics

**Time Complexity**: O(n) where n is the number of tools in the catalog  
**Space Complexity**: O(d * n) where d is embedding dimension (384)  
**Typical Performance**: 100-300ms for 1000 tools

**Optimization Strategies**:
- Vector indexing using approximate nearest neighbor (ANN)
- Embedding caching for common queries
- Batch processing for multiple queries
- GPU acceleration for large catalogs

### Advantages and Limitations

**Advantages**:
- Finds conceptually similar tools even with different terminology
- Robust to spelling variations and synonyms
- Captures semantic relationships between tools
- Effective for exploratory queries

**Limitations**:
- Less precise for exact name/keyword searches
- Requires quality training data for embeddings
- Computationally more expensive than keyword search
- May miss highly specific technical terms

## Keyword Search Algorithm

### Implementation Details

The keyword search algorithm uses traditional information retrieval techniques optimized for technical tool discovery.

#### BM25 Implementation
```python
class BM25SearchAlgorithm:
    def __init__(self):
        self.k1 = 1.5  # Term frequency saturation point
        self.b = 0.75  # Length normalization factor
        self.epsilon = 0.25  # IDF floor value
    
    def calculate_bm25_score(self, query_terms: List[str], document: Dict) -> float:
        """Calculate BM25 relevance score."""
        score = 0.0
        doc_text = f"{document['name']} {document['description']}"
        doc_terms = self.tokenize(doc_text.lower())
        doc_length = len(doc_terms)
        avg_doc_length = self.get_average_document_length()
        
        for term in query_terms:
            term_freq = doc_terms.count(term)
            if term_freq > 0:
                idf = self.calculate_idf(term)
                
                # BM25 formula
                numerator = term_freq * (self.k1 + 1)
                denominator = term_freq + self.k1 * (
                    1 - self.b + self.b * (doc_length / avg_doc_length)
                )
                
                score += idf * (numerator / denominator)
        
        return score
```

#### Enhanced Keyword Matching
```python
def enhanced_keyword_search(self, query: str, tools: List[Dict]) -> List[Dict]:
    """Enhanced keyword search with multiple matching strategies."""
    query_terms = self.tokenize_query(query)
    results = []
    
    for tool in tools:
        # Exact match scoring
        exact_score = self.calculate_exact_match_score(query_terms, tool)
        
        # Fuzzy match scoring  
        fuzzy_score = self.calculate_fuzzy_match_score(query_terms, tool)
        
        # Prefix match scoring
        prefix_score = self.calculate_prefix_match_score(query_terms, tool)
        
        # Combine scores with weights
        combined_score = (
            exact_score * 0.5 + 
            fuzzy_score * 0.3 + 
            prefix_score * 0.2
        )
        
        if combined_score > self.threshold:
            tool_result = tool.copy()
            tool_result['keyword_score'] = combined_score
            tool_result['match_components'] = {
                'exact': exact_score,
                'fuzzy': fuzzy_score, 
                'prefix': prefix_score
            }
            results.append(tool_result)
    
    return sorted(results, key=lambda x: x['keyword_score'], reverse=True)
```

### Performance Optimization

**Indexing Strategy**:
```python
class KeywordIndex:
    def __init__(self):
        self.term_index = {}  # term -> list of (tool_id, frequency)
        self.tool_lengths = {}  # tool_id -> document length
        self.total_docs = 0
        self.avg_doc_length = 0.0
    
    def build_index(self, tools: List[Dict]):
        """Build inverted index for fast keyword search."""
        for tool in tools:
            doc_id = tool['tool_id']
            text = f"{tool['name']} {tool['description']}"
            terms = self.tokenize(text.lower())
            
            self.tool_lengths[doc_id] = len(terms)
            
            # Build inverted index
            term_counts = Counter(terms)
            for term, count in term_counts.items():
                if term not in self.term_index:
                    self.term_index[term] = []
                self.term_index[term].append((doc_id, count))
        
        self.total_docs = len(tools)
        self.avg_doc_length = sum(self.tool_lengths.values()) / self.total_docs
```

## Hybrid Search Algorithm

### Reciprocal Rank Fusion (RRF)

The hybrid search algorithm combines semantic and keyword search results using Reciprocal Rank Fusion, which has been proven effective in information retrieval systems.

```python
class HybridSearchAlgorithm:
    def __init__(self):
        self.rrf_k = 60  # RRF constant parameter
        self.semantic_weight = 0.7
        self.keyword_weight = 0.3
    
    async def hybrid_search(
        self, 
        query: str, 
        tools: List[Dict],
        k: int = 10
    ) -> List[Dict]:
        """Execute hybrid search combining semantic and keyword results."""
        
        # Execute both search strategies in parallel
        semantic_results = await self.semantic_search(query, tools, k * 2)
        keyword_results = await self.keyword_search(query, tools, k * 2)
        
        # Apply RRF fusion
        fused_results = self.apply_rrf_fusion(
            semantic_results, 
            keyword_results,
            self.rrf_k
        )
        
        # Re-rank with weighted scores
        final_results = self.apply_weighted_reranking(
            fused_results,
            semantic_results,
            keyword_results
        )
        
        return final_results[:k]
    
    def apply_rrf_fusion(
        self, 
        semantic_results: List[Dict],
        keyword_results: List[Dict], 
        k: int = 60
    ) -> List[Dict]:
        """Apply Reciprocal Rank Fusion to combine result lists."""
        
        # Create ranking maps
        semantic_ranks = {
            tool['tool_id']: rank + 1 
            for rank, tool in enumerate(semantic_results)
        }
        keyword_ranks = {
            tool['tool_id']: rank + 1
            for rank, tool in enumerate(keyword_results)
        }
        
        # Calculate RRF scores
        all_tools = set(semantic_ranks.keys()) | set(keyword_ranks.keys())
        rrf_scores = {}
        
        for tool_id in all_tools:
            rrf_score = 0.0
            
            if tool_id in semantic_ranks:
                rrf_score += 1.0 / (k + semantic_ranks[tool_id])
            
            if tool_id in keyword_ranks:
                rrf_score += 1.0 / (k + keyword_ranks[tool_id])
            
            rrf_scores[tool_id] = rrf_score
        
        # Sort by RRF score and return
        sorted_tools = sorted(
            all_tools, 
            key=lambda x: rrf_scores[x], 
            reverse=True
        )
        
        return [self.get_tool_by_id(tool_id) for tool_id in sorted_tools]
```

### Adaptive Weight Adjustment

The system can dynamically adjust the weights between semantic and keyword search based on query characteristics and user feedback.

```python
class AdaptiveWeightAdjuster:
    def __init__(self):
        self.query_patterns = {
            'exact_name': {'semantic': 0.3, 'keyword': 0.7},
            'conceptual': {'semantic': 0.8, 'keyword': 0.2},
            'mixed': {'semantic': 0.6, 'keyword': 0.4}
        }
    
    def detect_query_type(self, query: str) -> str:
        """Detect query type to adjust search weights."""
        query_lower = query.lower()
        
        # Check for exact tool names or technical terms
        if any(term in query_lower for term in self.exact_terms):
            return 'exact_name'
        
        # Check for conceptual/descriptive language
        if any(word in query_lower for word in self.conceptual_words):
            return 'conceptual'
        
        return 'mixed'
    
    def get_adaptive_weights(self, query: str, context: Dict = None) -> Dict[str, float]:
        """Get adaptive weights based on query and context."""
        query_type = self.detect_query_type(query)
        base_weights = self.query_patterns[query_type]
        
        # Apply context adjustments
        if context:
            if context.get('user_expertise') == 'expert':
                # Experts prefer more precise keyword matching
                base_weights['keyword'] *= 1.2
                base_weights['semantic'] *= 0.8
            
            if context.get('exploration_mode'):
                # Exploration favors semantic discovery
                base_weights['semantic'] *= 1.3
                base_weights['keyword'] *= 0.7
        
        # Normalize weights
        total = sum(base_weights.values())
        return {k: v/total for k, v in base_weights.items()}
```

## Domain-Specific Search Algorithms

### Technical Domain Enhancement

For technical tool discovery, specialized algorithms enhance search accuracy within specific domains.

```python
class TechnicalDomainSearch:
    def __init__(self):
        self.domain_vocabularies = {
            'web_development': {
                'synonyms': {'api': 'endpoint', 'auth': 'authentication'},
                'expansions': {'crud': 'create read update delete'},
                'technical_terms': ['http', 'rest', 'json', 'cors']
            },
            'data_science': {
                'synonyms': {'ml': 'machine learning', 'ai': 'artificial intelligence'},
                'expansions': {'etl': 'extract transform load'},
                'technical_terms': ['pandas', 'numpy', 'sklearn', 'jupyter']
            }
        }
    
    def enhance_query_for_domain(self, query: str, domain: str) -> str:
        """Enhance query with domain-specific knowledge."""
        if domain not in self.domain_vocabularies:
            return query
        
        vocab = self.domain_vocabularies[domain]
        enhanced_query = query
        
        # Apply synonym expansion
        for abbrev, full_form in vocab['synonyms'].items():
            enhanced_query = enhanced_query.replace(abbrev, f"{abbrev} {full_form}")
        
        # Apply technical term expansion
        for expansion, full_form in vocab['expansions'].items():
            if expansion in enhanced_query.lower():
                enhanced_query += f" {full_form}"
        
        return enhanced_query
```

### Context-Aware Ranking

Tools are ranked differently based on the user's context and requirements.

```python
class ContextAwareRanking:
    def __init__(self):
        self.context_factors = {
            'user_expertise': {'weight': 0.3, 'boost_complex': True},
            'performance_critical': {'weight': 0.4, 'prefer_optimized': True},
            'safety_critical': {'weight': 0.5, 'prefer_validated': True},
            'learning_context': {'weight': 0.2, 'prefer_documented': True}
        }
    
    def apply_context_ranking(
        self, 
        results: List[Dict], 
        context: Dict[str, Any]
    ) -> List[Dict]:
        """Apply context-aware ranking adjustments."""
        
        for result in results:
            base_score = result.get('similarity_score', 0.5)
            adjustments = []
            
            # Apply context-specific adjustments
            for context_key, context_value in context.items():
                if context_key in self.context_factors:
                    factor = self.context_factors[context_key]
                    adjustment = self.calculate_context_adjustment(
                        result, context_key, context_value, factor
                    )
                    adjustments.append(adjustment * factor['weight'])
            
            # Calculate final adjusted score
            total_adjustment = sum(adjustments)
            result['context_adjusted_score'] = min(1.0, base_score + total_adjustment)
            result['context_adjustments'] = adjustments
        
        # Re-sort by adjusted score
        return sorted(results, key=lambda x: x['context_adjusted_score'], reverse=True)
```

## Performance Optimization Strategies

### Caching Architecture

Multi-level caching improves response times for common queries.

```python
class SearchCache:
    def __init__(self):
        self.query_cache = TTLCache(maxsize=1000, ttl=3600)
        self.embedding_cache = LRUCache(maxsize=5000)
        self.result_cache = TTLCache(maxsize=500, ttl=1800)
    
    def get_cached_results(self, query_hash: str) -> Optional[List[Dict]]:
        """Get cached search results."""
        return self.query_cache.get(query_hash)
    
    def cache_results(self, query_hash: str, results: List[Dict]):
        """Cache search results."""
        self.query_cache[query_hash] = results
    
    def get_cached_embedding(self, text: str) -> Optional[List[float]]:
        """Get cached embedding."""
        return self.embedding_cache.get(text)
    
    def cache_embedding(self, text: str, embedding: List[float]):
        """Cache embedding for reuse."""
        self.embedding_cache[text] = embedding
```

### Database Query Optimization

Optimized database queries reduce search latency.

```sql
-- Optimized vector search with early termination
WITH ranked_tools AS (
    SELECT 
        tool_id,
        name, 
        description,
        list_dot_product(embedding, $1) as similarity_score,
        ROW_NUMBER() OVER (ORDER BY list_dot_product(embedding, $1) DESC) as rank
    FROM mcp_tools 
    WHERE list_dot_product(embedding, $1) > $2
)
SELECT * FROM ranked_tools 
WHERE rank <= $3
ORDER BY similarity_score DESC;

-- Keyword search with index utilization
SELECT 
    t.tool_id,
    t.name,
    t.description,
    ts_rank_cd(search_vector, plainto_tsquery($1)) as keyword_score
FROM mcp_tools t
WHERE search_vector @@ plainto_tsquery($1)
ORDER BY keyword_score DESC
LIMIT $2;
```

### Parallel Processing

Parallel execution improves performance for complex searches.

```python
class ParallelSearchExecutor:
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
    
    async def execute_parallel_search(
        self, 
        query: str, 
        search_strategies: List[SearchStrategy]
    ) -> List[SearchResult]:
        """Execute multiple search strategies in parallel."""
        
        loop = asyncio.get_event_loop()
        futures = []
        
        for strategy in search_strategies:
            future = loop.run_in_executor(
                self.executor,
                strategy.search,
                query
            )
            futures.append(future)
        
        # Wait for all searches to complete
        results = await asyncio.gather(*futures, return_exceptions=True)
        
        # Filter out exceptions and return successful results
        successful_results = [
            result for result in results 
            if not isinstance(result, Exception)
        ]
        
        return successful_results
```

## Algorithm Selection Strategy

### Automatic Algorithm Selection

The system automatically selects the optimal search algorithm based on query characteristics.

```python
class AlgorithmSelector:
    def __init__(self):
        self.selection_rules = {
            'exact_match': {
                'conditions': ['quoted_terms', 'tool_name_pattern'],
                'algorithm': 'keyword',
                'confidence': 0.9
            },
            'exploratory': {
                'conditions': ['question_words', 'vague_terms'],
                'algorithm': 'semantic', 
                'confidence': 0.8
            },
            'mixed_intent': {
                'conditions': ['technical_and_conceptual'],
                'algorithm': 'hybrid',
                'confidence': 0.7
            }
        }
    
    def select_algorithm(self, query: str, context: Dict = None) -> str:
        """Select optimal algorithm for the given query."""
        query_features = self.extract_query_features(query)
        
        best_match = None
        best_confidence = 0.0
        
        for intent, rule in self.selection_rules.items():
            confidence = self.calculate_rule_confidence(query_features, rule)
            
            if confidence > best_confidence:
                best_confidence = confidence
                best_match = rule['algorithm']
        
        # Apply context adjustments
        if context and best_match:
            best_match = self.adjust_for_context(best_match, context)
        
        return best_match or 'hybrid'  # Default to hybrid
```

## Algorithm Performance Metrics

### Evaluation Metrics

The system tracks multiple metrics to evaluate algorithm performance:

**Relevance Metrics**:
- **Precision@K**: Percentage of relevant results in top K results
- **Recall@K**: Percentage of relevant tools found in top K results  
- **Mean Reciprocal Rank (MRR)**: Average reciprocal rank of first relevant result
- **Normalized Discounted Cumulative Gain (NDCG)**: Ranking quality metric

**Efficiency Metrics**:
- **Query Latency**: Time from query to result delivery
- **Throughput**: Queries processed per second
- **Cache Hit Rate**: Percentage of queries served from cache
- **Resource Utilization**: CPU, memory, and I/O usage

**User Experience Metrics**:
- **Click-Through Rate**: Percentage of results users interact with
- **Task Success Rate**: Percentage of queries leading to successful tool usage
- **User Satisfaction**: Explicit feedback ratings
- **Query Refinement Rate**: Percentage of queries that are reformulated

### Performance Benchmarking

```python
class AlgorithmBenchmark:
    def __init__(self):
        self.test_queries = self.load_benchmark_queries()
        self.ground_truth = self.load_ground_truth_labels()
    
    async def benchmark_algorithm(
        self, 
        algorithm: SearchAlgorithm,
        metric_types: List[str] = None
    ) -> Dict[str, float]:
        """Benchmark algorithm performance."""
        results = {}
        
        if not metric_types:
            metric_types = ['precision', 'recall', 'mrr', 'latency']
        
        for query_id, query in self.test_queries.items():
            start_time = time.time()
            search_results = await algorithm.search(query['text'])
            latency = time.time() - start_time
            
            # Calculate relevance metrics
            if 'precision' in metric_types:
                results[f'precision@5_{query_id}'] = self.calculate_precision_at_k(
                    search_results[:5], self.ground_truth[query_id]
                )
            
            if 'latency' in metric_types:
                results[f'latency_{query_id}'] = latency
        
        # Aggregate results
        return self.aggregate_benchmark_results(results)
```

This technical reference provides comprehensive documentation of the search algorithms powering the MCP Tool Search System. The algorithms are designed to be modular, performant, and adaptive to different use cases and user contexts.