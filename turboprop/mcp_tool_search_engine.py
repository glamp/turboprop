#!/usr/bin/env python3
"""
MCP Tool Search Engine

This module provides the core semantic search engine for MCP tools, enabling
natural language queries to find tools by functionality, purpose, and capabilities.
"""

import time
from typing import Any, Dict, List, Optional

from .database_manager import DatabaseManager
from .embedding_helper import EmbeddingGenerator
from .logging_config import get_logger
from .mcp_metadata_types import MCPToolMetadata, ParameterAnalysis, ToolExample, ToolId
from .parameter_utils import calculate_parameter_counts
from .search_result_formatter import SearchResultFormatter
from .tool_matching_algorithms import ToolMatchingAlgorithms
from .tool_query_processor import ToolQueryProcessor
from .tool_search_results import ProcessedQuery, ToolSearchResponse, ToolSearchResult

logger = get_logger(__name__)

# Search performance configuration
SEARCH_OPTIMIZATIONS = {
    "embedding_cache_size": 1000,  # Cache frequently accessed embeddings
    "result_cache_ttl": 300,  # Cache search results for 5 minutes
    "max_vector_search_results": 100,  # Limit initial vector search
    "similarity_threshold": 0.1,  # Filter very low similarity results
    "batch_embedding_size": 50,  # Batch size for embedding operations
}

# Confidence level thresholds for search results
CONFIDENCE_THRESHOLDS = {
    "high": 0.7,  # High confidence threshold for combined scores
    "medium": 0.4,  # Medium confidence threshold for combined scores
}

# Search strategies
SEARCH_STRATEGIES = {
    "semantic_only": {"semantic_weight": 1.0, "keyword_weight": 0.0},
    "hybrid_balanced": {"semantic_weight": 0.7, "keyword_weight": 0.3},
    "keyword_focused": {"semantic_weight": 0.3, "keyword_weight": 0.7},
    "comprehensive": {"use_examples": True, "use_parameters": True, "use_relationships": True},
}

# Default search parameters
DEFAULT_SEARCH_PARAMS = {
    "k": 10,
    "similarity_threshold": 0.1,
    "semantic_weight": 0.7,
    "keyword_weight": 0.3,
    "ranking_strategy": "relevance",
    "enable_clustering": False,
    "max_clusters": 5,
}


class SearchResultsCache:
    """Simple in-memory cache for search results."""

    def __init__(self, max_size: int = 100, ttl_seconds: int = 300):
        """Initialize the cache with size and TTL limits."""
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: Dict[str, Dict[str, Any]] = {}

    def get(self, cache_key: str) -> Optional[ToolSearchResponse]:
        """Get cached result if still valid."""
        if cache_key not in self._cache:
            return None

        entry = self._cache[cache_key]
        if time.time() - entry["timestamp"] > self.ttl_seconds:
            del self._cache[cache_key]
            return None

        return entry["result"]

    def put(self, cache_key: str, result: ToolSearchResponse) -> None:
        """Store result in cache with timestamp."""
        # Simple LRU: remove oldest if at capacity
        if len(self._cache) >= self.max_size:
            oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k]["timestamp"])
            del self._cache[oldest_key]

        self._cache[cache_key] = {"result": result, "timestamp": time.time()}


class MCPToolSearchEngine:
    """Semantic search engine for MCP tools."""

    def __init__(
        self,
        db_manager: DatabaseManager,
        embedding_generator: EmbeddingGenerator,
        query_processor: Optional[ToolQueryProcessor] = None,
        enable_caching: bool = True,
    ):
        """
        Initialize the MCP tool search engine.

        Args:
            db_manager: Database manager for tool storage and retrieval
            embedding_generator: Generator for semantic embeddings
            query_processor: Optional custom query processor
            enable_caching: Whether to enable result caching
        """
        self.db_manager = db_manager
        self.embedding_generator = embedding_generator
        self.query_processor = query_processor or ToolQueryProcessor()
        self.matching_algorithms = ToolMatchingAlgorithms()
        self.result_formatter = SearchResultFormatter()

        # Initialize caching
        self.enable_caching = enable_caching
        self._results_cache = SearchResultsCache() if enable_caching else None

        logger.info("Initialized MCPToolSearchEngine with caching=%s", enable_caching)

    def search_by_functionality(
        self,
        query: str,
        k: int = DEFAULT_SEARCH_PARAMS["k"],
        category_filter: Optional[str] = None,
        tool_type_filter: Optional[str] = None,
        similarity_threshold: float = DEFAULT_SEARCH_PARAMS["similarity_threshold"],
    ) -> ToolSearchResponse:
        """
        Search tools by functional description.

        Args:
            query: Natural language description of desired functionality
            k: Maximum number of results to return
            category_filter: Optional filter by tool category
            tool_type_filter: Optional filter by tool type
            similarity_threshold: Minimum similarity score threshold

        Returns:
            ToolSearchResponse with search results and metadata
        """
        start_time = time.time()

        try:
            logger.info("Starting functionality search for: '%s'", query)

            # Check cache first
            cache_key = self._generate_cache_key(query, k, category_filter, tool_type_filter)
            cached_result = self._check_search_cache(cache_key, query)
            if cached_result:
                return cached_result

            # Perform semantic search
            search_results = self._perform_semantic_search(
                query, k, category_filter, tool_type_filter, similarity_threshold
            )

            # Create final response with ranking and suggestions
            response = self._create_search_response(query, search_results, k, start_time, cache_key)

            logger.info(
                "Functionality search completed: %d results in %.2fs",
                len(response.results),
                response.execution_time,
            )

            return response

        except Exception as e:
            logger.error("Error in functionality search: %s", e)
            execution_time = time.time() - start_time
            return self._create_error_response(query, str(e), execution_time)

    def search_hybrid(
        self,
        query: str,
        k: int = DEFAULT_SEARCH_PARAMS["k"],
        semantic_weight: float = DEFAULT_SEARCH_PARAMS["semantic_weight"],
        keyword_weight: float = DEFAULT_SEARCH_PARAMS["keyword_weight"],
    ) -> ToolSearchResponse:
        """
        Hybrid search combining semantic and keyword matching.

        Args:
            query: Search query
            k: Maximum number of results to return
            semantic_weight: Weight for semantic similarity (0.0-1.0)
            keyword_weight: Weight for keyword matching (0.0-1.0)

        Returns:
            ToolSearchResponse with hybrid search results
        """
        start_time = time.time()

        try:
            logger.info("Starting hybrid search for: '%s'", query)

            # Process query
            processed_query = self.query_processor.process_query(query)

            # Perform semantic search
            query_embedding = self.embedding_generator.encode(processed_query.cleaned_query).tolist()
            semantic_results = self._perform_vector_search(query_embedding, k * 2)

            # Perform keyword search
            keyword_results = self._perform_keyword_search(query, k * 2)

            # Combine results with weighted scoring
            combined_results = self._combine_search_results(
                semantic_results, keyword_results, semantic_weight, keyword_weight
            )

            # Convert to ToolSearchResult objects
            search_results = self._convert_to_search_results(combined_results, processed_query, query_embedding, 0.05)

            # Rank and filter
            final_results = self.matching_algorithms.rank_results(search_results, "relevance")[:k]

            # Generate response
            execution_time = time.time() - start_time
            response = ToolSearchResponse(
                query=query,
                results=final_results,
                total_results=len(final_results),
                execution_time=execution_time,
                processed_query=processed_query,
                suggested_refinements=self.query_processor.suggest_query_refinements(query, len(final_results)),
                search_strategy="hybrid_balanced",
            )

            logger.info("Hybrid search completed: %d results in %.2fs", len(final_results), execution_time)

            return response

        except Exception as e:
            logger.error("Error in hybrid search: %s", e)
            execution_time = time.time() - start_time
            return self._create_error_response(query, str(e), execution_time)

    def search_by_capability(
        self,
        capability_description: str,
        required_parameters: Optional[List[str]] = None,
        optional_parameters: Optional[List[str]] = None,
        k: int = DEFAULT_SEARCH_PARAMS["k"],
    ) -> ToolSearchResponse:
        """
        Search tools by specific capability requirements.

        Args:
            capability_description: Description of required capability
            required_parameters: List of parameter names that must be present
            optional_parameters: List of parameter names that are preferred
            k: Maximum number of results to return

        Returns:
            ToolSearchResponse with capability-matched results
        """
        start_time = time.time()

        try:
            logger.info("Starting capability search for: '%s'", capability_description)

            # Process capability description
            processed_query = self.query_processor.process_query(capability_description)

            # Generate embedding for capability
            query_embedding = self.embedding_generator.encode(processed_query.cleaned_query).tolist()

            # Perform vector search
            raw_results = self._perform_vector_search(query_embedding, k * 3)

            # Filter by parameter requirements
            if required_parameters or optional_parameters:
                raw_results = self._filter_by_parameter_requirements(
                    raw_results, required_parameters or [], optional_parameters or []
                )

            # Convert to search results
            search_results = self._convert_to_search_results(raw_results, processed_query, query_embedding, 0.1)

            # Enhance with parameter compatibility scoring
            for result in search_results:
                param_score = self._calculate_parameter_match_score(
                    result, required_parameters or [], optional_parameters or []
                )
                result.relevance_score = (result.relevance_score * 0.7) + (param_score * 0.3)

            # Rank and limit results
            final_results = self.matching_algorithms.rank_results(search_results, "relevance")[:k]

            # Generate response
            execution_time = time.time() - start_time
            response = ToolSearchResponse(
                query=capability_description,
                results=final_results,
                total_results=len(final_results),
                execution_time=execution_time,
                processed_query=processed_query,
                search_strategy="capability_focused",
            )

            logger.info(
                "Capability search completed: %d results in %.2fs",
                len(final_results),
                execution_time,
            )

            return response

        except Exception as e:
            logger.error("Error in capability search: %s", e)
            execution_time = time.time() - start_time
            return self._create_error_response(capability_description, str(e), execution_time)

    def get_tool_alternatives(self, tool_id: ToolId, k: int = 5) -> ToolSearchResponse:
        """
        Find alternative tools to a given tool.

        Args:
            tool_id: ID of the tool to find alternatives for
            k: Maximum number of alternatives to return

        Returns:
            ToolSearchResponse with alternative tools
        """
        start_time = time.time()

        try:
            # Get the reference tool
            tool_data = self.db_manager.get_mcp_tool(tool_id)
            if not tool_data:
                raise ValueError(f"Tool {tool_id} not found")

            # Search for tools with similar functionality but different implementation
            query = f"alternative to {tool_data['name']} {tool_data['description']}"
            processed_query = self.query_processor.process_query(query)

            # Generate embedding
            query_embedding = self.embedding_generator.encode(processed_query.cleaned_query).tolist()

            # Search for similar tools
            raw_results = self._perform_vector_search(query_embedding, k * 2)

            # Filter out the original tool
            raw_results = [r for r in raw_results if r[0] != tool_id]

            # Convert to search results
            search_results = self._convert_to_search_results(raw_results, processed_query, query_embedding, 0.2)

            # Rank by alternative suitability
            final_results = self.matching_algorithms.rank_results(search_results, "similarity")[:k]

            # Create response
            execution_time = time.time() - start_time
            response = ToolSearchResponse(
                query=f"Alternatives to {tool_data['name']}",
                results=final_results,
                total_results=len(final_results),
                execution_time=execution_time,
                processed_query=processed_query,
                search_strategy="alternative_finding",
            )

            return response

        except Exception as e:
            logger.error("Error finding tool alternatives: %s", e)
            execution_time = time.time() - start_time
            return self._create_error_response(f"alternatives to {tool_id}", str(e), execution_time)

    def _perform_vector_search(
        self,
        query_embedding: List[float],
        limit: int,
        category_filter: Optional[str] = None,
        tool_type_filter: Optional[str] = None,
    ) -> List[tuple]:
        """
        Perform vector similarity search in the database.

        Args:
            query_embedding: Query embedding vector
            limit: Maximum results to return
            category_filter: Optional category filter
            tool_type_filter: Optional tool type filter

        Returns:
            List of tuples (tool_id, similarity_score, tool_data)
        """
        try:
            results = self.db_manager.search_mcp_tools_by_embedding(
                query_embedding=query_embedding,
                limit=min(limit, SEARCH_OPTIMIZATIONS["max_vector_search_results"]),
                category=category_filter,
                tool_type=tool_type_filter,
            )

            # Filter by similarity threshold
            filtered_results = []
            for result in results:
                similarity_score = result.get("similarity_score", 0.0)
                if similarity_score >= SEARCH_OPTIMIZATIONS["similarity_threshold"]:
                    filtered_results.append((result["id"], similarity_score, result))

            logger.debug("Vector search returned %d results", len(filtered_results))
            return filtered_results

        except Exception as e:
            logger.error("Error in vector search: %s", e)
            return []

    def _perform_keyword_search(self, query: str, limit: int) -> List[tuple]:
        """
        Perform keyword-based search using enhanced full-text search with ranking.

        Args:
            query: Search query string
            limit: Maximum results to return

        Returns:
            List of tuples (tool_id, keyword_score, tool_data)
        """
        try:
            with self.db_manager.get_connection() as conn:
                # Prepare normalized query terms for better matching
                normalized_terms = self._normalize_search_terms(query)

                # Enhanced full-text search with better ranking
                search_results = self._execute_enhanced_keyword_search(conn, normalized_terms, limit)

                # Convert to expected format
                keyword_results = []
                for row in search_results:
                    tool_data = {
                        "id": row[0],
                        "name": row[1],
                        "description": row[2],
                        "category": row[3],
                        "tool_type": row[4],
                        "metadata_json": row[5],
                    }
                    keyword_results.append((row[0], row[6], tool_data))

                logger.debug("Enhanced keyword search returned %d results", len(keyword_results))
                return keyword_results

        except Exception as e:
            logger.error("Error in enhanced keyword search: %s", e)
            # Fallback to basic search
            return self._perform_basic_keyword_search(query, limit)

    def _normalize_search_terms(self, query: str) -> List[str]:
        """Normalize search terms for better matching with stemming and expansion."""
        import re

        # Clean and split query into terms
        terms = re.findall(r"\b\w+\b", query.lower())

        # Remove very short terms
        terms = [term for term in terms if len(term) > 2]

        # Basic stemming - remove common suffixes
        normalized_terms = []
        for term in terms:
            # Remove common programming suffixes
            stemmed = term
            suffixes = ["ing", "ed", "er", "ly", "ion", "tion", "ness"]
            for suffix in suffixes:
                if stemmed.endswith(suffix) and len(stemmed) > len(suffix) + 2:
                    stemmed = stemmed[: -len(suffix)]
                    break
            normalized_terms.append(stemmed)

        return list(set(normalized_terms))  # Remove duplicates

    def _execute_enhanced_keyword_search(self, conn, terms: List[str], limit: int) -> List:
        """Execute enhanced keyword search with improved ranking algorithm."""
        # Build dynamic search query with weighted scoring
        search_conditions = []
        score_conditions = []
        params = []

        for i, term in enumerate(terms):
            pattern = f"%{term}%"
            search_conditions.append("(LOWER(name) LIKE ? OR LOWER(description) LIKE ?)")
            score_conditions.append(
                """
                CASE
                    WHEN LOWER(name) LIKE ? THEN 2.0
                    WHEN LOWER(description) LIKE ? THEN 1.0
                    ELSE 0.0
                END
            """
            )
            # Add parameters for conditions and scoring
            params.extend([pattern, pattern, pattern, pattern])

        if not search_conditions:
            return []

        search_sql = f"""
            SELECT id, name, description, category, tool_type, metadata_json,
                   ({' + '.join(score_conditions)}) / {len(terms)} as keyword_score
            FROM mcp_tools
            WHERE {' OR '.join(search_conditions)}
            ORDER BY keyword_score DESC, name ASC
            LIMIT ?
        """

        params.append(limit)
        return conn.execute(search_sql, params).fetchall()

    def _perform_basic_keyword_search(self, query: str, limit: int) -> List[tuple]:
        """Fallback to basic keyword search when enhanced search fails."""
        try:
            with self.db_manager.get_connection() as conn:
                search_sql = """
                    SELECT id, name, description, category, tool_type, metadata_json,
                           CASE
                               WHEN LOWER(name) LIKE LOWER(?) THEN 1.0
                               WHEN LOWER(description) LIKE LOWER(?) THEN 0.8
                               ELSE 0.5
                           END as keyword_score
                    FROM mcp_tools
                    WHERE LOWER(name) LIKE LOWER(?)
                       OR LOWER(description) LIKE LOWER(?)
                    ORDER BY keyword_score DESC
                    LIMIT ?
                """

                query_pattern = f"%{query}%"
                results = conn.execute(
                    search_sql, (query_pattern, query_pattern, query_pattern, query_pattern, limit)
                ).fetchall()

                return results

        except Exception as e:
            logger.error("Error in basic keyword search fallback: %s", e)
            return []

    def _combine_search_results(
        self,
        semantic_results: List[tuple],
        keyword_results: List[tuple],
        semantic_weight: float,
        keyword_weight: float,
    ) -> List[tuple]:
        """
        Combine semantic and keyword search results with weighted scoring.

        Args:
            semantic_results: Results from semantic search
            keyword_results: Results from keyword search
            semantic_weight: Weight for semantic scores
            keyword_weight: Weight for keyword scores

        Returns:
            Combined results with weighted scores
        """
        combined_scores = {}

        # Process semantic results
        for tool_id, semantic_score, tool_data in semantic_results:
            combined_scores[tool_id] = {"semantic_score": semantic_score, "keyword_score": 0.0, "tool_data": tool_data}

        # Process keyword results
        for tool_id, keyword_score, tool_data in keyword_results:
            if tool_id in combined_scores:
                combined_scores[tool_id]["keyword_score"] = keyword_score
            else:
                combined_scores[tool_id] = {
                    "semantic_score": 0.0,
                    "keyword_score": keyword_score,
                    "tool_data": tool_data,
                }

        # Calculate combined scores
        combined_results = []
        for tool_id, scores in combined_scores.items():
            combined_score = scores["semantic_score"] * semantic_weight + scores["keyword_score"] * keyword_weight
            combined_results.append((tool_id, combined_score, scores["tool_data"]))

        # Sort by combined score
        combined_results.sort(key=lambda x: x[1], reverse=True)

        return combined_results

    def _convert_to_search_results(
        self,
        raw_results: List[tuple],
        processed_query: ProcessedQuery,
        query_embedding: List[float],
        similarity_threshold: float,
    ) -> List[ToolSearchResult]:
        """
        Convert database results to ToolSearchResult objects.

        Args:
            raw_results: Raw database results
            processed_query: Processed query object
            query_embedding: Query embedding vector
            similarity_threshold: Minimum similarity threshold

        Returns:
            List of ToolSearchResult objects
        """
        search_results = []

        for tool_id, similarity_score, tool_data in raw_results:
            try:
                search_result = self._process_single_search_result(
                    tool_id, similarity_score, tool_data, processed_query, similarity_threshold
                )
                if search_result:
                    search_results.append(search_result)
            except Exception as e:
                logger.error("Error converting result for tool %s: %s", tool_id, e)
                continue

        return search_results

    def _process_single_search_result(
        self,
        tool_id: str,
        similarity_score: float,
        tool_data: Dict[str, Any],
        processed_query: ProcessedQuery,
        similarity_threshold: float,
    ) -> Optional[ToolSearchResult]:
        """Process a single search result and return ToolSearchResult if it meets criteria."""
        # Skip if below threshold
        if similarity_score < similarity_threshold:
            return None

        # Get tool metadata
        tool_metadata = self._build_tool_metadata_from_db_result(tool_data)

        # Calculate scores and confidence
        relevance_score = self.matching_algorithms.calculate_relevance_score(
            similarity_score, tool_metadata, processed_query
        )
        confidence_level = self._determine_confidence_level(similarity_score, relevance_score)

        # Generate match reasons
        scores = {
            "semantic_similarity": similarity_score,
            "relevance_score": relevance_score,
        }
        match_reasons = self.matching_algorithms.explain_match_reasons(tool_metadata, processed_query, scores)

        # Get parameters and examples
        parameters = self._get_tool_parameters(ToolId(tool_id))
        examples = self._get_tool_examples(ToolId(tool_id))

        # Create and return search result
        return ToolSearchResult(
            tool_id=ToolId(tool_id),
            name=tool_data.get("name", "Unknown"),
            description=tool_data.get("description", ""),
            category=tool_data.get("category", "unknown"),
            tool_type=tool_data.get("tool_type", "unknown"),
            similarity_score=similarity_score,
            relevance_score=relevance_score,
            confidence_level=confidence_level,
            match_reasons=match_reasons,
            parameters=parameters,
            parameter_count=calculate_parameter_counts(parameters)[0],
            required_parameter_count=calculate_parameter_counts(parameters)[1],
            complexity_score=tool_metadata.complexity_analysis.overall_complexity
            if tool_metadata.complexity_analysis
            else 0.0,
            examples=examples,
        )

    def _build_tool_metadata_from_db_result(self, tool_data: Dict[str, Any]) -> MCPToolMetadata:
        """Build MCPToolMetadata from database result."""
        # This is a simplified version - in practice you'd reconstruct full metadata
        return MCPToolMetadata(
            name=tool_data.get("name", ""),
            description=tool_data.get("description", ""),
            category=tool_data.get("category", "unknown"),
        )

    def _get_tool_parameters(self, tool_id: ToolId) -> List[ParameterAnalysis]:
        """Get parameter analysis for a tool."""
        try:
            param_dicts = self.db_manager.get_tool_parameters(tool_id)
            parameters = []

            for param_dict in param_dicts:
                param = ParameterAnalysis(
                    name=param_dict["parameter_name"],
                    type=param_dict["parameter_type"] or "string",
                    required=param_dict["is_required"],
                    description=param_dict["description"] or "",
                )
                parameters.append(param)

            return parameters
        except Exception as e:
            logger.error("Error getting parameters for tool %s: %s", tool_id, e)
            return []

    def _get_tool_examples(self, tool_id: ToolId) -> List[ToolExample]:
        """Get examples for a tool."""
        try:
            example_dicts = self.db_manager.get_tool_examples(tool_id)
            examples = []

            for example_dict in example_dicts:
                example = ToolExample(
                    use_case=example_dict["use_case"] or "",
                    example_call=example_dict["example_call"] or "",
                    expected_output=example_dict["expected_output"] or "",
                    context=example_dict["context"] or "",
                    effectiveness_score=example_dict["effectiveness_score"] or 0.8,
                )
                examples.append(example)

            return examples
        except Exception as e:
            logger.error("Error getting examples for tool %s: %s", tool_id, e)
            return []

    def _determine_confidence_level(self, similarity_score: float, relevance_score: float) -> str:
        """Determine confidence level based on scores."""
        combined_score = (similarity_score + relevance_score) / 2

        if combined_score > CONFIDENCE_THRESHOLDS["high"]:
            return "high"
        elif combined_score > CONFIDENCE_THRESHOLDS["medium"]:
            return "medium"
        else:
            return "low"

    def _generate_cache_key(
        self, query: str, k: int, category_filter: Optional[str], tool_type_filter: Optional[str]
    ) -> str:
        """Generate cache key for search parameters."""
        return f"{hash(query)}_{k}_{category_filter}_{tool_type_filter}"

    def _check_search_cache(self, cache_key: str, query: str) -> Optional[ToolSearchResponse]:
        """Check if search results are cached and return them if found."""
        if self._results_cache:
            cached_result = self._results_cache.get(cache_key)
            if cached_result:
                # Validate that cached result is actually a ToolSearchResponse
                if isinstance(cached_result, ToolSearchResponse):
                    logger.debug("Returning cached result for query: '%s'", query)
                    return cached_result
                else:
                    # Cache corruption detected, remove invalid entry
                    logger.warning("Cache corruption detected for query: '%s', removing invalid entry", query)
                    # Note: we can't directly remove from cache here since we don't have access to the cache's internals
        return None

    def _perform_semantic_search(
        self,
        query: str,
        k: int,
        category_filter: Optional[str],
        tool_type_filter: Optional[str],
        similarity_threshold: float,
    ) -> List[ToolSearchResult]:
        """Perform semantic search and convert results to ToolSearchResult objects."""
        # Process and expand query
        processed_query = self.query_processor.process_query(query)

        # Generate query embedding
        query_embedding = self.embedding_generator.encode(processed_query.cleaned_query).tolist()

        # Perform vector similarity search
        raw_results = self._perform_vector_search(query_embedding, k * 2, category_filter, tool_type_filter)

        # Convert to ToolSearchResult objects
        return self._convert_to_search_results(raw_results, processed_query, query_embedding, similarity_threshold)

    def _create_search_response(
        self, query: str, search_results: List[ToolSearchResult], k: int, start_time: float, cache_key: str
    ) -> ToolSearchResponse:
        """Create final search response with ranking, suggestions, and caching."""
        # Apply ranking and filtering
        final_results = self.matching_algorithms.rank_results(search_results, "relevance")[:k]

        # Get processed query for suggestions
        processed_query = self.query_processor.process_query(query)

        # Generate suggestions
        suggestions = self.query_processor.suggest_query_refinements(query, len(final_results))

        # Create response
        execution_time = time.time() - start_time
        response = ToolSearchResponse(
            query=query,
            results=final_results,
            total_results=len(final_results),
            execution_time=execution_time,
            processed_query=processed_query,
            suggested_refinements=suggestions,
            search_strategy="semantic_only",
        )

        # Cache result
        if self._results_cache:
            self._results_cache.put(cache_key, response)

        return response

    def _filter_by_parameter_requirements(
        self, raw_results: List[tuple], required_params: List[str], optional_params: List[str]
    ) -> List[tuple]:
        """Filter results by parameter requirements."""
        # Placeholder implementation - would check actual parameters
        return raw_results  # Return all for now

    def _calculate_parameter_match_score(
        self, result: ToolSearchResult, required_params: List[str], optional_params: List[str]
    ) -> float:
        """Calculate how well tool parameters match requirements."""
        if not required_params and not optional_params:
            return 0.8  # Neutral score

        # Check if required parameters are present
        tool_param_names = {p.name.lower() for p in result.parameters}
        required_matches = sum(1 for req in required_params if req.lower() in tool_param_names)
        optional_matches = sum(1 for opt in optional_params if opt.lower() in tool_param_names)

        total_required = len(required_params)
        total_optional = len(optional_params)

        if total_required > 0:
            required_score = required_matches / total_required
        else:
            required_score = 1.0

        if total_optional > 0:
            optional_score = optional_matches / total_optional
        else:
            optional_score = 0.5

        # Weight required parameters more heavily
        return (required_score * 0.7) + (optional_score * 0.3)

    def _create_error_response(self, query: str, error_message: str, execution_time: float) -> ToolSearchResponse:
        """Create error response for failed searches."""
        return ToolSearchResponse(
            query=query,
            results=[],
            total_results=0,
            execution_time=execution_time,
            suggested_refinements=[f"Error: {error_message}"],
            search_strategy="error",
        )
