#!/usr/bin/env python3
"""
mcp_response_types.py: Structured response data types for MCP tools.

This module defines comprehensive dataclasses for MCP tool responses that provide
structured JSON data while maintaining backward compatibility. These types allow
Claude to process responses programmatically with rich metadata and suggestions.

Classes:
- SearchResponse: Comprehensive search results with metadata and suggestions
- IndexResponse: Indexing operation results and statistics
- StatusResponse: Index status and health information
- QueryAnalysis: Analysis of search queries with suggestions
- ResultCluster: Grouped results by similarity or characteristics
"""

import json
import time
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Any, Union
from pathlib import Path

from search_result_types import CodeSearchResult, SearchMetadata


@dataclass
class QueryAnalysis:
    """
    Analysis of a search query with suggestions and insights.
    
    Provides metadata about the search query to help users understand
    and refine their searches for better results.
    """
    original_query: str
    suggested_refinements: List[str] = field(default_factory=list)
    query_complexity: str = "medium"  # low, medium, high
    estimated_result_count: Optional[int] = None
    search_hints: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass 
class ResultCluster:
    """
    A cluster of related search results grouped by similarity or characteristics.
    
    Groups results to help users understand patterns and relationships
    in their search results.
    """
    cluster_name: str
    cluster_type: str  # language, directory, similarity, etc.
    results: List[CodeSearchResult] = field(default_factory=list)
    cluster_score: float = 0.0
    cluster_description: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'cluster_name': self.cluster_name,
            'cluster_type': self.cluster_type,
            'results': [result.to_dict() for result in self.results],
            'cluster_score': self.cluster_score,
            'cluster_description': self.cluster_description,
            'result_count': len(self.results)
        }


@dataclass
class SearchResponse:
    """
    Comprehensive structured response for search operations.
    
    Provides rich metadata, suggestions, and organized results that Claude
    can process programmatically while maintaining human readability.
    """
    # Core search results
    query: str
    results: List[CodeSearchResult] = field(default_factory=list)
    total_results: int = 0
    
    # Execution metadata
    execution_time: Optional[float] = None
    timestamp: Optional[str] = None
    version: str = "1.0"
    
    # Enhanced metadata
    query_analysis: Optional[QueryAnalysis] = None
    result_clusters: List[ResultCluster] = field(default_factory=list)
    confidence_distribution: Dict[str, int] = field(default_factory=dict)
    language_breakdown: Dict[str, int] = field(default_factory=dict)
    
    # Suggestions and improvements
    suggested_queries: List[str] = field(default_factory=list)
    navigation_hints: List[str] = field(default_factory=list)
    performance_notes: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Initialize computed fields after construction."""
        if self.timestamp is None:
            self.timestamp = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
        
        if self.total_results == 0:
            self.total_results = len(self.results)
            
        # Auto-compute confidence distribution if not provided
        if not self.confidence_distribution and self.results:
            self._compute_confidence_distribution()
            
        # Auto-compute language breakdown if not provided  
        if not self.language_breakdown and self.results:
            self._compute_language_breakdown()
    
    def _compute_confidence_distribution(self):
        """Compute confidence level distribution from results."""
        self.confidence_distribution = {'high': 0, 'medium': 0, 'low': 0}
        for result in self.results:
            if result.confidence_level:
                self.confidence_distribution[result.confidence_level] += 1
    
    def _compute_language_breakdown(self):
        """Compute language distribution from results."""
        self.language_breakdown = {}
        for result in self.results:
            if result.file_metadata and 'language' in result.file_metadata:
                lang = result.file_metadata['language']
                self.language_breakdown[lang] = self.language_breakdown.get(lang, 0) + 1
    
    def add_cluster(self, cluster: ResultCluster) -> None:
        """Add a result cluster to the response."""
        self.result_clusters.append(cluster)
    
    def add_suggestion(self, suggestion: str) -> None:
        """Add a suggested query refinement."""
        if suggestion not in self.suggested_queries:
            self.suggested_queries.append(suggestion)
    
    def add_navigation_hint(self, hint: str) -> None:
        """Add a navigation hint for IDE integration."""
        if hint not in self.navigation_hints:
            self.navigation_hints.append(hint)
            
    def to_json(self) -> str:
        """Convert to JSON string for MCP tool responses."""
        return json.dumps(self.to_dict(), indent=2)
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result_dict = {
            'query': self.query,
            'results': [result.to_dict() for result in self.results],
            'total_results': self.total_results,
            'execution_time': self.execution_time,
            'timestamp': self.timestamp,
            'version': self.version,
            'confidence_distribution': self.confidence_distribution,
            'language_breakdown': self.language_breakdown,
            'suggested_queries': self.suggested_queries,
            'navigation_hints': self.navigation_hints,
            'performance_notes': self.performance_notes
        }
        
        # Include optional fields if present
        if self.query_analysis:
            result_dict['query_analysis'] = self.query_analysis.to_dict()
            
        if self.result_clusters:
            result_dict['result_clusters'] = [cluster.to_dict() for cluster in self.result_clusters]
            result_dict['cluster_count'] = len(self.result_clusters)
            
        return result_dict


@dataclass
class IndexResponse:
    """
    Structured response for indexing operations.
    
    Provides detailed information about indexing operations including
    progress, statistics, and any issues encountered.
    """
    operation: str  # index, reindex, incremental_update
    status: str  # success, partial, failed, in_progress
    message: str
    
    # File processing statistics
    files_processed: int = 0
    files_skipped: int = 0
    files_failed: int = 0
    total_files_scanned: int = 0
    
    # Performance metrics
    execution_time: Optional[float] = None
    processing_rate: Optional[float] = None  # files per second
    
    # Index statistics
    total_embeddings: int = 0
    database_size_mb: Optional[float] = None
    
    # Configuration used
    max_file_size_mb: Optional[float] = None
    repository_path: Optional[str] = None
    
    # Issues and warnings
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    
    # Metadata
    timestamp: Optional[str] = None
    version: str = "1.0"
    
    def __post_init__(self):
        """Initialize computed fields after construction."""
        if self.timestamp is None:
            self.timestamp = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
            
        # Compute processing rate if possible
        if self.execution_time and self.execution_time > 0 and self.files_processed > 0:
            self.processing_rate = self.files_processed / self.execution_time
    
    def add_warning(self, warning: str) -> None:
        """Add a warning message."""
        if warning not in self.warnings:
            self.warnings.append(warning)
            
    def add_error(self, error: str) -> None:
        """Add an error message.""" 
        if error not in self.errors:
            self.errors.append(error)
            
    def is_successful(self) -> bool:
        """Check if the indexing operation was successful."""
        return self.status == "success" and not self.errors
        
    def to_json(self) -> str:
        """Convert to JSON string for MCP tool responses."""
        return json.dumps(self.to_dict(), indent=2)
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass  
class StatusResponse:
    """
    Structured response for index status and health information.
    
    Provides comprehensive information about the current state of the
    code index including health metrics and recommendations.
    """
    # Index health
    status: str  # healthy, degraded, offline, building
    is_ready_for_search: bool
    
    # Index statistics
    total_files: int = 0
    files_with_embeddings: int = 0
    total_embeddings: int = 0
    
    # Database information
    database_path: Optional[str] = None
    database_size_mb: Optional[float] = None
    
    # Configuration
    repository_path: Optional[str] = None
    embedding_model: Optional[str] = None
    embedding_dimensions: Optional[int] = None
    
    # File watcher status
    watcher_active: bool = False
    watcher_status: Optional[str] = None
    
    # Freshness information
    last_index_time: Optional[str] = None
    files_needing_update: int = 0
    is_index_fresh: bool = True
    freshness_reason: Optional[str] = None
    
    # File type breakdown
    file_types: Dict[str, int] = field(default_factory=dict)
    
    # Health recommendations
    recommendations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Metadata
    timestamp: Optional[str] = None
    version: str = "1.0"
    
    def __post_init__(self):
        """Initialize computed fields after construction."""
        if self.timestamp is None:
            self.timestamp = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    
    def add_recommendation(self, recommendation: str) -> None:
        """Add a health recommendation."""
        if recommendation not in self.recommendations:
            self.recommendations.append(recommendation)
            
    def add_warning(self, warning: str) -> None:
        """Add a warning message."""
        if warning not in self.warnings:
            self.warnings.append(warning)
            
    def compute_health_score(self) -> float:
        """Compute a health score from 0-100."""
        score = 100.0
        
        # Reduce score for missing files
        if self.total_files > 0:
            embedding_ratio = self.files_with_embeddings / self.total_files
            if embedding_ratio < 1.0:
                score -= (1.0 - embedding_ratio) * 30
                
        # Reduce score for stale index
        if not self.is_index_fresh:
            score -= 20
            
        # Reduce score for warnings
        score -= len(self.warnings) * 5
        
        # Reduce score if not ready for search
        if not self.is_ready_for_search:
            score -= 30
            
        return max(0.0, score)
    
    def to_json(self) -> str:
        """Convert to JSON string for MCP tool responses."""
        return json.dumps(self.to_dict(), indent=2)
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result_dict = asdict(self)
        result_dict['health_score'] = self.compute_health_score()
        return result_dict


# Utility functions for creating responses

def create_search_response_from_results(
    query: str,
    results: List[CodeSearchResult], 
    execution_time: Optional[float] = None,
    add_clusters: bool = True,
    add_suggestions: bool = True
) -> SearchResponse:
    """
    Create a comprehensive SearchResponse from search results.
    
    Args:
        query: Original search query
        results: List of search results
        execution_time: Optional execution time in seconds
        add_clusters: Whether to add result clustering
        add_suggestions: Whether to add query suggestions
        
    Returns:
        SearchResponse with enhanced metadata
    """
    response = SearchResponse(
        query=query,
        results=results,
        execution_time=execution_time
    )
    
    # Add clustering if requested and we have results
    if add_clusters and results:
        response.result_clusters = _create_result_clusters(results)
        
    # Add suggestions if requested
    if add_suggestions:
        response.suggested_queries = _generate_query_suggestions(query, results)
        response.navigation_hints = _generate_navigation_hints(results)
        
    return response


def _create_result_clusters(results: List[CodeSearchResult]) -> List[ResultCluster]:
    """
    Create result clusters from search results.
    
    Groups results by programming language and directory structure.
    """
    clusters = []
    
    # Cluster by language
    language_groups = {}
    for result in results:
        if result.file_metadata and 'language' in result.file_metadata:
            lang = result.file_metadata['language']
            if lang not in language_groups:
                language_groups[lang] = []
            language_groups[lang].append(result)
    
    for lang, lang_results in language_groups.items():
        if len(lang_results) > 1:  # Only create clusters with multiple results
            cluster = ResultCluster(
                cluster_name=f"{lang} Files",
                cluster_type="language",
                results=lang_results,
                cluster_score=sum(r.similarity_score for r in lang_results) / len(lang_results),
                cluster_description=f"Results from {lang} source files"
            )
            clusters.append(cluster)
    
    # Cluster by directory if we have multiple results from same directories
    directory_groups = {}
    for result in results:
        directory = str(Path(result.file_path).parent)
        if directory not in directory_groups:
            directory_groups[directory] = []
        directory_groups[directory].append(result)
    
    for directory, dir_results in directory_groups.items():
        if len(dir_results) > 1:  # Only create clusters with multiple results
            cluster = ResultCluster(
                cluster_name=f"{Path(directory).name}/ directory",
                cluster_type="directory", 
                results=dir_results,
                cluster_score=sum(r.similarity_score for r in dir_results) / len(dir_results),
                cluster_description=f"Results from {directory} directory"
            )
            clusters.append(cluster)
    
    return clusters


def _generate_query_suggestions(query: str, results: List[CodeSearchResult]) -> List[str]:
    """Generate query refinement suggestions based on results."""
    suggestions = []
    
    if not results:
        suggestions.extend([
            f"Try broader terms related to '{query}'",
            "Check if the repository has been indexed recently",
            "Consider using synonyms or related technical terms"
        ])
    elif len(results) < 3:
        suggestions.extend([
            f"Try more specific terms to narrow down '{query}'",
            "Add programming language or framework context",
            "Include related function or class names"
        ])
    elif len(results) > 15:
        suggestions.extend([
            f"Add more specific context to '{query}' to narrow results",
            "Include file type or directory constraints",
            "Focus on particular implementation patterns"
        ])
    
    # Add language-specific suggestions based on results
    languages = set()
    for result in results:
        if result.file_metadata and 'language' in result.file_metadata:
            languages.add(result.file_metadata['language'])
    
    if len(languages) > 1:
        for lang in languages:
            suggestions.append(f"Search specifically in {lang} files: '{query} {lang.lower()}'")
    
    return suggestions[:5]  # Limit to top 5 suggestions


def _generate_navigation_hints(results: List[CodeSearchResult]) -> List[str]:
    """Generate navigation hints for IDE integration."""
    hints = []
    
    if results:
        hints.append(f"Found {len(results)} matches - use Cmd/Ctrl+Click to navigate")
        
        # Check for related files in same directory
        directories = set(str(Path(r.file_path).parent) for r in results)
        if len(directories) == 1:
            hints.append("All results are in the same directory - consider exploring related files")
        elif len(directories) <= 3:
            hints.append("Results span multiple related directories - check for patterns")
            
        # Check for similar file types
        extensions = set(Path(r.file_path).suffix for r in results)
        if len(extensions) == 1:
            ext = extensions.pop()
            hints.append(f"All results are {ext} files - consider searching in other file types")
    
    return hints