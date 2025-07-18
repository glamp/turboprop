#!/usr/bin/env python3
"""
server.py: FastAPI MCP server wrapper around code_index functions.

This module provides a REST API interface to the code indexing functionality,
making it accessible over HTTP for integration with other tools and services.
The server exposes the core code_index operations through web endpoints.

Key features:
- RESTful API for indexing and searching code repositories
- Automatic background watching of the current directory
- JSON request/response format for easy integration
- FastAPI with automatic OpenAPI documentation

Endpoints:
- POST /index: Build or rebuild the code index for a repository
- GET /search: Search the index for semantically similar code

This server is particularly useful for:
- Integration with IDEs and editors
- Building web-based code search interfaces
- Providing code search as a microservice
- MCP (Model Context Protocol) server implementations
"""

# Standard library imports
import contextlib
import threading
from pathlib import Path

# Web framework and data validation
from fastapi import FastAPI
from pydantic import BaseModel

# Import our core indexing functionality
# Note: reindex_all is referenced but may need to be implemented
from code_index import (
    DIMENSIONS,
    EMBED_MODEL,
    TABLE_NAME,
    init_db,
    reindex_all,
    search_index,
    watch_mode,
)
from config import config
from embedding_helper import EmbeddingGenerator


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handle application startup and shutdown events using lifespan context manager.

    This replaces the deprecated @app.on_event decorators with the modern
    lifespan approach recommended by FastAPI.
    """
    # Startup: Start background file watching
    print("[server] Starting background file watcher for current directory")
    watcher_thread = threading.Thread(
        # watch current dir with configurable settings
        target=lambda: watch_mode(
            config.server.WATCH_DIRECTORY,
            config.server.WATCH_MAX_FILE_SIZE_MB,
            config.server.WATCH_DEBOUNCE_SECONDS,
        ),
        daemon=True,  # Allow clean server shutdown without waiting for this thread
    )
    watcher_thread.start()

    yield  # Server is running

    # Shutdown: Clean up database connections
    if db_manager:
        db_manager.cleanup()
        print("[server] Database connections cleaned up")


# Initialize FastAPI application with metadata
app = FastAPI(
    title="Code Index MCP",
    description="Semantic code search and indexing API",
    version="0.1.9",
    lifespan=lifespan,
)

# Initialize shared resources that persist across requests
# These are created once when the server starts
current_dir = Path(".").resolve()  # Use current working directory as repo path
db_manager = init_db(current_dir)  # Database manager
# Initialize ML model with MPS compatibility handling
try:
    embedder = EmbeddingGenerator(EMBED_MODEL)  # ML model for embeddings
except Exception as e:
    print(f"❌ Failed to initialize embedding generator: {e}")
    raise e


# Pydantic models for request/response validation and documentation
class IndexRequest(BaseModel):
    """
    Request model for the /index endpoint.

    Attributes:
        repo: Path to the repository to index (can be relative or absolute)
        max_mb: Maximum file size in megabytes to include in indexing
               Files larger than this will be skipped to avoid memory issues
               and excessive processing time. Default is 1.0 MB.
    """

    repo: str
    max_mb: float = 1.0


class SearchResponse(BaseModel):
    """
    Response model for individual search results.

    This model defines the structure of each search result returned by
    the /search endpoint. FastAPI automatically generates JSON schema
    documentation from this model.

    Attributes:
        path: Absolute path to the file containing the matched code
        snippet: First 300 characters of the file content for preview
        distance: Cosine distance score (0.0 = identical, 1.0 = completely different)
                 Lower values indicate higher similarity to the search query
    """

    path: str
    snippet: str
    distance: float


# API endpoint definitions
@app.post("/index")
def http_index(req: IndexRequest):
    """
    Build or rebuild the code index for a specified repository.

    This endpoint triggers a full reindexing of the specified repository,
    which involves:
    1. Scanning all code files in the repository
    2. Generating semantic embeddings for each file
    3. Storing embeddings in the DuckDB database
    4. Building the HNSW search index for fast similarity queries

    This operation can take significant time for large repositories,
    especially on first run when the ML model needs to process all files.

    Args:
        req: IndexRequest containing repository path and size limits

    Returns:
        JSON response with indexing status and file count

    Example:
        POST /index
        {
            "repo": "/path/to/my/project",
            "max_mb": 2.0
        }

        Response:
        {
            "status": "indexed",
            "files": 1247
        }
    """
    try:
        # Convert megabytes to bytes for internal processing
        max_bytes = int(req.max_mb * 1024**2)

        # Trigger full reindexing of the specified repository
        total_files, processed_files, elapsed = reindex_all(
            Path(req.repo), max_bytes, db_manager, embedder
        )

        # Count total files in database to report back to user
        count_result = db_manager.execute_with_retry(
            f"SELECT count(*) FROM {TABLE_NAME}"
        )
        count = count_result[0][0] if count_result and count_result[0] else 0

        return {"status": "indexed", "files": count}

    except Exception as e:
        from fastapi import HTTPException

        raise HTTPException(status_code=500, detail=f"Indexing failed: {str(e)}")


@app.get("/search", response_model=list[SearchResponse])
def http_search(query: str, k: int = 5):
    """
    Search the code index for files semantically similar to a query.

    This endpoint performs semantic search over the indexed code files,
    returning the most relevant matches based on the meaning of the code
    rather than just keyword matching.

    The search process:
    1. Generate an embedding for the search query using the same ML model
    2. Use HNSW index to find files with similar embeddings (cosine similarity)
    3. Retrieve file details from the database
    4. Return ranked results with similarity scores

    Args:
        query: Natural language or code snippet to search for
               Examples: "function to parse JSON", "JWT authentication", "def calculate_tax"
        k: Maximum number of results to return (default: 5, max recommended: 20)

    Returns:
        List of SearchResponse objects, ordered by similarity (best matches first)

    Example:
        GET /search?query=parse%20JSON%20data&k=3

        Response:
        [
            {
                "path": "/src/utils/json_parser.py",
                "snippet": "def parse_json_data(raw_data):\n    \"\"\"Parse JSON string into Python dict...",
                "distance": 0.234
            },
            ...
        ]
    """
    try:
        # Perform semantic search using the core search function
        results = search_index(db_manager, embedder, query, k)

        # Convert results to the standardized response format
        # The list comprehension transforms tuples to structured objects
        return [SearchResponse(path=p, snippet=s, distance=d) for p, s, d in results]

    except Exception as e:
        from fastapi import HTTPException

        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.get("/status")
def http_status():
    """
    Get the current status of the code index.

    This endpoint provides information about the current state of the index,
    including the number of indexed files, database size, and whether the
    search index is ready for queries.

    Useful for:
    - Checking if indexing is complete
    - Monitoring index health in CI/CD pipelines
    - Debugging search issues
    - MCP tool status reporting

    Returns:
        JSON response with detailed index status information

    Example:
        GET /status

        Response:
        {
            "files_indexed": 1247,
            "database_size_mb": 125.6,
            "search_index_ready": true,
            "last_updated": "2025-07-13T10:30:00Z",
            "embedding_dimensions": 384
        }
    """
    # Get file count from database
    count_result = db_manager.execute_with_retry(f"SELECT count(*) FROM {TABLE_NAME}")
    file_count = count_result[0][0] if count_result and count_result[0] else 0

    # Check if database file exists and get its size
    db_path = current_dir / ".turboprop" / "code_index.duckdb"
    db_size_mb = 0
    if db_path.exists():
        db_size_mb = db_path.stat().st_size / (1024 * 1024)

    # With DuckDB vector search, index is always ready if files exist
    index_ready = file_count > 0

    # Get latest file timestamp from database (if any files exist)
    last_updated = None
    if file_count > 0:
        try:
            # This is a simple timestamp - in production you might want to store
            # actual timestamps
            last_updated = "Recent"  # Simplified for now
        except Exception:
            last_updated = "Unknown"

    return {
        "files_indexed": file_count,
        "database_size_mb": round(db_size_mb, 2),
        "search_index_ready": index_ready,
        "last_updated": last_updated,
        "embedding_dimensions": DIMENSIONS,
        "model_name": EMBED_MODEL,
    }
