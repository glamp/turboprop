# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Turboprop is a semantic code search and indexing system that uses DuckDB for both storage and vector similarity search with ML embeddings for intelligent code discovery. It consists of a CLI tool and HTTP API server for searching code repositories using natural language queries.

## Development Commands

### Installation & Setup
```bash
# Setup virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install duckdb sentence-transformers watchdog fastapi uvicorn

# Or install with development dependencies
pip install -e ".[dev]"
```

### Running the Application

**CLI Usage:**
```bash
# Index a repository
python code_index.py index /path/to/repo --max-mb 1.0

# Search the index
python code_index.py search "your search query" --k 5

# Watch for changes (continuous mode)
python code_index.py watch /path/to/repo --max-mb 1.0 --debounce-sec 5.0
```

**HTTP API Server:**
```bash
# Start the FastAPI server
uvicorn server:app --reload

# Alternative using uvx
uvx server:app --reload
```

### Testing
```bash
# Run unit tests
pytest tests/

# Run tests with coverage
pytest --cov=. tests/

# Test basic functionality
python code_index.py index example-codebases/bashplotlib
python code_index.py search "histogram plotting function"

# Test server endpoints
uvicorn server:app --reload
# Visit http://localhost:8000/docs for API documentation
```

### Code Quality & Linting
```bash
# Format code with Black
python -m black .

# Sort imports with isort
python -m isort .

# Run both formatting and import sorting together
python -m black . && python -m isort .
```

### Troubleshooting

**Apple Silicon MPS Compatibility Issue:**
If you see an error like "Cannot copy out of meta tensor; no data! Please use torch.nn.Module.to_empty() instead", this is automatically handled by falling back to CPU processing. This occurs due to PyTorch MPS backend compatibility issues with certain SentenceTransformer model configurations.

**MCP Server Output:**
The MCP server uses stderr for status messages to comply with the Model Context Protocol. When indexing completes, you'll see:
```
✅ Indexing complete! Processed X files with Y embeddings.
🎯 Repository 'path' is ready for semantic search!
```

**Progress Indication:**
During indexing, the system shows:
- File scanning progress
- Embedding generation status  
- Index building completion
- Final success message with file/embedding counts

## Architecture

### Core Components

**code_index.py** - Main CLI application with three primary functions:
- `scan_repo()`: Discovers code files using Git ls-files (respects .gitignore)
- `embed_and_store()`: Generates sentence embeddings and stores in DuckDB
- `build_full_index()`: Validates embeddings exist in database (no separate index needed)
- `search_index()`: Performs semantic search using DuckDB's native vector operations
- `watch_mode()`: Real-time file monitoring with debounced updates

**server.py** - FastAPI wrapper providing HTTP endpoints:
- `POST /index`: Trigger full repository reindexing
- `GET /search`: Query the index for similar code
- Background watcher automatically monitors current directory

### Key Technical Details

**Database Schema (DuckDB):**
```sql
CREATE TABLE code_files (
  id VARCHAR PRIMARY KEY,        -- SHA-256 hash of path + content
  path VARCHAR,                  -- Absolute file path
  content TEXT,                  -- Full file content
  embedding DOUBLE[384]          -- 384-dimension vector embeddings
);
```

**ML Model:** Uses SentenceTransformer "all-MiniLM-L6-v2" (384 dimensions) for semantic embeddings

**Search Algorithm:** DuckDB's native vector operations with cosine similarity for exact nearest neighbor search

**File Filtering:** Processes all Git-tracked files (regardless of extension) and respects Git ignore rules

### Database Files
- `.turboprop/code_index.duckdb`: Main database containing file content and embeddings with native vector search (stored in `./.turboprop/` directory within each repository)

**Note:** Add `.turboprop/` to your `.gitignore` file to avoid committing index files to version control.

## Common Development Patterns

### File Type Support
The system now indexes all Git-tracked files regardless of extension. This includes:
- All source code files (.py, .js, .ts, .java, .go, .rs, .cpp, etc.)
- Configuration files (.json, .yaml, .toml, .ini, etc.)
- Documentation files (.md, .rst, .txt, etc.)
- Any other files tracked in your Git repository

### Modifying Search Parameters
- **Vector search**: Uses DuckDB's `list_dot_product()` for cosine similarity calculations
- **Embedding model**: Change `EMBED_MODEL` constant (requires reindexing)
- **Result limits**: Modify default `k` values in CLI args or API endpoints

### Performance Tuning
- **File size limits**: Adjust `max_mb` parameters to balance completeness vs. performance
- **Debounce timing**: Tune `debounce_sec` for watch mode responsiveness
- **Batch processing**: `embed_and_store()` processes files in batches for efficiency

## Configuration

Turboprop supports configuration through YAML files and environment variables, with environment variables taking precedence.

### YAML Configuration

Create a `.turboprop.yml` file in your repository root to customize settings:

```yaml
# Database configuration
database:
  memory_limit: "2GB"              # Memory limit for DuckDB
  threads: 8                       # Number of threads for database operations
  max_retries: 3                   # Maximum connection retry attempts
  connection_timeout: 60.0         # Connection timeout (seconds)
  auto_vacuum: true                # Enable automatic database optimization

# File processing configuration
file_processing:
  max_file_size_mb: 2.0            # Maximum file size to index (MB)
  debounce_seconds: 3.0            # Debounce delay for file watching (seconds)
  batch_size: 200                  # Batch size for processing files
  max_workers: 8                   # Maximum parallel workers

# Search configuration
search:
  default_max_results: 10          # Default number of search results
  max_results_limit: 50            # Maximum allowed search results
  min_similarity: 0.2              # Minimum similarity threshold

# Embedding model configuration
embedding:
  model: "all-MiniLM-L6-v2"        # SentenceTransformer model name
  device: "cpu"                    # Device: "cpu", "cuda", or "mps"
  batch_size: 64                   # Batch size for embedding generation

# HTTP server configuration
server:
  host: "127.0.0.1"                # Server bind address
  port: 8080                       # Server port
  request_timeout: 60.0            # Request timeout (seconds)

# Logging configuration
logging:
  level: "DEBUG"                   # Log level: DEBUG, INFO, WARNING, ERROR
  file: "turboprop.log"            # Log file path (null for console only)
```

### Environment Variables

All YAML settings can be overridden with environment variables using the `TURBOPROP_` prefix:

```bash
# Database settings
export TURBOPROP_DB_MEMORY_LIMIT="4GB"
export TURBOPROP_DB_THREADS="16"

# File processing
export TURBOPROP_MAX_FILE_SIZE_MB="5.0"
export TURBOPROP_DEBOUNCE_SECONDS="10.0"

# Search settings
export TURBOPROP_DEFAULT_MAX_RESULTS="15"
export TURBOPROP_MIN_SIMILARITY="0.3"

# Embedding model
export TURBOPROP_EMBED_MODEL="sentence-transformers/all-mpnet-base-v2"
export TURBOPROP_DEVICE="cuda"

# Server settings
export TURBOPROP_HOST="0.0.0.0"
export TURBOPROP_PORT="9000"

# Logging
export TURBOPROP_LOG_LEVEL="INFO"
export TURBOPROP_LOG_FILE="./logs/turboprop.log"
```

### Configuration Priority

Configuration is loaded in the following priority order (highest to lowest):
1. **Environment Variables** - `TURBOPROP_*` variables
2. **YAML Configuration** - `.turboprop.yml` file in current directory
3. **Default Values** - Built-in sensible defaults

### Configuration Validation

All configuration values are validated on startup. Invalid values will cause the application to exit with an error message indicating the problem.

**Add `.turboprop.yml` to your `.gitignore`** if you want repository-specific configuration without committing it to version control.

## Project Structure

```
turboprop/
├── code_index.py      # Main CLI application
├── server.py          # FastAPI HTTP server
├── main.py           # Simple entry point
├── pyproject.toml    # Dependencies and project config
├── example-codebase/ # Sample repository for testing
└── *.duckdb, *.idx   # Generated database and index files
```

## Dependencies

**Core:**
- `duckdb`: Fast analytical database for embeddings storage and vector search
- `sentence-transformers`: ML models for semantic embeddings
- `watchdog`: File system monitoring

**Web API:**
- `fastapi`: Modern Python web framework
- `uvicorn`: ASGI server for running FastAPI

This system is designed for semantic code discovery - finding code by meaning rather than exact text matches, making it ideal for AI-assisted development workflows. The use of DuckDB for vector operations eliminates the need for separate indexing libraries and provides excellent performance for most use cases.

## Testing Strategies

- Use the tests/docker/run_tests.sh to test packaging and distribution