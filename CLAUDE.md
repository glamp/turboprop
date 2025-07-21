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