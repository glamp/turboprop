# Turboprop 🚀

Lightning-fast semantic code search with AI embeddings. Transform your codebase into a searchable knowledge base using natural language queries.

## ✨ Key Features

- 🔍 **Semantic Search**: Find code by meaning, not just keywords ("JWT authentication" finds auth logic)
- 🐆 **Lightning Fast**: DuckDB vector operations for sub-second search across large codebases
- 🔄 **Live Updates**: Watch mode with intelligent debouncing - your index stays fresh as you code
- 🤖 **MCP Ready**: Perfect integration with Claude and other AI coding assistants
- 📁 **Git-Aware**: Respects .gitignore and only indexes what matters
- 💻 **Simple CLI**: Clean command-line interface with helpful guidance

## 🚀 Quick Start

### MCP Configuration (Claude Integration) - Front and Center!

Add this to your Claude Code `.mcp` configuration file:

```json
{
  "mcpServers": {
    "turboprop": {
      "command": "uvx",
      "args": ["turboprop@latest", "mcp", "--repository", ".", "--auto-index"],
      "env": {}
    }
  }
}
```

Then restart Claude Desktop and start asking questions about your code:

- "Find JWT authentication code"
- "Show me error handling patterns"
- "Where is the database connection setup?"

### Installation & Basic Usage

```bash
# Install globally
pip install turboprop

# Index your codebase
turboprop index .

# Search with natural language
turboprop search "JWT authentication"

# Watch for changes (keeps index updated)
turboprop watch .
```

## ⚙️ CLI Usage

### `index` - Build Search Index

```bash
turboprop index <repository_path> [options]

Options:
  --max-mb FLOAT       Maximum file size in MB to index (default: 1.0)
  --workers INTEGER    Number of parallel workers (default: CPU count)
  --force-all         Force reprocessing of all files

Examples:
  turboprop index .                    # Index current directory
  turboprop index ~/my-project         # Index specific project
  turboprop index . --max-mb 2.0      # Allow larger files
```

### `search` - Semantic Code Search

```bash
turboprop search "<query>" [options]

Options:
  --repo PATH         Repository path (default: current directory)
  --k INTEGER         Number of results to return (default: 5)

Query Examples:
  turboprop search "JWT authentication"              # Find auth-related code
  turboprop search "parse JSON response"             # Discover JSON parsing logic
  turboprop search "error handling middleware"       # Locate error handling patterns
  turboprop search "database connection setup"       # Find DB initialization code
  turboprop search "React component for forms"       # Find form-related components
```

### `watch` - Live Index Updates

```bash
turboprop watch <repository_path> [options]

Options:
  --max-mb FLOAT        Maximum file size in MB (default: 1.0)
  --debounce-sec FLOAT  Seconds to wait before processing changes (default: 5.0)

Example:
  turboprop watch . --debounce-sec 3.0  # Faster updates
```

### `mcp` - Start MCP Server

```bash
turboprop mcp [options]

Options:
  --repository PATH     Repository to index and monitor (default: current directory)
  --max-mb FLOAT       Maximum file size in MB (default: 1.0)
  --debounce-sec FLOAT Seconds to wait before processing changes (default: 5.0)
  --auto-index         Automatically index on startup (default: True)
  --no-auto-index      Don't automatically index on startup
  --auto-watch         Automatically watch for changes (default: True)
  --no-auto-watch      Don't automatically watch for changes

Examples:
  turboprop mcp --repository .                     # Start MCP server for current directory
  turboprop mcp --repository /path/to/repo         # Index specific repository
  turboprop mcp --repository . --max-mb 2.0        # Allow larger files
  turboprop mcp --repository . --no-auto-index     # Don't auto-index on startup
```

### Features Available in Claude

When using the MCP server with Claude:

- **🔍 Semantic Code Search**: Ask Claude to find code using natural language
- **📁 Repository Indexing**: Automatically index and watch your codebase
- **🔄 Real-time Updates**: Index stays fresh as you code
- **💬 Natural Queries**: "Find JWT authentication code", "Show error handling patterns"

## 💡 Pro Tips & Search Examples

### Query Examples

- `"JWT authentication"` → Find auth-related code
- `"parse JSON response"` → Discover JSON parsing logic
- `"error handling middleware"` → Locate error handling patterns
- `"database connection setup"` → Find DB initialization code
- `"React component for forms"` → Find form-related components

### Performance Tips

- **File size limit**: Adjust `--max-mb` based on your repository size and available memory
- **Debounce timing**: Lower `--debounce-sec` for faster updates, higher for less CPU usage
- **Search results**: Increase `--k` to see more results, decrease for faster queries

## 📄 License

MIT License - feel free to use this in your projects!

---

**Ready to supercharge your code exploration? Get started in 60 seconds!** 🚀✨

## 🧠 Optimized for Claude Code

Add `.claude/code-index.commands.md` for slash commands.

---

That’s it—**fucking easy as pie**. 🍰🚀

## 💻 Enhanced CLI Experience

Turboprop now features a beautiful, user-friendly CLI with:

### Rich Help System

```bash
uv run python code_index.py --help     # Main help with examples
uv run python code_index.py index --help    # Detailed command help
uv run python code_index.py search --help   # Search-specific guidance
```

### Visual Feedback

- 🚀 Branded startup messages
- ⚡ Progress indicators with emojis
- 📊 Rich search result formatting
- 💡 Helpful tips and suggestions
- ❌ Clear error messages with solutions

### Smart Error Handling

- Missing index detection with guidance
- File size limit recommendations
- Search result explanations
- Recovery suggestions

## 🔗 MCP Integration (Claude & AI Assistants)

### Quick Setup for Claude

1. **Start the MCP server:**

   ```bash
   uv run uvicorn server:app --host localhost --port 8000
   ```

2. **Index your repository:**

   ```bash
   uv run python code_index.py index /path/to/your/repo
   ```

3. **Use in Claude with slash commands:**
   - `/search JWT authentication` - Find auth-related code
   - `/search React form validation` - Find form components
   - `/status` - Check index health

### Available MCP Tools

- **`index_repository`** - Build searchable index from code repository
- **`search_code`** - Search code using natural language queries
- **`get_index_status`** - Check current state of the code index
- **`watch_repository`** - Monitor repository for changes
- **`list_indexed_files`** - Show files currently in the index

### MCP Configuration

The repository includes ready-to-use MCP configuration:

- `.mcp/config.json` - Server configuration
- `.claude/turboprop-commands.md` - Claude slash commands

## ⚙️ Complete CLI Reference

### `index` - Build Search Index

```bash
uv run python code_index.py index <repository_path> [options]

Options:
  --max-mb FLOAT    Maximum file size in MB to index (default: 1.0)

Examples:
  uv run python code_index.py index .                    # Index current directory
  uv run python code_index.py index ~/my-project         # Index specific project
  uv run python code_index.py index . --max-mb 2.0      # Allow larger files
```

### `search` - Semantic Code Search

```bash
uv run python code_index.py search "<query>" [options]

Options:
  --k INTEGER      Number of results to return (default: 5)

Query Examples:
  "JWT authentication"              → Find auth-related code
  "parse JSON response"             → Discover JSON parsing logic
  "error handling middleware"       → Locate error handling patterns
  "database connection setup"       → Find DB initialization code
  "React component for forms"       → Find form-related components
```

### `watch` - Live Index Updates

```bash
uv run python code_index.py watch <repository_path> [options]

Options:
  --max-mb FLOAT        Maximum file size in MB (default: 1.0)
  --debounce-sec FLOAT  Seconds to wait before processing changes (default: 5.0)

Example:
  uv run python code_index.py watch . --debounce-sec 3.0  # Faster updates
```

## 🌐 HTTP API Server

### Start the Server

```bash
# Development mode with auto-reload (using uv)
uv run uvicorn server:app --reload --host 0.0.0.0 --port 8000

# Production mode
uv run uvicorn server:app --host 0.0.0.0 --port 8000

# OR with activated virtual environment
uvicorn server:app --reload --host 0.0.0.0 --port 8000
```

### API Endpoints

#### `POST /index` - Build Index

```bash
curl -X POST "http://localhost:8000/index" \
  -H "Content-Type: application/json" \
  -d '{"repo": "/path/to/repository", "max_mb": 1.0}'

# Response: {"status": "indexed", "files": 1247}
```

#### `GET /search` - Search Code

```bash
curl "http://localhost:8000/search?query=JWT%20authentication&k=5"

# Response:
[
  {
    "path": "/src/auth/middleware.py",
    "snippet": "def verify_jwt_token(token: str):\n    \"\"\"Verify JWT token and extract claims...",
    "distance": 0.234
  }
]
```

#### `GET /status` - Index Status

```bash
curl "http://localhost:8000/status"

# Response:
{
  "files_indexed": 1247,
  "database_size_mb": 125.6,
  "search_index_ready": true,
  "last_updated": "Recent",
  "embedding_dimensions": 384,
  "model_name": "all-MiniLM-L6-v2"
}
```

### Interactive API Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 💡 Pro Tips & Best Practices

### Search Query Optimization

- **Use descriptive phrases**: "authentication middleware" vs just "auth"
- **Ask conceptual questions**: "how to handle errors" vs "try catch"
- **Combine multiple concepts**: "JWT token validation middleware"
- **Be domain-specific**: "React form validation" vs "form validation"

### Performance Tuning

- **File size limit**: Adjust `--max-mb` based on your repository size and available memory
- **Debounce timing**: Lower `--debounce-sec` for faster updates, higher for less CPU usage
- **Search results**: Increase `--k` to see more results, decrease for faster queries

### File Management

- **Index files**: `code_index.duckdb` (database) and `hnsw_code.idx` (search index)
- **Location**: Created in the current working directory
- **Cleanup**: Delete these files to reset the index completely
- **Backup**: Copy these files to backup your index

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Git Repository │───▶│  File Scanner    │───▶│  Code Files     │
│   (.gitignore    │    │  (scan_repo)     │    │  (.py, .js, etc)│
│    aware)        │    └──────────────────┘    └─────────────────┘
└─────────────────┘                                       │
                                                          ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Search Results │◀───│  HNSW Index     │◀───│  Embeddings     │
│   (ranked by    │    │  (fast vector   │    │  (ML model:     │
│    similarity)  │    │   search)        │    │   all-MiniLM)   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                 ▲                        │
                                 │                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   User Query    │───▶│  Query Embedding │    │  DuckDB Storage │
│   ("parse JSON") │    │  (same model)   │    │  (persistent)   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                 ▲
                                 │
                   ┌──────────────────┐
                   │   FastAPI Server │ ←── MCP Integration
                   │   (HTTP API)     │     Claude, etc.
                   └──────────────────┘
```

## 🤝 Contributing

We welcome contributions! Key areas for improvement:

- Additional programming language support
- Performance optimizations for large repositories
- IDE/editor integrations
- Advanced search features (regex, file filters, etc.)
- Better error handling and user feedback
- Enhanced MCP tool capabilities

## 📄 License

MIT License - feel free to use this in your projects!

---

**Ready to supercharge your code exploration? Get started in 60 seconds!** 🚀✨

_Turboprop: Because finding code should be as smooth as flying._
