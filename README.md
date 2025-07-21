# Turboprop 🚀

**AI-powered semantic code search for developers**

Find code by describing what it does, not just what it's called. Perfect for exploring unfamiliar codebases, debugging, and AI-assisted development with Claude Code.

## Quickstart

### With Claude Code (Recommended)
Add to your Claude Code MCP configuration:
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

Then use natural language with Claude:
- "Use turboprop to find JWT authentication code"
- "Search for error handling middleware patterns"
- "Find React components that handle forms"

### Standalone CLI
```bash
# Install and index your codebase
uvx turboprop index .

# Search with natural language
uvx turboprop search "JWT authentication middleware"
uvx turboprop search "database connection setup"
uvx turboprop search "error handling patterns"
```

## Features

### 🧠 Semantic Code Search
Find code by describing what it does, not just keywords:
- "JWT token validation" finds auth code across languages
- "form validation logic" discovers input handling
- "database connection setup" locates data layer code

### 🔍 Hybrid Search Modes
- **AUTO** - Automatically picks the best search strategy
- **HYBRID** - Combines semantic understanding with exact text matching  
- **SEMANTIC** - Pure conceptual search for similar functionality
- **TEXT** - Fast exact text matching

### 📊 Rich Results
Every search result includes:
- **Confidence scores** (0.0-1.0) showing match quality
- **Language detection** and file types
- **Code context** with syntax highlighting
- **Match explanations** - why each result was selected
- **IDE navigation links** for VS Code, PyCharm, etc.

## Usage

### MCP Tools (with Claude Code)

**Core Search & Indexing:**
- `index_repository` - Build searchable index from your codebase
- `index_repository_structured` - Advanced indexing with detailed JSON response
- `search_code` - Perform semantic search with natural language
- `search_code_structured` - Advanced search with rich JSON metadata
- `search_code_hybrid` - Configurable hybrid semantic + keyword search

**Index Management:**
- `get_index_status` - Check index health and file counts
- `get_index_status_structured` - Comprehensive index status with JSON metadata
- `check_index_freshness_tool` - Validate index currency and freshness
- `watch_repository` - Auto-update index when files change
- `list_indexed_files` - Browse all files in the search index

**Construct-Level Search:**
- `search_functions` - Find functions and methods by purpose
- `search_classes` - Discover classes by functionality (with optional method inclusion)
- `search_imports` - Locate imports and dependencies semantically
- `search_hybrid_constructs` - Multi-granularity construct search with configurable weights

**AI Tool Discovery:**
- `search_mcp_tools` - Find tools using natural language queries with category filtering
- `get_tool_details` - Deep dive into any tool's capabilities with comprehensive metadata
- `list_tool_categories` - Overview of available tool categories and contents
- `search_tools_by_capability` - Search tools by specific technical capabilities

**Tool Analysis & Planning:**
- `recommend_tools_for_task` - Get intelligent tool recommendations with explanations
- `analyze_task_requirements` - Understand task complexity and technical needs
- `suggest_tool_alternatives` - Explore alternative tools for your primary choice
- `recommend_tool_sequence` - Plan optimal multi-step development workflows

**Tool Comparison & Analysis:**
- `compare_mcp_tools` - Compare multiple tools across various dimensions
- `find_tool_alternatives` - Discover similar tools with similarity analysis
- `analyze_tool_relationships` - Analyze relationships between tools and ecosystems

**Tool Browsing:**
- `browse_tools_by_category` - Explore tools within specific functional categories
- `get_category_overview` - High-level view of the entire tool ecosystem
- `get_tool_selection_guidance` - Structured decision support for optimal tool choice

**Quick Commands (Slash Commands):**
- `/search <query>` - Fast semantic search (3 results)
- `/index_current` - Reindex current repository
- `/status` - Show index status
- `/files [limit]` - List indexed files
- `/search_by_type <type> <query>` - Search specific file types
- `/help_commands` - Show available commands

### CLI Commands
```bash
# Index management
turboprop index .                     # Index current directory
turboprop index ~/project --max-mb 2  # Index with larger file limit

# Search
turboprop search "query" --mode auto  # Smart search (recommended)
turboprop search "query" --mode hybrid --explain  # Show match reasoning
turboprop search "query" --k 10       # Get 10 results

# Live updates
turboprop watch .                     # Monitor for file changes
```

## Search Query Tips

**Be descriptive and specific:**
- ✅ "JWT token validation middleware"
- ❌ "auth"

**Ask conceptual questions:**
- ✅ "how to handle database connection errors"  
- ❌ "try catch db"

**Combine multiple concepts:**
- ✅ "React form validation with custom hooks"
- ❌ "react forms"

**Example queries:**
- "JWT token validation and refresh logic"
- "REST API error handling patterns"
- "React component state management"
- "database query optimization"
- "OAuth2 authorization flow implementation"

## Architecture & Technical Details

**Storage:** DuckDB with 384-dimension vector embeddings  
**ML Model:** SentenceTransformer "all-MiniLM-L6-v2"  
**Search:** Native vector operations with cosine similarity  
**Files:** Indexes all Git-tracked files, respects `.gitignore`  
**Index Location:** `.turboprop/code_index.duckdb` in each repository

For detailed technical information, see [ARCHITECTURE.md](ARCHITECTURE.md).

## License

MIT License - use freely in your projects!

---

*Find code by meaning, not just by name.*