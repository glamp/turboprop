# Turboprop 🚀

**Lightning-fast semantic code search with AI embeddings**

Transform your codebase into a searchable knowledge base using natural language queries. Perfect for AI-assisted development with Claude Code and other AI coding assistants.

## ✨ What Makes Turboprop Special

🔍 **Hybrid Search** - Combines semantic understanding with exact text matching using advanced fusion algorithms  
🎯 **Smart Ranking** - Multi-factor ranking considers file type, recency, construct type, and semantic relevance  
🧠 **Structured Results** - Rich metadata with confidence scores, match explanations, and IDE navigation links  
🐆 **Lightning Fast** - DuckDB vector operations deliver sub-second search across massive codebases  
🔄 **Live Updates** - Watch mode with intelligent debouncing keeps your index fresh as you code  
🤖 **Claude Code Enhanced** - Advanced MCP integration with structured responses and query analysis  
🔒 **Safe Concurrent Access** - Advanced file locking prevents corruption during multi-process operations  
📁 **Git-Aware** - Respects .gitignore and only indexes what matters  
💻 **Beautiful CLI** - Rich terminal interface with progress indicators and helpful guidance

## 🆕 Enhanced Search Capabilities

Turboprop's enhanced search system provides sophisticated code discovery that goes far beyond simple semantic similarity:

### 🔄 Hybrid Search Modes

**AUTO Mode** (Recommended) - Automatically chooses the best search strategy for your query
```bash
turboprop search "JWT authentication middleware" --mode auto
```

**HYBRID Mode** - Combines semantic understanding with exact text matching
```bash  
turboprop search "error handling patterns" --mode hybrid --explain
```

**SEMANTIC Mode** - Pure conceptual search for finding similar functionality
```bash
turboprop search "user input validation logic" --mode semantic
```

**TEXT Mode** - Fast exact text matching for specific syntax
```bash
turboprop search "def authenticate(" --mode text
```

### 📊 Rich Search Results

Every result includes comprehensive metadata:

```bash
# Example enhanced result
2. src/auth/validators.py:28-35 (confidence: 0.92) 🐍 python [function]
   def validate_jwt_token(token: str) -> bool:
       """Validates JWT token signature and expiration."""
       try:
           payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
   
   ✨ Why this matches: Strong semantic match for JWT validation logic
   💡 Navigate: vscode://file/src/auth/validators.py:28
   🕒 Modified: 2 days ago
```

**Rich metadata includes:**
- 🎯 **Confidence scores** - How well results match your query (0.0-1.0)
- 🏷️ **Language detection** - Automatic programming language identification  
- 🧩 **Construct types** - Functions, classes, methods, constants, etc.
- 💭 **Match explanations** - Clear reasons why each result was selected
- 🔗 **IDE navigation** - Direct links to VS Code, PyCharm, and other editors
- ⏰ **File recency** - Git-based modification timestamps

### 🎯 Advanced Ranking System

Multi-factor ranking considers multiple relevance signals:

- **Semantic similarity** (40%) - How well the meaning matches
- **File type relevance** (20%) - Language and file type matching  
- **Construct type match** (15%) - Code structure alignment
- **File recency** (15%) - Recently modified files boost
- **File size optimization** (10%) - Prefers appropriately-sized files

### 🔧 Powerful Configuration

Fine-tune search behavior with comprehensive configuration:

```bash
# Environment variables
export TURBOPROP_SEARCH_MODE=hybrid
export TURBOPROP_MAX_FILE_SIZE_MB=2.0
export TURBOPROP_SNIPPET_CONTEXT_LINES=5
export TURBOPROP_RRF_K=80

# Or use configuration files
echo '{"search": {"mode": "hybrid", "max_results": 15}}' > .turboprop/config.json
```

## 🚀 Quick Start with Claude Code

### MCP Installation

Add this to your Claude Code MCP configuration:

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

### Sample Claude Code Prompts

Once installed, try these prompts with Claude Code:

**Search for specific patterns:**
- "Use the turboprop tools to find JWT authentication code in this repository"
- "Search the codebase for error handling middleware patterns"  
- "Find React components that handle form validation"
- "Look for database connection setup code"

**Index management:**
- "Index this repository with turboprop and then search for API route handlers"
- "Check the turboprop index status and tell me what files are indexed"
- "Reindex this codebase and search for logging implementations"

**Development workflow:**
- "Use turboprop to find code similar to what I'm working on and explain the patterns"
- "Search for examples of how authentication is implemented in this project"
- "Find all places where JSON parsing happens and show me the different approaches"

### Available MCP Tools

Claude Code can use these tools automatically:

- **`index_repository`** - Build searchable index from your codebase
- **`search_code`** - Perform semantic search with natural language  
- **`get_index_status`** - Check index health and file counts
- **`watch_repository`** - Monitor for changes (auto-enabled by default)
- **`list_indexed_files`** - Show what files are in the index

### Custom Slash Commands

Turboprop includes custom slash commands for Claude Code. Add these to your `.claude/` directory:

**`/search [query]`** - Quick semantic search
```
/search JWT authentication
/search error handling patterns  
/search React form components
```

**`/index [path]`** - Index a repository  
```
/index .
/index /path/to/project
```

**`/status`** - Check index status
```
/status
```

## ⚙️ Standalone CLI Usage

### Installation

```bash
# Install globally
pip install turboprop

# Or with uv (recommended)
uvx turboprop
```

### Core Commands

**Index your codebase:**
```bash
turboprop index .                    # Index current directory  
turboprop index ~/my-project         # Index specific project
turboprop index . --max-mb 2.0      # Allow larger files
turboprop index . --force-all        # Force reprocessing
```

**Search with natural language:**
```bash
turboprop search "JWT authentication"              # Find auth code
turboprop search "parse JSON response"             # JSON parsing logic  
turboprop search "error handling middleware"       # Error patterns
turboprop search "React component for forms"       # Form components
```

**Watch for live updates:**
```bash
turboprop watch .                    # Monitor current directory
turboprop watch . --debounce-sec 3.0 # Faster updates
```

**Start MCP server:**
```bash
turboprop mcp --repository . --auto-index    # Full auto mode
turboprop mcp --repository . --no-auto-watch # Manual updates only
```

## 🔧 Advanced Features

### Concurrent Access Protection

Turboprop uses advanced file locking to prevent database corruption:

- **Process-safe indexing** - Multiple processes can safely access the same repository
- **Atomic operations** - Index updates are completed fully or rolled back  
- **Deadlock prevention** - Smart lock ordering prevents system hangs
- **Graceful recovery** - Automatic cleanup of stale locks on restart

### Performance Optimization

**Smart file filtering:**
- Respects `.gitignore` automatically
- Configurable file size limits (`--max-mb`)
- Skips binary and generated files

**Efficient indexing:**
- Parallel processing with worker pools
- Incremental updates (only changed files)
- Memory-efficient batch processing

**Fast search:**
- Native DuckDB vector operations  
- 384-dimension embeddings for accuracy
- Cosine similarity ranking

### Configuration Options

**File size limits:**
```bash
--max-mb 1.0    # Default: 1MB max file size
--max-mb 5.0    # Allow larger files
--max-mb 0.1    # Strict limit for huge repos
```

**Watch mode timing:**
```bash
--debounce-sec 5.0   # Default: 5 second debounce
--debounce-sec 1.0   # Faster updates  
--debounce-sec 10.0  # Less CPU usage
```

**Search results:**
```bash
--k 5     # Default: 5 results
--k 10    # More results
--k 1     # Just the best match
```

## 💡 Search Query Tips

### Effective Query Patterns

**Be descriptive and specific:**
- ✅ "JWT token validation middleware"  
- ❌ "auth"

**Ask conceptual questions:**
- ✅ "how to handle database connection errors"
- ❌ "try catch db"

**Combine multiple concepts:**
- ✅ "React form validation with custom hooks"
- ❌ "react forms"

**Use domain-specific language:**
- ✅ "OAuth2 authorization flow implementation"  
- ❌ "login stuff"

### Example Queries by Use Case

**Authentication & Security:**
- "JWT token validation and refresh logic"
- "password hashing and salt generation"  
- "OAuth2 provider integration code"
- "session management middleware"

**API & Data:**
- "REST API error handling patterns"
- "JSON schema validation logic"
- "database query optimization"
- "caching layer implementation"

**Frontend & UI:**
- "React component state management"
- "form validation with error messages"  
- "responsive design utility classes"
- "event handler patterns"

## 🏗️ Architecture & Technical Details

### Database Schema
```sql
CREATE TABLE code_files (
  id VARCHAR PRIMARY KEY,        -- SHA-256 hash of path + content
  path VARCHAR,                  -- Absolute file path  
  content TEXT,                  -- Full file content
  embedding DOUBLE[384]          -- 384-dimension vector embeddings
);
```

### ML Model
- **Model**: SentenceTransformer "all-MiniLM-L6-v2"
- **Dimensions**: 384 (balanced accuracy/speed)
- **Similarity**: Cosine similarity via DuckDB vector operations

### File System
- **Index location**: `.turboprop/code_index.duckdb` in each repository
- **Git integration**: Uses `git ls-files` for file discovery
- **Ignore handling**: Respects `.gitignore` automatically

## 📚 Comprehensive Documentation

Turboprop includes extensive documentation to help you master enhanced search capabilities:

### 📖 User Guides
- **[Enhanced Search Guide](docs/user/search_guide.md)** - Master search modes, query techniques, and result interpretation
- **[Migration Guide](docs/migration/upgrade_guide.md)** - Step-by-step upgrade from basic to enhanced search system

### 🔧 Developer Resources  
- **[API Documentation](docs/api/enhanced_search.md)** - Complete API reference with examples for all enhanced features
- **[Architecture Guide](docs/developer/architecture.md)** - System design, extension patterns, and contribution guidelines

### 🚀 Quick References
- **Search Modes**: AUTO (recommended), HYBRID, SEMANTIC, TEXT
- **Result Metadata**: Confidence scores, match explanations, IDE navigation, language detection
- **Configuration**: Environment variables, config files, response detail levels
- **MCP Integration**: Structured responses, query analysis, Claude Code prompts

### 💡 Learning Path

1. **Start Here**: Read the [Enhanced Search Guide](docs/user/search_guide.md) for effective query techniques
2. **Upgrading**: Follow the [Migration Guide](docs/migration/upgrade_guide.md) if coming from basic search
3. **Deep Dive**: Explore the [API Documentation](docs/api/enhanced_search.md) for programmatic usage
4. **Contributing**: Check the [Architecture Guide](docs/developer/architecture.md) for development setup

## 🤝 Contributing

Key areas for contribution:

- Language-specific improvements (better syntax highlighting, smart parsing)
- Performance optimizations for enormous codebases  
- IDE/editor plugin development
- Advanced search features (regex filters, file type limits)
- Better error recovery and user guidance

## 📄 License

MIT License - use freely in your projects!

---

**Ready to supercharge your code exploration with Claude Code?** 🚀✨

*Turboprop: Because finding code should be as smooth as flying.*