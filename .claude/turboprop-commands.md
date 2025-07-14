# Turboprop MCP Tools

Slash commands for semantic code search in Claude.

## Available Commands

### `/search` - Semantic Code Search

Search your indexed codebase using natural language.

**Usage:** `/search [query]`

**Examples:**

- `/search JWT authentication` - Find authentication-related code
- `/search parse JSON response` - Discover JSON parsing logic
- `/search error handling patterns` - Locate error handling code
- `/search database connection` - Find DB initialization code

### `/index` - Build Code Index

Index a repository for semantic search.

**Usage:** `/index [path] [--max-mb SIZE]`

**Examples:**

- `/index .` - Index current directory
- `/index /path/to/project` - Index specific project
- `/index . --max-mb 2.0` - Index with larger file size limit

### `/status` - Index Status

Check the current state of your code index.

**Usage:** `/status`

Shows:

- Number of indexed files
- Database size
- Last updated timestamp
- Available search index status

## Setup Instructions

1. **Start the MCP server:**

   ```bash
   uv run uvicorn server:app --host localhost --port 8000
   ```

2. **Index your repository:**

   ```bash
   uv run python code_index.py index /path/to/your/repo
   ```

3. **Use slash commands in Claude:**
   - Type `/search` followed by your query
   - Get semantic search results directly in your conversation

## Tips for Better Results

- Use descriptive phrases: "authentication middleware" vs just "auth"
- Ask conceptual questions: "how to handle errors" vs "try catch"
- Combine multiple concepts: "JWT token validation middleware"
- Be specific about the domain: "React form validation" vs "form validation"

## Architecture

The MCP server provides HTTP endpoints that Claude can call:

- `POST /index` - Build/rebuild index
- `GET /search` - Semantic search
- `GET /status` - Index status

All tools respect your repository's `.gitignore` and only index recognized code files.
