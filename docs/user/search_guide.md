# User Guide: Mastering Enhanced Search

This guide will help you get the most out of Turboprop's enhanced search capabilities. Learn how to write effective queries, understand search results, and use advanced features to find exactly what you're looking for.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Understanding Search Modes](#understanding-search-modes)
3. [Writing Effective Queries](#writing-effective-queries)
4. [Understanding Results](#understanding-results)
5. [Advanced Search Techniques](#advanced-search-techniques)
6. [Claude Code Integration](#claude-code-integration)
7. [Performance Tips](#performance-tips)
8. [Troubleshooting](#troubleshooting)

## Getting Started

### Basic Search

The simplest way to search is with a natural language query:

```bash
# Search for authentication-related code
turboprop search "user authentication and login"

# Find error handling patterns
turboprop search "error handling middleware"

# Look for database queries
turboprop search "SQL database connection"
```

### Your First Enhanced Search

Let's walk through a complete search example:

```bash
# Search with enhanced features enabled
turboprop search "JWT token validation" --mode hybrid --explain --k 10

# Output:
# 🔍 Searching: "JWT token validation" (hybrid mode)
# 📊 Found 10 results in 0.15s
# 
# 1. src/auth/middleware.py:42-45 (confidence: 0.92) 🐍 python
#    def validate_jwt_token(token: str) -> bool:
#        """Validates JWT token signature and expiration."""
#        try:
#            payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
#    
#    ✨ Why this matches: Strong semantic match for JWT validation logic
#    💡 Navigate: vscode://file/src/auth/middleware.py:42
# 
# 2. src/utils/auth_helpers.py:18-22 (confidence: 0.89) 🐍 python
#    class JWTValidator:
#        """Helper class for JWT token operations."""
#        def __init__(self, secret_key: str):
#    
#    ✨ Why this matches: Contains JWT validation class definition
```

### Key Improvements Over Basic Search

The enhanced system provides:

- **🎯 Better Relevance**: Hybrid search combines meaning with exact matches
- **📊 Rich Metadata**: File language, construct types, confidence scores
- **💡 Explanations**: Clear reasons why each result matches
- **🔗 IDE Integration**: Direct navigation links to your editor
- **⚡ Speed**: Optimized performance with smart caching

## Understanding Search Modes

Turboprop offers multiple search modes for different use cases:

### AUTO Mode (Recommended)

```bash
turboprop search "authentication middleware" --mode auto
```

**Best for:** Most searches - automatically picks the optimal strategy
- Analyzes your query and chooses the best approach
- Balances semantic understanding with exact matching
- Adapts to query complexity and code patterns

### HYBRID Mode

```bash
turboprop search "JWT authentication" --mode hybrid
```

**Best for:** Complex queries that benefit from both semantic and exact matching
- Combines semantic search with text matching
- Uses Reciprocal Rank Fusion to merge results
- Ideal for technical concepts with specific keywords

### SEMANTIC Mode

```bash
turboprop search "user login functionality" --mode semantic
```

**Best for:** Conceptual queries and finding similar patterns
- Pure semantic search based on meaning
- Great for finding code that does similar things
- Best when you don't know exact keywords

### TEXT Mode

```bash
turboprop search "def authenticate" --mode text
```

**Best for:** Exact code patterns and specific syntax
- Direct text matching without semantic interpretation
- Fastest search mode for exact matches
- Perfect for finding specific function names or syntax

## Writing Effective Queries

### Query Structure Best Practices

#### ✅ DO: Be Descriptive and Specific

```bash
# Good queries
turboprop search "JWT token validation with expiration check"
turboprop search "React component for user profile editing"
turboprop search "database transaction rollback on error"
turboprop search "async HTTP request with retry logic"
```

#### ❌ DON'T: Use Vague or Too-Short Queries

```bash
# Poor queries
turboprop search "auth"           # Too vague
turboprop search "function"       # Too generic
turboprop search "error"          # Too broad
turboprop search "js"             # Not descriptive
```

### Query Types and Examples

#### 1. Functional Queries

Ask what the code does:

```bash
# What you want to find
turboprop search "validate user input for XSS attacks"
turboprop search "encrypt password before storing in database"
turboprop search "parse JSON response from API"
turboprop search "convert image to base64 encoding"
```

#### 2. Pattern Queries  

Find specific programming patterns:

```bash
# Code patterns and architectures
turboprop search "singleton pattern implementation"
turboprop search "factory method with dependency injection"
turboprop search "observer pattern for event handling"
turboprop search "middleware chain for request processing"
```

#### 3. Problem-Solution Queries

Find solutions to specific problems:

```bash
# Common programming challenges
turboprop search "handle file upload with progress tracking"
turboprop search "implement rate limiting for API endpoints"
turboprop search "retry failed network requests with backoff"
turboprop search "validate email address format"
```

#### 4. Technology-Specific Queries

Include relevant technology keywords:

```bash
# Framework/library specific
turboprop search "React useState hook for form data"
turboprop search "Express.js middleware for authentication"
turboprop search "SQLAlchemy database model relationships"
turboprop search "asyncio task scheduling and execution"
```

#### 5. Language-Specific Queries

Reference language-specific concepts:

```bash
# Python
turboprop search "Python decorator for caching function results"
turboprop search "context manager for database connections"

# JavaScript
turboprop search "JavaScript promise chain error handling"
turboprop search "async/await function with try-catch"

# Rust
turboprop search "Rust error handling with Result type"
turboprop search "borrowing and ownership in function parameters"
```

### Query Enhancement Tips

#### Use Domain Language

```bash
# Instead of generic terms, use domain-specific language
❌ "check data"
✅ "validate user input data"

❌ "make connection" 
✅ "establish database connection"

❌ "handle mistake"
✅ "catch and handle exceptions"
```

#### Combine Concepts

```bash
# Combine related concepts for better results
turboprop search "user authentication with session management"
turboprop search "form validation with error message display" 
turboprop search "API rate limiting with Redis caching"
```

#### Include Context

```bash
# Add context about the use case
turboprop search "file upload validation for image files"
turboprop search "password hashing for user registration"
turboprop search "JWT authentication for REST API endpoints"
```

## Understanding Results

### Result Anatomy

Each search result includes rich metadata:

```
2. src/auth/validators.py:28-35 (confidence: 0.87) 🐍 python [function]
   def validate_password_strength(password: str) -> bool:
       """Check if password meets security requirements."""
       if len(password) < 8:
           return False
       return re.match(r'^(?=.*[A-Z])(?=.*[a-z])(?=.*\d)', password)

   ✨ Match reasons: Contains password validation logic, Security pattern match
   💡 IDE: vscode://file/src/auth/validators.py:28
   🕒 Modified: 3 days ago
```

**Understanding the components:**

- **File Path & Lines**: Exact location in your codebase
- **Confidence Score**: How confident the system is (0.0-1.0)
- **Language Icon**: Programming language detected
- **Construct Type**: [function], [class], [method], [constant], etc.
- **Code Snippet**: Relevant code with context
- **Match Reasons**: Why this result was selected
- **IDE Link**: Direct navigation to your editor
- **Recency**: When the file was last modified

### Confidence Scores

| Score Range | Interpretation | Action |
|-------------|---------------|---------|
| 0.90 - 1.00 | Excellent match | Very likely what you need |
| 0.80 - 0.89 | Strong match | Good candidate to examine |
| 0.70 - 0.79 | Good match | Worth checking out |
| 0.60 - 0.69 | Moderate match | Consider if other results don't fit |
| 0.50 - 0.59 | Weak match | Probably not what you're looking for |

### Match Explanations

The system provides clear explanations for why results were selected:

**Common match reasons:**
- **Semantic match**: Code meaning aligns with your query
- **Keyword match**: Contains specific terms from your query  
- **Pattern match**: Code structure matches requested pattern
- **Context match**: Surrounding code provides relevant context
- **Language match**: File type is relevant to your query
- **Recency bonus**: Recently modified files boosted

## Advanced Search Techniques

### Using Search Filters

```bash
# Limit results to specific number
turboprop search "authentication" --k 5

# Get detailed explanations
turboprop search "error handling" --explain

# Verbose output with performance metrics
turboprop search "database query" --verbose

# Search specific file types
turboprop search "React component" --file-types "*.tsx,*.jsx"
```

### Construct-Specific Searches

Find specific types of code constructs:

```bash
# Find function definitions
turboprop search "function authentication validation"

# Find class definitions  
turboprop search "class user management"

# Find method implementations
turboprop search "method password validation"

# Find constants and configuration
turboprop search "constant API configuration"
```

### Combining Search Strategies

```bash
# Start broad, then narrow down
turboprop search "authentication" --k 20
# Review results, then search more specifically:
turboprop search "JWT authentication middleware" --k 10

# Use different modes for different needs:
turboprop search "auth" --mode text      # Find exact "auth" text
turboprop search "authentication logic" --mode semantic  # Find auth concepts
```

### Contextual Searching

Use information from your current work:

```bash
# Include relevant technology stack
turboprop search "React component form validation with hooks"

# Mention your specific use case
turboprop search "user registration with email verification"

# Include error or problem context
turboprop search "fix memory leak in event listeners"
```

## Claude Code Integration

### Basic Usage with Claude Code

Once Turboprop is installed as an MCP tool, use these prompts:

```
# Basic search prompts
"Search the codebase for authentication middleware patterns"
"Use turboprop to find JWT validation code"
"Look for React components that handle form submission"

# Analysis prompts
"Search for error handling patterns and explain the different approaches used"
"Find database query code and analyze it for potential SQL injection vulnerabilities"
"Look for API rate limiting implementations and compare their strategies"

# Development prompts  
"Search for similar authentication code to what I'm writing and suggest improvements"
"Find examples of how file uploads are handled in this codebase"
"Look for logging patterns and help me implement consistent logging"
```

### Advanced Claude Code Usage

```
# Multi-step analysis
"First search for authentication code, then analyze the security patterns and suggest improvements"

# Code quality assessment
"Search for error handling code and evaluate it for completeness and best practices"

# Architecture exploration
"Search for API endpoint definitions and create a map of the application's REST API structure"

# Refactoring assistance
"Find all places where user validation happens and help me consolidate the logic"
```

### Custom Search Commands

You can create custom slash commands for common searches:

```bash
# Create ~/.claude/commands/search-auth.md
Search for authentication and authorization code patterns in the current repository.

Use the turboprop search tool to find:
1. Authentication middleware
2. Authorization checks  
3. JWT token handling
4. Session management
5. Password validation

Analyze the results and provide a summary of the authentication architecture.
```

Then use: `/search-auth`

## Performance Tips

### Optimize Your Queries

```bash
# Faster searches
turboprop search "auth" --mode text           # Fastest for exact matches
turboprop search "authentication" --k 5       # Fewer results = faster

# Balance speed vs accuracy
turboprop search "user login" --mode semantic  # Good balance
turboprop search "user authentication system" --mode hybrid  # More thorough
```

### Manage Index Size

```bash
# Check index status
turboprop status

# Optimize for large codebases
turboprop index . --max-mb 0.5  # Skip very large files

# Selective indexing
turboprop index ./src ./lib  # Index only specific directories
```

### Use Appropriate Result Limits

```bash
# Quick exploration
turboprop search "error handling" --k 3

# Thorough investigation  
turboprop search "authentication" --k 15

# Finding specific implementation
turboprop search "JWT decode" --k 1
```

## Troubleshooting

### Common Issues and Solutions

#### No Results Found

```bash
# Try different search modes
turboprop search "your query" --mode text
turboprop search "your query" --mode semantic

# Check if index is up to date
turboprop status
turboprop index . --force-reindex

# Try broader or more specific queries
turboprop search "auth"              # Broader
turboprop search "JWT authentication middleware"  # More specific
```

#### Results Not Relevant

```bash
# Use more specific language
❌ turboprop search "function"
✅ turboprop search "authentication function with password validation"

# Try different search modes
turboprop search "your query" --mode hybrid
turboprop search "your query" --mode semantic

# Add context to your query
❌ "validation" 
✅ "form input validation with error handling"
```

#### Slow Search Performance

```bash
# Use faster search modes
turboprop search "query" --mode text  # Fastest

# Reduce result count
turboprop search "query" --k 5

# Check index size
turboprop status
# If index is very large, consider selective indexing
```

#### Missing Recent Changes

```bash
# Check if files are indexed
turboprop status

# Reindex if needed
turboprop index .

# Use watch mode for automatic updates
turboprop watch .
```

### Getting Help

```bash
# Check system status
turboprop status --verbose

# Run diagnostics  
turboprop diagnose

# View configuration
turboprop config --show

# Test with example query
turboprop search "test query" --explain --verbose
```

### Best Practices Summary

1. **Start Specific**: Use detailed, specific queries rather than generic terms
2. **Use AUTO Mode**: Let the system choose the best search strategy
3. **Read Explanations**: Pay attention to match reasons to understand results
4. **Iterate Queries**: Refine your search based on initial results
5. **Use Confidence Scores**: Focus on high-confidence results first
6. **Keep Index Fresh**: Update your index regularly, especially for active development
7. **Experiment**: Try different search modes and query styles to find what works best

With these techniques, you'll be able to find exactly the code you need quickly and efficiently. The enhanced search system learns from your usage patterns and gets better at understanding what you're looking for over time.