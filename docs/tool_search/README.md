# MCP Tool Search System Documentation

This directory contains comprehensive documentation for the MCP Tool Search System - the intelligent tool discovery and recommendation engine that transforms how Claude Code and other AI agents find and select MCP tools.

## 📚 Documentation Structure

### 🎯 Core Documentation

#### [User Guide](user_guide.md)
Complete guide for end users covering:
- Natural language tool discovery
- Intelligent recommendations
- Tool comparison and selection
- Best practices and troubleshooting
- Advanced configuration options

#### [API Reference](api_reference.md)
Comprehensive technical documentation including:
- All MCP tool functions and parameters
- Response data structures
- Error handling patterns
- Performance considerations
- Configuration options

#### [Migration Guide](migration_guide.md)
Step-by-step guide for upgrading:
- From manual to intelligent tool selection
- Database migration procedures
- Code migration examples
- Testing and validation
- Rollback procedures

#### [Integration Guide](integration_guide.md)
Integration patterns and examples for:
- Claude Code integration
- MCP client integration
- IDE and editor plugins
- Custom applications
- REST API wrappers

#### [Architecture Documentation](architecture.md)
System design and technical details:
- Component architecture
- Data flow diagrams
- Database schema
- Performance optimization
- Extension points

### 💡 Examples and Tutorials

#### [Basic Usage Examples](examples/basic_usage.py)
Working code examples for common patterns:
- Simple tool search
- Task-based recommendations
- Tool comparison
- Context-aware discovery
- Error handling

#### [Advanced Workflows](examples/advanced_workflows.py)
Sophisticated usage patterns:
- Intelligent workflow planning
- Adaptive tool selection
- Multi-stage development workflows
- Performance optimization
- Error recovery chains

#### [Custom Integrations](examples/custom_integrations.py)
Custom integration examples:
- Domain-specific search algorithms
- Custom tool categories
- Specialized scoring systems
- External system integration

### 📖 Reference Materials

#### [MCP Tools Reference](reference/mcp_tools_reference.md)
Complete reference for all MCP tools:
- Tool descriptions and capabilities
- Parameter specifications
- Usage examples
- Performance characteristics

#### [Search Algorithms](reference/search_algorithms.md)
Technical documentation on search algorithms:
- Semantic search implementation
- Hybrid search strategies
- Keyword matching algorithms
- Ranking and scoring methods

#### [Data Structures](reference/data_structures.md)
Complete data structure documentation:
- Request and response formats
- Tool metadata schemas
- Configuration structures
- Error response formats

## 🚀 Quick Start

### New Users
1. Start with the [User Guide](user_guide.md) to understand core concepts
2. Review the [Basic Usage Examples](examples/basic_usage.py)
3. Try the search features in Claude Code

### Developers
1. Read the [API Reference](api_reference.md) for technical details
2. Study the [Architecture Documentation](architecture.md)
3. Explore [Advanced Workflows](examples/advanced_workflows.py)

### Migration
1. Follow the [Migration Guide](migration_guide.md) step-by-step
2. Test with the validation examples
3. Review [Integration Guide](integration_guide.md) for client updates

## 🔍 Key Features Overview

### Natural Language Tool Discovery
Search for tools using plain English descriptions:
```python
search_mcp_tools("read configuration files safely")
search_mcp_tools("execute shell commands with timeout support")
```

### Intelligent Recommendations
Get personalized tool suggestions based on task context:
```python
recommend_tools_for_task(
    "process CSV files and generate reports",
    context="performance critical, large files"
)
```

### Tool Comparison
Compare multiple tools to understand trade-offs:
```python
compare_mcp_tools(["read", "write", "edit"])
```

### Context-Aware Selection
Tools adapt to your specific requirements and environment:
```python
search_mcp_tools(
    "data processing tools",
    context="beginner user, safety first"
)
```

## 🎯 Search Strategies

The system supports multiple search modes:

- **Hybrid Search** (Recommended) - Combines semantic understanding with keyword matching
- **Semantic Search** - Pure conceptual search for finding tools by purpose  
- **Keyword Search** - Fast exact term matching for specific tool names

## 📈 Benefits

### For Claude Code Users
- **Proactive suggestions** - Get better tool recommendations automatically
- **Context awareness** - Tools selected based on conversation context
- **Learning system** - Recommendations improve over time
- **Reasoning explanations** - Understand why tools were chosen

### For Developers
- **Reduced cognitive load** - No need to memorize all available tools
- **Better tool discovery** - Find tools you didn't know existed
- **Intelligent fallbacks** - Automatic alternatives when tools fail
- **Performance insights** - Data-driven tool selection

### For Teams
- **Consistency** - Everyone uses optimal tools for tasks
- **Knowledge sharing** - Best practices encoded in recommendations
- **Onboarding** - New team members discover tools naturally
- **Productivity** - Less time spent searching, more time coding

## 🔧 Configuration

### Environment Variables
```bash
# Enable tool search
export TOOL_SEARCH_ENABLED=true

# Search behavior
export TOOL_SEARCH_DEFAULT_MODE=hybrid
export TOOL_SEARCH_MAX_RESULTS=20

# Performance tuning
export TOOL_SEARCH_CACHE_SIZE=1000
export TOOL_SEARCH_CACHE_TTL=3600
```

### Configuration Files
```json
{
  "tool_search": {
    "enabled": true,
    "search_mode": "hybrid",
    "learning": {
      "enabled": true,
      "feedback_collection": true
    },
    "performance": {
      "max_results": 20,
      "timeout": 30
    }
  }
}
```

## 📊 Performance

### Typical Response Times
- **Basic search**: < 500ms
- **Recommendations**: < 1 second
- **Tool comparison**: < 2 seconds
- **Complex analysis**: < 3 seconds

### Scalability
- **Tool catalog**: Supports 10,000+ tools
- **Concurrent users**: 100+ simultaneous searches
- **Cache hit rate**: > 70% for common queries
- **Memory usage**: < 500MB for typical deployments

## 🤝 Contributing

We welcome contributions to improve the tool search system:

### Documentation Improvements
- Update examples with new patterns
- Add troubleshooting scenarios
- Improve clarity and completeness

### Feature Enhancements
- New search algorithms
- Advanced filtering options
- Performance optimizations
- Integration patterns

### Testing
- Add test cases for edge scenarios
- Performance benchmarking
- Integration testing

## 📝 License

The MCP Tool Search System is part of Turboprop and is released under the MIT License.

## 💬 Support

- **Documentation Issues**: Check existing docs or create an issue
- **Feature Requests**: Describe your use case and requirements
- **Bug Reports**: Include steps to reproduce and system information
- **Questions**: Start with the troubleshooting section in the User Guide

---

**Ready to revolutionize your tool discovery?** Start with the [User Guide](user_guide.md) and experience intelligent tool selection with Claude Code! 🚀