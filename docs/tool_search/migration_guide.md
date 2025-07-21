# Migration Guide: Upgrading to MCP MCP Tool Search System

## Overview

This guide helps you migrate from manual tool selection to the intelligent MCP MCP Tool Search System. The new system is fully backward compatible - existing code continues to work while new search capabilities are added.

## Migration Timeline

### Phase 1: Basic Integration (Week 1)
- Enable MCP MCP Tool Search System alongside existing tools
- Begin using `search_mcp_tools()` for tool discovery
- Familiarize team with search capabilities

### Phase 2: Enhanced Usage (Week 2-3)  
- Integrate `recommend_tools_for_task()` into workflows
- Use `compare_mcp_tools()` for tool selection decisions
- Begin leveraging advanced search features

### Phase 3: Full Adoption (Week 4+)
- Enable automatic tool suggestions
- Customize search behavior for your environment
- Optimize based on usage patterns and feedback

## Before You Begin

### Prerequisites
- Turboprop version 2.0 or later
- Python 3.8+ environment
- DuckDB database with tool catalog
- Sufficient disk space for embeddings (approximately 100MB)

### Backup Considerations
```bash
# Backup existing database
cp .turboprop/code_index.duckdb .turboprop/code_index_backup.duckdb

# Backup configuration
cp -r .turboprop/config/ .turboprop/config_backup/
```

## Database Migration

### Automatic Migration
The system automatically detects and migrates existing databases:

```python
# Migration happens automatically on first startup
from mcp_server import FastMCP
app = FastMCP("turboprop-with-search")

# Check migration status
app.check_migration_status()
```

### Manual Migration (If Needed)
```bash
# Run migration scripts manually
python -m turboprop.migrations.add_tool_search_schema

# Verify migration
python -m turboprop.migrations.verify_tool_search_migration
```

### Migration Rollback
If you need to rollback:

```bash
# Rollback database schema
python -m turboprop.migrations.rollback_tool_search

# Restore from backup
cp .turboprop/code_index_backup.duckdb .turboprop/code_index.duckdb
```

## Code Migration Examples

### Before: Manual Tool Usage
```python
# Old approach - manual tool selection
def process_files():
    # Developer had to know which tools exist and their capabilities
    file_content = read_tool(file_path="config.json")
    processed_data = process_data(file_content)
    write_tool(file_path="output.json", content=processed_data)
```

### After: Search-Enhanced Tool Usage  
```python
# New approach - intelligent tool discovery
def process_files():
    # Find optimal tools for the task
    recommendations = recommend_tools_for_task(
        "read JSON configuration, process data, write results",
        context="performance critical, error handling required"
    )
    
    # Use recommended tools with guidance
    best_reader = recommendations[0].tool
    file_content = use_tool(best_reader.tool_id, file_path="config.json")
    
    # Get processing recommendations
    processor_recs = recommend_tools_for_task(
        "transform JSON data structure",
        context=f"input from {best_reader.tool_id}"
    )
    
    processed_data = use_tool(processor_recs[0].tool.tool_id, data=file_content)
    
    # Find compatible output tool
    writer_recs = recommend_tools_for_task(
        "write processed data to file",
        context="JSON format, atomic write preferred"
    )
    
    use_tool(writer_recs[0].tool.tool_id, 
            file_path="output.json", 
            content=processed_data)
```

## Integration Patterns

### Claude Code Integration
```python
# Enable search suggestions in Claude Code responses
@mcp.tool()
def enhanced_task_handler(task_description: str) -> dict:
    # Get tool recommendations first
    recommendations = recommend_tools_for_task(task_description)
    
    # Use top recommendation
    primary_tool = recommendations[0]
    result = execute_tool(primary_tool.tool.tool_id, task_description)
    
    # Return result with suggestions for improvement
    return {
        "result": result,
        "tool_suggestions": {
            "used_tool": primary_tool.tool.name,
            "alternatives": [alt.tool_id for alt in recommendations[1:3]],
            "reasoning": primary_tool.recommendation_reasons
        }
    }
```

### Workflow Optimization
```python
# Before: Static workflow
def deployment_workflow():
    bash("./build.sh")
    bash("./test.sh") 
    bash("./deploy.sh")

# After: Optimized workflow
def deployment_workflow():
    # Analyze the complete workflow
    sequence = recommend_tool_sequence(
        "build application, run tests, deploy to production",
        optimization_goal="reliability"
    )
    
    # Execute optimized sequence
    for step in sequence[0].tools:
        result = execute_tool(step.tool_id, step.parameters)
        if not result.success:
            # Get recovery recommendations
            recovery = recommend_tools_for_task(
                f"recover from {step.tool_id} failure",
                context=f"error: {result.error}"
            )
            handle_failure(recovery)
```

## Configuration Migration

### Environment Variables
Add new configuration options:

```bash
# .env file additions
TOOL_SEARCH_ENABLED=true
TOOL_SEARCH_CACHE_SIZE=1000
TOOL_SEARCH_LEARNING_ENABLED=true

# Search behavior
TOOL_SEARCH_DEFAULT_MODE=hybrid
TOOL_SEARCH_SIMILARITY_THRESHOLD=0.3

# Performance settings  
TOOL_SEARCH_MAX_RESULTS=20
TOOL_SEARCH_CACHE_TTL=3600
```

### Configuration Files
```json
// turboprop_config.json
{
  "tool_search": {
    "enabled": true,
    "search_mode": "hybrid",
    "cache_settings": {
      "size": 1000,
      "ttl": 3600
    },
    "learning": {
      "enabled": true,
      "feedback_collection": true
    },
    "performance": {
      "max_results": 20,
      "timeout": 30,
      "concurrent_searches": 5
    }
  }
}
```

## Testing Your Migration

### Validation Steps
```python
# 1. Verify search functionality
results = search_mcp_tools("file operations")
assert len(results['results']) > 0
print(f"✅ Search working: found {len(results['results'])} tools")

# 2. Test recommendations
recs = recommend_tools_for_task("read configuration files")
assert len(recs['recommendations']) > 0
print(f"✅ Recommendations working: {len(recs['recommendations'])} suggestions")

# 3. Verify comparison
comparison = compare_mcp_tools(["read", "write"])
assert comparison['success'] is True
print("✅ Tool comparison working")

# 4. Check tool details
details = get_tool_details("read")
assert details['success'] is True
print("✅ Tool details working")
```

### Performance Validation
```python
import time

# Test search performance
start = time.time()
results = search_mcp_tools("data processing tools")
search_time = time.time() - start

assert search_time < 2.0  # Should be under 2 seconds
print(f"✅ Search performance: {search_time:.2f}s")

# Test recommendation performance  
start = time.time()
recs = recommend_tools_for_task("complex data transformation task")
rec_time = time.time() - start

assert rec_time < 3.0  # Should be under 3 seconds
print(f"✅ Recommendation performance: {rec_time:.2f}s")
```

## Common Migration Issues

### Issue: "Tool search system not responding"
**Symptoms:** Search queries return empty results or timeout
**Causes:** Database not properly migrated, embeddings not generated
**Solution:**
```bash
# Regenerate embeddings
python -m turboprop.tools.regenerate_embeddings

# Verify database schema
python -m turboprop.migrations.verify_schema
```

### Issue: "Search results seem irrelevant"
**Symptoms:** Search returns unexpected tools
**Causes:** Embedding model not properly loaded, query processing issues
**Solution:**
```python
# Test embedding generation
from embedding_helper import EmbeddingGenerator
gen = EmbeddingGenerator()
embedding = gen.generate_embedding("test query")
assert len(embedding) == 384
print("✅ Embeddings working correctly")
```

### Issue: "Performance degradation after migration"
**Symptoms:** Slower response times, high memory usage
**Causes:** Cache not properly configured, too many concurrent operations
**Solution:**
```bash
# Optimize cache settings
export TOOL_SEARCH_CACHE_SIZE=2000
export TOOL_SEARCH_CACHE_TTL=1800

# Monitor performance
python -m turboprop.tools.performance_monitor
```

## Rollback Procedures

If you need to rollback the migration:

### Complete Rollback
```bash
# Stop application
sudo systemctl stop turboprop-server

# Restore database backup
cp .turboprop/code_index_backup.duckdb .turboprop/code_index.duckdb

# Restore configuration
cp -r .turboprop/config_backup/* .turboprop/config/

# Disable tool search
export TOOL_SEARCH_ENABLED=false

# Restart application
sudo systemctl start turboprop-server
```

### Partial Rollback (Disable Features)
```python
# Disable specific features while keeping basic search
config = {
    "tool_search": {
        "enabled": True,
        "recommendations": False,  # Disable recommendations
        "comparisons": False,      # Disable comparisons  
        "learning": False,         # Disable learning
        "search_mode": "keyword"   # Use simpler search
    }
}
```

## Post-Migration Optimization

### Performance Tuning
```python
# Monitor and optimize based on usage
from mcp_response_optimizer import MCPResponseOptimizer
optimizer = MCPResponseOptimizer()

# Analyze performance metrics
metrics = optimizer.get_performance_report()
print(f"Average search time: {metrics['avg_search_time']:.2f}s")
print(f"Cache hit rate: {metrics['cache_hit_rate']:.1%}")

# Apply optimizations
if metrics['cache_hit_rate'] < 0.7:
    # Increase cache size
    optimizer.increase_cache_size(1500)
```

### Usage Analytics
```python
# Track usage patterns for optimization
from tool_usage_analytics import UsageAnalytics
analytics = UsageAnalytics()

# Most common queries
common_queries = analytics.get_common_queries(limit=10)
print("Most common queries:", common_queries)

# Tool popularity
tool_usage = analytics.get_tool_usage_stats()
print("Most recommended tools:", tool_usage['most_recommended'])

# Performance bottlenecks
bottlenecks = analytics.identify_bottlenecks()
if bottlenecks:
    print("Performance bottlenecks found:", bottlenecks)
```

## Migration Checklist

### Pre-Migration
- [ ] Backup existing database and configuration
- [ ] Verify system requirements (Python 3.8+, disk space)
- [ ] Review team training needs
- [ ] Plan rollback strategy

### During Migration
- [ ] Run database migration scripts
- [ ] Verify migration success with test queries
- [ ] Update configuration files
- [ ] Test basic functionality

### Post-Migration
- [ ] Validate search performance
- [ ] Train team on new capabilities
- [ ] Monitor usage patterns
- [ ] Collect feedback and optimize
- [ ] Document lessons learned

### Performance Benchmarks
- [ ] Search response time < 2 seconds
- [ ] Recommendation response time < 3 seconds
- [ ] Cache hit rate > 70%
- [ ] Zero critical errors in first week

## Next Steps

After successful migration:

1. **Train your team** on new search capabilities
2. **Monitor usage patterns** and optimize accordingly  
3. **Collect feedback** and adjust search behavior
4. **Explore advanced features** like automatic suggestions
5. **Integrate with development workflows** and documentation

## Support and Troubleshooting

### Logging and Diagnostics
```bash
# Enable detailed logging
export LOG_LEVEL=DEBUG
export TOOL_SEARCH_LOG_LEVEL=DEBUG

# Monitor logs
tail -f logs/turboprop.log | grep "tool_search"
```

### Getting Help
- Check the troubleshooting documentation
- Review performance metrics and logs
- Test with simple queries first
- Verify database integrity and embeddings

### Migration Support Script
```python
#!/usr/bin/env python3
"""Migration validation and support script."""

import sys
from pathlib import Path

def validate_migration():
    """Validate migration success."""
    print("🔍 Validating MCP Tool Search migration...")
    
    try:
        # Test basic functionality
        from tool_search_mcp_tools import search_mcp_tools
        results = search_mcp_tools("test query", max_results=1)
        
        if results.get('success'):
            print("✅ Basic search functionality working")
        else:
            print("❌ Search functionality failed")
            return False
            
        # Test database schema
        from database_manager import DatabaseManager
        db = DatabaseManager()
        tables = db.get_table_list()
        
        required_tables = ['mcp_tools', 'tool_parameters', 'tool_examples']
        missing_tables = [t for t in required_tables if t not in tables]
        
        if not missing_tables:
            print("✅ Database schema migration complete")
        else:
            print(f"❌ Missing tables: {missing_tables}")
            return False
            
        print("🎉 Migration validation successful!")
        return True
        
    except Exception as e:
        print(f"❌ Migration validation failed: {e}")
        return False

if __name__ == "__main__":
    success = validate_migration()
    sys.exit(0 if success else 1)
```

Remember: The migration is designed to be seamless and backward-compatible. Existing functionality continues to work while new capabilities are gradually adopted.