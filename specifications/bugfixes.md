Below are commands that were executed and the corresponding output. It includes the stack traces/errors

```
python -m turboprop search "vector matching"
🚀 Turboprop - Semantic Code Search
========================================
⚡ Initializing AI model...
✅ Initialized all-MiniLM-L6-v2 on cpu device
2025-07-21 06:30:03,120 - turboprop.code_index - INFO - Embedding generator initialized successfully

🔎 Searching for: "vector matching"
📊 Mode: auto | Results: 5
2025-07-21 06:30:03,134 - turboprop.database_manager - INFO - Starting schema migration for table code_files
2025-07-21 06:30:03,140 - turboprop.database_manager - INFO - Schema migration completed - no columns needed to be added
2025-07-21 06:30:03,140 - turboprop.database_manager - INFO - Creating repository_context table
2025-07-21 06:30:03,141 - turboprop.database_manager - INFO - Successfully created repository_context table with indexes
Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "/Users/glamp/workspace/github.com/glamp/turboprop/turboprop/__main__.py", line 34, in <module>
    main()
  File "/Users/glamp/workspace/github.com/glamp/turboprop/turboprop/__main__.py", line 30, in main
    cli_main()
  File "/Users/glamp/workspace/github.com/glamp/turboprop/code_index.py", line 1530, in main
    handle_search_command(args, embedder)
  File "/Users/glamp/workspace/github.com/glamp/turboprop/code_index.py", line 1450, in handle_search_command
    formatted_results = format_hybrid_search_results(
                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/glamp/workspace/github.com/glamp/turboprop/search_operations.py", line 1371, in format_hybrid_search_results
    lines.append(f"   📊 Similarity: {result.similarity_percentage:.1f}% " f"({result.confidence_level} confidence)")
                                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/glamp/workspace/github.com/glamp/turboprop/search_result_types.py", line 213, in similarity_percentage
    return self.similarity_score * 100.0
           ~~~~~~~~~~~~~~~~~~~~~~^~~~~~~
TypeError: unsupported operand type(s) for *: 'decimal.Decimal' and 'float'
```

```
python -m turboprop search "tool search entrypoint" --mode hybrid
🚀 Turboprop - Semantic Code Search
========================================
⚡ Initializing AI model...
✅ Initialized all-MiniLM-L6-v2 on cpu device
2025-07-21 06:30:38,150 - turboprop.code_index - INFO - Embedding generator initialized successfully

🔎 Searching for: "tool search entrypoint"
📊 Mode: hybrid | Results: 5
⚖️  Weights: semantic=0.6, text=0.4
2025-07-21 06:30:38,164 - turboprop.database_manager - INFO - Starting schema migration for table code_files
2025-07-21 06:30:38,171 - turboprop.database_manager - INFO - Schema migration completed - no columns needed to be added
2025-07-21 06:30:38,171 - turboprop.database_manager - INFO - Creating repository_context table
2025-07-21 06:30:38,171 - turboprop.database_manager - INFO - Successfully created repository_context table with indexes
2025-07-21 06:30:38,182 - turboprop.database_manager - INFO - Creating full-text search index for table code_files
2025-07-21 06:30:38,198 - turboprop.database_manager - WARNING - PRAGMA FTS creation failed: Catalog Error: Table with name code_files_fts does not exist!
Did you mean "code_index.code_files_fts"?. Trying alternative approach.
2025-07-21 06:30:38,204 - turboprop.database_manager - ERROR - Database operation failed: Catalog Error: Scalar Function with name to_tsvector does not exist!
Did you mean "to_seconds"?

LINE 1: ... EXISTS idx_code_files_fts_content ON code_files_fts USING gin(to_tsvector('english', content))
                                                                          ^
2025-07-21 06:30:38,204 - turboprop.database_manager - ERROR - Failed to create FTS index: Database operation failed: Catalog Error: Scalar Function with name to_tsvector does not exist!
Did you mean "to_seconds"?

LINE 1: ... EXISTS idx_code_files_fts_content ON code_files_fts USING gin(to_tsvector('english', content))
                                                                          ^
2025-07-21 06:30:38,204 - turboprop.database_manager - WARNING - FTS index creation failed, full-text search will be limited
🔀 Found 5 hybrid search results for: 'tool search entrypoint'
============================================================
⚠️ [1] issues/complete/000023_step.md
   📊 Similarity: 55.6% (low confidence)
   📄 Type: markdown (16.5KB)
   💻 Lines 1-16: # Step 000023: Semantic Tool Search Implementation

## Overview
Implement semantic search capabilities over the MCP tool catalog, enabling natural language queries to find tools by functionality, purp...

⚠️ [2] issues/complete/000027_step.md
   📊 Similarity: 53.5% (low confidence)
   📄 Type: markdown (21.0KB)
   💻 Lines 1-16: # Step 000027: MCP Tools for Tool Search (search_mcp_tools, get_tool_details)

## Overview
Implement core MCP tools that expose the tool search functionality to Claude Code and other MCP clients. This...

⚠️ [3] issues/complete/000031_step.md
   📊 Similarity: 48.3% (low confidence)
   📄 Type: markdown (23.0KB)
   💻 Lines 1-16: # Step 000031: Automatic Tool Selection Engine

## Overview
Implement an intelligent automatic tool selection system that can proactively analyze Claude Code's usage patterns and automatically suggest...

⚠️ [4] issues/complete/000020_step.md
   📊 Similarity: 48.0% (low confidence)
   📄 Type: markdown (7.9KB)
   💻 Lines 1-16: # Step 000020: Tool Discovery Framework for System Tools

## Overview
Implement a comprehensive tool discovery system that can automatically identify and catalog all available MCP tools, starting with...

⚠️ [5] parameter_search_engine.py
   📊 Similarity: 47.9% (low confidence)
   📄 Type: python (25.7KB)
   💻 Lines 27-37: from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from mcp_metadata_types import ParameterAnalysis, ToolId

class ToolChainStep:
    """A single step in a to...


✨ Done!
```
