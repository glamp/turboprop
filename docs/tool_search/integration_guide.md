# Integration Guide: MCP Tool Search System

## Overview

This guide provides comprehensive instructions for integrating the MCP Tool Search System with Claude Code, MCP clients, and other development tools. The system is designed to work seamlessly with existing MCP protocols while adding powerful tool discovery capabilities.

## Claude Code Integration

### Basic Integration

Claude Code automatically benefits from the tool search system once it's enabled:

```python
# Claude Code can now discover tools intelligently
# Instead of guessing which tools to use, Claude can search:

# Old approach
# User: "Read this config file"
# Claude: Uses 'read' tool (hoping it exists)

# New approach with tool search
# User: "Read this config file safely with error handling"
# Claude: Searches for optimal tool and finds 'read' with safety features
```

### Enhanced Claude Code Workflows

#### 1. Proactive Tool Suggestions
Enable Claude Code to suggest better tools during conversations:

```python
@mcp.tool()
def enhanced_file_operation(file_path: str, operation: str) -> dict:
    """Enhanced file operation with tool suggestions."""
    
    # Get recommendations for the specific operation
    recommendations = recommend_tools_for_task(
        f"{operation} file operations",
        context=f"file: {file_path}, safety critical"
    )
    
    if recommendations['recommendations']:
        best_tool = recommendations['recommendations'][0]
        
        # Execute with the recommended tool
        result = execute_tool_operation(best_tool, file_path)
        
        # Return result with suggestions for next time
        return {
            "result": result,
            "suggestions": {
                "recommended_tool": best_tool['tool']['name'],
                "reasoning": best_tool['recommendation_reasons'],
                "alternatives": [alt['tool_id'] for alt in recommendations['recommendations'][1:3]]
            }
        }
    
    # Fallback to default behavior
    return execute_default_operation(file_path, operation)
```

#### 2. Context-Aware Tool Selection
Help Claude Code choose tools based on conversation context:

```python
class ContextAwareToolSelector:
    def __init__(self):
        self.conversation_context = {}
        
    def select_tool_for_task(self, task_description: str, conversation_history: List[str]) -> dict:
        """Select optimal tool based on conversation context."""
        
        # Analyze conversation for context clues
        context = self._analyze_conversation_context(conversation_history)
        
        # Get contextualized recommendations
        recommendations = recommend_tools_for_task(
            task_description,
            context=context,
            complexity_preference=self._infer_user_skill_level(conversation_history)
        )
        
        return recommendations
    
    def _analyze_conversation_context(self, history: List[str]) -> str:
        """Extract context from conversation history."""
        context_clues = []
        
        for message in history[-5:]:  # Last 5 messages
            if "error" in message.lower():
                context_clues.append("error handling important")
            if "large file" in message.lower():
                context_clues.append("performance critical")
            if "beginner" in message.lower() or "new to" in message.lower():
                context_clues.append("beginner-friendly preferred")
                
        return ", ".join(context_clues)
```

#### 3. Tool Chain Optimization
Enable Claude Code to optimize sequences of tool usage:

```python
@mcp.tool()
def optimize_tool_sequence(task_description: str, current_tools: List[str]) -> dict:
    """Optimize a sequence of tool operations."""
    
    # Analyze the current tool sequence
    analysis = analyze_tool_sequence(current_tools, task_description)
    
    if analysis['optimization_potential'] > 0.3:
        # Get optimized sequence
        optimized = recommend_tool_sequence(
            task_description,
            current_sequence=current_tools,
            optimization_goals=["efficiency", "reliability"]
        )
        
        return {
            "optimized_sequence": optimized['recommended_sequence'],
            "improvements": optimized['improvements'],
            "estimated_time_savings": optimized['time_savings'],
            "reliability_improvement": optimized['reliability_boost']
        }
    
    return {"message": "Current tool sequence is already optimal"}
```

### Claude Code Configuration

Add tool search configuration to Claude Code settings:

```json
{
  "tool_search": {
    "enabled": true,
    "proactive_suggestions": true,
    "suggestion_threshold": 0.8,
    "context_analysis": true,
    "learning_mode": true
  },
  "tool_selection": {
    "default_mode": "smart",
    "fallback_mode": "traditional",
    "confidence_threshold": 0.7
  }
}
```

## MCP Client Integration

### Standard MCP Client Integration

For MCP clients that follow the standard protocol:

```typescript
// TypeScript/JavaScript MCP Client
import { MCPClient } from '@modelcontextprotocol/client';

class EnhancedMCPClient extends MCPClient {
    async findToolsForTask(taskDescription: string, options?: SearchOptions): Promise<ToolRecommendations> {
        return await this.call('recommend_tools_for_task', {
            task_description: taskDescription,
            max_recommendations: options?.maxResults || 5,
            include_alternatives: options?.includeAlternatives ?? true,
            explain_reasoning: options?.explainReasoning ?? true
        });
    }
    
    async searchTools(query: string, options?: SearchOptions): Promise<ToolSearchResults> {
        return await this.call('search_mcp_tools', {
            query,
            max_results: options?.maxResults || 10,
            search_mode: options?.searchMode || 'hybrid',
            include_examples: options?.includeExamples ?? true
        });
    }
    
    async compareTools(toolIds: string[], context?: string): Promise<ToolComparison> {
        return await this.call('compare_mcp_tools', {
            tool_ids: toolIds,
            comparison_context: context,
            include_decision_guidance: true
        });
    }
}
```

### Python MCP Client Integration

```python
from mcp import Client
from typing import List, Optional, Dict, Any

class ToolSearchEnabledClient(Client):
    """MCP Client with integrated tool search capabilities."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tool_cache = {}
        self.search_cache = {}
    
    async def smart_tool_selection(self, task: str, context: Optional[str] = None) -> Dict[str, Any]:
        """Intelligently select tools for a given task."""
        
        # Check cache first
        cache_key = f"{task}:{context}"
        if cache_key in self.search_cache:
            return self.search_cache[cache_key]
        
        # Get recommendations
        recommendations = await self.call('recommend_tools_for_task', {
            'task_description': task,
            'context': context,
            'max_recommendations': 3
        })
        
        # Cache results
        self.search_cache[cache_key] = recommendations
        return recommendations
    
    async def execute_with_fallback(self, primary_tool: str, fallback_tools: List[str], **params) -> Dict[str, Any]:
        """Execute tool with intelligent fallback options."""
        
        try:
            return await self.call(primary_tool, params)
        except Exception as e:
            # Try fallback tools
            for fallback in fallback_tools:
                try:
                    result = await self.call(fallback, params)
                    # Log successful fallback for learning
                    self._log_fallback_success(primary_tool, fallback, str(e))
                    return result
                except Exception:
                    continue
            
            # All tools failed, re-raise original exception
            raise e
    
    def _log_fallback_success(self, primary: str, fallback: str, error: str):
        """Log successful fallback for system learning."""
        # Implementation depends on your logging infrastructure
        pass
```

### Integration Patterns

#### 1. Smart Tool Discovery Pattern
```python
class SmartToolDiscovery:
    """Pattern for intelligent tool discovery in MCP clients."""
    
    def __init__(self, client: MCPClient):
        self.client = client
        self.discovery_cache = {}
    
    async def discover_tools_for_capability(self, capability: str) -> List[str]:
        """Discover tools that provide a specific capability."""
        
        if capability in self.discovery_cache:
            return self.discovery_cache[capability]
        
        # Search for tools with this capability
        results = await self.client.call('search_mcp_tools', {
            'query': f"tools with {capability} capability",
            'max_results': 10,
            'search_mode': 'semantic'
        })
        
        tool_ids = [tool['tool_id'] for tool in results['results']]
        self.discovery_cache[capability] = tool_ids
        return tool_ids
    
    async def get_best_tool_for_context(self, task: str, context: Dict[str, Any]) -> str:
        """Get the best tool for a specific context."""
        
        context_str = ", ".join([f"{k}: {v}" for k, v in context.items()])
        
        recommendations = await self.client.call('recommend_tools_for_task', {
            'task_description': task,
            'context': context_str,
            'max_recommendations': 1
        })
        
        if recommendations['recommendations']:
            return recommendations['recommendations'][0]['tool']['tool_id']
        
        return None
```

#### 2. Tool Chain Optimization Pattern
```python
class ToolChainOptimizer:
    """Optimize sequences of tool operations."""
    
    def __init__(self, client: MCPClient):
        self.client = client
        
    async def optimize_workflow(self, workflow_description: str, current_tools: List[str]) -> Dict[str, Any]:
        """Optimize a workflow by suggesting better tool sequences."""
        
        # Analyze current workflow
        analysis = await self.client.call('analyze_tool_sequence', {
            'sequence': current_tools,
            'workflow_description': workflow_description
        })
        
        if analysis.get('optimization_potential', 0) > 0.2:
            # Get optimized sequence
            optimization = await self.client.call('recommend_tool_sequence', {
                'task_description': workflow_description,
                'current_sequence': current_tools,
                'optimization_goals': ['efficiency', 'reliability']
            })
            
            return {
                'should_optimize': True,
                'current_sequence': current_tools,
                'optimized_sequence': optimization['recommended_sequence'],
                'benefits': optimization['improvements']
            }
        
        return {'should_optimize': False, 'current_sequence': current_tools}
```

## IDE and Editor Integration

### Visual Studio Code Extension

Create a VSCode extension that leverages tool search:

```typescript
// VSCode Extension for Tool Search
import * as vscode from 'vscode';
import { MCPClient } from './mcp-client';

export class ToolSearchProvider {
    private client: MCPClient;
    
    constructor(client: MCPClient) {
        this.client = client;
    }
    
    async provideToolSuggestions(document: vscode.TextDocument, position: vscode.Position): Promise<vscode.CompletionItem[]> {
        // Analyze current context
        const context = this.analyzeContext(document, position);
        
        if (context.isToolContext) {
            // Get tool recommendations
            const recommendations = await this.client.searchTools(context.query);
            
            return recommendations.results.map(tool => {
                const item = new vscode.CompletionItem(tool.name, vscode.CompletionItemKind.Function);
                item.detail = tool.description;
                item.documentation = new vscode.MarkdownString(this.formatToolDocumentation(tool));
                item.insertText = this.generateToolUsage(tool);
                return item;
            });
        }
        
        return [];
    }
    
    private analyzeContext(document: vscode.TextDocument, position: vscode.Position): any {
        // Analyze the current code context to determine if tool suggestions are needed
        const line = document.lineAt(position);
        const text = line.text;
        
        // Look for patterns that indicate tool usage
        if (text.includes('mcp.call(') || text.includes('@mcp.tool')) {
            return {
                isToolContext: true,
                query: this.extractIntentFromContext(document, position)
            };
        }
        
        return { isToolContext: false };
    }
}
```

### Vim/Neovim Plugin

```lua
-- Neovim plugin for tool search integration
local M = {}

local function search_tools(query)
    local client = require('mcp-client')
    local results = client.call('search_mcp_tools', {
        query = query,
        max_results = 10,
        include_examples = true
    })
    
    return results.results or {}
end

local function show_tool_picker(tools)
    local pickers = require('telescope.pickers')
    local finders = require('telescope.finders')
    local conf = require('telescope.config').values
    
    pickers.new({}, {
        prompt_title = "MCP Tool Search",
        finder = finders.new_table {
            results = tools,
            entry_maker = function(entry)
                return {
                    value = entry,
                    display = entry.name .. " - " .. entry.description,
                    ordinal = entry.name,
                }
            end,
        },
        sorter = conf.generic_sorter({}),
        attach_mappings = function(prompt_bufnr, map)
            map('i', '<CR>', function()
                local selection = require('telescope.actions.state').get_selected_entry()
                require('telescope.actions').close(prompt_bufnr)
                M.insert_tool_usage(selection.value)
            end)
            return true
        end,
    }):find()
end

function M.search_and_insert()
    local query = vim.fn.input('Search for tools: ')
    if query ~= '' then
        local tools = search_tools(query)
        if #tools > 0 then
            show_tool_picker(tools)
        else
            print('No tools found for: ' .. query)
        end
    end
end

function M.insert_tool_usage(tool)
    local lines = {
        'result = mcp.call("' .. tool.tool_id .. '", {',
        '    # Parameters for ' .. tool.name,
    }
    
    -- Add parameter placeholders
    for _, param in ipairs(tool.parameters or {}) do
        table.insert(lines, '    ' .. param.name .. ' = "",  # ' .. param.description)
    end
    
    table.insert(lines, '})')
    
    -- Insert at cursor
    local row, col = unpack(vim.api.nvim_win_get_cursor(0))
    vim.api.nvim_buf_set_lines(0, row, row, false, lines)
end

return M
```

## Custom Application Integration

### Web Application Integration

```javascript
// JavaScript/React integration
class ToolSearchIntegration {
    constructor(mcpEndpoint) {
        this.mcpEndpoint = mcpEndpoint;
        this.cache = new Map();
    }
    
    async searchTools(query, options = {}) {
        const cacheKey = `search:${query}:${JSON.stringify(options)}`;
        
        if (this.cache.has(cacheKey)) {
            return this.cache.get(cacheKey);
        }
        
        const response = await fetch(`${this.mcpEndpoint}/search_mcp_tools`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                query,
                max_results: options.maxResults || 10,
                search_mode: options.searchMode || 'hybrid'
            })
        });
        
        const results = await response.json();
        this.cache.set(cacheKey, results);
        
        // Cache for 5 minutes
        setTimeout(() => this.cache.delete(cacheKey), 5 * 60 * 1000);
        
        return results;
    }
    
    async getRecommendations(taskDescription, context = null) {
        const response = await fetch(`${this.mcpEndpoint}/recommend_tools_for_task`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                task_description: taskDescription,
                context: context,
                max_recommendations: 5
            })
        });
        
        return await response.json();
    }
}

// React component for tool search
import React, { useState, useCallback } from 'react';

const ToolSearchComponent = ({ onToolSelect }) => {
    const [query, setQuery] = useState('');
    const [results, setResults] = useState([]);
    const [loading, setLoading] = useState(false);
    
    const integration = new ToolSearchIntegration('/api/mcp');
    
    const handleSearch = useCallback(async () => {
        if (!query.trim()) return;
        
        setLoading(true);
        try {
            const searchResults = await integration.searchTools(query);
            setResults(searchResults.results || []);
        } catch (error) {
            console.error('Search failed:', error);
        } finally {
            setLoading(false);
        }
    }, [query]);
    
    return (
        <div className="tool-search">
            <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
                placeholder="Search for tools..."
                className="search-input"
            />
            <button onClick={handleSearch} disabled={loading}>
                {loading ? 'Searching...' : 'Search'}
            </button>
            
            <div className="results">
                {results.map((tool) => (
                    <div key={tool.tool_id} className="tool-result" onClick={() => onToolSelect(tool)}>
                        <h3>{tool.name}</h3>
                        <p>{tool.description}</p>
                        <div className="confidence">Confidence: {Math.round(tool.similarity_score * 100)}%</div>
                        <div className="match-reasons">
                            {tool.match_reasons.map((reason, i) => (
                                <span key={i} className="reason-badge">{reason}</span>
                            ))}
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
};

export default ToolSearchComponent;
```

### REST API Integration

```python
# Flask/FastAPI wrapper for tool search
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import asyncio

app = FastAPI(title="Tool Search API")

class SearchRequest(BaseModel):
    query: str
    max_results: int = 10
    search_mode: str = 'hybrid'
    category: Optional[str] = None

class RecommendationRequest(BaseModel):
    task_description: str
    context: Optional[str] = None
    max_recommendations: int = 5
    complexity_preference: str = 'balanced'

@app.post("/search")
async def search_tools(request: SearchRequest):
    """Search for tools by functionality."""
    try:
        # Import here to avoid circular imports
        from tool_search_mcp_tools import search_mcp_tools
        
        result = await search_mcp_tools(
            query=request.query,
            max_results=request.max_results,
            search_mode=request.search_mode,
            category=request.category
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/recommend")
async def recommend_tools(request: RecommendationRequest):
    """Get tool recommendations for a task."""
    try:
        from tool_recommendation_mcp_tools import recommend_tools_for_task
        
        result = await recommend_tools_for_task(
            task_description=request.task_description,
            context=request.context,
            max_recommendations=request.max_recommendations,
            complexity_preference=request.complexity_preference
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/tools/{tool_id}")
async def get_tool_details(tool_id: str):
    """Get detailed information about a specific tool."""
    try:
        from tool_search_mcp_tools import get_tool_details
        
        result = await get_tool_details(tool_id, include_examples=True)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

## Testing Integration

### Integration Testing Framework

```python
import pytest
from unittest.mock import Mock, patch
from tool_search_integration import ToolSearchIntegration

class TestToolSearchIntegration:
    """Test suite for tool search integration."""
    
    @pytest.fixture
    def integration(self):
        return ToolSearchIntegration()
    
    @pytest.mark.asyncio
    async def test_basic_search_integration(self, integration):
        """Test basic search functionality."""
        results = await integration.search_tools("file operations")
        
        assert results['success'] is True
        assert len(results['results']) > 0
        assert all('tool_id' in tool for tool in results['results'])
    
    @pytest.mark.asyncio
    async def test_recommendation_integration(self, integration):
        """Test recommendation functionality."""
        recommendations = await integration.get_recommendations(
            "read configuration files safely"
        )
        
        assert recommendations['success'] is True
        assert len(recommendations['recommendations']) > 0
    
    @pytest.mark.asyncio
    async def test_client_fallback_behavior(self, integration):
        """Test fallback behavior when search fails."""
        with patch('tool_search_mcp_tools.search_mcp_tools', side_effect=Exception("Search failed")):
            # Should fall back to basic tool discovery
            result = await integration.search_tools_with_fallback("file operations")
            
            assert result is not None
            # Should contain some basic tools even if search fails
```

## Performance Considerations

### Caching Strategies

```python
class IntegrationCache:
    """Caching layer for tool search integration."""
    
    def __init__(self):
        self.search_cache = TTLCache(maxsize=1000, ttl=3600)
        self.recommendation_cache = TTLCache(maxsize=500, ttl=1800)
        self.tool_details_cache = TTLCache(maxsize=200, ttl=7200)
    
    def get_cached_search(self, query: str, options: dict) -> Optional[dict]:
        cache_key = f"search:{query}:{hash(frozenset(options.items()))}"
        return self.search_cache.get(cache_key)
    
    def cache_search_result(self, query: str, options: dict, result: dict):
        cache_key = f"search:{query}:{hash(frozenset(options.items()))}"
        self.search_cache[cache_key] = result
```

### Rate Limiting

```python
from functools import wraps
import time
import asyncio

class RateLimiter:
    def __init__(self, calls_per_second: int = 10):
        self.calls_per_second = calls_per_second
        self.last_called = 0
        self.call_count = 0
    
    def __call__(self, func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            current_time = time.time()
            
            # Reset counter every second
            if current_time - self.last_called >= 1:
                self.call_count = 0
                self.last_called = current_time
            
            # Check rate limit
            if self.call_count >= self.calls_per_second:
                sleep_time = 1 - (current_time - self.last_called)
                await asyncio.sleep(sleep_time)
                self.call_count = 0
                self.last_called = time.time()
            
            self.call_count += 1
            return await func(*args, **kwargs)
        
        return wrapper
```

This integration guide provides comprehensive patterns for integrating the MCP Tool Search System with various clients, IDEs, and applications while maintaining performance and reliability.