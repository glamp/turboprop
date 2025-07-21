# Integration Guide: MCP Tool Search System

## Overview

This guide provides comprehensive instructions for integrating the MCP Tool Search System with Claude Code, MCP clients, and other development tools. The system is designed to work seamlessly with existing MCP protocols while adding powerful tool discovery capabilities.

## Table of Contents

### Core Integration
- [Claude Code Integration](#claude-code-integration)
  - [Basic Integration](#basic-integration)
  - [Enhanced Claude Code Workflows](#enhanced-claude-code-workflows)
    - [1. Proactive Tool Suggestions](#1-proactive-tool-suggestions)
    - [2. Context-Aware Tool Selection](#2-context-aware-tool-selection)  
    - [3. Tool Chain Optimization](#3-tool-chain-optimization)
  - [Claude Code Configuration](#claude-code-configuration)
- [MCP Client Integration](#mcp-client-integration)
  - [Standard MCP Client Integration](#standard-mcp-client-integration)
  - [Python MCP Client Integration](#python-mcp-client-integration)
  - [Integration Patterns](#integration-patterns)
    - [1. Smart Tool Discovery Pattern](#1-smart-tool-discovery-pattern)
    - [2. Tool Chain Optimization Pattern](#2-tool-chain-optimization-pattern)

### Development Environment Integration
- [IDE and Editor Integration](#ide-and-editor-integration)
  - [Visual Studio Code Extension](#visual-studio-code-extension)
  - [Vim/Neovim Plugin](#vimneovim-plugin)
- [Custom Application Integration](#custom-application-integration)
  - [Web Application Integration](#web-application-integration)
  - [REST API Integration](#rest-api-integration)

### Testing and Performance
- [Testing Integration](#testing-integration)
  - [Integration Testing Framework](#integration-testing-framework)
  - [Automated Integration Tests](#automated-integration-tests)
- [Performance Considerations](#performance-considerations)
  - [Caching Strategies](#caching-strategies)
  - [Rate Limiting](#rate-limiting)

### Troubleshooting and Advanced Topics
- [Troubleshooting Integration Issues](#troubleshooting-integration-issues)
  - [Common Integration Problems](#common-integration-problems)
    - [1. Connection Issues](#1-connection-issues)
    - [2. Authentication Issues](#2-authentication-issues)
    - [3. Tool Discovery Issues](#3-tool-discovery-issues)
    - [4. Performance Issues](#4-performance-issues)
- [Advanced Integration Patterns](#advanced-integration-patterns)
  - [1. Intelligent Tool Chains](#1-intelligent-tool-chains)
  - [2. Context-Aware Tool Selection](#2-context-aware-tool-selection-1)

---

## Claude Code Integration

### Basic Integration

Claude Code automatically benefits from the MCP Tool Search System once it's enabled:

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

## Troubleshooting Integration Issues

### Common Integration Problems

#### 1. Connection Issues

**Problem**: Cannot connect to MCP Tool Search System
```
ConnectionError: Unable to connect to tool search service
```

**Solutions**:
```bash
# Check service status
curl -f http://localhost:8000/health || echo "Service not responding"

# Check network connectivity
telnet localhost 8000

# Verify MCP server is running
ps aux | grep mcp | grep -v grep

# Check logs for errors
tail -f /var/log/mcp-tool-search/service.log
```

**Code-level diagnostics**:
```python
async def diagnose_connection_issue():
    """Diagnose MCP Tool Search connection problems."""
    try:
        # Test basic connectivity
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.get('http://localhost:8000/health') as resp:
                if resp.status == 200:
                    print("✅ Service is reachable")
                    health_data = await resp.json()
                    print(f"Service status: {health_data}")
                else:
                    print(f"⚠️  Service returned status {resp.status}")
    
    except aiohttp.ClientConnectorError:
        print("❌ Cannot connect to service")
        print("Solutions:")
        print("1. Check if service is running on correct port")
        print("2. Verify firewall settings")
        print("3. Check service configuration")
    
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
```

#### 2. Authentication Issues

**Problem**: Authentication failures
```
PermissionError: Invalid API key or insufficient permissions
```

**Solutions**:
```python
# Verify API key configuration
import os

def check_auth_config():
    """Check authentication configuration."""
    api_key = os.environ.get('MCP_TOOL_SEARCH_API_KEY')
    
    if not api_key:
        print("❌ API key not set")
        print("Set environment variable: export MCP_TOOL_SEARCH_API_KEY=your_key")
        return False
    
    if len(api_key) < 16:
        print("⚠️  API key seems too short")
        return False
    
    # Test key validity
    import requests
    headers = {'Authorization': f'Bearer {api_key}'}
    
    try:
        response = requests.get('http://localhost:8000/api/tools/validate', headers=headers)
        if response.status_code == 200:
            print("✅ API key is valid")
            return True
        elif response.status_code == 401:
            print("❌ API key is invalid")
            return False
        else:
            print(f"⚠️  Unexpected response: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error validating API key: {e}")
        return False
```

#### 3. Tool Discovery Issues

**Problem**: No tools found or poor search results
```
{"success": true, "results": [], "message": "No tools found"}
```

**Solutions**:
```python
async def diagnose_search_issues(query: str):
    """Diagnose tool search problems."""
    from tool_search_mcp_tools import search_mcp_tools
    
    print(f"🔍 Diagnosing search for: '{query}'")
    
    # Test different search modes
    search_modes = ['semantic', 'keyword', 'hybrid']
    
    for mode in search_modes:
        try:
            results = await search_mcp_tools(
                query=query,
                search_mode=mode,
                max_results=5
            )
            
            if results['success']:
                count = len(results['results'])
                print(f"  {mode}: {count} results")
                
                if count == 0:
                    # Try broader search
                    broad_results = await search_mcp_tools(
                        query=query.split()[0],  # First word only
                        search_mode=mode,
                        max_results=5
                    )
                    if broad_results['success'] and broad_results['results']:
                        print(f"    Broader search found {len(broad_results['results'])} results")
            else:
                print(f"  {mode}: ERROR - {results.get('error', 'Unknown')}")
                
        except Exception as e:
            print(f"  {mode}: EXCEPTION - {e}")
    
    # Check catalog status
    try:
        catalog_info = await search_mcp_tools(query="*", max_results=1)
        if catalog_info['success']:
            total = catalog_info.get('total_results', 0)
            print(f"📊 Total tools in catalog: {total}")
            
            if total == 0:
                print("❌ Tool catalog is empty - check indexing")
            elif total < 10:
                print("⚠️  Tool catalog seems small - may need re-indexing")
        
    except Exception as e:
        print(f"❌ Could not check catalog status: {e}")

# Usage
await diagnose_search_issues("file operations")
```

#### 4. Performance Issues

**Problem**: Slow search responses
```
Search taking >5 seconds, timeouts occurring
```

**Solutions**:
```python
import time
import asyncio

class PerformanceDiagnostic:
    """Diagnose and optimize performance issues."""
    
    async def run_performance_tests(self):
        """Run comprehensive performance tests."""
        
        # Test different query complexities
        test_queries = [
            "read",  # Simple
            "file operations",  # Medium
            "complex data processing with error handling and logging"  # Complex
        ]
        
        results = {}
        
        for query in test_queries:
            print(f"Testing query: '{query}'")
            
            # Measure search time
            start_time = time.time()
            
            try:
                search_result = await search_mcp_tools(
                    query=query,
                    max_results=10
                )
                
                end_time = time.time()
                response_time = end_time - start_time
                
                results[query] = {
                    'response_time': response_time,
                    'success': search_result['success'],
                    'result_count': len(search_result.get('results', []))
                }
                
                # Performance assessment
                if response_time < 0.5:
                    status = "✅ Excellent"
                elif response_time < 1.0:
                    status = "🟡 Good"
                elif response_time < 2.0:
                    status = "⚠️  Acceptable"
                else:
                    status = "❌ Poor"
                
                print(f"  Response time: {response_time:.3f}s - {status}")
                
            except asyncio.TimeoutError:
                print(f"  ⏱️  TIMEOUT")
                results[query] = {'timeout': True}
            except Exception as e:
                print(f"  ❌ ERROR: {e}")
                results[query] = {'error': str(e)}
        
        return results
    
    def recommend_optimizations(self, results: dict):
        """Recommend performance optimizations."""
        print("\n🔧 Performance Optimization Recommendations:")
        
        slow_queries = [q for q, r in results.items() 
                       if r.get('response_time', 0) > 1.0]
        
        if slow_queries:
            print("• Enable result caching:")
            print("  export TOOL_SEARCH_CACHE_ENABLED=true")
            print("  export TOOL_SEARCH_CACHE_SIZE=1000")
            
            print("• Reduce max_results for faster searches:")
            print("  Use max_results=5 instead of 10+ for interactive use")
            
            print("• Use simpler search modes for speed:")
            print("  search_mode='keyword' for exact matches")
            print("  search_mode='hybrid' for balanced performance")
        
        timeouts = [q for q, r in results.items() if r.get('timeout')]
        if timeouts:
            print("• Increase timeout values:")
            print("  export TOOL_SEARCH_TIMEOUT=30")
        
        errors = [q for q, r in results.items() if r.get('error')]
        if errors:
            print("• Check system resources:")
            print("  Monitor CPU and memory usage during searches")
            print("  Consider scaling to multiple instances")

# Usage
diagnostic = PerformanceDiagnostic()
perf_results = await diagnostic.run_performance_tests()
diagnostic.recommend_optimizations(perf_results)
```

### Integration Testing

#### Automated Integration Tests

```python
import pytest
import asyncio
from tool_search_mcp_tools import search_mcp_tools, recommend_tools_for_task

class TestToolSearchIntegration:
    """Comprehensive integration tests."""
    
    @pytest.mark.asyncio
    async def test_basic_search_functionality(self):
        """Test basic search operations."""
        
        # Test successful search
        result = await search_mcp_tools("file operations", max_results=5)
        
        assert result['success'] is True
        assert isinstance(result['results'], list)
        assert len(result['results']) <= 5
        
        # Verify result structure
        if result['results']:
            tool = result['results'][0]
            required_fields = ['tool_id', 'name', 'description', 'similarity_score']
            
            for field in required_fields:
                assert field in tool, f"Missing field: {field}"
    
    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling scenarios."""
        
        # Test empty query
        result = await search_mcp_tools("", max_results=5)
        
        # Should handle gracefully
        assert 'success' in result
        if not result['success']:
            assert 'error' in result
    
    @pytest.mark.asyncio
    async def test_recommendation_system(self):
        """Test recommendation functionality."""
        
        result = await recommend_tools_for_task(
            "process CSV files",
            context="performance critical",
            max_recommendations=3
        )
        
        assert result['success'] is True
        assert 'recommendations' in result
        assert len(result['recommendations']) <= 3
    
    @pytest.mark.asyncio  
    async def test_search_modes(self):
        """Test different search modes."""
        
        query = "web scraping"
        modes = ['semantic', 'keyword', 'hybrid']
        
        for mode in modes:
            result = await search_mcp_tools(
                query=query,
                search_mode=mode,
                max_results=3
            )
            
            # All modes should work
            assert result['success'] is True, f"Search mode {mode} failed"
    
    def test_concurrent_searches(self):
        """Test system under concurrent load."""
        
        async def concurrent_test():
            tasks = []
            
            for i in range(10):
                task = search_mcp_tools(f"test query {i}", max_results=3)
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Count successful vs failed requests
            successes = sum(1 for r in results 
                          if isinstance(r, dict) and r.get('success'))
            
            # At least 80% should succeed under load
            assert successes >= 8, f"Only {successes}/10 requests succeeded"
        
        asyncio.run(concurrent_test())

# Run tests
# pytest test_integration.py -v --asyncio-mode=auto
```

## Advanced Integration Patterns

### 1. Intelligent Tool Chains

```python
class IntelligentToolChain:
    """Build and execute intelligent tool chains."""
    
    def __init__(self):
        self.execution_history = []
        self.context_memory = {}
    
    async def build_adaptive_chain(self, goal: str, context: dict = None):
        """Build a tool chain that adapts based on execution results."""
        
        # Start with initial recommendations
        initial_recs = await recommend_tools_for_task(
            goal,
            context=self._format_context(context),
            max_recommendations=3
        )
        
        if not initial_recs['success']:
            raise ValueError("Could not get initial tool recommendations")
        
        # Build adaptive chain
        chain = []
        current_context = context or {}
        
        for step, rec in enumerate(initial_recs['recommendations']):
            tool_id = rec['tool']['tool_id']
            
            # Add context-aware configuration
            tool_config = {
                'tool_id': tool_id,
                'step': step,
                'context': current_context.copy(),
                'fallbacks': self._get_fallback_tools(rec),
                'success_criteria': self._define_success_criteria(rec, goal)
            }
            
            chain.append(tool_config)
            
            # Update context for next step
            current_context.update({
                'previous_tool': tool_id,
                'chain_progress': f"{step + 1}/{len(initial_recs['recommendations'])}"
            })
        
        return chain
    
    async def execute_chain(self, chain: list):
        """Execute tool chain with adaptive error recovery."""
        
        results = []
        context = {}
        
        for step_config in chain:
            step_result = await self._execute_step_with_recovery(step_config, context)
            results.append(step_result)
            
            # Update context with step results
            if step_result['success']:
                context.update(step_result.get('context_updates', {}))
            else:
                # Handle step failure
                recovery_result = await self._handle_step_failure(
                    step_config, step_result, context
                )
                
                if recovery_result['recovered']:
                    results[-1] = recovery_result['result']
                    context.update(recovery_result['result'].get('context_updates', {}))
                else:
                    # Chain failed - attempt emergency recovery
                    return await self._emergency_recovery(results, chain, context)
        
        return {
            'success': True,
            'chain_results': results,
            'final_context': context,
            'performance_metrics': self._calculate_metrics(results)
        }
    
    def _format_context(self, context):
        """Format context for API calls."""
        if not context:
            return "default context"
        
        context_parts = []
        for key, value in context.items():
            context_parts.append(f"{key}: {value}")
        
        return ", ".join(context_parts)
    
    def _get_fallback_tools(self, recommendation):
        """Get fallback tools for a recommendation."""
        return recommendation.get('alternative_tools', [])[:2]  # Top 2 alternatives
    
    def _define_success_criteria(self, recommendation, goal):
        """Define success criteria for a tool execution."""
        return {
            'min_confidence': 0.7,
            'max_execution_time': 30.0,
            'required_output_fields': ['success', 'result']
        }

# Usage Example
chain_builder = IntelligentToolChain()

# Build chain for complex task
chain = await chain_builder.build_adaptive_chain(
    "Deploy web application with monitoring and backup",
    context={
        'environment': 'production',
        'priority': 'high_availability',
        'constraints': ['security_compliance', 'performance_monitoring']
    }
)

# Execute with adaptive recovery
result = await chain_builder.execute_chain(chain)
print(f"Chain execution: {'✅ Success' if result['success'] else '❌ Failed'}")
```

### 2. Context-Aware Tool Selection

```python
class ContextualToolSelector:
    """Select tools based on rich contextual information."""
    
    def __init__(self):
        self.context_analyzers = [
            self._analyze_user_expertise,
            self._analyze_project_constraints,
            self._analyze_performance_requirements,
            self._analyze_security_requirements,
            self._analyze_integration_context
        ]
    
    async def select_optimal_tools(self, task: str, context: dict):
        """Select optimal tools using comprehensive context analysis."""
        
        # Analyze context through multiple lenses
        enriched_context = await self._enrich_context(context)
        
        # Get base recommendations
        base_recs = await recommend_tools_for_task(
            task,
            context=self._format_context(enriched_context),
            max_recommendations=10
        )
        
        if not base_recs['success']:
            return base_recs
        
        # Apply contextual scoring
        scored_tools = []
        
        for rec in base_recs['recommendations']:
            contextual_score = await self._calculate_contextual_score(
                rec, enriched_context, task
            )
            
            enhanced_rec = rec.copy()
            enhanced_rec['contextual_score'] = contextual_score
            enhanced_rec['final_score'] = (
                rec['recommendation_score'] * 0.6 + 
                contextual_score * 0.4
            )
            
            scored_tools.append(enhanced_rec)
        
        # Re-rank by contextual score
        scored_tools.sort(key=lambda x: x['final_score'], reverse=True)
        
        return {
            'success': True,
            'recommendations': scored_tools[:5],
            'context_analysis': enriched_context,
            'selection_reasoning': self._explain_selection(scored_tools[:3], enriched_context)
        }
    
    async def _enrich_context(self, base_context: dict):
        """Enrich context with additional analysis."""
        enriched = base_context.copy()
        
        for analyzer in self.context_analyzers:
            analysis_result = await analyzer(base_context)
            enriched.update(analysis_result)
        
        return enriched
    
    async def _analyze_user_expertise(self, context: dict):
        """Analyze user expertise level."""
        expertise_indicators = {
            'beginner': ['new to', 'learning', 'first time', 'help'],
            'intermediate': ['familiar with', 'some experience', 'usually'],
            'expert': ['optimize', 'performance', 'advanced', 'complex']
        }
        
        # Analyze context text for expertise clues
        context_text = str(context).lower()
        
        expertise_scores = {}
        for level, indicators in expertise_indicators.items():
            score = sum(1 for indicator in indicators if indicator in context_text)
            expertise_scores[level] = score
        
        # Determine most likely expertise level
        likely_expertise = max(expertise_scores, key=expertise_scores.get)
        
        return {
            'user_expertise': likely_expertise,
            'expertise_confidence': expertise_scores[likely_expertise] / 4.0,
            'preferences': {
                'beginner': {'prioritize': 'safety', 'avoid': 'complexity'},
                'intermediate': {'prioritize': 'efficiency', 'balance': 'features'},
                'expert': {'prioritize': 'power', 'accept': 'complexity'}
            }.get(likely_expertise, {})
        }

# Usage Example
selector = ContextualToolSelector()

# Rich context for tool selection
context = {
    'user_message': "I'm new to deployment but need to deploy this critical application quickly",
    'project_type': 'web_application',
    'environment': 'production',
    'constraints': ['time_sensitive', 'reliability_critical'],
    'team_size': 3,
    'experience_level': 'mixed'
}

# Get contextually optimized tool recommendations
recommendations = await selector.select_optimal_tools(
    "Deploy web application with monitoring", 
    context
)

if recommendations['success']:
    print("🎯 Contextually optimized recommendations:")
    for i, rec in enumerate(recommendations['recommendations'][:3], 1):
        print(f"{i}. {rec['tool']['name']} (score: {rec['final_score']:.3f})")
        print(f"   Context fit: {rec['contextual_score']:.3f}")
        print(f"   Why: {rec['recommendation_reasons'][0]}")
```

This comprehensive integration guide provides complete patterns for integrating the MCP Tool Search System with various clients, IDEs, and applications while addressing common issues and providing advanced capabilities.