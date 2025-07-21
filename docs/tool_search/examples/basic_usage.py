#!/usr/bin/env python3
"""
Basic Usage Examples for MCP Tool Search System

This file demonstrates common usage patterns for the tool search system
with working examples that can be tested against a live system.
"""

import asyncio
from typing import Dict, List, Any

# Example 1: Basic Tool Search
async def basic_tool_search():
    """Demonstrate basic tool search functionality."""
    print("=== Basic Tool Search ===")
    
    # Import the MCP tools (these would be available in a real MCP environment)
    from tool_search_mcp_tools import search_mcp_tools
    
    # Simple search for file operations
    results = await search_mcp_tools(
        query="read configuration files safely",
        max_results=5,
        include_examples=True
    )
    
    if results['success']:
        print(f"Found {len(results['results'])} tools for file operations:")
        
        for i, tool in enumerate(results['results'], 1):
            print(f"\n{i}. {tool['name']} (confidence: {tool['similarity_score']:.2f})")
            print(f"   Description: {tool['description']}")
            print(f"   Match reasons: {', '.join(tool['match_reasons'])}")
            
            if tool.get('examples'):
                print(f"   Example usage: {tool['examples'][0]['code'][:50]}...")
    else:
        print(f"Search failed: {results.get('error', 'Unknown error')}")

# Example 2: Task-Based Recommendations
async def task_based_recommendations():
    """Demonstrate intelligent task-based tool recommendations."""
    print("\n=== Task-Based Recommendations ===")
    
    from tool_recommendation_mcp_tools import recommend_tools_for_task
    
    # Get recommendations for a specific development task
    recommendations = await recommend_tools_for_task(
        task_description="process CSV files and generate reports",
        context="performance critical, large files, Python environment",
        max_recommendations=3,
        complexity_preference="balanced"
    )
    
    if recommendations['success']:
        print("Recommended tools for CSV processing:")
        
        for i, rec in enumerate(recommendations['recommendations'], 1):
            tool = rec['tool']
            print(f"\n{i}. {tool['name']} (score: {rec['recommendation_score']:.2f})")
            print(f"   Task alignment: {rec['task_alignment']:.2f}")
            print(f"   Reasons: {rec['recommendation_reasons'][0]}")
            print(f"   When to use: {rec['when_to_use']}")
            
            if rec['usage_guidance']:
                print(f"   Guidance: {rec['usage_guidance'][0]}")

# Example 3: Tool Comparison
async def tool_comparison():
    """Demonstrate tool comparison functionality."""
    print("\n=== Tool Comparison ===")
    
    from tool_comparison_mcp_tools import compare_mcp_tools
    
    # Compare different file operation tools
    comparison = await compare_mcp_tools(
        tool_ids=["read", "write", "edit"],
        comparison_context="configuration file management",
        include_decision_guidance=True
    )
    
    if comparison['success']:
        tools = comparison['comparison']['tools']
        summary = comparison['comparison']['summary']
        
        print("Tool Comparison for configuration file management:")
        print(f"Recommended choice: {summary['recommended_choice']}")
        print(f"Reasoning: {summary['reasoning']}")
        
        print("\nDetailed comparison:")
        for tool in tools:
            print(f"\n• {tool['tool_id'].upper()}")
            print(f"  Overall score: {tool['overall_score']:.2f}")
            print(f"  Strengths: {', '.join(tool['strengths'])}")
            print(f"  Best for: {', '.join(tool['best_for'])}")

# Example 4: Advanced Search with Filters
async def advanced_search():
    """Demonstrate advanced search with filtering options."""
    print("\n=== Advanced Search with Filters ===")
    
    from tool_search_mcp_tools import search_mcp_tools
    
    # Search with category filtering
    results = await search_mcp_tools(
        query="execute system commands",
        category="execution",
        max_results=3,
        search_mode="hybrid"
    )
    
    if results['success']:
        print("Execution tools found:")
        
        for tool in results['results']:
            print(f"\n• {tool['name']}")
            print(f"  Confidence: {tool['confidence_level']} ({tool['similarity_score']:.2f})")
            print(f"  Complexity: {tool.get('complexity_score', 'N/A')}")
            
            # Show parameter information
            if tool.get('parameters'):
                print("  Key parameters:")
                for param in tool['parameters'][:2]:  # Show first 2 parameters
                    required = "required" if param.get('required') else "optional"
                    print(f"    - {param['name']} ({param['type']}, {required})")

# Example 5: Context-Aware Discovery
async def context_aware_discovery():
    """Demonstrate context-aware tool discovery."""
    print("\n=== Context-Aware Discovery ===")
    
    from tool_search_mcp_tools import search_mcp_tools
    from tool_recommendation_mcp_tools import recommend_tools_for_task
    
    # Scenario: New developer needs beginner-friendly tools
    context = "beginner user, safety first, clear error messages"
    
    recommendations = await recommend_tools_for_task(
        task_description="read and modify configuration files",
        context=context,
        complexity_preference="simple",
        max_recommendations=2
    )
    
    if recommendations['success']:
        print("Beginner-friendly tools for configuration management:")
        
        for rec in recommendations['recommendations']:
            tool = rec['tool']
            print(f"\n• {tool['name']}")
            print(f"  Why recommended: {rec['recommendation_reasons'][0]}")
            print(f"  Complexity alignment: {rec['complexity_alignment']:.2f}")
            
            if rec['alternative_tools']:
                print(f"  Alternatives: {', '.join(rec['alternative_tools'])}")

# Example 6: Search Result Analysis
async def analyze_search_results():
    """Demonstrate analysis of search results."""
    print("\n=== Search Result Analysis ===")
    
    from tool_search_mcp_tools import search_mcp_tools
    
    results = await search_mcp_tools(
        query="web scraping with error handling",
        max_results=5,
        include_examples=True
    )
    
    if results['success']:
        print("Analysis of web scraping tools:")
        print(f"Query processed in {results['execution_time']:.3f} seconds")
        print(f"Total results: {results['total_results']}")
        
        # Category breakdown
        if 'category_breakdown' in results:
            print("\nCategory distribution:")
            for category, count in results['category_breakdown'].items():
                print(f"  {category}: {count} tools")
        
        # Confidence analysis
        scores = [tool['similarity_score'] for tool in results['results']]
        avg_score = sum(scores) / len(scores) if scores else 0
        print(f"\nAverage confidence: {avg_score:.2f}")
        
        high_confidence = [t for t in results['results'] if t['similarity_score'] > 0.8]
        print(f"High confidence results: {len(high_confidence)}")

# Example 7: Learning from User Feedback
async def user_feedback_example():
    """Demonstrate how to provide feedback for system learning."""
    print("\n=== User Feedback Example ===")
    
    # This would be part of a real implementation
    print("Example feedback workflow:")
    print("1. User searches for 'process JSON data'")
    print("2. System recommends 'json_parser' tool")
    print("3. User successfully uses the tool")
    print("4. Positive feedback improves future recommendations")
    
    # Simulated feedback
    feedback = {
        "query": "process JSON data",
        "recommended_tool": "json_parser",
        "user_action": "accepted",
        "outcome": "successful",
        "user_rating": 5
    }
    
    print(f"\nFeedback recorded: {feedback}")
    print("This feedback helps improve future recommendations for similar queries.")

# Example 8: Error Handling and Recovery
async def error_handling_example():
    """Demonstrate proper error handling."""
    print("\n=== Error Handling and Recovery ===")
    
    from tool_search_mcp_tools import search_mcp_tools
    
    try:
        # Intentionally problematic query
        results = await search_mcp_tools(
            query="",  # Empty query should cause validation error
            max_results=10
        )
        
        if not results['success']:
            error = results.get('error', {})
            print(f"Error occurred: {error.get('message', 'Unknown error')}")
            print(f"Error type: {error.get('error_type', 'unknown')}")
            
            # Show recovery suggestions
            if error.get('recovery_options'):
                print("Recovery options:")
                for option in error['recovery_options']:
                    print(f"  • {option}")
    
    except Exception as e:
        print(f"Exception caught: {str(e)}")
        print("Falling back to basic tool discovery...")
        
        # Fallback mechanism
        fallback_tools = ["read", "write", "edit", "bash"]
        print(f"Available fallback tools: {', '.join(fallback_tools)}")

# Main execution function
async def run_all_examples():
    """Run all examples in sequence."""
    examples = [
        basic_tool_search,
        task_based_recommendations,
        tool_comparison,
        advanced_search,
        context_aware_discovery,
        analyze_search_results,
        user_feedback_example,
        error_handling_example
    ]
    
    print("MCP Tool Search System - Basic Usage Examples")
    print("=" * 50)
    
    for example in examples:
        try:
            await example()
            print("\n" + "-" * 50)
        except Exception as e:
            print(f"\nExample {example.__name__} failed: {str(e)}")
            print("-" * 50)

# Standalone usage functions
def demonstrate_search_patterns():
    """Show effective search query patterns."""
    print("=== Effective Search Query Patterns ===")
    
    patterns = {
        "Specific functionality": [
            "read configuration files with validation",
            "execute shell commands with timeout",
            "parse JSON with error handling"
        ],
        "Conceptual queries": [
            "tools for data transformation",
            "web scraping utilities",
            "file system operations"
        ],
        "Context-aware queries": [
            "beginner-friendly file tools",
            "performance-critical data processing",
            "secure authentication methods"
        ],
        "Domain-specific queries": [
            "REST API interaction tools",
            "database connection utilities",
            "logging and monitoring tools"
        ]
    }
    
    for category, queries in patterns.items():
        print(f"\n{category}:")
        for query in queries:
            print(f"  ✅ \"{query}\"")

def show_configuration_examples():
    """Show configuration options for different scenarios."""
    print("=== Configuration Examples ===")
    
    configs = {
        "Performance-focused": {
            "TOOL_SEARCH_CACHE_SIZE": 2000,
            "TOOL_SEARCH_CACHE_TTL": 1800,
            "TOOL_SEARCH_MAX_RESULTS": 10
        },
        "Accuracy-focused": {
            "TOOL_SEARCH_SIMILARITY_THRESHOLD": 0.7,
            "TOOL_SEARCH_SEMANTIC_WEIGHT": 0.8,
            "TOOL_SEARCH_INCLUDE_EXAMPLES": True
        },
        "Learning-enabled": {
            "TOOL_SEARCH_ENABLE_LEARNING": True,
            "TOOL_SEARCH_FEEDBACK_COLLECTION": True,
            "TOOL_SEARCH_LEARNING_RATE": 0.1
        }
    }
    
    for scenario, config in configs.items():
        print(f"\n{scenario}:")
        for key, value in config.items():
            print(f"  export {key}={value}")

if __name__ == "__main__":
    # Run basic demonstrations
    demonstrate_search_patterns()
    print("\n" + "=" * 50)
    show_configuration_examples()
    
    # Run async examples (would need proper MCP environment)
    print("\n" + "=" * 50)
    print("To run async examples, use: python -m asyncio basic_usage.run_all_examples")