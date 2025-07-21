#!/usr/bin/env python3
"""
Advanced Workflow Examples for MCP Tool Search System

This file demonstrates sophisticated usage patterns, including tool chaining,
workflow optimization, and integration with development processes.
"""

import asyncio
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

# Workflow state management
class WorkflowState(Enum):
    PLANNING = "planning"
    ANALYZING = "analyzing"
    EXECUTING = "executing"
    OPTIMIZING = "optimizing"
    COMPLETE = "complete"

@dataclass
class ToolStep:
    """Represents a step in a tool workflow."""
    tool_id: str
    parameters: Dict[str, Any]
    depends_on: List[str] = None
    optional: bool = False
    retry_count: int = 0
    max_retries: int = 3

@dataclass 
class WorkflowContext:
    """Context for workflow execution."""
    task_description: str
    environment: str
    constraints: List[str]
    performance_requirements: Dict[str, Any]
    user_preferences: Dict[str, Any]

# Example 1: Intelligent Workflow Planning
class IntelligentWorkflowPlanner:
    """Plan optimal tool sequences for complex tasks."""
    
    def __init__(self):
        self.workflow_cache = {}
        
    async def plan_workflow(self, task: str, context: WorkflowContext) -> List[ToolStep]:
        """Plan an optimal sequence of tools for a complex task."""
        print(f"=== Planning Workflow for: {task} ===")
        
        # Step 1: Analyze task requirements
        from tool_recommendation_mcp_tools import analyze_task_requirements
        
        analysis = await analyze_task_requirements(
            task_description=task,
            detail_level="comprehensive"
        )
        
        if not analysis['success']:
            raise ValueError("Failed to analyze task requirements")
        
        requirements = analysis['analysis']
        print(f"Task complexity: {requirements['complexity_level']}")
        print(f"Required capabilities: {', '.join(requirements['required_capabilities'])}")
        
        # Step 2: Get tool recommendations for each capability
        workflow_steps = []
        
        for capability in requirements['required_capabilities']:
            tools = await self._find_tools_for_capability(capability, context)
            if tools:
                step = ToolStep(
                    tool_id=tools[0]['tool_id'],
                    parameters=self._generate_parameters(tools[0], context)
                )
                workflow_steps.append(step)
                print(f"Selected {step.tool_id} for {capability}")
        
        # Step 3: Optimize step sequence
        optimized_steps = await self._optimize_sequence(workflow_steps, task)
        
        return optimized_steps
    
    async def _find_tools_for_capability(self, capability: str, context: WorkflowContext) -> List[Dict]:
        """Find tools that provide a specific capability."""
        from tool_search_mcp_tools import search_mcp_tools
        
        context_str = f"{context.environment}, {', '.join(context.constraints)}"
        
        results = await search_mcp_tools(
            query=f"tools with {capability} capability",
            context=context_str,
            max_results=3,
            search_mode="semantic"
        )
        
        return results.get('results', []) if results['success'] else []
    
    def _generate_parameters(self, tool: Dict, context: WorkflowContext) -> Dict[str, Any]:
        """Generate appropriate parameters for a tool based on context."""
        params = {}
        
        # Extract parameter requirements from tool info
        for param_info in tool.get('parameters', []):
            param_name = param_info['name']
            
            # Apply context-specific parameter values
            if param_name == 'timeout' and 'performance-critical' in context.constraints:
                params[param_name] = 60  # Longer timeout for critical operations
            elif param_name == 'max_results' and 'comprehensive' in context.constraints:
                params[param_name] = 20  # More results for comprehensive analysis
            elif param_info.get('required') and param_info.get('default_value'):
                params[param_name] = param_info['default_value']
        
        return params
    
    async def _optimize_sequence(self, steps: List[ToolStep], task: str) -> List[ToolStep]:
        """Optimize the sequence of tool steps."""
        from tool_recommendation_mcp_tools import recommend_tool_sequence
        
        tool_ids = [step.tool_id for step in steps]
        
        optimization = await recommend_tool_sequence(
            task_description=task,
            current_sequence=tool_ids,
            optimization_goals=["efficiency", "reliability"]
        )
        
        if optimization['success'] and optimization.get('improvements'):
            print(f"Workflow optimized: {optimization['improvements']}")
            # Apply optimizations to steps
            return self._apply_optimizations(steps, optimization)
        
        return steps
    
    def _apply_optimizations(self, steps: List[ToolStep], optimization: Dict) -> List[ToolStep]:
        """Apply optimization recommendations to workflow steps."""
        # This would implement the actual optimization logic
        # For now, return the original steps
        return steps

# Example 2: Adaptive Tool Selection
class AdaptiveToolSelector:
    """Select tools that adapt based on context and feedback."""
    
    def __init__(self):
        self.usage_history = []
        self.performance_metrics = {}
    
    async def select_adaptive_tool(
        self, 
        task: str, 
        context: Dict[str, Any],
        previous_failures: List[str] = None
    ) -> Dict[str, Any]:
        """Select tool with adaptive learning from past usage."""
        print(f"=== Adaptive Tool Selection for: {task} ===")
        
        from tool_recommendation_mcp_tools import recommend_tools_for_task
        
        # Get initial recommendations
        recommendations = await recommend_tools_for_task(
            task_description=task,
            context=str(context),
            max_recommendations=5,
            explain_reasoning=True
        )
        
        if not recommendations['success']:
            raise ValueError("Failed to get tool recommendations")
        
        # Filter out previously failed tools
        filtered_recs = []
        for rec in recommendations['recommendations']:
            tool_id = rec['tool']['tool_id']
            
            if previous_failures and tool_id in previous_failures:
                print(f"Skipping {tool_id} due to previous failure")
                continue
                
            # Apply historical performance data
            performance = self.performance_metrics.get(tool_id, {})
            adjusted_score = self._adjust_score_by_performance(
                rec['recommendation_score'], 
                performance
            )
            rec['adjusted_score'] = adjusted_score
            filtered_recs.append(rec)
        
        if not filtered_recs:
            raise ValueError("No suitable tools found after filtering")
        
        # Sort by adjusted score
        best_tool = max(filtered_recs, key=lambda x: x['adjusted_score'])
        
        print(f"Selected: {best_tool['tool']['name']}")
        print(f"Original score: {best_tool['recommendation_score']:.2f}")
        print(f"Adjusted score: {best_tool['adjusted_score']:.2f}")
        
        return best_tool
    
    def _adjust_score_by_performance(self, base_score: float, performance: Dict) -> float:
        """Adjust recommendation score based on historical performance."""
        if not performance:
            return base_score
        
        success_rate = performance.get('success_rate', 1.0)
        avg_execution_time = performance.get('avg_execution_time', 1.0)
        
        # Boost score for reliable, fast tools
        reliability_factor = success_rate
        performance_factor = min(1.0, 2.0 / avg_execution_time)  # Favor faster tools
        
        return base_score * reliability_factor * performance_factor
    
    def record_usage(self, tool_id: str, execution_time: float, success: bool):
        """Record tool usage for adaptive learning."""
        if tool_id not in self.performance_metrics:
            self.performance_metrics[tool_id] = {
                'success_count': 0,
                'total_count': 0,
                'total_time': 0.0
            }
        
        metrics = self.performance_metrics[tool_id]
        metrics['total_count'] += 1
        metrics['total_time'] += execution_time
        
        if success:
            metrics['success_count'] += 1
        
        # Calculate derived metrics
        metrics['success_rate'] = metrics['success_count'] / metrics['total_count']
        metrics['avg_execution_time'] = metrics['total_time'] / metrics['total_count']

# Example 3: Multi-Stage Development Workflow
async def demonstrate_development_workflow():
    """Demonstrate a complete development workflow with tool search."""
    print("=== Multi-Stage Development Workflow ===")
    
    # Stage 1: Project Analysis
    print("\nStage 1: Analyzing Project Structure")
    from tool_search_mcp_tools import search_mcp_tools
    
    analysis_tools = await search_mcp_tools(
        query="analyze project structure and dependencies",
        category="analysis",
        max_results=3
    )
    
    if analysis_tools['success']:
        analyzer = analysis_tools['results'][0]
        print(f"Using {analyzer['name']} for project analysis")
        # Simulate using the analyzer
        project_info = {
            "language": "python",
            "framework": "fastapi",
            "complexity": "medium",
            "dependencies": ["fastapi", "pydantic", "uvicorn"]
        }
        print(f"Project analysis complete: {project_info}")
    
    # Stage 2: Development Tool Selection
    print("\nStage 2: Selecting Development Tools")
    from tool_recommendation_mcp_tools import recommend_tools_for_task
    
    dev_recommendations = await recommend_tools_for_task(
        task_description="develop and test FastAPI application",
        context="Python, REST API, automated testing",
        complexity_preference="balanced"
    )
    
    if dev_recommendations['success']:
        for i, rec in enumerate(dev_recommendations['recommendations'][:3], 1):
            tool = rec['tool']
            print(f"{i}. {tool['name']}: {rec['when_to_use']}")
    
    # Stage 3: Testing Strategy
    print("\nStage 3: Determining Testing Strategy")
    
    test_tools = await search_mcp_tools(
        query="automated testing tools for web APIs",
        context="Python, FastAPI, integration tests"
    )
    
    if test_tools['success']:
        for tool in test_tools['results'][:2]:
            print(f"• {tool['name']}: {tool['description']}")
    
    # Stage 4: Deployment Planning
    print("\nStage 4: Planning Deployment")
    
    deployment_recommendations = await recommend_tools_for_task(
        task_description="deploy FastAPI application with monitoring",
        context="cloud deployment, automated CI/CD, monitoring"
    )
    
    if deployment_recommendations['success']:
        best_deploy = deployment_recommendations['recommendations'][0]
        print(f"Recommended deployment approach: {best_deploy['tool']['name']}")
        print(f"Reasoning: {best_deploy['recommendation_reasons'][0]}")

# Example 4: Context-Sensitive Tool Chains
class ContextSensitiveChain:
    """Build tool chains that adapt to changing context."""
    
    def __init__(self):
        self.context_stack = []
        self.active_tools = []
    
    async def build_chain(self, primary_task: str, context_updates: List[Dict]) -> List[str]:
        """Build a tool chain that adapts to context changes."""
        print(f"=== Building Context-Sensitive Chain for: {primary_task} ===")
        
        chain = []
        current_context = {}
        
        # Process each context update
        for i, context_update in enumerate(context_updates):
            print(f"\nContext Update {i+1}: {context_update}")
            current_context.update(context_update)
            
            # Get tools for current context
            tools = await self._get_contextual_tools(primary_task, current_context)
            
            if tools:
                selected_tool = tools[0]['tool_id']
                chain.append(selected_tool)
                print(f"Added tool: {selected_tool}")
                
                # Explain context sensitivity
                print(f"Context factors: {list(current_context.keys())}")
        
        return chain
    
    async def _get_contextual_tools(self, task: str, context: Dict) -> List[Dict]:
        """Get tools appropriate for current context."""
        from tool_recommendation_mcp_tools import recommend_tools_for_task
        
        context_str = ", ".join([f"{k}: {v}" for k, v in context.items()])
        
        recommendations = await recommend_tools_for_task(
            task_description=task,
            context=context_str,
            max_recommendations=3
        )
        
        return recommendations.get('recommendations', []) if recommendations['success'] else []

# Example 5: Performance-Optimized Tool Selection
async def demonstrate_performance_optimization():
    """Demonstrate performance-optimized tool selection."""
    print("=== Performance-Optimized Tool Selection ===")
    
    from tool_search_mcp_tools import search_mcp_tools
    from tool_comparison_mcp_tools import compare_mcp_tools
    
    # Scenario: Need fast file processing for large datasets
    performance_query = "file processing tools optimized for large datasets"
    
    tools = await search_mcp_tools(
        query=performance_query,
        context="performance critical, large files, streaming preferred",
        max_results=5
    )
    
    if tools['success']:
        print("Performance-optimized tools found:")
        
        # Get top candidates for comparison
        top_tools = [tool['tool_id'] for tool in tools['results'][:3]]
        
        comparison = await compare_mcp_tools(
            tool_ids=top_tools,
            comparison_criteria=['performance', 'reliability', 'scalability'],
            comparison_context="large dataset processing"
        )
        
        if comparison['success']:
            best_tool = comparison['comparison']['summary']['recommended_choice']
            reasoning = comparison['comparison']['summary']['reasoning']
            
            print(f"\nRecommended for performance: {best_tool}")
            print(f"Reasoning: {reasoning}")
            
            # Show performance characteristics
            for tool in comparison['comparison']['tools']:
                if tool['tool_id'] == best_tool:
                    perf_score = tool['dimension_scores'].get('performance', 0)
                    print(f"Performance score: {perf_score:.2f}")
                    print(f"Strengths: {', '.join(tool['strengths'])}")

# Example 6: Error Recovery and Fallback Chains
class ErrorRecoveryChain:
    """Implement intelligent error recovery with tool fallbacks."""
    
    def __init__(self):
        self.fallback_history = {}
    
    async def execute_with_fallbacks(
        self, 
        primary_tool: str,
        task_description: str,
        max_fallbacks: int = 3
    ) -> Dict[str, Any]:
        """Execute tool with intelligent fallback chain."""
        print(f"=== Executing with Fallbacks: {primary_tool} ===")
        
        execution_log = []
        current_tool = primary_tool
        
        for attempt in range(max_fallbacks + 1):
            try:
                print(f"\nAttempt {attempt + 1}: Using {current_tool}")
                
                # Simulate tool execution (would be actual MCP call)
                result = await self._simulate_tool_execution(current_tool)
                
                if result['success']:
                    execution_log.append({
                        'tool': current_tool,
                        'attempt': attempt + 1,
                        'result': 'success'
                    })
                    print(f"✅ Success with {current_tool}")
                    return {'success': True, 'tool_used': current_tool, 'log': execution_log}
                else:
                    raise Exception(result['error'])
                    
            except Exception as e:
                print(f"❌ Failed with {current_tool}: {str(e)}")
                execution_log.append({
                    'tool': current_tool,
                    'attempt': attempt + 1,
                    'result': 'failed',
                    'error': str(e)
                })
                
                if attempt < max_fallbacks:
                    # Get fallback tool
                    fallback = await self._get_fallback_tool(
                        current_tool, 
                        task_description, 
                        str(e)
                    )
                    
                    if fallback:
                        current_tool = fallback
                        print(f"🔄 Falling back to: {fallback}")
                    else:
                        print("No more fallback options")
                        break
        
        return {'success': False, 'log': execution_log}
    
    async def _simulate_tool_execution(self, tool_id: str) -> Dict[str, Any]:
        """Simulate tool execution with random success/failure."""
        import random
        
        # Simulate different failure rates for different tools
        failure_rates = {
            'experimental_tool': 0.7,
            'stable_tool': 0.1,
            'fallback_tool': 0.3
        }
        
        failure_rate = failure_rates.get(tool_id, 0.2)
        
        if random.random() < failure_rate:
            return {'success': False, 'error': f'{tool_id} execution failed'}
        else:
            return {'success': True, 'result': f'{tool_id} completed successfully'}
    
    async def _get_fallback_tool(
        self, 
        failed_tool: str, 
        task_description: str, 
        error: str
    ) -> Optional[str]:
        """Get an appropriate fallback tool."""
        from tool_search_mcp_tools import find_tool_alternatives
        
        alternatives = await find_tool_alternatives(
            reference_tool=failed_tool,
            context_filter="stable, reliable",
            max_alternatives=3
        )
        
        if alternatives['success'] and alternatives['alternatives']:
            # Return the most reliable alternative
            return alternatives['alternatives'][0]['tool_id']
        
        return None

# Main demonstration runner
async def run_advanced_examples():
    """Run all advanced workflow examples."""
    print("MCP Tool Search System - Advanced Workflow Examples")
    print("=" * 60)
    
    # Example 1: Workflow Planning
    planner = IntelligentWorkflowPlanner()
    context = WorkflowContext(
        task_description="Deploy microservice with monitoring",
        environment="cloud",
        constraints=["high-availability", "automated"],
        performance_requirements={"response_time": "<100ms"},
        user_preferences={"complexity": "balanced"}
    )
    
    try:
        workflow = await planner.plan_workflow(
            "Deploy microservice with monitoring", 
            context
        )
        print(f"Planned workflow with {len(workflow)} steps")
    except Exception as e:
        print(f"Workflow planning failed: {e}")
    
    print("\n" + "-" * 60)
    
    # Example 2: Adaptive Selection
    selector = AdaptiveToolSelector()
    
    try:
        adaptive_tool = await selector.select_adaptive_tool(
            "process large CSV files",
            {"size": "large", "format": "CSV", "performance": "critical"}
        )
        print(f"Adaptively selected: {adaptive_tool['tool']['name']}")
    except Exception as e:
        print(f"Adaptive selection failed: {e}")
    
    print("\n" + "-" * 60)
    
    # Example 3: Development Workflow
    try:
        await demonstrate_development_workflow()
    except Exception as e:
        print(f"Development workflow demo failed: {e}")
    
    print("\n" + "-" * 60)
    
    # Example 4: Context-Sensitive Chains
    chain_builder = ContextSensitiveChain()
    
    context_updates = [
        {"environment": "development"},
        {"data_size": "large"},
        {"user_type": "expert"},
        {"performance_requirement": "high"}
    ]
    
    try:
        chain = await chain_builder.build_chain(
            "data processing pipeline",
            context_updates
        )
        print(f"Built context-sensitive chain: {' -> '.join(chain)}")
    except Exception as e:
        print(f"Chain building failed: {e}")
    
    print("\n" + "-" * 60)
    
    # Example 5: Performance Optimization
    try:
        await demonstrate_performance_optimization()
    except Exception as e:
        print(f"Performance optimization demo failed: {e}")
    
    print("\n" + "-" * 60)
    
    # Example 6: Error Recovery
    recovery_chain = ErrorRecoveryChain()
    
    try:
        result = await recovery_chain.execute_with_fallbacks(
            "experimental_tool",
            "process data with experimental algorithm"
        )
        print(f"Recovery chain result: {'Success' if result['success'] else 'Failed'}")
    except Exception as e:
        print(f"Recovery chain demo failed: {e}")

if __name__ == "__main__":
    # Run demonstrations
    print("To run async examples, use: python -m asyncio advanced_workflows.run_advanced_examples")
    
    # Show configuration for advanced usage
    print("\n=== Advanced Configuration Examples ===")
    
    advanced_configs = {
        "High-Performance Setup": {
            "TOOL_SEARCH_CACHE_SIZE": 5000,
            "TOOL_SEARCH_CONCURRENT_SEARCHES": 10,
            "TOOL_SEARCH_TIMEOUT": 60
        },
        "Learning-Optimized Setup": {
            "TOOL_SEARCH_ENABLE_LEARNING": True,
            "TOOL_SEARCH_FEEDBACK_COLLECTION": True,
            "TOOL_SEARCH_ADAPTATION_RATE": 0.2
        },
        "Enterprise Setup": {
            "TOOL_SEARCH_AUDIT_LOGGING": True,
            "TOOL_SEARCH_PERFORMANCE_MONITORING": True,
            "TOOL_SEARCH_USAGE_ANALYTICS": True
        }
    }
    
    for scenario, config in advanced_configs.items():
        print(f"\n{scenario}:")
        for key, value in config.items():
            print(f"  export {key}={value}")