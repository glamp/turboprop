#!/usr/bin/env python3
"""
context_analyzer.py: Context-Aware Analysis for Tool Recommendations

This module analyzes user context, project requirements, and environmental constraints
to enable personalized and context-aware tool recommendations.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class UserContext:
    """User-specific context for recommendations."""
    skill_level: str  # 'beginner', 'intermediate', 'advanced'
    tool_familiarity: Dict[str, float] = field(default_factory=dict)  # tool_id -> familiarity_score
    complexity_preference: str = "balanced"  # 'simple', 'balanced', 'powerful'
    error_tolerance: str = "medium"  # 'low', 'medium', 'high'
    learning_preference: str = "guided"  # 'guided', 'exploratory', 'efficient'

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class ProjectContext:
    """Project-specific context for recommendations."""
    project_type: str
    domain: str
    team_size: int
    existing_tools: List[str] = field(default_factory=list)
    technology_stack: List[str] = field(default_factory=list)
    performance_requirements: Dict[str, Any] = field(default_factory=dict)
    compliance_requirements: List[str] = field(default_factory=list)
    integration_patterns: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class EnvironmentalConstraints:
    """Environmental constraints affecting tool choice."""
    cpu_cores: int
    memory_gb: int
    gpu_available: bool
    operating_system: str
    security_requirements: List[str] = field(default_factory=list)
    resource_limits: Dict[str, Any] = field(default_factory=dict)
    compliance_frameworks: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class TaskContext:
    """Complete context for task-based recommendations."""
    user_context: Optional[UserContext] = None
    project_context: Optional[ProjectContext] = None
    environmental_constraints: Optional[EnvironmentalConstraints] = None
    time_constraints: Optional[Dict[str, Any]] = None
    quality_requirements: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class UserProfileAnalyzer:
    """Analyze user profiles and infer preferences."""
    
    def __init__(self):
        """Initialize the user profile analyzer."""
        self.skill_thresholds = {
            "beginner": {"max_complexity_score": 0.3, "min_success_rate": 0.8},
            "intermediate": {"max_complexity_score": 0.7, "min_success_rate": 0.6},
            "advanced": {"min_complexity_score": 0.7, "min_success_rate": 0.5}
        }
        
        self.complexity_weights = {
            "simple": 1.0,
            "moderate": 2.0,
            "complex": 3.0
        }
    
    def infer_skill_level(self, usage_history: List[Dict[str, Any]]) -> str:
        """Infer user skill level from usage history."""
        if not usage_history:
            return "beginner"
        
        # Calculate average complexity and success rate
        total_complexity = 0
        successful_uses = 0
        total_uses = len(usage_history)
        
        for use in usage_history:
            complexity = use.get("complexity", "simple")
            complexity_weight = self.complexity_weights.get(complexity, 1.0)
            total_complexity += complexity_weight
            
            if use.get("success", False):
                successful_uses += 1
        
        avg_complexity = total_complexity / total_uses
        success_rate = successful_uses / total_uses
        
        # Determine skill level based on thresholds
        if avg_complexity >= 2.5 and success_rate >= 0.7:
            return "advanced"
        elif avg_complexity >= 1.5 and success_rate >= 0.6:
            return "intermediate"
        else:
            return "beginner"
    
    def calculate_tool_familiarity(self, tool_id: str, tool_history: List[Dict[str, Any]]) -> float:
        """Calculate familiarity score for a specific tool."""
        tool_uses = [use for use in tool_history if use.get("tool") == tool_id]
        
        if not tool_uses:
            return 0.0
        
        # Calculate based on usage frequency and success rate
        usage_count = len(tool_uses)
        successful_uses = sum(1 for use in tool_uses if use.get("success", False))
        success_rate = successful_uses / usage_count if usage_count > 0 else 0
        
        # Base familiarity from usage frequency (normalized to 0-0.5)
        frequency_score = min(usage_count * 0.1, 0.5)
        
        # Success rate contribution (0-0.5)
        success_score = success_rate * 0.5
        
        return frequency_score + success_score
    
    def analyze_complexity_preference(self, usage_patterns: List[Dict[str, Any]]) -> str:
        """Analyze user's complexity preference from usage patterns."""
        if not usage_patterns:
            return "balanced"
        
        # Calculate weighted preference scores
        complexity_scores = {"simple": 0, "moderate": 0, "complex": 0}
        
        for pattern in usage_patterns:
            complexity = pattern.get("complexity", "moderate")
            preference_score = pattern.get("preference_score", 0.5)
            
            if complexity in complexity_scores:
                complexity_scores[complexity] += preference_score
        
        # Find the highest scoring complexity
        max_complexity = max(complexity_scores, key=complexity_scores.get)
        max_score = complexity_scores[max_complexity]
        
        # If the preference is not strong enough, default to balanced
        if max_score < 0.6:
            return "balanced"
        
        # Map to our preference categories
        if max_complexity == "simple":
            return "simple"
        elif max_complexity == "complex":
            return "powerful"
        else:
            return "balanced"
    
    def determine_error_tolerance(self, failure_patterns: List[Dict[str, Any]]) -> str:
        """Determine user's error tolerance from failure recovery patterns."""
        if not failure_patterns:
            return "medium"
        
        # Analyze recovery behavior after failures
        quick_recoveries = sum(1 for p in failure_patterns if p.get("recovery_time_minutes", 60) < 15)
        total_failures = len(failure_patterns)
        
        if quick_recoveries / total_failures > 0.8:
            return "high"  # User handles errors well
        elif quick_recoveries / total_failures < 0.3:
            return "low"   # User struggles with errors
        else:
            return "medium"
    
    def analyze_learning_preference(self, interaction_patterns: List[Dict[str, Any]]) -> str:
        """Analyze user's learning preference from interaction patterns."""
        if not interaction_patterns:
            return "guided"
        
        # Look for patterns in how users approach new tools
        documentation_usage = sum(1 for p in interaction_patterns if p.get("used_documentation", False))
        trial_error_usage = sum(1 for p in interaction_patterns if p.get("trial_and_error", False))
        tutorial_usage = sum(1 for p in interaction_patterns if p.get("used_tutorials", False))
        
        total_patterns = len(interaction_patterns)
        
        if tutorial_usage / total_patterns > 0.6:
            return "guided"
        elif trial_error_usage / total_patterns > 0.6:
            return "exploratory"
        elif documentation_usage / total_patterns > 0.6:
            return "efficient"
        else:
            return "guided"  # Default for unclear patterns


class ProjectAnalyzer:
    """Analyze project context and requirements."""
    
    def __init__(self):
        """Initialize the project analyzer."""
        self.technology_indicators = {
            "python": ["python", "pandas", "numpy", "scipy", "jupyter", "django", "flask"],
            "javascript": ["nodejs", "npm", "react", "vue", "angular", "webpack", "babel"],
            "java": ["java", "spring", "maven", "gradle", "junit"],
            "data_science": ["pandas", "numpy", "scikit-learn", "tensorflow", "pytorch", "jupyter"],
            "web_development": ["react", "vue", "angular", "express", "fastapi", "django", "flask"],
            "machine_learning": ["scikit-learn", "tensorflow", "pytorch", "keras", "xgboost"]
        }
        
        self.performance_priorities = {
            "fast": {"priority": "high", "focus": "speed"},
            "high": {"priority": "high", "focus": "throughput"},
            "low": {"priority": "high", "focus": "efficiency"},
            "medium": {"priority": "standard", "focus": "balanced"}
        }
    
    def identify_technology_stack(self, existing_tools: List[str]) -> List[str]:
        """Identify technology stack from existing tools."""
        identified_stack = []
        tool_set = set(tool.lower() for tool in existing_tools)
        
        for technology, indicators in self.technology_indicators.items():
            matches = sum(1 for indicator in indicators if indicator in tool_set)
            if matches >= 2:  # Require at least 2 matches
                identified_stack.append(technology)
        
        return identified_stack
    
    def analyze_performance_requirements(self, requirements: Dict[str, str]) -> Dict[str, Any]:
        """Analyze performance requirements and determine priorities."""
        analysis = {"priority": "standard", "optimization_focus": "balanced"}
        
        # Check individual requirements
        high_performance_indicators = 0
        focus_priorities = []
        
        for requirement, value in requirements.items():
            if value in ["fast", "high", "low"]:  # low memory usage is high performance
                high_performance_indicators += 1
                perf_info = self.performance_priorities.get(value, {"priority": "standard", "focus": "balanced"})
                analysis.update(perf_info)
                focus_priorities.append(perf_info.get("focus", "balanced"))
        
        # Set optimization focus based on requirements
        if focus_priorities:
            # Use the most specific focus, prioritizing speed and efficiency
            if "speed" in focus_priorities:
                analysis["optimization_focus"] = "speed"
            elif "efficiency" in focus_priorities:
                analysis["optimization_focus"] = "efficiency"
            elif "throughput" in focus_priorities:
                analysis["optimization_focus"] = "throughput"
            else:
                analysis["optimization_focus"] = focus_priorities[0]
        
        # Overall priority based on number of high-performance requirements
        if high_performance_indicators >= 2:
            analysis["priority"] = "high"
        elif high_performance_indicators == 1:
            analysis["priority"] = "standard"  # Keep standard for single high-perf requirement
        elif all(value == "medium" for value in requirements.values()):
            analysis["priority"] = "standard"  # Medium requirements are standard
        else:
            analysis["priority"] = "low"
        
        return analysis
    
    def identify_integration_patterns(self, existing_tools: List[str], project_type: str) -> List[str]:
        """Identify integration patterns from tools and project type."""
        patterns = []
        tool_set = set(tool.lower() for tool in existing_tools)
        
        # API integration patterns
        api_tools = ["fastapi", "express", "requests", "axios", "oauth", "swagger"]
        if any(tool in tool_set for tool in api_tools):
            patterns.append("api_integration")
        
        # Database integration patterns
        db_tools = ["postgresql", "mysql", "mongodb", "redis", "sqlite"]
        if any(tool in tool_set for tool in db_tools):
            patterns.append("database_integration")
        
        # Data processing patterns
        data_tools = ["pandas", "spark", "kafka", "airflow"]
        if any(tool in tool_set for tool in data_tools):
            patterns.append("data_pipeline")
        
        # Project type specific patterns
        if project_type == "web_service":
            patterns.append("rest_services")
        elif project_type == "data_analysis":
            patterns.append("analytical_workflows")
        elif project_type == "machine_learning":
            patterns.append("ml_pipelines")
        
        return patterns
    
    def assess_team_collaboration_needs(self, team_size: int) -> Dict[str, Any]:
        """Assess collaboration requirements based on team size."""
        if team_size <= 2:
            return {
                "collaboration_level": "minimal",
                "shared_tools_priority": "low",
                "standardization_needs": "low"
            }
        elif team_size <= 5:
            return {
                "collaboration_level": "moderate",
                "shared_tools_priority": "medium",
                "standardization_needs": "medium"
            }
        else:
            return {
                "collaboration_level": "high",
                "shared_tools_priority": "high",
                "standardization_needs": "high"
            }


class EnvironmentAnalyzer:
    """Analyze environmental constraints and capabilities."""
    
    def __init__(self):
        """Initialize the environment analyzer."""
        self.compute_thresholds = {
            "high": {"min_cores": 8, "min_memory": 16, "supports_ml": True},
            "medium": {"min_cores": 4, "min_memory": 8, "supports_ml": False},
            "limited": {"max_cores": 4, "max_memory": 8, "memory_constraints": True}
        }
        
        self.security_levels = {
            "high": ["SOC2", "HIPAA", "PCI-DSS", "FISMA"],
            "medium": ["ISO27001", "GDPR"],
            "basic": []
        }
    
    def assess_compute_capacity(self, system_info: Dict[str, Any]) -> Dict[str, Any]:
        """Assess computational capacity and capabilities."""
        cores = system_info.get("cpu_cores", 2)
        memory = system_info.get("memory_gb", 4)
        gpu = system_info.get("gpu_available", False)
        
        # Determine capacity level
        if cores >= 8 and memory >= 16:
            level = "high"
            supports_ml = True
            parallel_processing = True
        elif cores >= 4 and memory >= 8:
            level = "medium"
            supports_ml = gpu  # ML support depends on GPU for medium systems
            parallel_processing = True
        else:
            level = "limited"
            supports_ml = False
            parallel_processing = False
        
        return {
            "level": level,
            "cpu_cores": cores,
            "memory_gb": memory,
            "gpu_available": gpu,
            "supports_ml": supports_ml,
            "parallel_processing": parallel_processing,
            "memory_constraints": memory < 8
        }
    
    def analyze_security_requirements(self, security_info: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze security requirements and compliance needs."""
        frameworks = security_info.get("compliance_frameworks", [])
        requirements = security_info.get("security_requirements", [])
        network_restrictions = security_info.get("network_restrictions", [])
        
        # Determine security level
        security_level = "basic"
        for level, level_frameworks in self.security_levels.items():
            if any(framework in frameworks for framework in level_frameworks):
                security_level = level
                break
        
        # Analyze specific requirements
        encryption_required = any("encryption" in req for req in requirements)
        audit_required = any("audit" in req for req in requirements)
        
        # Identify restricted operations
        restricted_operations = []
        if "no_external_apis" in network_restrictions:
            restricted_operations.append("external_api_calls")
        if "firewall_restricted" in network_restrictions:
            restricted_operations.append("network_operations")
        
        return {
            "security_level": security_level,
            "compliance_frameworks": frameworks,
            "encryption_required": encryption_required,
            "audit_required": audit_required,
            "restricted_operations": restricted_operations,
            "network_constraints": network_restrictions
        }
    
    def evaluate_resource_constraints(self, constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate resource constraints and their impact."""
        # Parse memory constraints
        memory_limit = constraints.get("max_memory_usage", "unlimited")
        memory_limited = "GB" in memory_limit and int(memory_limit.split("GB")[0]) <= 4
        
        # Parse CPU constraints
        cpu_limit = constraints.get("max_cpu_usage", "100%")
        cpu_limited = "%" in cpu_limit and int(cpu_limit.split("%")[0]) <= 50
        
        # Evaluate overall constraint level
        constraint_level = "relaxed"
        if memory_limited and cpu_limited:
            constraint_level = "strict"
        elif memory_limited or cpu_limited:
            constraint_level = "moderate"
        
        return {
            "constraint_level": constraint_level,
            "memory_limited": memory_limited,
            "cpu_limited": cpu_limited,
            "network_limited": constraints.get("network_bandwidth") == "limited",
            "storage_limited": "GB" in constraints.get("disk_space", "unlimited")
        }


class ContextAnalyzer:
    """Analyze context for personalized recommendations."""
    
    def __init__(self):
        """Initialize the context analyzer."""
        self.user_analyzer = UserProfileAnalyzer()
        self.project_analyzer = ProjectAnalyzer()
        self.environment_analyzer = EnvironmentAnalyzer()
        
        logger.info("Context analyzer initialized")
    
    def analyze_user_context(self, context_data: Dict[str, Any]) -> UserContext:
        """Analyze user skill level and preferences."""
        logger.debug("Analyzing user context")
        
        usage_history = context_data.get("tool_usage_history", [])
        task_history = context_data.get("task_completion_history", [])
        preferences = context_data.get("preferences", {})
        
        # Infer skill level from usage history
        skill_level = self.user_analyzer.infer_skill_level(usage_history)
        
        # Calculate tool familiarity scores
        tool_familiarity = {}
        for use in usage_history:
            tool_id = use.get("tool", "")
            if tool_id:
                familiarity = self.user_analyzer.calculate_tool_familiarity(tool_id, usage_history)
                tool_familiarity[tool_id] = familiarity
        
        # Analyze complexity preference
        usage_patterns = [
            {"complexity": use.get("complexity", "moderate"), "preference_score": 0.8 if use.get("success") else 0.3}
            for use in usage_history
        ]
        complexity_preference = self.user_analyzer.analyze_complexity_preference(usage_patterns)
        
        # Extract other preferences
        error_tolerance = preferences.get("error_tolerance", "medium")
        learning_preference = preferences.get("learning_preference", "guided")
        
        user_context = UserContext(
            skill_level=skill_level,
            tool_familiarity=tool_familiarity,
            complexity_preference=complexity_preference,
            error_tolerance=error_tolerance,
            learning_preference=learning_preference
        )
        
        logger.debug(f"User context analyzed: skill={skill_level}, complexity_pref={complexity_preference}")
        return user_context
    
    def analyze_project_context(self, project_info: Dict[str, Any]) -> ProjectContext:
        """Analyze project context for tailored recommendations."""
        logger.debug("Analyzing project context")
        
        project_type = project_info.get("project_type", "general")
        domain = project_info.get("domain", "general")
        team_size = project_info.get("team_size", 1)
        existing_tools = project_info.get("existing_tools", [])
        
        # Identify technology stack
        technology_stack = self.project_analyzer.identify_technology_stack(existing_tools)
        
        # Analyze performance requirements
        performance_reqs = project_info.get("performance_requirements", {})
        
        # Extract compliance and integration requirements
        compliance_requirements = project_info.get("compliance_requirements", [])
        integration_patterns = self.project_analyzer.identify_integration_patterns(existing_tools, project_type)
        
        project_context = ProjectContext(
            project_type=project_type,
            domain=domain,
            team_size=team_size,
            existing_tools=existing_tools,
            technology_stack=technology_stack,
            performance_requirements=performance_reqs,
            compliance_requirements=compliance_requirements,
            integration_patterns=integration_patterns
        )
        
        logger.debug(f"Project context analyzed: type={project_type}, stack={technology_stack}")
        return project_context
    
    def analyze_environmental_constraints(self, env_info: Dict[str, Any]) -> EnvironmentalConstraints:
        """Analyze environmental constraints affecting tool choice."""
        logger.debug("Analyzing environmental constraints")
        
        system_capabilities = env_info.get("system_capabilities", {})
        security_requirements = env_info.get("security_requirements", [])
        resource_constraints = env_info.get("resource_constraints", {})
        compliance_frameworks = env_info.get("compliance_frameworks", [])
        
        # Extract system information
        cpu_cores = system_capabilities.get("cpu_cores", 2)
        memory_gb = system_capabilities.get("memory_gb", 4)
        gpu_available = system_capabilities.get("gpu_available", False)
        operating_system = system_capabilities.get("os", "unknown")
        
        env_constraints = EnvironmentalConstraints(
            cpu_cores=cpu_cores,
            memory_gb=memory_gb,
            gpu_available=gpu_available,
            operating_system=operating_system,
            security_requirements=security_requirements,
            resource_limits=resource_constraints,
            compliance_frameworks=compliance_frameworks
        )
        
        logger.debug(f"Environmental constraints analyzed: cores={cpu_cores}, memory={memory_gb}GB")
        return env_constraints
    
    def create_task_context(self,
                          user_context: Optional[UserContext] = None,
                          project_context: Optional[ProjectContext] = None,
                          environmental_constraints: Optional[EnvironmentalConstraints] = None,
                          time_constraints: Optional[Dict[str, Any]] = None,
                          quality_requirements: Optional[Dict[str, Any]] = None) -> TaskContext:
        """Create a complete task context from components."""
        logger.debug("Creating integrated task context")
        
        task_context = TaskContext(
            user_context=user_context,
            project_context=project_context,
            environmental_constraints=environmental_constraints,
            time_constraints=time_constraints,
            quality_requirements=quality_requirements
        )
        
        logger.debug("Task context created successfully")
        return task_context