#!/usr/bin/env python3
"""
Tests for context_analyzer.py - Context-Aware Analysis for Tool Recommendations

This module tests the context analysis functionality that personalizes recommendations
based on user context, project requirements, and environmental constraints.
"""

import pytest
from dataclasses import asdict, field
from typing import Any, Dict, List, Optional

from context_analyzer import (
    ContextAnalyzer,
    UserContext,
    ProjectContext,
    EnvironmentalConstraints,
    TaskContext,
    UserProfileAnalyzer,
    ProjectAnalyzer,
    EnvironmentAnalyzer,
)


class TestContextAnalyzer:
    """Test cases for ContextAnalyzer class."""

    @pytest.fixture
    def context_analyzer(self):
        """Create a ContextAnalyzer instance for testing."""
        return ContextAnalyzer()

    @pytest.fixture
    def basic_user_data(self):
        """Create basic user context data for testing."""
        return {
            "tool_usage_history": [
                {"tool": "csv_reader", "success": True, "complexity": "simple"},
                {"tool": "data_processor", "success": True, "complexity": "moderate"},
                {"tool": "ml_analyzer", "success": False, "complexity": "complex"}
            ],
            "task_completion_history": [
                {"task_type": "file_operation", "success_rate": 0.95},
                {"task_type": "data_processing", "success_rate": 0.80},
                {"task_type": "machine_learning", "success_rate": 0.45}
            ],
            "preferences": {
                "complexity_tolerance": "moderate",
                "learning_preference": "guided",
                "error_tolerance": "medium"
            }
        }

    @pytest.fixture
    def project_info(self):
        """Create project context information for testing."""
        return {
            "project_type": "data_analysis",
            "domain": "finance",
            "team_size": 3,
            "existing_tools": ["pandas", "numpy", "jupyter"],
            "performance_requirements": {
                "response_time": "fast",
                "throughput": "medium",
                "memory_usage": "low"
            },
            "compliance_requirements": ["data_privacy", "audit_trail"],
            "integration_constraints": ["python_only", "no_external_apis"]
        }

    @pytest.fixture
    def environment_info(self):
        """Create environmental constraint information for testing."""
        return {
            "system_capabilities": {
                "cpu_cores": 8,
                "memory_gb": 16,
                "gpu_available": False,
                "os": "linux"
            },
            "security_requirements": ["encryption_at_rest", "secure_communication"],
            "resource_constraints": {
                "max_memory_usage": "8GB",
                "max_cpu_usage": "70%",
                "network_bandwidth": "limited"
            },
            "compliance_frameworks": ["SOC2", "GDPR"]
        }

    def test_analyze_user_context_skill_inference(self, context_analyzer, basic_user_data):
        """Test user context analysis and skill level inference."""
        user_context = context_analyzer.analyze_user_context(basic_user_data)
        
        assert isinstance(user_context, UserContext)
        assert user_context.skill_level in ["beginner", "intermediate", "advanced"]
        assert len(user_context.tool_familiarity) > 0
        assert user_context.complexity_preference in ["simple", "balanced", "powerful"]
        assert user_context.error_tolerance in ["low", "medium", "high"]
        assert user_context.learning_preference in ["guided", "exploratory", "efficient"]

    def test_analyze_user_context_tool_familiarity(self, context_analyzer, basic_user_data):
        """Test tool familiarity scoring from usage history."""
        user_context = context_analyzer.analyze_user_context(basic_user_data)
        
        # Should have familiarity scores for tools used
        assert "csv_reader" in user_context.tool_familiarity
        assert "data_processor" in user_context.tool_familiarity
        assert "ml_analyzer" in user_context.tool_familiarity
        
        # Successful usage should result in higher familiarity
        assert user_context.tool_familiarity["csv_reader"] > user_context.tool_familiarity["ml_analyzer"]

    def test_analyze_project_context_requirements(self, context_analyzer, project_info):
        """Test project context analysis and requirement extraction."""
        project_context = context_analyzer.analyze_project_context(project_info)
        
        assert isinstance(project_context, ProjectContext)
        assert project_context.project_type == "data_analysis"
        assert project_context.domain == "finance"
        assert project_context.team_size == 3
        assert len(project_context.existing_tools) > 0
        assert len(project_context.performance_requirements) > 0
        assert len(project_context.compliance_requirements) > 0

    def test_analyze_project_context_tool_ecosystem(self, context_analyzer, project_info):
        """Test analysis of existing tool ecosystem."""
        project_context = context_analyzer.analyze_project_context(project_info)
        
        # Should identify Python ecosystem
        assert "python" in project_context.technology_stack or any("python" in tool.lower() for tool in project_context.existing_tools)
        assert len(project_context.integration_patterns) >= 0

    def test_analyze_environmental_constraints_system(self, context_analyzer, environment_info):
        """Test environmental constraint analysis."""
        env_constraints = context_analyzer.analyze_environmental_constraints(environment_info)
        
        assert isinstance(env_constraints, EnvironmentalConstraints)
        assert env_constraints.cpu_cores == 8
        assert env_constraints.memory_gb == 16
        assert env_constraints.gpu_available == False
        assert env_constraints.operating_system == "linux"
        assert len(env_constraints.security_requirements) > 0
        assert len(env_constraints.resource_limits) > 0

    def test_analyze_environmental_constraints_compliance(self, context_analyzer, environment_info):
        """Test compliance framework identification."""
        env_constraints = context_analyzer.analyze_environmental_constraints(environment_info)
        
        assert len(env_constraints.compliance_frameworks) > 0
        assert "SOC2" in env_constraints.compliance_frameworks
        assert "GDPR" in env_constraints.compliance_frameworks

    def test_context_integration_complete(self, context_analyzer, basic_user_data, project_info, environment_info):
        """Test integration of all context components."""
        user_context = context_analyzer.analyze_user_context(basic_user_data)
        project_context = context_analyzer.analyze_project_context(project_info)
        env_constraints = context_analyzer.analyze_environmental_constraints(environment_info)
        
        # Create complete task context
        task_context = TaskContext(
            user_context=user_context,
            project_context=project_context,
            environmental_constraints=env_constraints,
            time_constraints={"deadline": "1_week"},
            quality_requirements={"accuracy": "high"}
        )
        
        assert task_context.user_context is not None
        assert task_context.project_context is not None
        assert task_context.environmental_constraints is not None


class TestUserProfileAnalyzer:
    """Test cases for UserProfileAnalyzer class."""

    @pytest.fixture
    def profile_analyzer(self):
        """Create a UserProfileAnalyzer instance for testing."""
        return UserProfileAnalyzer()

    def test_infer_skill_level_beginner(self, profile_analyzer):
        """Test skill level inference for beginner users."""
        usage_history = [
            {"tool": "basic_tool", "success": True, "complexity": "simple"},
            {"tool": "another_simple_tool", "success": True, "complexity": "simple"}
        ]
        
        skill_level = profile_analyzer.infer_skill_level(usage_history)
        
        assert skill_level == "beginner"

    def test_infer_skill_level_intermediate(self, profile_analyzer):
        """Test skill level inference for intermediate users."""
        usage_history = [
            {"tool": "simple_tool", "success": True, "complexity": "simple"},
            {"tool": "moderate_tool", "success": True, "complexity": "moderate"},
            {"tool": "another_moderate", "success": True, "complexity": "moderate"}
        ]
        
        skill_level = profile_analyzer.infer_skill_level(usage_history)
        
        assert skill_level == "intermediate"

    def test_infer_skill_level_advanced(self, profile_analyzer):
        """Test skill level inference for advanced users.""" 
        usage_history = [
            {"tool": "complex_tool", "success": True, "complexity": "complex"},
            {"tool": "advanced_tool", "success": True, "complexity": "complex"},
            {"tool": "expert_tool", "success": True, "complexity": "complex"}
        ]
        
        skill_level = profile_analyzer.infer_skill_level(usage_history)
        
        assert skill_level == "advanced"

    def test_calculate_tool_familiarity_high_success(self, profile_analyzer):
        """Test tool familiarity calculation for high success rate."""
        tool_history = [
            {"tool": "csv_reader", "success": True},
            {"tool": "csv_reader", "success": True},
            {"tool": "csv_reader", "success": True}
        ]
        
        familiarity = profile_analyzer.calculate_tool_familiarity("csv_reader", tool_history)
        
        assert 0.8 <= familiarity <= 1.0

    def test_calculate_tool_familiarity_mixed_success(self, profile_analyzer):
        """Test tool familiarity calculation for mixed success rate."""
        tool_history = [
            {"tool": "data_processor", "success": True},
            {"tool": "data_processor", "success": False},
            {"tool": "data_processor", "success": True}
        ]
        
        familiarity = profile_analyzer.calculate_tool_familiarity("data_processor", tool_history)
        
        assert 0.4 <= familiarity <= 0.8

    def test_analyze_complexity_preference_conservative(self, profile_analyzer):
        """Test complexity preference analysis for conservative users."""
        usage_patterns = [
            {"complexity": "simple", "preference_score": 0.9},
            {"complexity": "moderate", "preference_score": 0.3},
            {"complexity": "complex", "preference_score": 0.1}
        ]
        
        preference = profile_analyzer.analyze_complexity_preference(usage_patterns)
        
        assert preference == "simple"

    def test_analyze_complexity_preference_balanced(self, profile_analyzer):
        """Test complexity preference analysis for balanced users."""
        usage_patterns = [
            {"complexity": "simple", "preference_score": 0.5},
            {"complexity": "moderate", "preference_score": 0.8},
            {"complexity": "complex", "preference_score": 0.4}
        ]
        
        preference = profile_analyzer.analyze_complexity_preference(usage_patterns)
        
        assert preference == "balanced"


class TestProjectAnalyzer:
    """Test cases for ProjectAnalyzer class."""

    @pytest.fixture
    def project_analyzer(self):
        """Create a ProjectAnalyzer instance for testing."""
        return ProjectAnalyzer()

    def test_identify_technology_stack_python(self, project_analyzer):
        """Test technology stack identification for Python projects."""
        existing_tools = ["pandas", "numpy", "scikit-learn", "jupyter"]
        
        tech_stack = project_analyzer.identify_technology_stack(existing_tools)
        
        assert "python" in tech_stack
        assert "data_science" in tech_stack

    def test_identify_technology_stack_javascript(self, project_analyzer):
        """Test technology stack identification for JavaScript projects."""
        existing_tools = ["nodejs", "express", "react", "webpack"]
        
        tech_stack = project_analyzer.identify_technology_stack(existing_tools)
        
        assert "javascript" in tech_stack
        assert "web_development" in tech_stack

    def test_analyze_performance_requirements_high(self, project_analyzer):
        """Test performance requirement analysis for high-performance needs."""
        requirements = {
            "response_time": "fast",
            "throughput": "high",
            "memory_usage": "low"
        }
        
        analysis = project_analyzer.analyze_performance_requirements(requirements)
        
        assert analysis["priority"] == "high"
        assert analysis["optimization_focus"] in ["speed", "efficiency"]

    def test_analyze_performance_requirements_standard(self, project_analyzer):
        """Test performance requirement analysis for standard needs."""
        requirements = {
            "response_time": "medium",
            "throughput": "medium",
            "memory_usage": "medium"
        }
        
        analysis = project_analyzer.analyze_performance_requirements(requirements)
        
        assert analysis["priority"] in ["medium", "standard"]

    def test_identify_integration_patterns_api_focused(self, project_analyzer):
        """Test integration pattern identification for API-focused projects."""
        existing_tools = ["fastapi", "requests", "oauth", "swagger"]
        project_type = "web_service"
        
        patterns = project_analyzer.identify_integration_patterns(existing_tools, project_type)
        
        assert "api_integration" in patterns
        assert "rest_services" in patterns or "web_api" in patterns


class TestEnvironmentAnalyzer:
    """Test cases for EnvironmentAnalyzer class."""

    @pytest.fixture
    def env_analyzer(self):
        """Create an EnvironmentAnalyzer instance for testing."""
        return EnvironmentAnalyzer()

    def test_assess_compute_capacity_high(self, env_analyzer):
        """Test compute capacity assessment for high-spec systems."""
        system_info = {
            "cpu_cores": 16,
            "memory_gb": 32,
            "gpu_available": True,
            "storage_type": "ssd"
        }
        
        capacity = env_analyzer.assess_compute_capacity(system_info)
        
        assert capacity["level"] == "high"
        assert capacity["supports_ml"] == True
        assert capacity["parallel_processing"] == True

    def test_assess_compute_capacity_limited(self, env_analyzer):
        """Test compute capacity assessment for limited systems."""
        system_info = {
            "cpu_cores": 2,
            "memory_gb": 4,
            "gpu_available": False,
            "storage_type": "hdd"
        }
        
        capacity = env_analyzer.assess_compute_capacity(system_info)
        
        assert capacity["level"] == "limited"
        assert capacity["supports_ml"] == False
        assert capacity["memory_constraints"] == True

    def test_analyze_security_requirements_high_security(self, env_analyzer):
        """Test security requirement analysis for high-security environments."""
        security_info = {
            "compliance_frameworks": ["SOC2", "HIPAA", "PCI-DSS"],
            "security_requirements": ["encryption_at_rest", "secure_communication", "audit_logging"],
            "network_restrictions": ["no_external_apis", "firewall_restricted"]
        }
        
        analysis = env_analyzer.analyze_security_requirements(security_info)
        
        assert analysis["security_level"] == "high"
        assert analysis["encryption_required"] == True
        assert analysis["audit_required"] == True
        assert len(analysis["restricted_operations"]) > 0

    def test_analyze_security_requirements_standard(self, env_analyzer):
        """Test security requirement analysis for standard environments."""
        security_info = {
            "compliance_frameworks": [],
            "security_requirements": ["basic_auth"],
            "network_restrictions": []
        }
        
        analysis = env_analyzer.analyze_security_requirements(security_info)
        
        assert analysis["security_level"] in ["standard", "basic"]
        assert analysis["encryption_required"] in [False, None]

    def test_evaluate_resource_constraints_strict(self, env_analyzer):
        """Test resource constraint evaluation for strict limits."""
        constraints = {
            "max_memory_usage": "2GB",
            "max_cpu_usage": "25%",
            "network_bandwidth": "limited",
            "disk_space": "10GB"
        }
        
        evaluation = env_analyzer.evaluate_resource_constraints(constraints)
        
        assert evaluation["constraint_level"] == "strict"
        assert evaluation["memory_limited"] == True
        assert evaluation["cpu_limited"] == True


class TestContextDataTypes:
    """Test cases for context data type classes."""

    def test_user_context_serialization(self):
        """Test UserContext serialization to dictionary."""
        user_context = UserContext(
            skill_level="intermediate",
            tool_familiarity={"tool1": 0.8, "tool2": 0.6},
            complexity_preference="balanced",
            error_tolerance="medium",
            learning_preference="guided"
        )
        
        context_dict = user_context.to_dict()
        
        assert isinstance(context_dict, dict)
        assert context_dict["skill_level"] == "intermediate"
        assert context_dict["tool_familiarity"]["tool1"] == 0.8

    def test_project_context_serialization(self):
        """Test ProjectContext serialization to dictionary."""
        project_context = ProjectContext(
            project_type="data_analysis",
            domain="finance",
            team_size=5,
            existing_tools=["pandas", "numpy"],
            technology_stack=["python"],
            performance_requirements={"speed": "high"},
            compliance_requirements=["GDPR"],
            integration_patterns=["api_integration"]
        )
        
        context_dict = project_context.to_dict()
        
        assert isinstance(context_dict, dict)
        assert context_dict["project_type"] == "data_analysis"
        assert "pandas" in context_dict["existing_tools"]

    def test_task_context_completeness(self):
        """Test TaskContext with all components."""
        user_context = UserContext("intermediate", {}, "balanced", "medium", "guided")
        project_context = ProjectContext("web_app", "ecommerce", 3, [], [], {}, [], [])
        env_constraints = EnvironmentalConstraints(4, 8, False, "windows", [], {}, [])
        
        task_context = TaskContext(
            user_context=user_context,
            project_context=project_context,
            environmental_constraints=env_constraints,
            time_constraints={"deadline": "2_weeks"},
            quality_requirements={"performance": "high"}
        )
        
        # Verify all components are properly referenced
        assert task_context.user_context.skill_level == "intermediate"
        assert task_context.project_context.project_type == "web_app"
        assert task_context.environmental_constraints.cpu_cores == 4
        assert task_context.time_constraints["deadline"] == "2_weeks"