#!/usr/bin/env python3
"""
Integration test for the complete tool recommendation system.

This test verifies that all components work together to provide
comprehensive tool recommendations with explanations and context awareness.
"""

from unittest.mock import Mock

import pytest

from context_analyzer import ContextAnalyzer
from task_analyzer import TaskAnalyzer
from tool_recommendation_engine import RecommendationRequest, ToolRecommendationEngine


class TestRecommendationSystemIntegration:
    """Integration tests for the complete recommendation system."""

    @pytest.fixture
    def complete_system(self):
        """Create a complete recommendation system for testing."""
        # Create real components (not mocks) to test integration
        task_analyzer = TaskAnalyzer()
        context_analyzer = ContextAnalyzer()

        # Mock the search engines since we don't have real tool catalogs
        mock_tool_search = Mock()
        mock_tool_search.search_tools.return_value = [
            Mock(
                tool_id="csv_reader",
                name="CSV Reader",
                description="Read CSV files",
                score=0.9,
                metadata={"capabilities": ["read", "csv"], "complexity": "simple"},
            ),
            Mock(
                tool_id="data_processor",
                name="Data Processor",
                description="Process data",
                score=0.8,
                metadata={"capabilities": ["process", "data"], "complexity": "moderate"},
            ),
        ]

        mock_param_search = Mock()
        mock_param_search.search_by_parameters.return_value = []

        return ToolRecommendationEngine(
            tool_search_engine=mock_tool_search,
            parameter_search_engine=mock_param_search,
            task_analyzer=task_analyzer,
            context_analyzer=context_analyzer,
        )

    def test_complete_recommendation_flow_basic(self, complete_system):
        """Test the complete recommendation flow with basic request."""
        request = RecommendationRequest(
            task_description="Read CSV files and process the data for analysis",
            max_recommendations=3,
            include_alternatives=True,
            include_explanations=True,
            context_data={
                "user_skill_level": "intermediate",
                "project_type": "data_analysis",
                "existing_tools": ["pandas", "numpy"],
            },
        )

        response = complete_system.recommend_for_task(request)

        # Verify complete response structure
        assert response.recommendations is not None
        assert len(response.recommendations) > 0
        assert response.task_analysis is not None
        assert response.context is not None
        assert response.explanations is not None
        assert response.alternatives is not None

        # Verify task analysis worked correctly
        assert response.task_analysis.task_category in ["data_processing", "file_operation", "data_extraction"]
        assert (
            "read" in response.task_analysis.required_capabilities
            or "csv" in response.task_analysis.required_capabilities
        )
        assert response.task_analysis.complexity_level in ["simple", "moderate", "complex"]

        # Verify context analysis worked
        if response.context.user_context:
            assert response.context.user_context.skill_level in ["beginner", "intermediate", "advanced"]
        if response.context.project_context:
            assert response.context.project_context.project_type == "data_analysis"

        # Verify recommendations have proper structure
        for rec in response.recommendations:
            assert rec.tool is not None
            assert rec.recommendation_score > 0
            assert rec.confidence_level in ["low", "medium", "high"]
            assert len(rec.recommendation_reasons) > 0

        # Verify explanations are comprehensive
        for explanation in response.explanations:
            assert explanation.capability_match_explanation != ""
            assert explanation.complexity_fit_explanation != ""
            assert explanation.confidence_explanation != ""

    def test_complete_recommendation_flow_complex(self, complete_system):
        """Test the complete recommendation flow with complex ML task."""
        request = RecommendationRequest(
            task_description="Build a machine learning pipeline with data validation, model training, and performance monitoring",
            max_recommendations=5,
            include_alternatives=True,
            include_explanations=True,
            context_data={
                "tool_usage_history": [
                    {"tool": "scikit-learn", "success": True, "complexity": "complex"},
                    {"tool": "tensorflow", "success": True, "complexity": "complex"},
                    {"tool": "mlflow", "success": True, "complexity": "complex"},
                ],
                "project_type": "machine_learning",
                "existing_tools": ["scikit-learn", "tensorflow"],
                "performance_requirements": {"latency": "low", "throughput": "high"},
                "system_capabilities": {"cpu_cores": 16, "memory_gb": 32, "gpu_available": True},
                "compliance_requirements": ["audit_trail", "data_privacy"],
            },
        )

        response = complete_system.recommend_for_task(request)

        # Verify complex task analysis
        assert response.task_analysis.task_category == "machine_learning"
        assert response.task_analysis.complexity_level in ["moderate", "complex"]
        assert response.task_analysis.skill_level_required in ["intermediate", "advanced"]
        assert response.task_analysis.estimated_steps > 3

        # Verify context integration with all components
        assert response.context.user_context is not None
        assert response.context.project_context is not None
        assert response.context.environmental_constraints is not None

        # Verify advanced user context
        assert response.context.user_context.skill_level == "advanced"

        # Verify project context
        assert response.context.project_context.project_type == "machine_learning"
        assert "scikit-learn" in response.context.project_context.existing_tools

        # Verify environmental constraints
        assert response.context.environmental_constraints.cpu_cores == 16
        assert response.context.environmental_constraints.memory_gb == 32
        assert response.context.environmental_constraints.gpu_available == True

    def test_end_to_end_performance(self, complete_system):
        """Test end-to-end performance and caching."""
        request = RecommendationRequest(
            task_description="Simple file processing task",
            max_recommendations=2,
            include_alternatives=False,
            include_explanations=False,
        )

        # First request
        import time

        start_time = time.time()
        response1 = complete_system.recommend_for_task(request)
        first_duration = time.time() - start_time

        # Second identical request (should use cache)
        start_time = time.time()
        response2 = complete_system.recommend_for_task(request)
        second_duration = time.time() - start_time

        # Verify both requests succeeded
        assert len(response1.recommendations) > 0
        assert len(response2.recommendations) > 0

        # Verify performance (second request should be much faster due to caching)
        assert second_duration <= first_duration  # Should be faster or equal
        assert first_duration < 1.0  # Should complete within 1 second
        assert second_duration < 0.1  # Cached request should be very fast

    def test_error_handling_graceful_degradation(self, complete_system):
        """Test graceful degradation when components fail."""
        # Create a request that might cause some analysis to fail
        request = RecommendationRequest(
            task_description="Process data with malformed context",
            context_data={
                "invalid_context": "this should not break the system",
                "user_skill_level": "invalid_level",  # Invalid but should be handled
            },
        )

        # Should not raise exception despite invalid context
        response = complete_system.recommend_for_task(request)

        # Should still provide basic recommendations
        assert response.recommendations is not None
        assert len(response.recommendations) > 0
        assert response.task_analysis is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
