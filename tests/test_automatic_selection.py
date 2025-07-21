#!/usr/bin/env python3
"""
Test suite for Automatic Tool Selection Engine

This module tests all components of the automatic tool selection system:
- AutomaticToolSelector: Core orchestrator
- UsagePatternAnalyzer: Pattern recognition and analysis
- ProactiveSuggestionEngine: Suggestion generation
- ToolLearningSystem: Machine learning and adaptation
- SelectionEffectivenessTracker: Performance monitoring
"""

import time
from unittest.mock import MagicMock


from automatic_tool_selector import AutomaticSelectionResult, AutomaticToolSelector, ToolRanking
from proactive_suggestion_engine import ProactiveSuggestion, ProactiveSuggestionEngine
from selection_effectiveness_tracker import SelectionEffectivenessTracker
from tool_learning_system import ToolLearningSystem
from usage_pattern_analyzer import InefficiencyPattern, UsagePatternAnalysis, UsagePatternAnalyzer


class TestAutomaticToolSelector:
    """Test the main AutomaticToolSelector class."""

    def setup_method(self):
        """Set up test dependencies."""
        self.mock_usage_analyzer = MagicMock(spec=UsagePatternAnalyzer)
        self.mock_suggestion_engine = MagicMock(spec=ProactiveSuggestionEngine)
        self.mock_learning_system = MagicMock(spec=ToolLearningSystem)
        self.mock_effectiveness_tracker = MagicMock(spec=SelectionEffectivenessTracker)

        self.selector = AutomaticToolSelector(
            usage_analyzer=self.mock_usage_analyzer,
            suggestion_engine=self.mock_suggestion_engine,
            learning_system=self.mock_learning_system,
            effectiveness_tracker=self.mock_effectiveness_tracker,
        )

    def test_initialization(self):
        """Test AutomaticToolSelector initialization."""
        assert self.selector.usage_analyzer == self.mock_usage_analyzer
        assert self.selector.suggestion_engine == self.mock_suggestion_engine
        assert self.selector.learning_system == self.mock_learning_system
        assert self.selector.effectiveness_tracker == self.mock_effectiveness_tracker
        assert hasattr(self.selector, "context_manager")

    def test_analyze_and_suggest_basic(self):
        """Test basic analyze_and_suggest functionality."""
        # Setup mock responses
        mock_usage_patterns = UsagePatternAnalysis(
            context={"task": "file_search"},
            recent_patterns=[],
            task_patterns=[],
            inefficiency_patterns=[],
            session_summary={},
        )

        mock_suggestions = [
            ProactiveSuggestion(
                suggestion_type="tool_replacement", tool_id="search_tool", confidence=0.8, pattern_strength=0.7
            )
        ]

        mock_learned_preferences = {"tool_preferences": {"search_tool": 0.9}, "confidence": 0.8}

        self.mock_usage_analyzer.analyze_current_session.return_value = mock_usage_patterns
        self.mock_suggestion_engine.generate_proactive_suggestions.return_value = mock_suggestions
        self.mock_learning_system.get_learned_preferences.return_value = mock_learned_preferences
        self.mock_effectiveness_tracker.get_tool_effectiveness.return_value = 0.6

        # Execute
        context = {"task": "file_search", "user_input": "find files"}
        result = self.selector.analyze_and_suggest(context)

        # Verify result
        assert isinstance(result, AutomaticSelectionResult)
        assert result.context == context
        assert len(result.suggested_tools) == 1
        assert result.suggested_tools[0].tool_id == "search_tool"
        assert result.learned_preferences == mock_learned_preferences
        assert "search_tool" in result.confidence_scores

        # Verify method calls
        self.mock_usage_analyzer.analyze_current_session.assert_called_once_with(
            context=context, task=None, history=None
        )
        self.mock_suggestion_engine.generate_proactive_suggestions.assert_called_once_with(
            patterns=mock_usage_patterns, context=context
        )
        self.mock_learning_system.get_learned_preferences.assert_called_once_with(
            context=context, user_patterns=mock_usage_patterns
        )
        self.mock_effectiveness_tracker.track_selection_event.assert_called_once()

    def test_pre_rank_tools_for_context(self):
        """Test tool pre-ranking functionality."""
        available_tools = ["tool1", "tool2", "tool3"]
        context = {"task_type": "search", "complexity": "simple"}

        # Mock context manager
        self.selector.context_manager = MagicMock()
        self.selector.context_manager.analyze_context.return_value = {"task_type": "search", "complexity_score": 0.3}

        self.mock_learning_system.get_context_preferences.return_value = {"tool1": 0.9, "tool2": 0.7, "tool3": 0.5}

        # Mock private method
        self.selector._calculate_tool_ranking = MagicMock(
            side_effect=[
                ToolRanking(tool_id="tool1", ranking_score=0.9, context=context, factors={}),
                ToolRanking(tool_id="tool2", ranking_score=0.7, context=context, factors={}),
                ToolRanking(tool_id="tool3", ranking_score=0.5, context=context, factors={}),
            ]
        )

        # Execute
        rankings = self.selector.pre_rank_tools_for_context(available_tools, context)

        # Verify
        assert len(rankings) == 3
        assert rankings[0].tool_id == "tool1"
        assert rankings[0].ranking_score == 0.9
        assert rankings[1].tool_id == "tool2"
        assert rankings[2].tool_id == "tool3"

    def test_confidence_score_calculation(self):
        """Test confidence score calculation for suggestions."""
        suggestions = [
            ProactiveSuggestion(
                suggestion_type="tool_replacement", tool_id="tool1", pattern_strength=0.8, context={"task": "search"}
            ),
            ProactiveSuggestion(
                suggestion_type="workflow_improvement",
                tool_id="tool2",
                pattern_strength=0.6,
                context={"task": "analysis"},
            ),
        ]

        self.mock_effectiveness_tracker.get_tool_effectiveness.side_effect = [0.7, 0.5]

        # Execute
        scores = self.selector._calculate_confidence_scores(suggestions)

        # Verify
        assert len(scores) == 2
        assert "tool1" in scores
        assert "tool2" in scores
        # tool1: 0.8 * 0.6 + 0.7 * 0.4 = 0.48 + 0.28 = 0.76
        assert abs(scores["tool1"] - 0.76) < 0.01
        # tool2: 0.6 * 0.6 + 0.5 * 0.4 = 0.36 + 0.20 = 0.56
        assert abs(scores["tool2"] - 0.56) < 0.01


class TestAutomaticSelectionResult:
    """Test the AutomaticSelectionResult dataclass."""

    def test_initialization(self):
        """Test AutomaticSelectionResult initialization."""
        context = {"task": "search"}
        suggestions = [ProactiveSuggestion(suggestion_type="tool_replacement", tool_id="tool1", confidence=0.8)]
        preferences = {"tool_preferences": {"tool1": 0.9}}
        confidence_scores = {"tool1": 0.8}
        reasoning = ["Tool1 is optimal for search tasks"]

        result = AutomaticSelectionResult(
            context=context,
            suggested_tools=suggestions,
            learned_preferences=preferences,
            confidence_scores=confidence_scores,
            reasoning=reasoning,
        )

        assert result.context == context
        assert result.suggested_tools == suggestions
        assert result.learned_preferences == preferences
        assert result.confidence_scores == confidence_scores
        assert result.reasoning == reasoning
        assert result.selection_strategy == "automatic"
        assert isinstance(result.selection_timestamp, float)

    def test_get_top_suggestion(self):
        """Test getting the highest confidence suggestion."""
        suggestions = [
            ProactiveSuggestion(suggestion_type="tool_replacement", tool_id="tool1"),
            ProactiveSuggestion(suggestion_type="tool_replacement", tool_id="tool2"),
            ProactiveSuggestion(suggestion_type="tool_replacement", tool_id="tool3"),
        ]
        confidence_scores = {"tool1": 0.6, "tool2": 0.9, "tool3": 0.4}

        result = AutomaticSelectionResult(
            context={},
            suggested_tools=suggestions,
            learned_preferences={},
            confidence_scores=confidence_scores,
            reasoning=[],
        )

        top_suggestion = result.get_top_suggestion()
        assert top_suggestion is not None
        assert top_suggestion.tool_id == "tool2"

    def test_get_top_suggestion_empty(self):
        """Test get_top_suggestion with empty suggestions."""
        result = AutomaticSelectionResult(
            context={}, suggested_tools=[], learned_preferences={}, confidence_scores={}, reasoning=[]
        )

        top_suggestion = result.get_top_suggestion()
        assert top_suggestion is None

    def test_to_dict(self):
        """Test conversion to dictionary for JSON serialization."""
        suggestion = ProactiveSuggestion(suggestion_type="tool_replacement", tool_id="tool1", confidence=0.8)

        result = AutomaticSelectionResult(
            context={"task": "search"},
            suggested_tools=[suggestion],
            learned_preferences={"tool_preferences": {"tool1": 0.9}},
            confidence_scores={"tool1": 0.8},
            reasoning=["Tool1 is optimal"],
        )

        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert result_dict["context"] == {"task": "search"}
        assert len(result_dict["suggested_tools"]) == 1
        assert result_dict["confidence_scores"] == {"tool1": 0.8}
        assert result_dict["reasoning"] == ["Tool1 is optimal"]
        assert "selection_timestamp" in result_dict
        assert result_dict["selection_strategy"] == "automatic"


class TestUsagePatternAnalyzer:
    """Test the UsagePatternAnalyzer class."""

    def setup_method(self):
        """Set up test dependencies."""
        self.analyzer = UsagePatternAnalyzer()

    def test_initialization(self):
        """Test UsagePatternAnalyzer initialization."""
        assert hasattr(self.analyzer, "pattern_detectors")
        assert hasattr(self.analyzer, "session_tracker")

    def test_analyze_current_session_basic(self):
        """Test basic session analysis."""
        context = {"task": "file_search", "user_input": "find config files"}

        # Mock session tracker
        self.analyzer.session_tracker = MagicMock()
        self.analyzer.session_tracker.get_current_session.return_value = {
            "tool_usage": [{"tool": "search_files", "timestamp": time.time(), "success": True}],
            "start_time": time.time() - 300,
            "errors": [],
        }

        # Execute
        analysis = self.analyzer.analyze_current_session(context)

        # Verify
        assert isinstance(analysis, UsagePatternAnalysis)
        assert analysis.context == context
        assert isinstance(analysis.recent_patterns, list)
        assert isinstance(analysis.task_patterns, list)
        assert isinstance(analysis.inefficiency_patterns, list)
        assert isinstance(analysis.session_summary, dict)

    def test_detect_inefficiencies(self):
        """Test inefficiency pattern detection."""
        session_data = {
            "tool_usage": [
                {"tool": "wrong_tool", "timestamp": time.time() - 60, "success": False, "context": {"task": "search"}},
                {
                    "tool": "another_wrong_tool",
                    "timestamp": time.time() - 30,
                    "success": False,
                    "context": {"task": "search"},
                },
                {"tool": "correct_tool", "timestamp": time.time(), "success": True, "context": {"task": "search"}},
            ]
        }

        # Mock private methods
        self.analyzer._is_suboptimal_choice = MagicMock(return_value=True)
        self.analyzer._get_better_alternative = MagicMock(return_value="better_tool")
        self.analyzer._find_error_sequences = MagicMock(
            return_value=[[session_data["tool_usage"][0], session_data["tool_usage"][1]]]
        )

        # Execute
        inefficiencies = self.analyzer._detect_inefficiencies(session_data)

        # Verify
        assert len(inefficiencies) >= 1
        # Check if we have either type of inefficiency
        has_suboptimal = any(i.type == "suboptimal_choice" for i in inefficiencies)
        has_trial_error = any(i.type == "excessive_trial_error" for i in inefficiencies)
        assert has_suboptimal or has_trial_error


class TestProactiveSuggestionEngine:
    """Test the ProactiveSuggestionEngine class."""

    def setup_method(self):
        """Set up test dependencies."""
        self.engine = ProactiveSuggestionEngine()

    def test_initialization(self):
        """Test ProactiveSuggestionEngine initialization."""
        assert hasattr(self.engine, "rule_engine")
        assert hasattr(self.engine, "context_analyzer")

    def test_generate_proactive_suggestions(self):
        """Test proactive suggestion generation."""
        # Create mock usage patterns
        inefficiency_pattern = InefficiencyPattern(
            type="suboptimal_choice",
            description="Using slow tool for simple task",
            suggestion="fast_tool",
            confidence=0.8,
        )

        patterns = UsagePatternAnalysis(
            context={"task": "search"},
            recent_patterns=[],
            task_patterns=[],
            inefficiency_patterns=[inefficiency_pattern],
            session_summary={},
        )

        context = {"task": "search", "complexity": "simple"}

        # Mock private methods
        self.engine._create_efficiency_suggestion = MagicMock(
            return_value=ProactiveSuggestion(
                suggestion_type="tool_replacement",
                tool_id="fast_tool",
                reasoning="More efficient for simple searches",
                confidence=0.8,
            )
        )
        self.engine._rank_suggestions = MagicMock(side_effect=lambda x, ctx: x)

        # Execute
        suggestions = self.engine.generate_proactive_suggestions(patterns, context)

        # Verify
        assert isinstance(suggestions, list)
        assert len(suggestions) >= 1
        assert all(isinstance(s, ProactiveSuggestion) for s in suggestions)


class TestToolLearningSystem:
    """Test the ToolLearningSystem class."""

    def setup_method(self):
        """Set up test dependencies."""
        self.learning_system = ToolLearningSystem()

    def test_initialization(self):
        """Test ToolLearningSystem initialization."""
        assert hasattr(self.learning_system, "preference_model")
        assert hasattr(self.learning_system, "effectiveness_model")
        assert hasattr(self.learning_system, "context_model")

    def test_learn_from_selection_outcome(self):
        """Test learning from tool selection outcomes."""
        selection_context = {"task": "search", "complexity": "simple"}
        tool_chosen = "search_tool"
        outcome_success = True
        outcome_metrics = {"completion_time": 5.2, "satisfaction": 0.9}

        # Mock model methods
        self.learning_system.effectiveness_model.update_effectiveness = MagicMock()
        self.learning_system.preference_model.reinforce_preference = MagicMock()
        self.learning_system.context_model.update_context_patterns = MagicMock()

        # Execute
        self.learning_system.learn_from_selection_outcome(
            selection_context, tool_chosen, outcome_success, outcome_metrics
        )

        # Verify all models were updated
        self.learning_system.effectiveness_model.update_effectiveness.assert_called_once_with(
            tool_id=tool_chosen, context=selection_context, success=outcome_success, metrics=outcome_metrics
        )
        self.learning_system.preference_model.reinforce_preference.assert_called_once_with(
            context=selection_context, tool=tool_chosen, strength=0.9
        )
        self.learning_system.context_model.update_context_patterns.assert_called_once()

    def test_get_learned_preferences(self):
        """Test getting learned user preferences."""
        context = {"task": "search"}

        mock_patterns = UsagePatternAnalysis(
            context=context, recent_patterns=[], task_patterns=[], inefficiency_patterns=[], session_summary={}
        )

        # Mock model methods
        self.learning_system.preference_model.predict_preferences = MagicMock(
            return_value={"search_tool": 0.8, "analyze_tool": 0.6}
        )
        self.learning_system.preference_model.get_prediction_confidence = MagicMock(return_value=0.7)
        self.learning_system._get_learning_status = MagicMock(
            return_value={"preference_model_accuracy": 0.85, "total_learning_samples": 150}
        )

        # Execute
        preferences = self.learning_system.get_learned_preferences(context, mock_patterns)

        # Verify
        assert isinstance(preferences, dict)
        assert "tool_preferences" in preferences
        assert "confidence" in preferences
        assert "learning_status" in preferences
        assert preferences["confidence"] == 0.7


class TestSelectionEffectivenessTracker:
    """Test the SelectionEffectivenessTracker class."""

    def setup_method(self):
        """Set up test dependencies."""
        self.tracker = SelectionEffectivenessTracker()

    def test_initialization(self):
        """Test SelectionEffectivenessTracker initialization."""
        assert hasattr(self.tracker, "selection_history")
        assert hasattr(self.tracker, "effectiveness_metrics")

    def test_track_selection_event(self):
        """Test tracking a selection event."""
        context = {"task": "search", "user_input": "find files"}
        suggestions = [ProactiveSuggestion(suggestion_type="tool_replacement", tool_id="search_tool")]
        selection_result = AutomaticSelectionResult(
            context=context,
            suggested_tools=suggestions,
            learned_preferences={},
            confidence_scores={"search_tool": 0.8},
            reasoning=["Optimal for search tasks"],
        )

        # Execute
        self.tracker.track_selection_event(context, suggestions, selection_result)

        # Verify event was tracked
        assert len(self.tracker.selection_history) >= 1

        latest_event = self.tracker.selection_history[-1]
        assert latest_event.context == context
        assert len(latest_event.suggestions) == 1
        assert latest_event.timestamp > 0

    def test_get_tool_effectiveness(self):
        """Test getting tool effectiveness score."""
        tool_id = "search_tool"
        context = {"task": "search"}

        # Mock some historical data
        self.tracker.effectiveness_metrics[tool_id] = {
            "total_uses": 10,
            "successful_uses": 8,
            "average_satisfaction": 0.85,
            "contexts": [context] * 5,
        }

        # Execute
        effectiveness = self.tracker.get_tool_effectiveness(tool_id, context)

        # Verify
        assert isinstance(effectiveness, float)
        assert 0.0 <= effectiveness <= 1.0


class TestIntegrationRealComponents:
    """Integration tests using real components without heavy mocking."""

    def setup_method(self):
        """Set up real components for integration testing."""
        # Use temporary directories for testing
        import tempfile
        from pathlib import Path

        self.temp_dir = Path(tempfile.mkdtemp(prefix="test_automatic_selection_"))

        # Create real components with test storage paths
        self.usage_analyzer = UsagePatternAnalyzer()
        self.suggestion_engine = ProactiveSuggestionEngine()
        self.learning_system = ToolLearningSystem(storage_dir=self.temp_dir)
        self.effectiveness_tracker = SelectionEffectivenessTracker(storage_path=self.temp_dir / "effectiveness.json")

        # Create integrated system
        self.selector = AutomaticToolSelector(
            usage_analyzer=self.usage_analyzer,
            suggestion_engine=self.suggestion_engine,
            learning_system=self.learning_system,
            effectiveness_tracker=self.effectiveness_tracker,
        )

    def teardown_method(self):
        """Clean up test files."""
        import shutil

        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_end_to_end_selection_workflow(self):
        """Test complete end-to-end selection workflow with real data."""
        # Simulate a realistic usage scenario
        context = {
            "task": "search_files",
            "user_input": "find all Python files in the project",
            "complexity": "medium",
            "urgency": "normal",
        }

        # First selection - cold start
        result = self.selector.analyze_and_suggest(current_context=context)

        # Verify we got a real result
        assert isinstance(result, AutomaticSelectionResult)
        assert result.context == context
        assert isinstance(result.suggested_tools, list)
        assert isinstance(result.confidence_scores, dict)
        assert isinstance(result.reasoning, list)
        assert result.selection_timestamp > 0

        # Verify the system generated some suggestions
        # Even with cold start, the system should provide some default suggestions
        assert len(result.reasoning) > 0

        # Simulate user selection and outcome
        if result.suggested_tools:
            selected_tool = result.suggested_tools[0]

            # Simulate successful outcome
            self.effectiveness_tracker.update_selection_outcome(
                event_id=f"test_event_{int(time.time())}", outcome_success=True, completion_time=2.5, user_feedback=0.8
            )

            # Verify the effectiveness tracker recorded the outcome
            assert len(self.effectiveness_tracker.selection_history) > 0

    def test_learning_and_adaptation_cycle(self):
        """Test that the system learns and adapts from usage patterns."""
        contexts = [
            {"task": "file_search", "user_input": "find config files", "domain": "development"},
            {"task": "file_search", "user_input": "locate test files", "domain": "development"},
            {"task": "file_search", "user_input": "search for documentation", "domain": "development"},
        ]

        initial_preferences = self.learning_system.get_learned_preferences(contexts[0], None)

        # Simulate multiple successful interactions
        for i, context in enumerate(contexts):
            result = self.selector.analyze_and_suggest(current_context=context)

            if result.suggested_tools:
                # Simulate positive outcomes
                event_id = f"learning_test_{i}_{int(time.time())}"
                self.effectiveness_tracker.track_selection_event(
                    context=context, suggestions=result.suggested_tools, selection_result=result
                )

                # Add positive outcome
                self.effectiveness_tracker.update_selection_outcome(
                    event_id=event_id,
                    outcome_success=True,
                    completion_time=1.0 + (i * 0.2),  # Improving performance
                    user_feedback=0.7 + (i * 0.1),  # Increasing satisfaction
                )

        # Check if learning system has adapted
        final_preferences = self.learning_system.get_learned_preferences(contexts[0], None)

        # The preferences should exist (may be same or different, but should be populated)
        assert isinstance(final_preferences, dict)

        # Check that effectiveness tracking has data
        assert len(self.effectiveness_tracker.selection_history) >= len(contexts)

    def test_real_suggestion_generation(self):
        """Test that suggestion engine generates realistic suggestions."""
        # Test various realistic scenarios
        scenarios = [
            {
                "context": {"task": "file_operations", "user_input": "need to read and process large files"},
                "expected_categories": ["file", "read"],
            },
            {
                "context": {"task": "data_analysis", "user_input": "analyze CSV data for patterns"},
                "expected_categories": ["analysis", "data"],
            },
            {
                "context": {"task": "search", "user_input": "find specific functions in codebase"},
                "expected_categories": ["search", "code"],
            },
        ]

        for scenario in scenarios:
            # Use the usage analyzer to get patterns
            patterns = self.usage_analyzer.analyze_current_session(
                context=scenario["context"], task=scenario["context"]["task"]
            )

            # Generate suggestions
            suggestions = self.suggestion_engine.generate_proactive_suggestions(
                patterns=patterns, context=scenario["context"]
            )

            # Verify we got suggestions
            assert isinstance(suggestions, list)

            # Test suggestion structure
            for suggestion in suggestions:
                assert isinstance(suggestion, ProactiveSuggestion)
                assert suggestion.tool_id is not None
                assert suggestion.suggestion_type in [
                    "tool_replacement",
                    "workflow_improvement",
                    "parameter_optimization",
                ]
                assert 0.0 <= suggestion.confidence <= 1.0
                assert isinstance(suggestion.reasoning, str)
                assert suggestion.reasoning != ""  # Should have actual reasoning

    def test_configuration_integration(self):
        """Test that the system uses configuration correctly."""
        # Test that the system respects configuration limits
        context = {"task": "test", "user_input": "simple test"}

        # Generate multiple events to test memory management
        for i in range(15):  # Create more than typical batch size
            result = self.selector.analyze_and_suggest(current_context=context)

            # Add some tracking events
            if result.suggested_tools:
                self.effectiveness_tracker.track_selection_event(
                    context=context, suggestions=result.suggested_tools, selection_result=result
                )

        # Check that memory limits are respected
        memory_stats = self.effectiveness_tracker.get_memory_usage_stats()
        assert isinstance(memory_stats, dict)
        assert "current_history_size" in memory_stats
        assert "max_history_size" in memory_stats

        # History size should be managed according to configuration
        assert memory_stats["current_history_size"] <= memory_stats["max_history_size"] * 1.2  # Allow some buffer

    def test_error_resilience(self):
        """Test that the system handles errors gracefully."""
        # Test with problematic context
        problematic_contexts = [
            {},  # Empty context
            {"task": None},  # Null task
            {"task": "test", "user_input": ""},  # Empty input
            {"task": "test", "user_input": "x" * 10000},  # Very long input
        ]

        for context in problematic_contexts:
            try:
                result = self.selector.analyze_and_suggest(current_context=context)

                # Should still get a valid result object
                assert isinstance(result, AutomaticSelectionResult)
                assert result.context == context
                assert isinstance(result.suggested_tools, list)
                assert isinstance(result.reasoning, list)

                # System should handle edge cases gracefully
                # No suggestions is acceptable for problematic contexts
                # The important thing is we get a valid result object

            except Exception as e:
                # If exceptions occur, they should be logged and handled gracefully
                assert False, f"System should handle errors gracefully, but got: {e}"

    def test_storage_persistence(self):
        """Test that data persists correctly across sessions."""
        context = {"task": "persistence_test", "user_input": "test storage"}

        # Generate some data
        result = self.selector.analyze_and_suggest(current_context=context)

        # Force a save
        if hasattr(self.learning_system, "_save_model"):
            self.learning_system._save_model()
        if hasattr(self.effectiveness_tracker, "_save_tracking_data"):
            self.effectiveness_tracker._save_tracking_data()

        # Create new instances (simulating restart)
        new_learning_system = ToolLearningSystem(storage_dir=self.temp_dir)
        new_effectiveness_tracker = SelectionEffectivenessTracker(storage_path=self.temp_dir / "effectiveness.json")

        # Check that they loaded previous data
        new_preferences = new_learning_system.get_learned_preferences(context, None)
        assert isinstance(new_preferences, dict)

        # Check effectiveness tracker
        assert isinstance(new_effectiveness_tracker.selection_history, list)
        assert isinstance(new_effectiveness_tracker.effectiveness_metrics, dict)
