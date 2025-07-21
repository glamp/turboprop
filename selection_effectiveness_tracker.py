#!/usr/bin/env python3
"""
Selection Effectiveness Tracker - Performance Monitoring

This module tracks the effectiveness of automatic tool selection suggestions,
monitors performance metrics, and provides feedback for learning system
improvement. It maintains selection history and calculates effectiveness scores.
"""

import json
import statistics
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

from automatic_selection_config import CONFIG
from automatic_tool_selector import AutomaticSelectionResult
from logging_config import get_logger
from proactive_suggestion_engine import ProactiveSuggestion
from storage_manager import get_storage_manager

logger = get_logger(__name__)


@dataclass
class SelectionEvent:
    """A tool selection event with outcome tracking."""

    event_id: str
    context: Dict[str, Any]
    suggestions: List[ProactiveSuggestion]
    user_selection: Optional[str] = None
    selection_time: Optional[float] = None
    outcome_success: Optional[bool] = None
    outcome_metrics: Dict[str, float] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    # Tracking metadata
    suggestion_quality_score: Optional[float] = None
    user_feedback: Optional[float] = None  # 0.0 to 1.0 satisfaction
    completion_time: Optional[float] = None


@dataclass
class ToolMetrics:
    """Metrics for a specific tool."""

    tool_id: str
    total_suggestions: int = 0
    total_selections: int = 0
    total_successes: int = 0
    total_failures: int = 0

    # Performance metrics
    average_confidence: float = 0.0
    average_completion_time: float = 0.0
    average_satisfaction: float = 0.0

    # Selection metrics
    selection_rate: float = 0.0  # How often suggested tool was selected
    success_rate: float = 0.0  # How often selected tool succeeded
    effectiveness_score: float = 0.0  # Combined metric

    # Recent performance tracking
    recent_successes: Deque[bool] = field(default_factory=lambda: deque(maxlen=20))
    recent_completion_times: Deque[float] = field(default_factory=lambda: deque(maxlen=20))
    recent_satisfaction_scores: Deque[float] = field(default_factory=lambda: deque(maxlen=20))

    last_updated: float = field(default_factory=time.time)


class PerformanceAnalyzer:
    """Analyzes performance trends and patterns."""

    def __init__(self):
        self.trend_window = 50  # Number of events for trend analysis
        logger.debug("Initialized Performance Analyzer")

    def analyze_suggestion_quality(
        self, event: SelectionEvent, historical_performance: Dict[str, ToolMetrics]
    ) -> float:
        """Analyze the quality of suggestions for this event."""

        try:
            if not event.suggestions:
                return 0.0

            quality_factors = []

            # Factor 1: Confidence of suggestions
            confidences = [s.confidence for s in event.suggestions if hasattr(s, "confidence")]
            if confidences:
                avg_confidence = statistics.mean(confidences)
                quality_factors.append(avg_confidence * 0.3)

            # Factor 2: Historical performance of suggested tools
            historical_scores = []
            for suggestion in event.suggestions:
                if suggestion.tool_id in historical_performance:
                    metrics = historical_performance[suggestion.tool_id]
                    historical_scores.append(metrics.effectiveness_score)

            if historical_scores:
                avg_historical = statistics.mean(historical_scores)
                quality_factors.append(avg_historical * 0.4)

            # Factor 3: Diversity of suggestions
            unique_tools = len(set(s.tool_id for s in event.suggestions))
            diversity_score = min(1.0, unique_tools / 3.0)  # Normalize to max 3 suggestions
            quality_factors.append(diversity_score * 0.2)

            # Factor 4: Contextual relevance (simplified scoring)
            context_score = self._assess_contextual_relevance(event.context, event.suggestions)
            quality_factors.append(context_score * 0.1)

            overall_quality = sum(quality_factors) if quality_factors else 0.3
            return min(1.0, overall_quality)

        except Exception as e:
            logger.error("Error analyzing suggestion quality: %s", e)
            return 0.3

    def detect_performance_trends(self, tool_metrics: Dict[str, ToolMetrics], lookback_days: int = 7) -> Dict[str, str]:
        """Detect performance trends for tools."""
        trends = {}

        try:
            cutoff_time = time.time() - (lookback_days * 86400)

            for tool_id, metrics in tool_metrics.items():
                if metrics.last_updated < cutoff_time:
                    trends[tool_id] = "stale"
                    continue

                # Analyze recent success trend
                recent_successes = list(metrics.recent_successes)
                if len(recent_successes) >= 10:
                    # Compare first half vs second half
                    mid_point = len(recent_successes) // 2
                    early_success_rate = sum(recent_successes[:mid_point]) / mid_point
                    late_success_rate = sum(recent_successes[mid_point:]) / (len(recent_successes) - mid_point)

                    if late_success_rate - early_success_rate > 0.1:
                        trends[tool_id] = "improving"
                    elif early_success_rate - late_success_rate > 0.1:
                        trends[tool_id] = "declining"
                    else:
                        trends[tool_id] = "stable"
                else:
                    trends[tool_id] = "insufficient_data"

        except Exception as e:
            logger.error("Error detecting performance trends: %s", e)

        return trends

    def identify_problematic_patterns(self, selection_history: List[SelectionEvent]) -> List[Dict[str, Any]]:
        """Identify patterns that indicate problems."""
        problems = []

        try:
            recent_events = [e for e in selection_history[-100:] if time.time() - e.timestamp < 86400]  # Last 24 hours

            if len(recent_events) < 5:
                return problems

            # Problem 1: Low suggestion acceptance rate
            events_with_selections = [e for e in recent_events if e.user_selection]
            if len(events_with_selections) < len(recent_events) * 0.3:
                problems.append(
                    {
                        "type": "low_acceptance_rate",
                        "description": "Users rarely select suggested tools",
                        "severity": "high",
                        "suggestion": "Review suggestion quality and relevance",
                    }
                )

            # Problem 2: High failure rate for selected tools
            events_with_outcomes = [e for e in recent_events if e.outcome_success is not None]
            if events_with_outcomes:
                failure_rate = sum(1 for e in events_with_outcomes if not e.outcome_success) / len(events_with_outcomes)
                if failure_rate > 0.4:
                    problems.append(
                        {
                            "type": "high_failure_rate",
                            "description": f"Tool selections failing {failure_rate:.1%} of the time",
                            "severity": "high",
                            "suggestion": "Improve tool effectiveness tracking and suggestions",
                        }
                    )

            # Problem 3: Consistently long selection times
            events_with_times = [e for e in recent_events if e.selection_time and e.selection_time > 0]
            if len(events_with_times) >= 10:
                avg_selection_time = statistics.mean(e.selection_time for e in events_with_times)
                if avg_selection_time > 30:  # 30 seconds
                    problems.append(
                        {
                            "type": "slow_selection",
                            "description": f"Average selection time: {avg_selection_time:.1f}s",
                            "severity": "medium",
                            "suggestion": "Simplify suggestion presentation or improve ranking",
                        }
                    )

        except Exception as e:
            logger.error("Error identifying problematic patterns: %s", e)

        return problems

    def _assess_contextual_relevance(self, context: Dict[str, Any], suggestions: List[ProactiveSuggestion]) -> float:
        """Assess how well suggestions match the context."""
        if not suggestions:
            return 0.0

        try:
            relevance_scores = []

            task_type = str(context.get("task", "")).lower()
            user_input = str(context.get("user_input", "")).lower()

            for suggestion in suggestions:
                score = 0.5  # Base score

                # Check tool name relevance to task
                tool_name = suggestion.tool_id.lower()
                if task_type and any(word in tool_name for word in task_type.split()):
                    score += 0.2
                if user_input and any(word in tool_name for word in user_input.split()[-5:]):
                    score += 0.2

                # Check reasoning relevance
                reasoning = getattr(suggestion, "reasoning", "").lower()
                if reasoning and task_type and task_type in reasoning:
                    score += 0.1

                relevance_scores.append(min(1.0, score))

            return statistics.mean(relevance_scores)

        except Exception as e:
            logger.debug("Error assessing contextual relevance: %s", e)
            return 0.5


class SelectionEffectivenessTracker:
    """Tracks and analyzes effectiveness of automatic tool selection."""

    def __init__(self, storage_path: Optional[Path] = None):
        """Initialize effectiveness tracker with optional storage path."""
        self.storage_manager = get_storage_manager()
        self.storage_path = storage_path or self.storage_manager.get_effectiveness_data_path()
        self.selection_history: List[SelectionEvent] = []
        self.effectiveness_metrics: Dict[str, ToolMetrics] = {}
        self.performance_analyzer = PerformanceAnalyzer()

        # Load existing data
        self._load_tracking_data()

        # Configuration from centralized config
        memory_config = CONFIG["memory_config"]
        self.max_history_size = memory_config["max_history_size"]
        self.cleanup_threshold = memory_config["history_cleanup_threshold"]
        self.batch_size = memory_config["batch_size"]
        self.save_frequency = 25  # Save every N events
        self.event_counter = 0

        logger.info("Initialized Selection Effectiveness Tracker with max_history_size=%d", self.max_history_size)

    def track_selection_event(
        self,
        context: Dict[str, Any],
        suggestions: List[ProactiveSuggestion],
        selection_result: AutomaticSelectionResult,
    ) -> str:
        """Track a selection event and return event ID."""

        try:
            # Generate unique event ID
            event_id = f"event_{int(time.time())}_{len(self.selection_history)}"

            # Create selection event
            event = SelectionEvent(
                event_id=event_id, context=context.copy(), suggestions=suggestions.copy(), timestamp=time.time()
            )

            # Analyze suggestion quality
            event.suggestion_quality_score = self.performance_analyzer.analyze_suggestion_quality(
                event, self.effectiveness_metrics
            )

            # Add to history
            self.selection_history.append(event)

            # Update tool metrics
            self._update_tool_metrics_from_suggestions(suggestions)

            # Maintain history size with configurable cleanup
            self._cleanup_history_if_needed()

            # Periodic save
            self.event_counter += 1
            if self.event_counter % self.save_frequency == 0:
                self._save_tracking_data()

            logger.debug("Tracked selection event: %s with %d suggestions", event_id, len(suggestions))

            return event_id

        except Exception as e:
            logger.error("Error tracking selection event: %s", e)
            return f"error_{int(time.time())}"

    def record_user_selection(self, event_id: str, selected_tool: str, selection_time: float):
        """Record which tool the user actually selected."""

        try:
            # Find the event
            event = self._find_event_by_id(event_id)
            if not event:
                logger.warning("Event not found for selection recording: %s", event_id)
                return

            event.user_selection = selected_tool
            event.selection_time = selection_time

            # Update tool metrics
            if selected_tool in self.effectiveness_metrics:
                metrics = self.effectiveness_metrics[selected_tool]
                metrics.total_selections += 1
                metrics.last_updated = time.time()

                # Update selection rate
                if metrics.total_suggestions > 0:
                    metrics.selection_rate = metrics.total_selections / metrics.total_suggestions

            logger.debug("Recorded user selection: %s for event %s", selected_tool, event_id)

        except Exception as e:
            logger.error("Error recording user selection: %s", e)

    def record_outcome(
        self,
        event_id: str,
        outcome_success: bool,
        outcome_metrics: Dict[str, float],
        user_feedback: Optional[float] = None,
    ):
        """Record the outcome of a tool selection."""

        try:
            # Find the event
            event = self._find_event_by_id(event_id)
            if not event:
                logger.warning("Event not found for outcome recording: %s", event_id)
                return

            event.outcome_success = outcome_success
            event.outcome_metrics = outcome_metrics.copy()
            event.user_feedback = user_feedback
            event.completion_time = outcome_metrics.get("completion_time", 0)

            # Update tool metrics if user made a selection
            if event.user_selection and event.user_selection in self.effectiveness_metrics:
                self._update_tool_outcome_metrics(event.user_selection, outcome_success, outcome_metrics, user_feedback)

            logger.debug("Recorded outcome: success=%s for event %s", outcome_success, event_id)

        except Exception as e:
            logger.error("Error recording outcome: %s", e)

    def get_tool_effectiveness(self, tool_id: str, context: Dict[str, Any] = None) -> float:
        """Get effectiveness score for a tool."""

        try:
            if tool_id not in self.effectiveness_metrics:
                return 0.5  # Default score for unknown tools

            metrics = self.effectiveness_metrics[tool_id]

            # Simple effectiveness calculation
            base_score = metrics.effectiveness_score

            # Adjust for recent performance if we have data
            if len(metrics.recent_successes) >= 5:
                recent_success_rate = sum(metrics.recent_successes) / len(metrics.recent_successes)
                # Weight recent performance more heavily
                adjusted_score = base_score * 0.6 + recent_success_rate * 0.4
                return adjusted_score

            return base_score

        except Exception as e:
            logger.error("Error getting tool effectiveness: %s", e)
            return 0.5

    def get_effectiveness_report(self) -> Dict[str, Any]:
        """Generate comprehensive effectiveness report."""

        try:
            report = {
                "overview": {
                    "total_events": len(self.selection_history),
                    "total_tools_tracked": len(self.effectiveness_metrics),
                    "tracking_period_days": self._calculate_tracking_period(),
                    "report_timestamp": time.time(),
                },
                "tool_performance": {},
                "system_performance": {},
                "trends": {},
                "problems": [],
            }

            # Tool-specific performance
            for tool_id, metrics in self.effectiveness_metrics.items():
                report["tool_performance"][tool_id] = {
                    "effectiveness_score": metrics.effectiveness_score,
                    "selection_rate": metrics.selection_rate,
                    "success_rate": metrics.success_rate,
                    "average_satisfaction": metrics.average_satisfaction,
                    "total_suggestions": metrics.total_suggestions,
                    "total_selections": metrics.total_selections,
                }

            # System-wide performance
            if self.selection_history:
                recent_events = self.selection_history[-50:]  # Last 50 events

                suggestion_acceptance_rate = len([e for e in recent_events if e.user_selection]) / len(recent_events)

                events_with_outcomes = [e for e in recent_events if e.outcome_success is not None]
                overall_success_rate = (
                    (sum(1 for e in events_with_outcomes if e.outcome_success) / len(events_with_outcomes))
                    if events_with_outcomes
                    else 0
                )

                avg_suggestion_quality = (
                    statistics.mean(
                        e.suggestion_quality_score for e in recent_events if e.suggestion_quality_score is not None
                    )
                    if any(e.suggestion_quality_score is not None for e in recent_events)
                    else 0
                )

                report["system_performance"] = {
                    "suggestion_acceptance_rate": suggestion_acceptance_rate,
                    "overall_success_rate": overall_success_rate,
                    "average_suggestion_quality": avg_suggestion_quality,
                    "events_analyzed": len(recent_events),
                }

            # Performance trends
            report["trends"] = self.performance_analyzer.detect_performance_trends(self.effectiveness_metrics)

            # Identify problems
            report["problems"] = self.performance_analyzer.identify_problematic_patterns(self.selection_history)

            return report

        except Exception as e:
            logger.error("Error generating effectiveness report: %s", e)
            return {"error": str(e)}

    def _find_event_by_id(self, event_id: str) -> Optional[SelectionEvent]:
        """Find an event by its ID."""
        for event in reversed(self.selection_history):  # Search recent first
            if event.event_id == event_id:
                return event
        return None

    def _update_tool_metrics_from_suggestions(self, suggestions: List[ProactiveSuggestion]):
        """Update tool metrics based on suggestions made."""
        for suggestion in suggestions:
            tool_id = suggestion.tool_id

            if tool_id not in self.effectiveness_metrics:
                self.effectiveness_metrics[tool_id] = ToolMetrics(tool_id=tool_id)

            metrics = self.effectiveness_metrics[tool_id]
            metrics.total_suggestions += 1
            metrics.last_updated = time.time()

            # Update average confidence
            if hasattr(suggestion, "confidence") and suggestion.confidence > 0:
                current_avg = metrics.average_confidence
                total_suggestions = metrics.total_suggestions
                metrics.average_confidence = (
                    current_avg * (total_suggestions - 1) + suggestion.confidence
                ) / total_suggestions

    def _update_tool_outcome_metrics(
        self, tool_id: str, outcome_success: bool, outcome_metrics: Dict[str, float], user_feedback: Optional[float]
    ):
        """Update tool metrics based on outcome."""

        if tool_id not in self.effectiveness_metrics:
            return

        metrics = self.effectiveness_metrics[tool_id]

        # Update success/failure counts
        if outcome_success:
            metrics.total_successes += 1
        else:
            metrics.total_failures += 1

        # Update success rate
        total_outcomes = metrics.total_successes + metrics.total_failures
        metrics.success_rate = metrics.total_successes / total_outcomes if total_outcomes > 0 else 0

        # Update completion time
        completion_time = outcome_metrics.get("completion_time", 0)
        if completion_time > 0:
            metrics.recent_completion_times.append(completion_time)
            metrics.average_completion_time = statistics.mean(metrics.recent_completion_times)

        # Update satisfaction
        if user_feedback is not None:
            metrics.recent_satisfaction_scores.append(user_feedback)
            metrics.average_satisfaction = statistics.mean(metrics.recent_satisfaction_scores)

        # Update recent success tracking
        metrics.recent_successes.append(outcome_success)

        # Calculate overall effectiveness score
        metrics.effectiveness_score = self._calculate_effectiveness_score(metrics)
        metrics.last_updated = time.time()

    def _calculate_effectiveness_score(self, metrics: ToolMetrics) -> float:
        """Calculate overall effectiveness score for a tool."""
        factors = []

        # Success rate factor (40% weight)
        if metrics.total_successes + metrics.total_failures > 0:
            factors.append(metrics.success_rate * 0.4)

        # Selection rate factor (20% weight)
        if metrics.total_suggestions > 0:
            factors.append(metrics.selection_rate * 0.2)

        # User satisfaction factor (25% weight)
        if metrics.average_satisfaction > 0:
            factors.append(metrics.average_satisfaction * 0.25)

        # Performance consistency factor (15% weight)
        if len(metrics.recent_successes) >= 5:
            recent_performance = sum(metrics.recent_successes) / len(metrics.recent_successes)
            factors.append(recent_performance * 0.15)

        return sum(factors) if factors else 0.3

    def _calculate_tracking_period(self) -> float:
        """Calculate how long we've been tracking (in days)."""
        if not self.selection_history:
            return 0

        earliest = min(event.timestamp for event in self.selection_history)
        latest = max(event.timestamp for event in self.selection_history)

        return (latest - earliest) / 86400  # Convert to days

    def _load_tracking_data(self):
        """Load tracking data from storage."""
        try:
            if not self.storage_path.exists():
                return

            with open(self.storage_path, "r") as f:
                data = json.load(f)

            # Load selection history
            if "selection_history" in data:
                for event_data in data["selection_history"]:
                    event = SelectionEvent(**event_data)
                    # Restore suggestions
                    if "suggestions" in event_data:
                        event.suggestions = [ProactiveSuggestion(**s) for s in event_data["suggestions"]]
                    self.selection_history.append(event)

            # Load effectiveness metrics
            if "effectiveness_metrics" in data:
                for tool_id, metrics_data in data["effectiveness_metrics"].items():
                    metrics = ToolMetrics(tool_id=tool_id)

                    # Update metrics with loaded data
                    for field_name, value in metrics_data.items():
                        if hasattr(metrics, field_name) and field_name != "tool_id":
                            if field_name.startswith("recent_"):
                                # Handle deque fields
                                deque_field = getattr(metrics, field_name)
                                deque_field.extend(value)
                            else:
                                setattr(metrics, field_name, value)

                    self.effectiveness_metrics[tool_id] = metrics

            logger.info(
                "Loaded tracking data: %d events, %d tools",
                len(self.selection_history),
                len(self.effectiveness_metrics),
            )

        except Exception as e:
            logger.warning("Failed to load tracking data: %s", e)

    def _cleanup_history_if_needed(self):
        """Clean up history when size limits are exceeded."""
        current_size = len(self.selection_history)

        if current_size <= self.max_history_size:
            return  # No cleanup needed

        # Aggressive cleanup when threshold is exceeded
        if current_size >= self.cleanup_threshold:
            # Keep only the most recent events up to max_history_size
            events_to_keep = self.max_history_size
            events_to_remove = current_size - events_to_keep

            logger.info(
                "Cleaning up history: removing %d old events, keeping %d recent events",
                events_to_remove,
                events_to_keep,
            )

            self.selection_history = self.selection_history[-events_to_keep:]

        else:
            # Gentle cleanup - remove a batch of old events
            events_to_remove = min(self.batch_size, current_size - self.max_history_size)

            logger.debug("Gentle cleanup: removing %d old events from history of %d", events_to_remove, current_size)

            self.selection_history = self.selection_history[events_to_remove:]

    def get_memory_usage_stats(self) -> Dict[str, Any]:
        """Get current memory usage statistics."""
        current_size = len(self.selection_history)
        metrics_count = len(self.effectiveness_metrics)

        # Calculate approximate memory usage
        estimated_bytes = current_size * 500  # Rough estimate per event
        estimated_mb = estimated_bytes / (1024 * 1024)

        return {
            "current_history_size": current_size,
            "max_history_size": self.max_history_size,
            "cleanup_threshold": self.cleanup_threshold,
            "total_metrics": metrics_count,
            "memory_utilization_percent": round((current_size / self.max_history_size) * 100, 1),
            "estimated_memory_mb": round(estimated_mb, 2),
            "cleanup_needed": current_size > self.max_history_size,
            "aggressive_cleanup_needed": current_size >= self.cleanup_threshold,
        }

    def _save_tracking_data(self):
        """Save tracking data to storage."""
        # Prepare data for JSON serialization
        data = {
            "selection_history": [],
            "effectiveness_metrics": {},
            "metadata": {"last_saved": time.time(), "version": "1.0", "total_events": len(self.selection_history)},
        }

        # Convert selection history - limit to prevent huge files
        events_to_save = min(500, len(self.selection_history))  # Save last 500 events
        for event in self.selection_history[-events_to_save:]:
            event_dict = asdict(event)
            # Convert suggestions to dicts
            event_dict["suggestions"] = [asdict(s) for s in event.suggestions]
            data["selection_history"].append(event_dict)

        # Convert effectiveness metrics
        for tool_id, metrics in self.effectiveness_metrics.items():
            metrics_dict = asdict(metrics)
            # Convert deques to lists for JSON
            for field_name, field_value in metrics_dict.items():
                if isinstance(field_value, deque):
                    metrics_dict[field_name] = list(field_value)
            data["effectiveness_metrics"][tool_id] = metrics_dict

        # Use storage manager for thread-safe, atomic saves
        if self.storage_manager.save_json_data(self.storage_path, data):
            logger.debug("Saved tracking data to %s", self.storage_path)
        else:
            logger.error("Failed to save tracking data to %s", self.storage_path)
