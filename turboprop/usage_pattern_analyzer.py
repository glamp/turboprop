#!/usr/bin/env python3
"""
Usage Pattern Analyzer - Real-time Pattern Recognition and Analysis

This module analyzes Claude Code's actual tool usage patterns over time,
detects inefficiencies, and identifies common task sequences for intelligent
tool selection suggestions.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class DetectedPattern:
    """Detected usage pattern."""

    pattern_type: str  # 'sequence', 'repetitive', 'error', 'efficiency'
    description: str
    frequency: int
    confidence: float
    context: Dict[str, Any]
    suggested_improvement: Optional[str] = None


@dataclass
class InefficiencyPattern:
    """Detected inefficiency pattern."""

    type: str
    description: str
    suggestion: str
    confidence: float
    impact_estimate: str = "medium"  # 'low', 'medium', 'high'


@dataclass
class UsagePatternAnalysis:
    """Result of usage pattern analysis."""

    context: Dict[str, Any]
    recent_patterns: List[DetectedPattern]
    task_patterns: List[DetectedPattern]
    inefficiency_patterns: List[InefficiencyPattern]
    session_summary: Dict[str, Any]

    # Analysis metadata
    analysis_timestamp: float = field(default_factory=time.time)
    confidence_level: float = 0.5


class SessionTracker:
    """Tracks current session data for pattern analysis."""

    def __init__(self):
        self.session_data = {
            "start_time": time.time(),
            "tool_usage": [],
            "errors": [],
            "successes": [],
            "context_switches": [],
        }

        logger.debug("Initialized session tracker")

    def get_current_session(self) -> Dict[str, Any]:
        """Get current session data."""
        # Add current timestamp
        self.session_data["current_time"] = time.time()
        self.session_data["session_duration"] = self.session_data["current_time"] - self.session_data["start_time"]

        return self.session_data.copy()

    def record_tool_usage(self, tool_name: str, success: bool, context: Dict[str, Any]):
        """Record a tool usage event."""
        usage_event = {
            "tool": tool_name,
            "timestamp": time.time(),
            "success": success,
            "context": context.copy(),
            "duration": context.get("execution_time", 0),
        }

        self.session_data["tool_usage"].append(usage_event)

        if success:
            self.session_data["successes"].append(usage_event)
        else:
            self.session_data["errors"].append(usage_event)

        logger.debug("Recorded tool usage: %s (success: %s)", tool_name, success)


class PatternDetector:
    """Base class for different pattern detection algorithms."""

    def __init__(self, pattern_type: str):
        self.pattern_type = pattern_type

    def detect_patterns(self, session_data: Dict[str, Any]) -> List[DetectedPattern]:
        """Detect patterns in session data. Override in subclasses."""
        return []


class SequencePatternDetector(PatternDetector):
    """Detects tool sequence patterns."""

    def __init__(self):
        super().__init__("sequence")

    def detect_patterns(self, session_data: Dict[str, Any]) -> List[DetectedPattern]:
        """Detect tool sequence patterns."""
        patterns = []
        tool_usage = session_data.get("tool_usage", [])

        if len(tool_usage) < 3:
            return patterns

        # Look for common 3-tool sequences
        sequences = {}
        for i in range(len(tool_usage) - 2):
            sequence = (tool_usage[i]["tool"], tool_usage[i + 1]["tool"], tool_usage[i + 2]["tool"])
            sequences[sequence] = sequences.get(sequence, 0) + 1

        # Identify frequent sequences
        for sequence, frequency in sequences.items():
            if frequency >= 2:  # Appeared at least twice
                patterns.append(
                    DetectedPattern(
                        pattern_type="sequence",
                        description=f"Common tool sequence: {' -> '.join(sequence)}",
                        frequency=frequency,
                        confidence=min(0.9, frequency * 0.3),
                        context={"sequence": sequence},
                        suggested_improvement="Consider creating a workflow for this sequence",
                    )
                )

        return patterns


class RepetitivePatternDetector(PatternDetector):
    """Detects repetitive tool usage patterns."""

    def __init__(self):
        super().__init__("repetitive")

    def detect_patterns(self, session_data: Dict[str, Any]) -> List[DetectedPattern]:
        """Detect repetitive patterns."""
        patterns = []
        tool_usage = session_data.get("tool_usage", [])

        if len(tool_usage) < 5:
            return patterns

        # Count tool usage frequency in recent history
        recent_tools = [usage["tool"] for usage in tool_usage[-10:]]
        tool_counts = {}
        for tool in recent_tools:
            tool_counts[tool] = tool_counts.get(tool, 0) + 1

        # Identify overly repetitive usage
        for tool, count in tool_counts.items():
            if count >= 4:  # Used 4+ times in last 10 actions
                patterns.append(
                    DetectedPattern(
                        pattern_type="repetitive",
                        description=f"Excessive use of {tool} ({count} times recently)",
                        frequency=count,
                        confidence=0.8,
                        context={"tool": tool, "count": count},
                        suggested_improvement="Consider alternative tools or batch operations",
                    )
                )

        return patterns


class ErrorPatternDetector(PatternDetector):
    """Detects error patterns in tool usage."""

    def __init__(self):
        super().__init__("error")

    def detect_patterns(self, session_data: Dict[str, Any]) -> List[DetectedPattern]:
        """Detect error patterns."""
        patterns = []
        errors = session_data.get("errors", [])

        if len(errors) < 2:
            return patterns

        # Group errors by tool
        error_by_tool = {}
        for error in errors:
            tool = error["tool"]
            if tool not in error_by_tool:
                error_by_tool[tool] = []
            error_by_tool[tool].append(error)

        # Identify tools with high error rates
        for tool, tool_errors in error_by_tool.items():
            if len(tool_errors) >= 3:
                patterns.append(
                    DetectedPattern(
                        pattern_type="error",
                        description=f"High error rate with {tool} ({len(tool_errors)} failures)",
                        frequency=len(tool_errors),
                        confidence=0.9,
                        context={"tool": tool, "errors": tool_errors},
                        suggested_improvement=f"Consider alternative to {tool} or check parameters",
                    )
                )

        return patterns


class UsagePatternAnalyzer:
    """Analyze Claude Code usage patterns for intelligent suggestions."""

    def __init__(self):
        """Initialize pattern detectors and session tracker."""
        self.pattern_detectors = self._initialize_pattern_detectors()
        self.session_tracker = SessionTracker()

        logger.info("Initialized Usage Pattern Analyzer with %d detectors", len(self.pattern_detectors))

    def _initialize_pattern_detectors(self) -> List[PatternDetector]:
        """Initialize all pattern detectors."""
        return [SequencePatternDetector(), RepetitivePatternDetector(), ErrorPatternDetector()]

    def analyze_current_session(
        self, context: Dict[str, Any], task: Optional[str] = None, history: Optional[List[Dict]] = None
    ) -> UsagePatternAnalysis:
        """Analyze current session for usage patterns."""

        try:
            logger.debug("Analyzing session patterns for context: %s", context)

            # Track current session data
            session_data = self.session_tracker.get_current_session()

            # Incorporate external history if provided
            if history:
                self._incorporate_history(session_data, history)

            # Detect patterns in recent tool usage
            recent_patterns = self._detect_recent_patterns(session_data, history)

            # Analyze task patterns if task is provided
            task_patterns = []
            if task:
                task_patterns = self._analyze_task_patterns(task, history)

            # Identify inefficiency patterns
            inefficiency_patterns = self._detect_inefficiencies(session_data)

            # Create session summary
            session_summary = self._create_session_summary(session_data)

            analysis = UsagePatternAnalysis(
                context=context,
                recent_patterns=recent_patterns,
                task_patterns=task_patterns,
                inefficiency_patterns=inefficiency_patterns,
                session_summary=session_summary,
                confidence_level=self._calculate_analysis_confidence(session_data),
            )

            logger.info(
                "Pattern analysis complete: %d recent patterns, %d inefficiencies",
                len(recent_patterns),
                len(inefficiency_patterns),
            )

            return analysis

        except Exception as e:
            logger.error("Error in analyze_current_session: %s", e)
            # Return empty analysis on error
            return UsagePatternAnalysis(
                context=context,
                recent_patterns=[],
                task_patterns=[],
                inefficiency_patterns=[],
                session_summary={"error": str(e)},
            )

    def _incorporate_history(self, session_data: Dict[str, Any], history: List[Dict]):
        """Incorporate external history into session data."""
        try:
            for event in history[-20:]:  # Last 20 events
                if "tool" in event and "success" in event:
                    usage_event = {
                        "tool": event["tool"],
                        "timestamp": event.get("timestamp", time.time()),
                        "success": event["success"],
                        "context": event.get("context", {}),
                        "duration": event.get("duration", 0),
                    }
                    session_data["tool_usage"].append(usage_event)

        except Exception as e:
            logger.warning("Error incorporating history: %s", e)

    def _detect_recent_patterns(
        self, session_data: Dict[str, Any], history: Optional[List[Dict]]
    ) -> List[DetectedPattern]:
        """Detect patterns in recent tool usage."""
        all_patterns = []

        try:
            # Run all pattern detectors
            for detector in self.pattern_detectors:
                try:
                    patterns = detector.detect_patterns(session_data)
                    all_patterns.extend(patterns)
                except Exception as e:
                    logger.warning("Pattern detector %s failed: %s", detector.pattern_type, e)

        except Exception as e:
            logger.error("Error in _detect_recent_patterns: %s", e)

        return all_patterns

    def _analyze_task_patterns(self, task: str, history: Optional[List[Dict]]) -> List[DetectedPattern]:
        """Analyze patterns specific to the given task."""
        patterns = []

        try:
            # Simple task-based pattern detection
            task_lower = task.lower()

            if "search" in task_lower:
                patterns.append(
                    DetectedPattern(
                        pattern_type="task",
                        description="Search-oriented task detected",
                        frequency=1,
                        confidence=0.8,
                        context={"task_type": "search"},
                        suggested_improvement="Consider using semantic search tools",
                    )
                )
            elif "file" in task_lower:
                patterns.append(
                    DetectedPattern(
                        pattern_type="task",
                        description="File operation task detected",
                        frequency=1,
                        confidence=0.8,
                        context={"task_type": "file_ops"},
                        suggested_improvement="Use batch file operations when possible",
                    )
                )

        except Exception as e:
            logger.warning("Error in _analyze_task_patterns: %s", e)

        return patterns

    def _detect_inefficiencies(self, session_data: Dict[str, Any]) -> List[InefficiencyPattern]:
        """Detect inefficient tool usage patterns."""
        inefficiencies = []

        try:
            tool_usage = session_data.get("tool_usage", [])

            # Detect suboptimal tool choices
            for usage in tool_usage[-10:]:  # Check recent usage
                if self._is_suboptimal_choice(usage):
                    better_alternative = self._get_better_alternative(usage)
                    inefficiencies.append(
                        InefficiencyPattern(
                            type="suboptimal_choice",
                            description=f"Tool '{usage['tool']}' may not be optimal for this task",
                            suggestion=better_alternative,
                            confidence=0.7,
                            impact_estimate="medium",
                        )
                    )

            # Detect excessive trial-and-error
            error_sequences = self._find_error_sequences(session_data)
            for sequence in error_sequences:
                if len(sequence) >= 3:
                    inefficiencies.append(
                        InefficiencyPattern(
                            type="excessive_trial_error",
                            description="Multiple failed attempts detected",
                            suggestion="Consider analyzing requirements before tool selection",
                            confidence=0.9,
                            impact_estimate="high",
                        )
                    )

        except Exception as e:
            logger.error("Error in _detect_inefficiencies: %s", e)

        return inefficiencies

    def _is_suboptimal_choice(self, usage: Dict[str, Any]) -> bool:
        """Determine if a tool usage was suboptimal."""
        # Simple heuristics for suboptimal choices
        tool = usage["tool"].lower()
        context = usage.get("context", {})
        task_type = str(context.get("task", "")).lower()

        # Example suboptimal patterns
        if "search" in task_type and "read" in tool:
            return True  # Using read for search tasks
        if "simple" in task_type and "complex" in tool:
            return True  # Using complex tool for simple tasks
        if usage.get("duration", 0) > 30 and "quick" in task_type:
            return True  # Tool took too long for quick task

        return False

    def _get_better_alternative(self, usage: Dict[str, Any]) -> str:
        """Get a better tool alternative for suboptimal usage."""
        tool = usage["tool"].lower()
        context = usage.get("context", {})
        task_type = str(context.get("task", "")).lower()

        # Simple alternative mapping
        if "search" in task_type:
            if "read" in tool:
                return "search_files"
            if "grep" in tool:
                return "semantic_search"

        if "file" in task_type:
            if "search" in tool:
                return "read_file"

        return "analyze_requirements"  # Default suggestion

    def _find_error_sequences(self, session_data: Dict[str, Any]) -> List[List[Dict[str, Any]]]:
        """Find sequences of consecutive errors."""
        sequences = []
        current_sequence = []

        for usage in session_data.get("tool_usage", []):
            if not usage.get("success", True):
                current_sequence.append(usage)
            else:
                if len(current_sequence) >= 2:
                    sequences.append(current_sequence.copy())
                current_sequence = []

        # Add final sequence if it exists
        if len(current_sequence) >= 2:
            sequences.append(current_sequence)

        return sequences

    def _create_session_summary(self, session_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summary of the current session."""
        tool_usage = session_data.get("tool_usage", [])
        errors = session_data.get("errors", [])
        successes = session_data.get("successes", [])

        summary = {
            "total_actions": len(tool_usage),
            "successful_actions": len(successes),
            "failed_actions": len(errors),
            "success_rate": len(successes) / max(1, len(tool_usage)),
            "session_duration": session_data.get("session_duration", 0),
            "most_used_tools": self._get_most_used_tools(tool_usage),
            "error_prone_tools": self._get_error_prone_tools(tool_usage),
        }

        return summary

    def _get_most_used_tools(self, tool_usage: List[Dict[str, Any]]) -> List[str]:
        """Get the most frequently used tools."""
        tool_counts = {}
        for usage in tool_usage:
            tool = usage["tool"]
            tool_counts[tool] = tool_counts.get(tool, 0) + 1

        # Sort by usage count and return top 5
        sorted_tools = sorted(tool_counts.items(), key=lambda x: x[1], reverse=True)
        return [tool for tool, count in sorted_tools[:5]]

    def _get_error_prone_tools(self, tool_usage: List[Dict[str, Any]]) -> List[str]:
        """Get tools with high error rates."""
        tool_stats = {}

        for usage in tool_usage:
            tool = usage["tool"]
            if tool not in tool_stats:
                tool_stats[tool] = {"total": 0, "errors": 0}

            tool_stats[tool]["total"] += 1
            if not usage.get("success", True):
                tool_stats[tool]["errors"] += 1

        # Find tools with >50% error rate and at least 2 uses
        error_prone = []
        for tool, stats in tool_stats.items():
            if stats["total"] >= 2:
                error_rate = stats["errors"] / stats["total"]
                if error_rate > 0.5:
                    error_prone.append(tool)

        return error_prone

    def _calculate_analysis_confidence(self, session_data: Dict[str, Any]) -> float:
        """Calculate confidence level for the analysis."""
        tool_usage_count = len(session_data.get("tool_usage", []))
        session_duration = session_data.get("session_duration", 0)

        # Base confidence on data availability
        confidence = 0.3  # Base confidence

        # Increase confidence based on usage data
        if tool_usage_count > 10:
            confidence += 0.3
        elif tool_usage_count > 5:
            confidence += 0.2
        elif tool_usage_count > 2:
            confidence += 0.1

        # Increase confidence for longer sessions
        if session_duration > 600:  # 10 minutes
            confidence += 0.2
        elif session_duration > 300:  # 5 minutes
            confidence += 0.1

        return min(1.0, confidence)
