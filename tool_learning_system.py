#!/usr/bin/env python3
"""
Tool Learning System - Machine Learning and Adaptation

This module implements a lightweight learning system for tool selection improvement
using statistical models and JSON-based persistence. It learns from user preferences,
tool effectiveness, and context patterns to improve automatic suggestions over time.
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from logging_config import get_logger
from storage_manager import get_storage_manager
from usage_pattern_analyzer import UsagePatternAnalysis

logger = get_logger(__name__)


@dataclass
class LearningEvent:
    """A learning event for tool selection."""

    tool_id: str
    context: Dict[str, Any]
    outcome_success: bool
    outcome_metrics: Dict[str, float]
    timestamp: float = field(default_factory=time.time)
    user_feedback: Optional[float] = None  # 0.0 to 1.0 satisfaction


class UserPreferenceModel:
    """Model for learning and predicting user tool preferences."""

    def __init__(self, storage_path: Optional[Path] = None):
        """Initialize preference model with optional storage path."""
        self.storage_manager = get_storage_manager()
        self.storage_path = storage_path or self.storage_manager.get_preferences_path()
        self.preferences_data = self._load_preferences()
        self.context_weights = self._initialize_context_weights()

        logger.debug("Initialized User Preference Model")

    def _load_preferences(self) -> Dict[str, Any]:
        """Load preferences from storage."""
        return self.storage_manager.load_json_data(
            self.storage_path,
            default={
                "tool_preferences": {},
                "context_preferences": {},
                "learning_metadata": {"total_samples": 0, "last_updated": time.time(), "model_version": "1.0"},
            },
        )

    def _save_preferences(self):
        """Save preferences to storage."""
        # Update metadata
        self.preferences_data["learning_metadata"]["last_updated"] = time.time()

        # Use storage manager for thread-safe, atomic saves
        if not self.storage_manager.save_json_data(self.storage_path, self.preferences_data):
            logger.error("Failed to save preferences to %s", self.storage_path)

    def _initialize_context_weights(self) -> Dict[str, float]:
        """Initialize weights for different context factors."""
        return {"task_type": 0.4, "complexity": 0.2, "urgency": 0.2, "user_experience": 0.1, "domain": 0.1}

    def reinforce_preference(self, context: Dict[str, Any], tool: str, strength: float):
        """Reinforce user preference for a tool in given context."""
        try:
            # Clamp strength to valid range
            strength = max(0.0, min(1.0, strength))

            # Update global tool preference
            if tool not in self.preferences_data["tool_preferences"]:
                self.preferences_data["tool_preferences"][tool] = {
                    "total_score": 0.0,
                    "sample_count": 0,
                    "last_reinforcement": time.time(),
                }

            tool_pref = self.preferences_data["tool_preferences"][tool]
            tool_pref["total_score"] += strength
            tool_pref["sample_count"] += 1
            tool_pref["last_reinforcement"] = time.time()

            # Update context-specific preferences
            context_key = self._create_context_key(context)
            if context_key not in self.preferences_data["context_preferences"]:
                self.preferences_data["context_preferences"][context_key] = {}

            if tool not in self.preferences_data["context_preferences"][context_key]:
                self.preferences_data["context_preferences"][context_key][tool] = {
                    "total_score": 0.0,
                    "sample_count": 0,
                }

            context_pref = self.preferences_data["context_preferences"][context_key][tool]
            context_pref["total_score"] += strength
            context_pref["sample_count"] += 1

            # Update metadata
            self.preferences_data["learning_metadata"]["total_samples"] += 1

            # Save periodically
            if self.preferences_data["learning_metadata"]["total_samples"] % 10 == 0:
                self._save_preferences()

            logger.debug("Reinforced preference: tool=%s, strength=%.2f", tool, strength)

        except Exception as e:
            logger.error("Error reinforcing preference: %s", e)

    def predict_preferences(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Predict user preferences for tools in given context."""
        predictions = {}

        try:
            context_key = self._create_context_key(context)

            # Get all tools from global and context preferences
            all_tools = set()
            all_tools.update(self.preferences_data["tool_preferences"].keys())

            if context_key in self.preferences_data["context_preferences"]:
                all_tools.update(self.preferences_data["context_preferences"][context_key].keys())

            # Calculate prediction for each tool
            for tool in all_tools:
                global_score = self._get_global_preference_score(tool)
                context_score = self._get_context_preference_score(tool, context_key)

                # Combine scores with weights
                combined_score = global_score * 0.6 + context_score * 0.4
                predictions[tool] = combined_score

        except Exception as e:
            logger.error("Error predicting preferences: %s", e)

        return predictions

    def get_prediction_confidence(self, context: Dict[str, Any]) -> float:
        """Get confidence level for predictions in given context."""
        try:
            context_key = self._create_context_key(context)
            total_samples = self.preferences_data["learning_metadata"]["total_samples"]

            # Base confidence on overall sample count
            base_confidence = min(0.9, total_samples / 100.0)

            # Adjust for context-specific data
            context_samples = 0
            if context_key in self.preferences_data["context_preferences"]:
                for tool_data in self.preferences_data["context_preferences"][context_key].values():
                    context_samples += tool_data["sample_count"]

            context_confidence = min(0.9, context_samples / 20.0)

            # Combine confidences
            return base_confidence * 0.7 + context_confidence * 0.3

        except Exception as e:
            logger.error("Error calculating prediction confidence: %s", e)
            return 0.3

    def get_accuracy(self) -> float:
        """Get current model accuracy estimate."""
        try:
            total_samples = self.preferences_data["learning_metadata"]["total_samples"]

            # Simple accuracy estimate based on sample count and age
            base_accuracy = min(0.85, 0.5 + (total_samples / 200.0))

            # Adjust for data freshness
            last_updated = self.preferences_data["learning_metadata"].get("last_updated", time.time())
            age_days = (time.time() - last_updated) / 86400
            freshness_factor = max(0.7, 1.0 - (age_days / 30.0))  # Decay over 30 days

            return base_accuracy * freshness_factor

        except Exception as e:
            logger.error("Error calculating accuracy: %s", e)
            return 0.6

    def get_sample_count(self) -> int:
        """Get total number of learning samples."""
        return self.preferences_data["learning_metadata"]["total_samples"]

    def _create_context_key(self, context: Dict[str, Any]) -> str:
        """Create a key for context-based storage."""
        key_components = []

        # Extract key context features
        for key in ["task_type", "complexity", "urgency"]:
            if key in context:
                key_components.append(f"{key}:{context[key]}")

        # Fallback to generic if no recognized context
        if not key_components:
            key_components.append("general")

        return "|".join(sorted(key_components))

    def _get_global_preference_score(self, tool: str) -> float:
        """Get global preference score for a tool."""
        if tool in self.preferences_data["tool_preferences"]:
            tool_data = self.preferences_data["tool_preferences"][tool]
            if tool_data["sample_count"] > 0:
                return tool_data["total_score"] / tool_data["sample_count"]

        return 0.5  # Default neutral preference

    def _get_context_preference_score(self, tool: str, context_key: str) -> float:
        """Get context-specific preference score for a tool."""
        if (
            context_key in self.preferences_data["context_preferences"]
            and tool in self.preferences_data["context_preferences"][context_key]
        ):
            tool_data = self.preferences_data["context_preferences"][context_key][tool]
            if tool_data["sample_count"] > 0:
                return tool_data["total_score"] / tool_data["sample_count"]

        return 0.5  # Default neutral preference


class ToolEffectivenessModel:
    """Model for tracking and predicting tool effectiveness."""

    def __init__(self, storage_path: Optional[Path] = None):
        """Initialize effectiveness model with optional storage path."""
        self.storage_manager = get_storage_manager()
        self.storage_path = storage_path or self.storage_manager.get_effectiveness_data_path()
        self.effectiveness_data = self._load_effectiveness()

        logger.debug("Initialized Tool Effectiveness Model")

    def _load_effectiveness(self) -> Dict[str, Any]:
        """Load effectiveness data from storage."""
        return self.storage_manager.load_json_data(
            self.storage_path,
            default={
                "tool_effectiveness": {},
                "context_effectiveness": {},
                "learning_metadata": {"total_updates": 0, "last_updated": time.time(), "model_version": "1.0"},
            },
        )

    def _save_effectiveness(self):
        """Save effectiveness data to storage."""
        # Update metadata
        self.effectiveness_data["learning_metadata"]["last_updated"] = time.time()

        # Use storage manager for thread-safe, atomic saves
        if not self.storage_manager.save_json_data(self.storage_path, self.effectiveness_data):
            logger.error("Failed to save effectiveness data to %s", self.storage_path)

    def update_effectiveness(self, tool_id: str, context: Dict[str, Any], success: bool, metrics: Dict[str, float]):
        """Update tool effectiveness based on outcome."""
        try:
            # Update global effectiveness
            if tool_id not in self.effectiveness_data["tool_effectiveness"]:
                self.effectiveness_data["tool_effectiveness"][tool_id] = {
                    "total_uses": 0,
                    "successful_uses": 0,
                    "total_satisfaction": 0.0,
                    "completion_times": [],
                    "last_updated": time.time(),
                }

            tool_data = self.effectiveness_data["tool_effectiveness"][tool_id]
            tool_data["total_uses"] += 1
            tool_data["last_updated"] = time.time()

            if success:
                tool_data["successful_uses"] += 1

            # Add satisfaction score
            satisfaction = metrics.get("satisfaction", 0.8 if success else 0.2)
            tool_data["total_satisfaction"] += satisfaction

            # Track completion time
            completion_time = metrics.get("completion_time", 0)
            if completion_time > 0:
                tool_data["completion_times"].append(completion_time)
                # Keep only recent times
                if len(tool_data["completion_times"]) > 50:
                    tool_data["completion_times"] = tool_data["completion_times"][-50:]

            # Update context-specific effectiveness
            context_key = self._create_context_key(context)
            if context_key not in self.effectiveness_data["context_effectiveness"]:
                self.effectiveness_data["context_effectiveness"][context_key] = {}

            if tool_id not in self.effectiveness_data["context_effectiveness"][context_key]:
                self.effectiveness_data["context_effectiveness"][context_key][tool_id] = {
                    "total_uses": 0,
                    "successful_uses": 0,
                    "avg_satisfaction": 0.0,
                }

            context_tool_data = self.effectiveness_data["context_effectiveness"][context_key][tool_id]
            context_tool_data["total_uses"] += 1

            if success:
                context_tool_data["successful_uses"] += 1

            # Update average satisfaction
            current_avg = context_tool_data["avg_satisfaction"]
            new_sample_count = context_tool_data["total_uses"]
            context_tool_data["avg_satisfaction"] = (
                current_avg * (new_sample_count - 1) + satisfaction
            ) / new_sample_count

            # Update metadata
            self.effectiveness_data["learning_metadata"]["total_updates"] += 1

            # Save periodically
            if self.effectiveness_data["learning_metadata"]["total_updates"] % 15 == 0:
                self._save_effectiveness()

            logger.debug("Updated effectiveness: tool=%s, success=%s", tool_id, success)

        except Exception as e:
            logger.error("Error updating effectiveness: %s", e)

    def get_tool_effectiveness(self, tool_id: str, context: Dict[str, Any] = None) -> float:
        """Get effectiveness score for a tool in given context."""
        try:
            # Get global effectiveness
            global_effectiveness = 0.5  # Default

            if tool_id in self.effectiveness_data["tool_effectiveness"]:
                tool_data = self.effectiveness_data["tool_effectiveness"][tool_id]
                if tool_data["total_uses"] > 0:
                    success_rate = tool_data["successful_uses"] / tool_data["total_uses"]
                    avg_satisfaction = tool_data["total_satisfaction"] / tool_data["total_uses"]
                    global_effectiveness = success_rate * 0.6 + avg_satisfaction * 0.4

            # Get context-specific effectiveness if context provided
            if context:
                context_key = self._create_context_key(context)
                if (
                    context_key in self.effectiveness_data["context_effectiveness"]
                    and tool_id in self.effectiveness_data["context_effectiveness"][context_key]
                ):
                    context_data = self.effectiveness_data["context_effectiveness"][context_key][tool_id]
                    if context_data["total_uses"] > 0:
                        context_success_rate = context_data["successful_uses"] / context_data["total_uses"]
                        context_satisfaction = context_data["avg_satisfaction"]
                        context_effectiveness = context_success_rate * 0.6 + context_satisfaction * 0.4

                        # Combine global and context effectiveness
                        # Weight context more heavily if we have enough data
                        context_weight = min(0.7, context_data["total_uses"] / 10.0)
                        return global_effectiveness * (1 - context_weight) + context_effectiveness * context_weight

            return global_effectiveness

        except Exception as e:
            logger.error("Error getting tool effectiveness: %s", e)
            return 0.5

    def get_accuracy(self) -> float:
        """Get current model accuracy estimate."""
        try:
            total_updates = self.effectiveness_data["learning_metadata"]["total_updates"]

            # Base accuracy on number of updates
            base_accuracy = min(0.80, 0.4 + (total_updates / 150.0))

            # Adjust for data recency
            last_updated = self.effectiveness_data["learning_metadata"].get("last_updated", time.time())
            age_days = (time.time() - last_updated) / 86400
            freshness_factor = max(0.6, 1.0 - (age_days / 14.0))  # Decay over 2 weeks

            return base_accuracy * freshness_factor

        except Exception as e:
            logger.error("Error calculating effectiveness accuracy: %s", e)
            return 0.5

    def _create_context_key(self, context: Dict[str, Any]) -> str:
        """Create a key for context-based storage."""
        key_components = []

        # Extract key context features
        for key in ["task_type", "complexity_level", "urgency_level"]:
            if key in context:
                key_components.append(f"{key}:{context[key]}")

        if not key_components:
            key_components.append("general")

        return "|".join(sorted(key_components))


class ContextPatternModel:
    """Model for learning context patterns and their outcomes."""

    def __init__(self, storage_path: Optional[Path] = None):
        """Initialize context pattern model."""
        self.storage_manager = get_storage_manager()
        self.storage_path = storage_path or self.storage_manager.get_custom_file_path("context_patterns.json")
        self.pattern_data = self._load_patterns()

        logger.debug("Initialized Context Pattern Model")

    def _load_patterns(self) -> Dict[str, Any]:
        """Load pattern data from storage."""
        return self.storage_manager.load_json_data(
            self.storage_path,
            default={
                "context_patterns": {},
                "pattern_outcomes": {},
                "learning_metadata": {"total_patterns": 0, "last_updated": time.time()},
            },
        )

    def update_context_patterns(self, context: Dict[str, Any], tool: str, outcome: bool):
        """Update context patterns with outcome data."""
        try:
            pattern_key = self._extract_pattern_key(context)

            # Update pattern frequency
            if pattern_key not in self.pattern_data["context_patterns"]:
                self.pattern_data["context_patterns"][pattern_key] = {
                    "frequency": 0,
                    "tools_used": {},
                    "success_rate": 0.0,
                    "last_seen": time.time(),
                }

            pattern = self.pattern_data["context_patterns"][pattern_key]
            pattern["frequency"] += 1
            pattern["last_seen"] = time.time()

            # Update tool usage for this pattern
            if tool not in pattern["tools_used"]:
                pattern["tools_used"][tool] = {"uses": 0, "successes": 0}

            tool_stats = pattern["tools_used"][tool]
            tool_stats["uses"] += 1
            if outcome:
                tool_stats["successes"] += 1

            # Update overall success rate
            total_successes = sum(t["successes"] for t in pattern["tools_used"].values())
            total_uses = sum(t["uses"] for t in pattern["tools_used"].values())
            pattern["success_rate"] = total_successes / total_uses if total_uses > 0 else 0.0

            self.pattern_data["learning_metadata"]["total_patterns"] = len(self.pattern_data["context_patterns"])
            self.pattern_data["learning_metadata"]["last_updated"] = time.time()

        except Exception as e:
            logger.error("Error updating context patterns: %s", e)

    def get_pattern_recommendations(self, context: Dict[str, Any]) -> List[Tuple[str, float]]:
        """Get tool recommendations based on context patterns."""
        recommendations = []

        try:
            pattern_key = self._extract_pattern_key(context)

            if pattern_key in self.pattern_data["context_patterns"]:
                pattern = self.pattern_data["context_patterns"][pattern_key]

                # Calculate recommendation scores for each tool
                for tool, stats in pattern["tools_used"].items():
                    if stats["uses"] >= 2:  # Need minimum usage
                        success_rate = stats["successes"] / stats["uses"]
                        # Weight by usage frequency
                        confidence = min(0.9, stats["uses"] / 10.0)
                        score = success_rate * confidence
                        recommendations.append((tool, score))

            # Sort by score
            recommendations.sort(key=lambda x: x[1], reverse=True)

        except Exception as e:
            logger.error("Error getting pattern recommendations: %s", e)

        return recommendations

    def _extract_pattern_key(self, context: Dict[str, Any]) -> str:
        """Extract a pattern key from context."""
        # Create pattern from key context features
        key_features = []

        for feature in ["task_type", "urgency_level", "complexity_level"]:
            if feature in context:
                key_features.append(f"{feature}={context[feature]}")

        if not key_features:
            key_features.append("general")

        return "|".join(sorted(key_features))


class ToolLearningSystem:
    """Machine learning system for tool selection improvement."""

    def __init__(self, storage_dir: Optional[Path] = None):
        """Initialize learning system with component models."""
        if storage_dir is None:
            storage_dir = Path(".turboprop")

        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.preference_model = UserPreferenceModel(storage_dir / "preferences.json")
        self.effectiveness_model = ToolEffectivenessModel(storage_dir / "effectiveness.json")
        self.context_model = ContextPatternModel(storage_dir / "context_patterns.json")

        logger.info("Initialized Tool Learning System with storage at: %s", storage_dir)

    def learn_from_selection_outcome(
        self,
        selection_context: Dict[str, Any],
        tool_chosen: str,
        outcome_success: bool,
        outcome_metrics: Dict[str, float],
    ) -> None:
        """Learn from tool selection outcomes."""

        try:
            logger.debug("Learning from outcome: tool=%s, success=%s", tool_chosen, outcome_success)

            # Update effectiveness model
            self.effectiveness_model.update_effectiveness(
                tool_id=tool_chosen, context=selection_context, success=outcome_success, metrics=outcome_metrics
            )

            # Update preference model if outcome was successful
            if outcome_success:
                satisfaction = outcome_metrics.get("satisfaction", 0.8)
                self.preference_model.reinforce_preference(
                    context=selection_context, tool=tool_chosen, strength=satisfaction
                )

            # Update context patterns
            self.context_model.update_context_patterns(
                context=selection_context, tool=tool_chosen, outcome=outcome_success
            )

        except Exception as e:
            logger.error("Error learning from selection outcome: %s", e)

    def get_learned_preferences(self, context: Dict[str, Any], user_patterns: UsagePatternAnalysis) -> Dict[str, Any]:
        """Get learned user preferences for context."""

        try:
            # Get preference predictions
            preferences = self.preference_model.predict_preferences(context)

            # Adjust based on usage patterns
            pattern_adjustments = self._calculate_pattern_adjustments(user_patterns)
            adjusted_preferences = self._apply_adjustments(preferences, pattern_adjustments)

            return {
                "tool_preferences": adjusted_preferences,
                "confidence": self.preference_model.get_prediction_confidence(context),
                "learning_status": self._get_learning_status(),
            }

        except Exception as e:
            logger.error("Error getting learned preferences: %s", e)
            return {"tool_preferences": {}, "confidence": 0.3, "learning_status": {"error": str(e)}}

    def get_context_preferences(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Get context-specific preferences."""
        try:
            # Get pattern-based recommendations
            pattern_recommendations = self.context_model.get_pattern_recommendations(context)

            # Convert to preference dictionary
            preferences = {}
            for tool, score in pattern_recommendations:
                preferences[tool] = score

            return preferences

        except Exception as e:
            logger.error("Error getting context preferences: %s", e)
            return {}

    def _calculate_pattern_adjustments(self, user_patterns: UsagePatternAnalysis) -> Dict[str, float]:
        """Calculate adjustments based on usage patterns."""
        adjustments = {}

        try:
            # Reduce scores for tools showing inefficiency patterns
            for inefficiency in user_patterns.inefficiency_patterns:
                if inefficiency.type == "suboptimal_choice":
                    # Extract tool from description (simplified)
                    description = inefficiency.description.lower()
                    for word in description.split():
                        if "tool" in word or len(word) > 4:
                            adjustments[word] = -0.2  # Penalty

            # Boost scores for tools in successful patterns
            for pattern in user_patterns.recent_patterns:
                if pattern.pattern_type == "efficiency" and pattern.confidence > 0.7:
                    # Extract tool names from pattern context
                    if "tool" in pattern.context:
                        tool = pattern.context["tool"]
                        adjustments[tool] = adjustments.get(tool, 0) + 0.1

        except Exception as e:
            logger.warning("Error calculating pattern adjustments: %s", e)

        return adjustments

    def _apply_adjustments(self, preferences: Dict[str, float], adjustments: Dict[str, float]) -> Dict[str, float]:
        """Apply adjustments to preferences."""
        adjusted = preferences.copy()

        for tool, adjustment in adjustments.items():
            if tool in adjusted:
                adjusted[tool] = max(0.0, min(1.0, adjusted[tool] + adjustment))

        return adjusted

    def _get_learning_status(self) -> Dict[str, Any]:
        """Get status of learning models."""
        try:
            return {
                "preference_model_accuracy": self.preference_model.get_accuracy(),
                "effectiveness_model_accuracy": self.effectiveness_model.get_accuracy(),
                "total_learning_samples": self.preference_model.get_sample_count(),
                "learning_confidence": self._calculate_overall_confidence(),
            }
        except Exception as e:
            logger.error("Error getting learning status: %s", e)
            return {"error": str(e)}

    def _calculate_overall_confidence(self) -> float:
        """Calculate overall learning confidence."""
        try:
            pref_accuracy = self.preference_model.get_accuracy()
            eff_accuracy = self.effectiveness_model.get_accuracy()
            sample_count = self.preference_model.get_sample_count()

            # Combine factors
            base_confidence = (pref_accuracy + eff_accuracy) / 2
            sample_factor = min(1.0, sample_count / 100.0)

            return base_confidence * sample_factor

        except Exception as e:
            logger.error("Error calculating overall confidence: %s", e)
            return 0.4
