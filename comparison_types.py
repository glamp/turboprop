#!/usr/bin/env python3
"""
comparison_types.py: Shared data types for tool comparison system.

This module contains the core data types shared between comparison components
to avoid circular import dependencies.
"""

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

from decision_support import SelectionGuidance, TradeOffAnalysis


@dataclass
class ToolComparisonResult:
    """Complete tool comparison analysis result."""

    compared_tools: List[str]
    comparison_matrix: Dict[str, Dict[str, float]]  # tool_id -> metric -> score

    # Rankings
    overall_ranking: List[str]  # tool_ids ordered by overall score
    category_rankings: Dict[str, List[str]]  # category -> ranked tool_ids

    # Analysis
    key_differentiators: List[str]
    trade_off_analysis: List[TradeOffAnalysis]
    decision_guidance: Optional[SelectionGuidance]

    # Metadata
    comparison_criteria: List[str]
    context_factors: List[str]
    confidence_scores: Dict[str, float]  # tool_id -> confidence

    # Performance metadata
    execution_time: Optional[float] = None
    tools_analyzed: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)

        # Convert trade-off analysis to dict format
        result["trade_off_analysis"] = [ta.to_dict() for ta in self.trade_off_analysis]

        # Convert decision guidance to dict format if present
        if self.decision_guidance:
            result["decision_guidance"] = self.decision_guidance.to_dict()

        return result


@dataclass
class TaskComparisonResult:
    """Tool comparison result specific to a task."""

    task_description: str
    candidate_tools: List[str]
    comparison_matrix: Dict[str, Dict[str, float]]
    task_specific_ranking: List[str]
    task_fit_scores: Dict[str, float]  # tool_id -> task fit score
    recommendation_confidence: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class DetailedComparison:
    """Detailed head-to-head comparison of two tools."""

    tool_a: str
    tool_b: str

    # Similarity analysis
    similarities: List[str]
    differences: List[str]

    # Advantage analysis
    tool_a_advantages: List[str]
    tool_b_advantages: List[str]

    # Use case analysis
    use_case_scenarios: Dict[str, List[str]]  # tool -> scenarios where it excels
    switching_guidance: List[str]

    # Metadata
    confidence: float
    analysis_depth: str = "standard"  # standard, detailed, comprehensive

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
