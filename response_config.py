#!/usr/bin/env python3
"""
response_config.py: Configuration constants for MCP response types.

This module contains configurable constants for response formatting,
clustering, and suggestion generation to avoid hard-coded values.
"""

# JSON formatting
JSON_INDENT = 2

# Response versioning
RESPONSE_VERSION = "1.0"

# Result clustering thresholds
MIN_CLUSTER_SIZE = 2  # Minimum results needed to form a cluster
MAX_CLUSTERS = 10  # Maximum number of clusters to generate

# Query suggestion thresholds
FEW_RESULTS_THRESHOLD = 3  # Below this considered "few results"
MANY_RESULTS_THRESHOLD = 15  # Above this considered "many results"
MAX_SUGGESTIONS = 5  # Maximum suggestions to return
MAX_NAVIGATION_HINTS = 5  # Maximum navigation hints to return

# Cross-reference limits
MAX_CROSS_REFERENCES = 5  # Maximum cross-references to generate
MAX_FILENAMES_IN_CROSS_REF = 3  # Max filenames to show in cross-reference

# Directory grouping thresholds
SAME_DIRECTORY_THRESHOLD = 1  # Exactly 1 directory = "same directory"
MULTIPLE_DIRECTORY_THRESHOLD = 3  # Up to 3 directories = "multiple related"

# Health score constants
BASE_HEALTH_SCORE = 100.0
MISSING_EMBEDDINGS_PENALTY = 30  # Points deducted per missing embedding ratio
STALE_INDEX_PENALTY = 20  # Points deducted for stale index
WARNING_PENALTY = 5  # Points deducted per warning
NOT_READY_PENALTY = 30  # Points deducted if not ready for search
MIN_HEALTH_SCORE = 0.0  # Minimum possible health score

# Confidence distribution
DEFAULT_CONFIDENCE_LEVELS = ['high', 'medium', 'low']

# Language breakdown
UNKNOWN_LANGUAGE = 'unknown'