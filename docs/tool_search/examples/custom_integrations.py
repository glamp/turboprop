#!/usr/bin/env python3
"""
Custom Integration Examples for MCP Tool Search System

This file demonstrates how to create custom integrations with the tool search system,
including custom search algorithms, specialized tool categories, and integration patterns.
"""

import asyncio
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, Tuple


# Custom integration interfaces
class SearchAlgorithm(Protocol):
    """Protocol for custom search algorithms."""

    async def search(self, query: str, tools: List[Dict], **kwargs) -> List[Dict]:
        """Execute custom search logic."""
        ...


class ToolFilter(Protocol):
    """Protocol for custom tool filtering."""

    def filter(self, tools: List[Dict], context: Dict[str, Any]) -> List[Dict]:
        """Filter tools based on custom criteria."""
        ...


class ScoreAdjuster(Protocol):
    """Protocol for custom score adjustment."""

    def adjust_score(self, tool: Dict, base_score: float, context: Dict) -> float:
        """Adjust tool score based on custom logic."""
        ...


# Example 1: Domain-Specific Search Algorithm
class DomainSpecificSearchAlgorithm:
    """Custom search algorithm for domain-specific tool discovery."""

    def __init__(self, domain_weights: Dict[str, float]):
        self.domain_weights = domain_weights
        self.domain_keywords = {
            "data_science": ["numpy", "pandas", "sklearn", "analysis", "statistics"],
            "web_development": ["http", "api", "server", "client", "web"],
            "devops": ["deploy", "container", "kubernetes", "monitoring", "cicd"],
            "security": ["auth", "encrypt", "secure", "validation", "permission"],
        }

    async def search(self, query: str, tools: List[Dict], domain_context: Optional[str] = None, **kwargs) -> List[Dict]:
        """Execute domain-aware search."""
        print(f"=== Domain-Specific Search for: {query} ===")

        # Detect domain from query if not provided
        if not domain_context:
            domain_context = self._detect_domain(query)

        print(f"Detected domain: {domain_context}")

        # Score tools based on domain relevance
        scored_tools = []
        for tool in tools:
            base_score = await self._calculate_base_similarity(query, tool)
            domain_score = self._calculate_domain_score(tool, domain_context)

            # Combine scores with domain weighting
            domain_weight = self.domain_weights.get(domain_context, 1.0)
            final_score = (base_score * 0.7) + (domain_score * 0.3 * domain_weight)

            tool_result = tool.copy()
            tool_result["similarity_score"] = final_score
            tool_result["domain_relevance"] = domain_score
            tool_result["detected_domain"] = domain_context

            scored_tools.append(tool_result)

        # Sort by final score
        scored_tools.sort(key=lambda x: x["similarity_score"], reverse=True)

        return scored_tools

    def _detect_domain(self, query: str) -> str:
        """Detect domain from query text."""
        query_lower = query.lower()
        domain_scores = {}

        for domain, keywords in self.domain_keywords.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            if score > 0:
                domain_scores[domain] = score

        if domain_scores:
            return max(domain_scores.items(), key=lambda x: x[1])[0]

        return "general"

    async def _calculate_base_similarity(self, query: str, tool: Dict) -> float:
        """Calculate base semantic similarity (simplified)."""
        # In real implementation, this would use actual embeddings
        query_words = set(query.lower().split())
        tool_words = set((tool.get("description", "") + " " + tool.get("name", "")).lower().split())

        if not query_words or not tool_words:
            return 0.0

        intersection = query_words.intersection(tool_words)
        union = query_words.union(tool_words)

        return len(intersection) / len(union) if union else 0.0

    def _calculate_domain_score(self, tool: Dict, domain: str) -> float:
        """Calculate domain-specific relevance score."""
        if domain not in self.domain_keywords:
            return 0.5  # Neutral score for unknown domains

        keywords = self.domain_keywords[domain]
        tool_text = (tool.get("description", "") + " " + tool.get("name", "")).lower()

        matches = sum(1 for keyword in keywords if keyword in tool_text)
        return min(1.0, matches / len(keywords))


# Example 2: Custom Tool Category Manager
class CustomToolCategoryManager:
    """Manage custom tool categories and classifications."""

    def __init__(self):
        self.custom_categories = {}
        self.category_rules = {}
        self.tool_classifications = {}

    def register_category(
        self, category_id: str, display_name: str, description: str, classification_rules: Dict[str, Any]
    ):
        """Register a new custom tool category."""
        print(f"=== Registering Category: {display_name} ===")

        self.custom_categories[category_id] = {
            "id": category_id,
            "name": display_name,
            "description": description,
            "created_at": time.time(),
        }

        self.category_rules[category_id] = classification_rules
        print(f"Category '{display_name}' registered with {len(classification_rules)} rules")

    async def classify_tool(self, tool: Dict) -> List[str]:
        """Classify a tool into custom categories."""
        classifications = []

        for category_id, rules in self.category_rules.items():
            if await self._matches_category_rules(tool, rules):
                classifications.append(category_id)

        # Cache classification
        self.tool_classifications[tool.get("tool_id")] = classifications

        return classifications

    async def _matches_category_rules(self, tool: Dict, rules: Dict[str, Any]) -> bool:
        """Check if tool matches category rules."""
        tool_text = (tool.get("description", "") + " " + tool.get("name", "")).lower()

        # Keyword-based rules
        if "keywords" in rules:
            required_keywords = rules["keywords"].get("required", [])
            optional_keywords = rules["keywords"].get("optional", [])

            # Check required keywords
            if required_keywords:
                if not all(keyword.lower() in tool_text for keyword in required_keywords):
                    return False

            # Check optional keywords (at least one must match)
            if optional_keywords:
                if not any(keyword.lower() in tool_text for keyword in optional_keywords):
                    return False

        # Parameter-based rules
        if "parameters" in rules:
            tool_params = [param.get("name", "") for param in tool.get("parameters", [])]
            required_params = rules["parameters"].get("required", [])

            if not all(param in tool_params for param in required_params):
                return False

        # Complexity rules
        if "complexity" in rules:
            tool_complexity = tool.get("complexity_score", 0.5)
            complexity_range = rules["complexity"]

            if not (complexity_range.get("min", 0) <= tool_complexity <= complexity_range.get("max", 1)):
                return False

        return True

    async def search_by_custom_category(
        self, category_id: str, additional_filters: Dict[str, Any] = None
    ) -> List[Dict]:
        """Search tools by custom category."""
        print(f"=== Searching Custom Category: {category_id} ===")

        # Get all tools in category (simplified - would query actual database)
        category_tools = []

        for tool_id, classifications in self.tool_classifications.items():
            if category_id in classifications:
                # Would fetch full tool data in real implementation
                tool_data = {"tool_id": tool_id, "categories": classifications}
                category_tools.append(tool_data)

        # Apply additional filters
        if additional_filters:
            category_tools = self._apply_filters(category_tools, additional_filters)

        print(f"Found {len(category_tools)} tools in category '{category_id}'")
        return category_tools

    def _apply_filters(self, tools: List[Dict], filters: Dict[str, Any]) -> List[Dict]:
        """Apply additional filters to tools."""
        filtered_tools = tools

        if "complexity_max" in filters:
            max_complexity = filters["complexity_max"]
            filtered_tools = [tool for tool in filtered_tools if tool.get("complexity_score", 0.5) <= max_complexity]

        if "has_examples" in filters and filters["has_examples"]:
            filtered_tools = [tool for tool in filtered_tools if tool.get("examples")]

        return filtered_tools


# Example 3: Custom Scoring System
class CustomScoringSystem:
    """Implement custom scoring algorithms for tool ranking."""

    def __init__(self):
        self.scoring_weights = {
            "semantic_similarity": 0.4,
            "popularity": 0.2,
            "reliability": 0.2,
            "recency": 0.1,
            "user_preference": 0.1,
        }
        self.user_preferences = {}
        self.tool_statistics = {}

    def set_user_preferences(self, user_id: str, preferences: Dict[str, Any]):
        """Set user-specific preferences for scoring."""
        print(f"=== Setting User Preferences for {user_id} ===")

        self.user_preferences[user_id] = {
            "complexity_preference": preferences.get("complexity", "balanced"),
            "reliability_importance": preferences.get("reliability_weight", 1.0),
            "performance_importance": preferences.get("performance_weight", 1.0),
            "familiarity_bonus": preferences.get("familiarity_bonus", 0.1),
            "preferred_categories": preferences.get("categories", []),
        }

        print(f"Preferences set: {list(preferences.keys())}")

    async def score_tools(
        self, tools: List[Dict], query: str, user_id: Optional[str] = None, context: Optional[Dict] = None
    ) -> List[Dict]:
        """Apply custom scoring to tools."""
        print(f"=== Custom Scoring for {len(tools)} tools ===")

        scored_tools = []
        user_prefs = self.user_preferences.get(user_id, {}) if user_id else {}

        for tool in tools:
            scores = await self._calculate_component_scores(tool, query, user_prefs, context)
            final_score = self._combine_scores(scores, user_prefs)

            scored_tool = tool.copy()
            scored_tool["custom_score"] = final_score
            scored_tool["score_components"] = scores

            scored_tools.append(scored_tool)

        # Sort by custom score
        scored_tools.sort(key=lambda x: x["custom_score"], reverse=True)

        print(f"Scoring complete. Top tool: {scored_tools[0]['name']} ({scored_tools[0]['custom_score']:.3f})")

        return scored_tools

    async def _calculate_component_scores(
        self, tool: Dict, query: str, user_prefs: Dict, context: Optional[Dict]
    ) -> Dict[str, float]:
        """Calculate individual component scores."""
        scores = {}

        # Semantic similarity (simplified)
        scores["semantic_similarity"] = await self._calculate_semantic_score(tool, query)

        # Popularity score
        scores["popularity"] = self._calculate_popularity_score(tool)

        # Reliability score
        scores["reliability"] = self._calculate_reliability_score(tool)

        # Recency score
        scores["recency"] = self._calculate_recency_score(tool)

        # User preference score
        scores["user_preference"] = self._calculate_preference_score(tool, user_prefs)

        return scores

    async def _calculate_semantic_score(self, tool: Dict, query: str) -> float:
        """Calculate semantic similarity score."""
        # Simplified implementation - would use actual embeddings
        return tool.get("similarity_score", 0.5)

    def _calculate_popularity_score(self, tool: Dict) -> float:
        """Calculate popularity score based on usage statistics."""
        tool_id = tool.get("tool_id")
        stats = self.tool_statistics.get(tool_id, {})

        usage_count = stats.get("usage_count", 0)
        max_usage = max([s.get("usage_count", 0) for s in self.tool_statistics.values()], default=1)

        return usage_count / max_usage if max_usage > 0 else 0.0

    def _calculate_reliability_score(self, tool: Dict) -> float:
        """Calculate reliability score."""
        tool_id = tool.get("tool_id")
        stats = self.tool_statistics.get(tool_id, {})

        return stats.get("success_rate", 0.5)

    def _calculate_recency_score(self, tool: Dict) -> float:
        """Calculate recency score based on last update."""
        last_updated = tool.get("last_updated", 0)
        current_time = time.time()

        # Score based on how recent the tool was updated (within last year)
        seconds_in_year = 365 * 24 * 3600
        age = current_time - last_updated

        if age < seconds_in_year:
            return 1.0 - (age / seconds_in_year)
        else:
            return 0.0

    def _calculate_preference_score(self, tool: Dict, user_prefs: Dict) -> float:
        """Calculate user preference score."""
        if not user_prefs:
            return 0.5

        score = 0.0
        factors = 0

        # Complexity preference
        if "complexity_preference" in user_prefs:
            tool_complexity = tool.get("complexity_score", 0.5)
            pref = user_prefs["complexity_preference"]

            if pref == "simple" and tool_complexity < 0.3:
                score += 1.0
            elif pref == "balanced" and 0.3 <= tool_complexity <= 0.7:
                score += 1.0
            elif pref == "powerful" and tool_complexity > 0.7:
                score += 1.0
            else:
                score += 0.5

            factors += 1

        # Category preferences
        if "preferred_categories" in user_prefs:
            tool_categories = tool.get("categories", [])
            preferred = user_prefs["preferred_categories"]

            if any(cat in preferred for cat in tool_categories):
                score += 1.0
            else:
                score += 0.3

            factors += 1

        return score / factors if factors > 0 else 0.5

    def _combine_scores(self, scores: Dict[str, float], user_prefs: Dict) -> float:
        """Combine component scores into final score."""
        # Apply user-specific weight adjustments
        weights = self.scoring_weights.copy()

        if user_prefs.get("reliability_importance", 1.0) > 1.0:
            weights["reliability"] *= user_prefs["reliability_importance"]

        if user_prefs.get("performance_importance", 1.0) > 1.0:
            weights["semantic_similarity"] *= user_prefs["performance_importance"]

        # Normalize weights
        total_weight = sum(weights.values())
        normalized_weights = {k: v / total_weight for k, v in weights.items()}

        # Calculate weighted sum
        final_score = sum(scores.get(component, 0.0) * weight for component, weight in normalized_weights.items())

        return final_score


# Example 4: Integration with External Systems
class ExternalSystemIntegration:
    """Integrate tool search with external systems."""

    def __init__(self):
        self.external_catalogs = {}
        self.sync_status = {}

    def register_external_catalog(self, catalog_id: str, catalog_config: Dict[str, Any]):
        """Register an external tool catalog."""
        print(f"=== Registering External Catalog: {catalog_id} ===")

        self.external_catalogs[catalog_id] = {
            "id": catalog_id,
            "type": catalog_config.get("type"),  # 'api', 'database', 'file'
            "endpoint": catalog_config.get("endpoint"),
            "auth": catalog_config.get("auth", {}),
            "sync_interval": catalog_config.get("sync_interval", 3600),
            "last_sync": 0,
        }

        print(f"Catalog registered: {catalog_config.get('type')} at {catalog_config.get('endpoint')}")

    async def sync_external_catalogs(self) -> Dict[str, Any]:
        """Synchronize with external tool catalogs."""
        print("=== Synchronizing External Catalogs ===")

        sync_results = {}

        for catalog_id, config in self.external_catalogs.items():
            try:
                result = await self._sync_catalog(catalog_id, config)
                sync_results[catalog_id] = result

                self.sync_status[catalog_id] = {
                    "last_sync": time.time(),
                    "status": "success",
                    "tools_synced": result.get("count", 0),
                }

                print(f"✅ {catalog_id}: {result.get('count', 0)} tools synced")

            except Exception as e:
                sync_results[catalog_id] = {"error": str(e)}
                self.sync_status[catalog_id] = {"last_sync": time.time(), "status": "failed", "error": str(e)}
                print(f"❌ {catalog_id}: {str(e)}")

        return sync_results

    async def _sync_catalog(self, catalog_id: str, config: Dict) -> Dict[str, Any]:
        """Sync individual external catalog."""
        catalog_type = config["type"]

        if catalog_type == "api":
            return await self._sync_api_catalog(config)
        elif catalog_type == "database":
            return await self._sync_database_catalog(config)
        elif catalog_type == "file":
            return await self._sync_file_catalog(config)
        else:
            raise ValueError(f"Unsupported catalog type: {catalog_type}")

    async def _sync_api_catalog(self, config: Dict) -> Dict[str, Any]:
        """Sync API-based catalog."""
        # Simulate API call
        print(f"Fetching from API: {config['endpoint']}")

        # In real implementation, would make HTTP request
        await asyncio.sleep(0.1)  # Simulate network delay

        return {"count": 25, "type": "api"}

    async def _sync_database_catalog(self, config: Dict) -> Dict[str, Any]:
        """Sync database-based catalog."""
        print(f"Querying database: {config['endpoint']}")

        # In real implementation, would query database
        await asyncio.sleep(0.1)

        return {"count": 150, "type": "database"}

    async def _sync_file_catalog(self, config: Dict) -> Dict[str, Any]:
        """Sync file-based catalog."""
        print(f"Reading file: {config['endpoint']}")

        # In real implementation, would read and parse file
        await asyncio.sleep(0.05)

        return {"count": 12, "type": "file"}

    async def search_federated(self, query: str, catalogs: List[str] = None) -> Dict[str, Any]:
        """Search across multiple catalogs."""
        print(f"=== Federated Search: {query} ===")

        if catalogs is None:
            catalogs = list(self.external_catalogs.keys())

        federated_results = {}

        for catalog_id in catalogs:
            if catalog_id in self.external_catalogs:
                try:
                    results = await self._search_catalog(catalog_id, query)
                    federated_results[catalog_id] = results
                    print(f"✅ {catalog_id}: {len(results.get('tools', []))} results")
                except Exception as e:
                    federated_results[catalog_id] = {"error": str(e)}
                    print(f"❌ {catalog_id}: {str(e)}")

        # Combine and deduplicate results
        combined_results = self._combine_federated_results(federated_results)

        return {
            "query": query,
            "catalogs_searched": catalogs,
            "individual_results": federated_results,
            "combined_results": combined_results,
        }

    async def _search_catalog(self, catalog_id: str, query: str) -> Dict[str, Any]:
        """Search individual catalog."""
        # Simulate catalog-specific search
        await asyncio.sleep(0.1)

        return {
            "catalog": catalog_id,
            "tools": [
                {"tool_id": f"{catalog_id}_tool_1", "name": f"Tool 1 from {catalog_id}"},
                {"tool_id": f"{catalog_id}_tool_2", "name": f"Tool 2 from {catalog_id}"},
            ],
        }

    def _combine_federated_results(self, results: Dict[str, Any]) -> List[Dict]:
        """Combine results from multiple catalogs."""
        combined = []
        seen_tools = set()

        for catalog_id, catalog_results in results.items():
            if "error" in catalog_results:
                continue

            for tool in catalog_results.get("tools", []):
                tool_id = tool.get("tool_id")

                if tool_id not in seen_tools:
                    tool_copy = tool.copy()
                    tool_copy["source_catalog"] = catalog_id
                    combined.append(tool_copy)
                    seen_tools.add(tool_id)

        return combined


# Main demonstration function
async def run_custom_integration_examples():
    """Run all custom integration examples."""
    print("MCP Tool Search System - Custom Integration Examples")
    print("=" * 65)

    # Example 1: Domain-Specific Search
    domain_algorithm = DomainSpecificSearchAlgorithm(
        {"data_science": 1.5, "web_development": 1.2, "devops": 1.3, "security": 1.4}
    )

    # Mock tools for demonstration
    mock_tools = [
        {"tool_id": "pandas_reader", "name": "Pandas Reader", "description": "Read data with pandas for analysis"},
        {"tool_id": "web_scraper", "name": "Web Scraper", "description": "HTTP web scraping tool for APIs"},
        {"tool_id": "docker_deploy", "name": "Docker Deploy", "description": "Deploy containers to kubernetes"},
        {"tool_id": "auth_validator", "name": "Auth Validator", "description": "Validate authentication tokens"},
    ]

    try:
        domain_results = await domain_algorithm.search("analyze data from CSV files", mock_tools)
        print(f"Domain search found {len(domain_results)} results")
        for tool in domain_results[:2]:
            print(f"  {tool['name']}: {tool['similarity_score']:.3f} (domain: {tool['detected_domain']})")
    except Exception as e:
        print(f"Domain search failed: {e}")

    print("\n" + "-" * 65)

    # Example 2: Custom Categories
    category_manager = CustomToolCategoryManager()

    # Register custom categories
    category_manager.register_category(
        "data_processing",
        "Data Processing Tools",
        "Tools for processing and transforming data",
        {
            "keywords": {"required": ["data"], "optional": ["process", "transform", "parse", "convert"]},
            "complexity": {"min": 0.2, "max": 0.8},
        },
    )

    # Example 3: Custom Scoring
    scoring_system = CustomScoringSystem()

    scoring_system.set_user_preferences(
        "user123",
        {"complexity": "balanced", "reliability_weight": 1.5, "categories": ["data_processing", "web_development"]},
    )

    try:
        scored_tools = await scoring_system.score_tools(mock_tools, "reliable data processing", user_id="user123")
        print(f"Custom scoring complete. Top tool scored: {scored_tools[0]['custom_score']:.3f}")
    except Exception as e:
        print(f"Custom scoring failed: {e}")

    print("\n" + "-" * 65)

    # Example 4: External System Integration
    external_integration = ExternalSystemIntegration()

    external_integration.register_external_catalog(
        "github_tools", {"type": "api", "endpoint": "https://api.github.com/tools", "auth": {"token": "mock_token"}}
    )

    external_integration.register_external_catalog(
        "internal_db",
        {
            "type": "database",
            "endpoint": "postgresql://internal/tools",
            "auth": {"username": "admin", "password": "secret"},
        },
    )

    try:
        sync_results = await external_integration.sync_external_catalogs()
        print(f"Sync completed for {len(sync_results)} catalogs")

        federated_results = await external_integration.search_federated("web development tools")
        combined_count = len(federated_results["combined_results"])
        print(f"Federated search found {combined_count} unique tools")

    except Exception as e:
        print(f"External integration failed: {e}")


# Utility functions for custom integrations
def create_custom_search_pipeline(components: List[Any]) -> callable:
    """Create a custom search pipeline from components."""

    async def pipeline(query: str, tools: List[Dict], **kwargs) -> List[Dict]:
        """Execute the custom search pipeline."""
        result = tools

        for component in components:
            if hasattr(component, "search"):
                result = await component.search(query, result, **kwargs)
            elif hasattr(component, "filter"):
                result = component.filter(result, kwargs.get("context", {}))
            elif hasattr(component, "score_tools"):
                result = await component.score_tools(result, query, **kwargs)

        return result

    return pipeline


def validate_integration_config(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate custom integration configuration."""
    errors = []

    required_fields = ["name", "type", "version"]
    for field in required_fields:
        if field not in config:
            errors.append(f"Missing required field: {field}")

    if config.get("type") not in ["search_algorithm", "filter", "scorer", "external_catalog"]:
        errors.append("Invalid integration type")

    return len(errors) == 0, errors


if __name__ == "__main__":
    # Run demonstration
    print("To run async examples, use: python -m asyncio custom_integrations.run_custom_integration_examples")

    # Show configuration template
    print("\n=== Custom Integration Configuration Template ===")

    config_template = {
        "name": "MyCustomIntegration",
        "type": "search_algorithm",
        "version": "1.0.0",
        "description": "Custom search algorithm for domain-specific tools",
        "parameters": {"domain_weights": {"data_science": 1.5, "web_development": 1.2}, "similarity_threshold": 0.3},
        "dependencies": ["numpy", "scikit-learn"],
        "metadata": {"author": "Your Name", "license": "MIT", "documentation_url": "https://example.com/docs"},
    }

    print(json.dumps(config_template, indent=2))

    # Validation example
    is_valid, errors = validate_integration_config(config_template)
    print(f"\nConfiguration valid: {is_valid}")
    if errors:
        print(f"Errors: {errors}")
