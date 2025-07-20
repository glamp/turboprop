#!/usr/bin/env python3
"""
Tool Query Processor

This module processes and analyzes tool search queries to understand search intent,
expand terms with synonyms, and prepare queries for semantic search.
"""

import re
from typing import Dict, List, Optional, Set

from logging_config import get_logger
from tool_search_results import ProcessedQuery, SearchIntent

logger = get_logger(__name__)

# Technical synonym mappings for tool functionality
FUNCTIONALITY_SYNONYMS = {
    "file": ["document", "file", "filesystem", "fs", "disk", "storage"],
    "read": ["load", "open", "retrieve", "fetch", "get", "input"],
    "write": ["save", "store", "output", "create", "generate"],
    "edit": ["modify", "update", "change", "alter", "revise"],
    "execute": ["run", "invoke", "call", "launch", "start"],
    "command": ["cmd", "shell", "bash", "terminal", "cli"],
    "search": ["find", "query", "lookup", "locate", "discover"],
    "web": ["internet", "http", "https", "url", "online", "network"],
    "data": ["information", "content", "text", "json", "xml"],
    "error": ["exception", "failure", "problem", "issue", "bug"],
    "timeout": ["delay", "wait", "duration", "time limit", "deadline"],
}

# Programming language synonyms
LANGUAGE_SYNONYMS = {
    "javascript": ["js", "javascript", "node", "nodejs", "ecmascript"],
    "python": ["py", "python", "python3", "py3"],
    "typescript": ["ts", "typescript", "tsx"],
    "java": ["java", "jvm"],
    "cpp": ["c++", "cpp", "cxx", "cc"],
    "csharp": ["c#", "csharp", "cs", "dotnet"],
    "rust": ["rust", "rs"],
    "go": ["go", "golang"],
}

# Tool category mappings
CATEGORY_KEYWORDS = {
    "file_ops": ["file", "directory", "folder", "filesystem", "path", "read", "write", "edit"],
    "execution": ["execute", "run", "command", "shell", "bash", "process", "script"],
    "search": ["search", "find", "query", "grep", "match", "filter"],
    "web": ["web", "http", "url", "fetch", "scrape", "download", "browser"],
    "data": ["json", "xml", "csv", "parse", "format", "convert", "transform"],
    "version_control": ["git", "commit", "branch", "repository", "diff", "merge"],
    "testing": ["test", "junit", "pytest", "spec", "assert", "mock"],
    "build": ["build", "compile", "make", "cmake", "gradle", "maven", "npm"],
}

# Common tool name patterns
TOOL_TYPE_PATTERNS = {
    "system": [r"\b(bash|grep|ls|cat|head|tail|find|sed|awk)\b"],
    "custom": [r"\b(read|write|edit|multiedit)\b"],
    "third_party": [r"\b(git|npm|pip|cargo|docker)\b"],
}


class SynonymExpander:
    """Expands queries with technical synonyms and related terms."""

    def __init__(self):
        """Initialize the synonym expander with technical vocabularies."""
        self.functionality_synonyms = FUNCTIONALITY_SYNONYMS
        self.language_synonyms = LANGUAGE_SYNONYMS

    def expand_functionality_terms(self, query: str) -> List[str]:
        """
        Expand query terms with functionality synonyms.

        Args:
            query: Original query string

        Returns:
            List of expanded terms
        """
        expanded_terms = set()
        query_lower = query.lower()

        # Add original terms
        for word in query_lower.split():
            expanded_terms.add(word)

        # Find and expand functionality terms
        for base_term, synonyms in self.functionality_synonyms.items():
            for synonym in synonyms:
                if synonym in query_lower:
                    expanded_terms.update(synonyms)
                    break

        return list(expanded_terms)

    def expand_language_terms(self, query: str) -> List[str]:
        """
        Expand query with programming language synonyms.

        Args:
            query: Original query string

        Returns:
            List of language-related terms
        """
        expanded_terms = set()
        query_lower = query.lower()

        # Find and expand language terms
        for base_lang, synonyms in self.language_synonyms.items():
            for synonym in synonyms:
                if synonym in query_lower:
                    expanded_terms.update(synonyms)
                    break

        return list(expanded_terms)

    def get_context_expansions(self, query: str) -> List[str]:
        """
        Get context-specific expansions based on query content.

        Args:
            query: Original query string

        Returns:
            List of context-specific expansions
        """
        expansions = []
        query_lower = query.lower()

        # Error handling context
        if any(term in query_lower for term in ["error", "exception", "fail"]):
            expansions.extend(["error handling", "exception handling", "robust", "resilient"])

        # Async/concurrency context
        if any(term in query_lower for term in ["async", "concurrent", "parallel"]):
            expansions.extend(["asynchronous", "concurrent", "parallel", "threading", "multiprocessing"])

        # Performance context
        if any(term in query_lower for term in ["fast", "quick", "performance", "speed"]):
            expansions.extend(["efficient", "optimized", "performance", "scalable", "high-performance"])

        # Security context
        if any(term in query_lower for term in ["secure", "safe", "auth"]):
            expansions.extend(["security", "authentication", "authorization", "safe", "secure"])

        return expansions


class QueryAnalyzer:
    """Analyzes queries to understand search intent and extract entities."""

    def __init__(self):
        """Initialize the query analyzer."""
        self.category_keywords = CATEGORY_KEYWORDS
        self.tool_type_patterns = TOOL_TYPE_PATTERNS

    def analyze_search_intent(self, query: str) -> SearchIntent:
        """
        Analyze query to understand search intent.

        Args:
            query: Search query string

        Returns:
            SearchIntent with analyzed intent information
        """
        query_lower = query.lower()
        intent = SearchIntent(query_type="general", confidence=0.5)

        # Detect specific tool queries
        if any(word in query_lower for word in ["tool", "function", "command"]):
            intent.query_type = "specific"
            intent.confidence = 0.8

        # Detect comparison queries
        if any(word in query_lower for word in ["vs", "versus", "compare", "alternative", "better"]):
            intent.query_type = "comparison"
            intent.confidence = 0.9

        # Detect alternative seeking queries
        if any(phrase in query_lower for phrase in ["alternative to", "instead of", "replace", "substitute"]):
            intent.query_type = "alternative"
            intent.confidence = 0.9

        # Extract target functionality
        functionality_terms = []
        for word in query_lower.split():
            # Look for action words
            if word in ["read", "write", "execute", "search", "find", "create", "delete", "update"]:
                functionality_terms.append(word)

        intent.target_functionality = functionality_terms

        # Extract constraints from query
        constraints = {}

        # Timeout constraints
        timeout_match = re.search(r"(\d+)\s*(second|minute|hour|ms|sec|min)s?\s*(timeout|limit)", query_lower)
        if timeout_match:
            constraints["timeout"] = f"{timeout_match.group(1)} {timeout_match.group(2)}"

        # Error handling requirements
        if any(term in query_lower for term in ["error handling", "robust", "safe", "reliable"]):
            constraints["error_handling"] = True

        # Performance requirements
        if any(term in query_lower for term in ["fast", "quick", "performance", "efficient"]):
            constraints["performance"] = "high"

        intent.constraints = constraints

        return intent

    def detect_category(self, query: str) -> Optional[str]:
        """
        Detect target category from query.

        Args:
            query: Search query string

        Returns:
            Detected category or None
        """
        query_lower = query.lower()
        category_scores = {}

        for category, keywords in self.category_keywords.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            if score > 0:
                category_scores[category] = score

        if category_scores:
            # Return category with highest score
            return max(category_scores, key=category_scores.get)

        return None

    def detect_tool_type(self, query: str) -> Optional[str]:
        """
        Detect target tool type from query.

        Args:
            query: Search query string

        Returns:
            Detected tool type or None
        """
        query_lower = query.lower()

        for tool_type, patterns in self.tool_type_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower, re.IGNORECASE):
                    return tool_type

        return None

    def extract_quoted_phrases(self, query: str) -> List[str]:
        """
        Extract quoted phrases from query for exact matching.

        Args:
            query: Search query string

        Returns:
            List of quoted phrases
        """
        return re.findall(r'"([^"]*)"', query)

    def extract_parameter_hints(self, query: str) -> Dict[str, str]:
        """
        Extract parameter hints from query.

        Args:
            query: Search query string

        Returns:
            Dictionary of parameter hints
        """
        hints = {}
        query_lower = query.lower()

        # Required parameters
        if "required" in query_lower:
            hints["has_required_params"] = "true"

        # Optional parameters
        if "optional" in query_lower:
            hints["has_optional_params"] = "true"

        # Simple/complex tool preferences
        if any(term in query_lower for term in ["simple", "basic", "easy"]):
            hints["complexity"] = "low"
        elif any(term in query_lower for term in ["complex", "advanced", "sophisticated"]):
            hints["complexity"] = "high"

        return hints


class ToolQueryProcessor:
    """Process and analyze tool search queries."""

    def __init__(self):
        """Initialize the query processor with analyzers."""
        self.synonym_expander = SynonymExpander()
        self.query_analyzer = QueryAnalyzer()

    def process_query(self, query: str) -> ProcessedQuery:
        """
        Process raw query into structured search parameters.

        Args:
            query: Raw search query string

        Returns:
            ProcessedQuery with analyzed and expanded query information
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        try:
            # Clean and normalize query
            cleaned_query = self._clean_query(query)

            # Analyze search intent
            search_intent = self.query_analyzer.analyze_search_intent(cleaned_query)

            # Expand query terms
            expanded_terms = self.expand_query_terms(cleaned_query)

            # Detect category and tool type
            detected_category = self.query_analyzer.detect_category(cleaned_query)
            detected_tool_type = self.query_analyzer.detect_tool_type(cleaned_query)

            # Calculate overall confidence
            confidence = self._calculate_query_confidence(cleaned_query, search_intent, detected_category)

            processed_query = ProcessedQuery(
                original_query=query,
                cleaned_query=cleaned_query,
                expanded_terms=expanded_terms,
                detected_category=detected_category,
                detected_tool_type=detected_tool_type,
                search_intent=search_intent,
                confidence=confidence,
            )

            logger.debug(
                "Processed query: %s -> category=%s, type=%s, confidence=%.2f",
                query,
                detected_category,
                detected_tool_type,
                confidence,
            )

            return processed_query

        except Exception as e:
            logger.error("Error processing query '%s': %s", query, e)
            # Return a basic processed query on error
            return ProcessedQuery(
                original_query=query,
                cleaned_query=query.strip(),
                expanded_terms=[],
                confidence=0.1,
            )

    def expand_query_terms(self, query: str) -> List[str]:
        """
        Expand query with related technical terms.

        Args:
            query: Clean query string

        Returns:
            List of expanded terms including synonyms and context
        """
        expanded_terms = set()

        # Add original terms
        expanded_terms.update(query.lower().split())

        # Expand functionality terms
        functionality_expansions = self.synonym_expander.expand_functionality_terms(query)
        expanded_terms.update(functionality_expansions)

        # Expand language terms
        language_expansions = self.synonym_expander.expand_language_terms(query)
        expanded_terms.update(language_expansions)

        # Add context-specific expansions
        context_expansions = self.synonym_expander.get_context_expansions(query)
        expanded_terms.update(context_expansions)

        return list(expanded_terms)

    def _clean_query(self, query: str) -> str:
        """
        Clean and normalize query text.

        Args:
            query: Raw query string

        Returns:
            Cleaned query string
        """
        # Remove extra whitespace
        cleaned = re.sub(r"\s+", " ", query.strip())

        # Normalize punctuation
        cleaned = re.sub(r'[^\w\s\-"\']', " ", cleaned)

        # Remove very short words (except common technical terms)
        words = cleaned.split()
        meaningful_words = []
        for word in words:
            if len(word) >= 2 or word.lower() in ["js", "py", "go", "r", "c"]:
                meaningful_words.append(word)

        return " ".join(meaningful_words)

    def _calculate_query_confidence(
        self, query: str, search_intent: SearchIntent, detected_category: Optional[str]
    ) -> float:
        """
        Calculate confidence score for query processing.

        Args:
            query: Cleaned query string
            search_intent: Analyzed search intent
            detected_category: Detected category

        Returns:
            Confidence score between 0.0 and 1.0
        """
        confidence = 0.5  # Base confidence

        # Boost confidence for longer, more specific queries
        word_count = len(query.split())
        if word_count >= 4:
            confidence += 0.2
        elif word_count >= 2:
            confidence += 0.1

        # Boost confidence for detected intent
        confidence += search_intent.confidence * 0.3

        # Boost confidence for detected category
        if detected_category:
            confidence += 0.2

        # Boost confidence for technical terms
        technical_terms = ["timeout", "error", "handling", "async", "sync", "performance"]
        if any(term in query.lower() for term in technical_terms):
            confidence += 0.1

        return min(confidence, 1.0)

    def suggest_query_refinements(self, query: str, results_found: int = 0) -> List[str]:
        """
        Suggest query refinements based on results and analysis.

        Args:
            query: Original query string
            results_found: Number of results found

        Returns:
            List of suggested refinements
        """
        suggestions = []
        query_lower = query.lower()

        # No results found
        if results_found == 0:
            suggestions.append("Try using more general terms")
            suggestions.append("Check spelling of technical terms")

            # Suggest category-specific refinements
            if "file" in query_lower:
                suggestions.append("Try 'file operations' or 'filesystem tools'")
            elif "command" in query_lower or "execute" in query_lower:
                suggestions.append("Try 'execution tools' or 'command line'")
            elif "search" in query_lower:
                suggestions.append("Try 'search tools' or 'text matching'")

        # Too many results
        elif results_found > 20:
            suggestions.append("Add more specific terms to narrow results")
            suggestions.append("Specify a category like 'file operations' or 'web tools'")

            # Suggest constraint refinements
            if "timeout" not in query_lower:
                suggestions.append("Add 'with timeout' for tools with time limits")
            if "error" not in query_lower and "handling" not in query_lower:
                suggestions.append("Add 'error handling' for robust tools")

        # Moderate results - suggest expansions
        else:
            suggestions.append("Try related terms for broader results")
            if "simple" not in query_lower and "basic" not in query_lower:
                suggestions.append("Add 'simple' for basic tools")
            if "advanced" not in query_lower and "complex" not in query_lower:
                suggestions.append("Add 'advanced' for sophisticated tools")

        return suggestions[:3]  # Return top 3 suggestions
