#!/usr/bin/env python3
"""
Tool Query Processor

This module processes and analyzes tool search queries to understand search intent,
expand terms with synonyms, and prepare queries for semantic search.
"""

import re
from typing import Any, Dict, List, Optional

from .logging_config import get_logger
from .tool_search_results import ProcessedQuery, SearchIntent

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

# Query confidence adjustment constants
CONFIDENCE_ADJUSTMENTS = {
    "base_confidence": 0.5,  # Base confidence level
    "long_query_boost": 0.2,  # Boost for queries with 4+ words
    "medium_query_boost": 0.1,  # Boost for queries with 2-3 words
    "intent_confidence_multiplier": 0.3,  # Multiplier for search intent confidence
    "category_detection_boost": 0.2,  # Boost when category is detected
    "technical_terms_boost": 0.1,  # Boost for technical terminology
}

# Input validation security constants
SECURITY_LIMITS = {
    "max_query_length": 1000,  # Maximum query length in characters
    "max_word_count": 50,  # Maximum number of words
    "max_word_length": 100,  # Maximum individual word length
    "min_query_length": 1,  # Minimum query length
}

# Patterns for security validation
SECURITY_PATTERNS = {
    "sql_injection": [
        r"(\bunion\b.*\bselect\b)",
        r"(\bselect\b.*\bfrom\b)",
        r"(\binsert\b.*\binto\b)",
        r"(\bdelete\b.*\bfrom\b)",
        r"(\bdrop\b.*\btable\b)",
        r"(\btruncate\b.*\btable\b)",
        r"(\balter\b.*\btable\b)",
        r"(\bcreate\b.*\btable\b)",
        r"(\bupdate\b.*\bset\b)",
        r"(\bexec\b\s+xp_)",
        r"(\bexecute\b\s+sp_)",
        r"(\bsp_\w+)",
        r"(\bxp_\w+)",
        r"(--)",
        r"(/\*)",
        r"(\*/)",
        r"(\bor\b.*=.*)",
        r"(\band\b.*=.*)",
        r"('.*--)",
        r"('.*\*/)",
        r"('.*\*)",
    ],
    "script_injection": [
        r"<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>",
        r"javascript:",
        r"vbscript:",
        r"onload\s*=",
        r"onerror\s*=",
        r"onclick\s*=",
        r"onmouseover\s*=",
        r"onfocus\s*=",
        r"<iframe\b",
        r"<object\b",
        r"<embed\b",
        r"<applet\b",
    ],
    "command_injection": [
        r"(\|\s*\w+)",
        r"(\&\&\s*\w+)",
        r"(\;\s*\w+)",
        r"(\$\()",
        r"(\`.*\`)",
        r"(\beval\s*\()",
        r"(\bexec\s*\()",
        r"(\bsystem\s*\()",
    ],
    "path_traversal": [
        r"(\.\.\/)",
        r"(\.\.\\)",
        r"(%2e%2e%2f)",
        r"(%2e%2e%5c)",
        r"(%2f)",
        r"(%5c)",
        r"(\.\./)",
        r"(\..\\/)",
        r"(/etc/passwd)",
        r"(/proc/)",
    ],
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
            constraints["error_handling"] = "required"

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

            # Analyze query components
            query_analysis = self._analyze_query_components(cleaned_query)

            # Build processed query object
            processed_query = self._build_processed_query(query, cleaned_query, query_analysis)

            logger.debug(
                "Processed query: %s -> category=%s, type=%s, confidence=%.2f",
                query,
                query_analysis["detected_category"],
                query_analysis["detected_tool_type"],
                processed_query.confidence,
            )

            return processed_query

        except (AttributeError, KeyError) as e:
            logger.error("Error processing query '%s' - missing analyzer attributes: %s", query, e)
            return self._create_fallback_query(query)
        except (TypeError, ValueError) as e:
            logger.error("Error processing query '%s' - invalid query format or data: %s", query, e)
            return self._create_fallback_query(query)
        except ImportError as e:
            logger.error("Error processing query '%s' - missing dependencies: %s", query, e)
            return self._create_fallback_query(query)
        except Exception as e:
            logger.error("Unexpected error processing query '%s': %s", query, e)
            return self._create_fallback_query(query)

    def _analyze_query_components(self, cleaned_query: str) -> Dict[str, Any]:
        """Analyze query components including intent, categories, and terms."""
        # Analyze search intent
        search_intent = self.query_analyzer.analyze_search_intent(cleaned_query)

        # Expand query terms
        expanded_terms = self.expand_query_terms(cleaned_query)

        # Detect category and tool type
        detected_category = self.query_analyzer.detect_category(cleaned_query)
        detected_tool_type = self.query_analyzer.detect_tool_type(cleaned_query)

        return {
            "search_intent": search_intent,
            "expanded_terms": expanded_terms,
            "detected_category": detected_category,
            "detected_tool_type": detected_tool_type,
        }

    def _build_processed_query(
        self, original_query: str, cleaned_query: str, query_analysis: Dict[str, Any]
    ) -> ProcessedQuery:
        """Build ProcessedQuery object from analysis components."""
        # Calculate overall confidence
        confidence = self._calculate_query_confidence(
            cleaned_query, query_analysis["search_intent"], query_analysis["detected_category"]
        )

        return ProcessedQuery(
            original_query=original_query,
            cleaned_query=cleaned_query,
            expanded_terms=query_analysis["expanded_terms"],
            detected_category=query_analysis["detected_category"],
            detected_tool_type=query_analysis["detected_tool_type"],
            search_intent=query_analysis["search_intent"],
            confidence=confidence,
        )

    def _create_fallback_query(self, query: str) -> ProcessedQuery:
        """Create a basic fallback ProcessedQuery when processing fails."""
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
        Clean and validate query text with comprehensive security checks.

        Args:
            query: Raw query string

        Returns:
            Cleaned and validated query string

        Raises:
            ValueError: If query fails security validation or size limits
        """
        # Input validation and security checks
        self._validate_query_security(query)

        # Remove extra whitespace
        cleaned = re.sub(r"\s+", " ", query.strip())

        # Remove control characters and potentially dangerous characters
        cleaned = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", cleaned)

        # Normalize punctuation (allow more technical characters)
        cleaned = re.sub(r'[^\w\s\-"\'\.\_\+\#\@]', " ", cleaned)

        # Remove very short words (except common technical terms)
        words = cleaned.split()
        meaningful_words = []
        for word in words:
            # Skip excessively long words (potential attack vectors)
            if len(word) > SECURITY_LIMITS["max_word_length"]:
                logger.warning("Skipping excessively long word in query: %d chars", len(word))
                continue

            if len(word) >= 2 or word.lower() in ["js", "py", "go", "r", "c", "c+", "c#"]:
                meaningful_words.append(word)

        return " ".join(meaningful_words)

    def _validate_query_security(self, query: str) -> None:
        """
        Validate query against security threats and size limits.

        Args:
            query: Query string to validate

        Raises:
            ValueError: If query fails security validation
        """
        if not query:
            raise ValueError("Query cannot be empty")

        # Check query length limits
        if len(query) > SECURITY_LIMITS["max_query_length"]:
            raise ValueError(f"Query too long: {len(query)} chars > {SECURITY_LIMITS['max_query_length']} limit")

        if len(query) < SECURITY_LIMITS["min_query_length"]:
            raise ValueError(f"Query too short: {len(query)} chars < {SECURITY_LIMITS['min_query_length']} minimum")

        # Check word count
        word_count = len(query.split())
        if word_count > SECURITY_LIMITS["max_word_count"]:
            raise ValueError(f"Too many words: {word_count} > {SECURITY_LIMITS['max_word_count']} limit")

        # Convert to lowercase for pattern matching
        query_lower = query.lower()

        # Check for SQL injection patterns
        for pattern in SECURITY_PATTERNS["sql_injection"]:
            if re.search(pattern, query_lower, re.IGNORECASE):
                logger.warning("Potential SQL injection detected in query: %s", pattern)
                raise ValueError("Query contains potentially malicious SQL patterns")

        # Check for script injection patterns
        for pattern in SECURITY_PATTERNS["script_injection"]:
            if re.search(pattern, query, re.IGNORECASE):
                logger.warning("Potential script injection detected in query: %s", pattern)
                raise ValueError("Query contains potentially malicious script patterns")

        # Check for command injection patterns
        for pattern in SECURITY_PATTERNS["command_injection"]:
            if re.search(pattern, query_lower, re.IGNORECASE):
                logger.warning("Potential command injection detected in query: %s", pattern)
                raise ValueError("Query contains potentially malicious command patterns")

        # Check for path traversal patterns
        for pattern in SECURITY_PATTERNS["path_traversal"]:
            if re.search(pattern, query, re.IGNORECASE):
                logger.warning("Potential path traversal detected in query: %s", pattern)
                raise ValueError("Query contains potentially malicious path patterns")

        # Additional security checks
        self._validate_encoding_attacks(query)
        self._validate_repeated_patterns(query)

    def _validate_encoding_attacks(self, query: str) -> None:
        """
        Validate against encoding-based attacks.

        Args:
            query: Query to validate

        Raises:
            ValueError: If encoding attack patterns detected
        """
        # Check for URL encoding attacks
        url_encoded_patterns = ["%00", "%0a", "%0d", "%3c", "%3e", "%22", "%27"]
        for pattern in url_encoded_patterns:
            if pattern in query.lower():
                raise ValueError("Query contains potentially malicious URL encoded characters")

        # Check for Unicode attacks (excessive non-ASCII)
        try:
            ascii_chars = sum(1 for c in query if ord(c) < 128)
            non_ascii_chars = len(query) - ascii_chars
            if len(query) > 0 and (non_ascii_chars / len(query)) > 0.5:
                logger.warning("Query contains high ratio of non-ASCII characters: %.2f", non_ascii_chars / len(query))
                # Don't reject, but log for monitoring
        except Exception:
            pass  # Ignore encoding issues in validation

    def _validate_repeated_patterns(self, query: str) -> None:
        """
        Validate against repeated pattern attacks.

        Args:
            query: Query to validate

        Raises:
            ValueError: If excessive repetition detected
        """
        # Check for excessive character repetition (potential DoS)
        if len(query) > 20:
            # Check for repeated characters
            for char in set(query):
                if query.count(char) > len(query) * 0.7:
                    raise ValueError("Query contains excessive character repetition")

            # Check for repeated substrings
            words = query.split()
            if len(words) > 5:
                for word in set(words):
                    if words.count(word) > len(words) * 0.6:
                        raise ValueError("Query contains excessive word repetition")

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
        confidence = CONFIDENCE_ADJUSTMENTS["base_confidence"]

        # Boost confidence for longer, more specific queries
        word_count = len(query.split())
        if word_count >= 4:
            confidence += CONFIDENCE_ADJUSTMENTS["long_query_boost"]
        elif word_count >= 2:
            confidence += CONFIDENCE_ADJUSTMENTS["medium_query_boost"]

        # Boost confidence for detected intent
        confidence += search_intent.confidence * CONFIDENCE_ADJUSTMENTS["intent_confidence_multiplier"]

        # Boost confidence for detected category
        if detected_category:
            confidence += CONFIDENCE_ADJUSTMENTS["category_detection_boost"]

        # Boost confidence for technical terms
        technical_terms = ["timeout", "error", "handling", "async", "sync", "performance"]
        if any(term in query.lower() for term in technical_terms):
            confidence += CONFIDENCE_ADJUSTMENTS["technical_terms_boost"]

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
