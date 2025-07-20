#!/usr/bin/env python3
"""
Tool Relationship Detector

This module detects and analyzes relationships between tools including alternatives,
complements, and prerequisites. It analyzes tool descriptions, parameters, and
usage patterns to identify how tools relate to each other.
"""

import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Set, Tuple

from logging_config import get_logger
from mcp_metadata_types import MCPToolMetadata, ParameterAnalysis

logger = get_logger(__name__)


@dataclass
class ToolRelationship:
    """Represents a relationship between two tools."""
    tool_a_id: str
    tool_a_name: str
    tool_b_id: str
    tool_b_name: str
    relationship_type: str  # 'alternative', 'complement', 'prerequisite'
    strength: float         # 0.0 to 1.0
    description: str
    confidence: float = 0.0  # Confidence in the relationship detection

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class RelationshipAnalysisResult:
    """Result of relationship analysis operation."""
    relationships_found: int
    alternatives: List[ToolRelationship]
    complements: List[ToolRelationship]
    prerequisites: List[ToolRelationship]
    analysis_time: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "relationships_found": self.relationships_found,
            "alternatives_count": len(self.alternatives),
            "complements_count": len(self.complements),
            "prerequisites_count": len(self.prerequisites),
            "analysis_time": self.analysis_time
        }


class ToolRelationshipDetector:
    """Detect and analyze relationships between tools."""

    def __init__(self):
        """Initialize the relationship detector."""
        self.similarity_threshold = 0.6    # Minimum similarity for alternatives
        self.complement_threshold = 0.4    # Minimum score for complements  
        self.prerequisite_threshold = 0.3  # Minimum score for prerequisites
        
        # Keyword patterns for different relationship types
        self.alternative_patterns = [
            r'\b(read|write|edit|create|delete|list|search|find|get|set)\b',
            r'\b(file|directory|folder|path)\b',
            r'\b(text|content|data|information)\b'
        ]
        
        self.complement_patterns = [
            ('read', 'write'), ('create', 'delete'), ('search', 'filter'),
            ('input', 'output'), ('encode', 'decode'), ('compress', 'decompress')
        ]
        
        self.prerequisite_patterns = [
            ('create', 'edit'), ('create', 'delete'), ('mkdir', 'write'),
            ('login', 'fetch'), ('configure', 'execute')
        ]
        
        logger.info("Initialized ToolRelationshipDetector")

    def detect_alternatives(self, tools: List[MCPToolMetadata]) -> List[ToolRelationship]:
        """
        Find tools that serve similar purposes.
        
        Compares tool descriptions and functionality, analyzes parameter similarity,
        calculates functional overlap scores, and identifies direct alternatives.
        
        Args:
            tools: List of tool metadata to analyze
            
        Returns:
            List of alternative relationships found
        """
        logger.info("Detecting alternative relationships among %d tools", len(tools))
        
        alternatives = []
        
        for i, tool_a in enumerate(tools):
            for j, tool_b in enumerate(tools[i + 1:], i + 1):
                # Skip same tool
                if tool_a.name == tool_b.name:
                    continue
                
                # Calculate similarity score
                similarity = self.calculate_similarity_score(tool_a.description, tool_b.description)
                
                # Check if they're in same category (stronger indication of alternatives)
                category_match = tool_a.category == tool_b.category
                if category_match:
                    similarity += 0.2  # Boost for same category
                
                # Check parameter similarity
                param_similarity = self._calculate_parameter_similarity(
                    getattr(tool_a, 'parameters', []),
                    getattr(tool_b, 'parameters', [])
                )
                
                # Combine scores
                total_score = (similarity * 0.7) + (param_similarity * 0.3)
                
                if total_score >= self.similarity_threshold:
                    relationship = ToolRelationship(
                        tool_a_id=f"tool_{tool_a.name.lower().replace(' ', '_')}",
                        tool_a_name=tool_a.name,
                        tool_b_id=f"tool_{tool_b.name.lower().replace(' ', '_')}",
                        tool_b_name=tool_b.name,
                        relationship_type="alternative",
                        strength=min(total_score, 1.0),
                        description=f"Alternative tools for {tool_a.category} operations",
                        confidence=total_score
                    )
                    alternatives.append(relationship)
                    
                    logger.debug(
                        "Found alternative relationship: %s <-> %s (strength: %.3f)",
                        tool_a.name, tool_b.name, total_score
                    )
        
        logger.info("Found %d alternative relationships", len(alternatives))
        return alternatives

    def detect_complements(self, tools: List[MCPToolMetadata]) -> List[ToolRelationship]:
        """
        Find tools that work well together.
        
        Analyzes input/output compatibility, identifies workflow patterns,
        finds tools that enhance each other, and calculates complementary strengths.
        
        Args:
            tools: List of tool metadata to analyze
            
        Returns:
            List of complement relationships found
        """
        logger.info("Detecting complement relationships among %d tools", len(tools))
        
        complements = []
        
        for i, tool_a in enumerate(tools):
            for j, tool_b in enumerate(tools[i + 1:], i + 1):
                # Skip same tool
                if tool_a.name == tool_b.name:
                    continue
                
                # Calculate complement score
                complement_score = self._calculate_complement_score(tool_a, tool_b)
                
                if complement_score >= self.complement_threshold:
                    # Determine direction (if any)
                    direction = self._determine_complement_direction(tool_a, tool_b)
                    
                    relationship = ToolRelationship(
                        tool_a_id=f"tool_{tool_a.name.lower().replace(' ', '_')}",
                        tool_a_name=tool_a.name,
                        tool_b_id=f"tool_{tool_b.name.lower().replace(' ', '_')}",
                        tool_b_name=tool_b.name,
                        relationship_type="complement",
                        strength=complement_score,
                        description=f"Complementary tools that work well together{direction}",
                        confidence=complement_score
                    )
                    complements.append(relationship)
                    
                    logger.debug(
                        "Found complement relationship: %s <-> %s (strength: %.3f)",
                        tool_a.name, tool_b.name, complement_score
                    )
        
        logger.info("Found %d complement relationships", len(complements))
        return complements

    def detect_prerequisites(self, tools: List[MCPToolMetadata]) -> List[ToolRelationship]:
        """
        Find prerequisite tool relationships.
        
        Identifies setup or preparation tools, analyzes tool dependencies,
        creates prerequisite chains, and calculates dependency strengths.
        
        Args:
            tools: List of tool metadata to analyze
            
        Returns:
            List of prerequisite relationships found
        """
        logger.info("Detecting prerequisite relationships among %d tools", len(tools))
        
        prerequisites = []
        
        for i, tool_a in enumerate(tools):
            for j, tool_b in enumerate(tools):
                # Skip same tool
                if i == j or tool_a.name == tool_b.name:
                    continue
                
                # Calculate prerequisite score (A is prerequisite for B)
                prereq_score = self._calculate_prerequisite_score(tool_a, tool_b)
                
                if prereq_score >= self.prerequisite_threshold:
                    relationship = ToolRelationship(
                        tool_a_id=f"tool_{tool_a.name.lower().replace(' ', '_')}",
                        tool_a_name=tool_a.name,
                        tool_b_id=f"tool_{tool_b.name.lower().replace(' ', '_')}",
                        tool_b_name=tool_b.name,
                        relationship_type="prerequisite",
                        strength=prereq_score,
                        description=f"{tool_a.name} should typically be used before {tool_b.name}",
                        confidence=prereq_score
                    )
                    prerequisites.append(relationship)
                    
                    logger.debug(
                        "Found prerequisite relationship: %s -> %s (strength: %.3f)",
                        tool_a.name, tool_b.name, prereq_score
                    )
        
        logger.info("Found %d prerequisite relationships", len(prerequisites))
        return prerequisites

    def calculate_similarity_score(self, description_a: str, description_b: str) -> float:
        """
        Calculate similarity score between two tool descriptions.
        
        Uses text analysis, keyword matching, and semantic similarity indicators
        to determine how similar two tools are.
        
        Args:
            description_a: First tool description
            description_b: Second tool description
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        if not description_a or not description_b:
            return 0.0
        
        # Exact match
        if description_a.lower().strip() == description_b.lower().strip():
            return 1.0
        
        # Tokenize and normalize
        tokens_a = self._tokenize_description(description_a.lower())
        tokens_b = self._tokenize_description(description_b.lower())
        
        if not tokens_a or not tokens_b:
            return 0.0
        
        # Calculate Jaccard similarity (intersection over union)
        intersection = len(tokens_a.intersection(tokens_b))
        union = len(tokens_a.union(tokens_b))
        
        if union == 0:
            return 0.0
        
        jaccard_score = intersection / union
        
        # Boost score for key action words
        action_words = {'read', 'write', 'create', 'delete', 'search', 'find', 'list', 'edit'}
        a_actions = tokens_a.intersection(action_words)
        b_actions = tokens_b.intersection(action_words)
        
        if a_actions and b_actions:
            action_overlap = len(a_actions.intersection(b_actions)) / len(a_actions.union(b_actions))
            jaccard_score += action_overlap * 0.3
        
        # Boost score for domain-specific terms
        domain_words = {'file', 'directory', 'web', 'http', 'database', 'text', 'image'}
        a_domains = tokens_a.intersection(domain_words) 
        b_domains = tokens_b.intersection(domain_words)
        
        if a_domains and b_domains:
            domain_overlap = len(a_domains.intersection(b_domains)) / len(a_domains.union(b_domains))
            jaccard_score += domain_overlap * 0.2
        
        return min(jaccard_score, 1.0)

    def analyze_all_relationships(self, tools: List[MCPToolMetadata]) -> RelationshipAnalysisResult:
        """
        Perform comprehensive relationship analysis on all tools.
        
        Args:
            tools: List of tools to analyze
            
        Returns:
            RelationshipAnalysisResult with all detected relationships
        """
        import time
        
        logger.info("Starting comprehensive relationship analysis for %d tools", len(tools))
        start_time = time.time()
        
        # Detect all relationship types
        alternatives = self.detect_alternatives(tools)
        complements = self.detect_complements(tools)
        prerequisites = self.detect_prerequisites(tools)
        
        total_relationships = len(alternatives) + len(complements) + len(prerequisites)
        analysis_time = time.time() - start_time
        
        result = RelationshipAnalysisResult(
            relationships_found=total_relationships,
            alternatives=alternatives,
            complements=complements,
            prerequisites=prerequisites,
            analysis_time=analysis_time
        )
        
        logger.info(
            "Relationship analysis completed: %d relationships found in %.2fs",
            total_relationships, analysis_time
        )
        
        return result

    def _calculate_parameter_similarity(self, params_a: List[ParameterAnalysis], params_b: List[ParameterAnalysis]) -> float:
        """Calculate similarity score between parameter lists."""
        if not params_a and not params_b:
            return 1.0  # Both have no parameters
        
        if not params_a or not params_b:
            return 0.0  # One has parameters, other doesn't
        
        # Extract parameter names and types
        names_a = {p.name.lower() for p in params_a}
        names_b = {p.name.lower() for p in params_b}
        
        types_a = {p.type.lower() for p in params_a}
        types_b = {p.type.lower() for p in params_b}
        
        # Calculate name overlap
        name_intersection = len(names_a.intersection(names_b))
        name_union = len(names_a.union(names_b))
        name_similarity = name_intersection / name_union if name_union > 0 else 0.0
        
        # Calculate type overlap
        type_intersection = len(types_a.intersection(types_b))
        type_union = len(types_a.union(types_b))
        type_similarity = type_intersection / type_union if type_union > 0 else 0.0
        
        # Combine scores
        return (name_similarity * 0.7) + (type_similarity * 0.3)

    def _calculate_complement_score(self, tool_a: MCPToolMetadata, tool_b: MCPToolMetadata) -> float:
        """Calculate complement score between two tools."""
        score = 0.0
        
        # Check for complementary action patterns
        desc_a = tool_a.description.lower()
        desc_b = tool_b.description.lower()
        
        for pattern_a, pattern_b in self.complement_patterns:
            if pattern_a in desc_a and pattern_b in desc_b:
                score += 0.4
            elif pattern_b in desc_a and pattern_a in desc_b:
                score += 0.4
        
        # Check for workflow compatibility (different categories that work together)
        if tool_a.category != tool_b.category:
            # Common workflow combinations
            workflow_combos = [
                ('file_ops', 'search'), ('web', 'file_ops'), 
                ('execution', 'file_ops'), ('development', 'file_ops')
            ]
            
            for cat_a, cat_b in workflow_combos:
                if (tool_a.category == cat_a and tool_b.category == cat_b) or \
                   (tool_a.category == cat_b and tool_b.category == cat_a):
                    score += 0.3
        
        # Check parameter compatibility
        params_a = getattr(tool_a, 'parameters', [])
        params_b = getattr(tool_b, 'parameters', [])
        
        if params_a and params_b:
            # Look for input/output parameter patterns
            a_outputs = any('output' in p.name.lower() or 'result' in p.name.lower() for p in params_a)
            b_inputs = any('input' in p.name.lower() or 'file' in p.name.lower() for p in params_b)
            
            if a_outputs and b_inputs:
                score += 0.2
        
        return min(score, 1.0)

    def _calculate_prerequisite_score(self, tool_a: MCPToolMetadata, tool_b: MCPToolMetadata) -> float:
        """Calculate prerequisite score (tool_a is prerequisite for tool_b)."""
        score = 0.0
        
        desc_a = tool_a.description.lower()
        desc_b = tool_b.description.lower()
        
        # Check for prerequisite patterns
        for pattern_a, pattern_b in self.prerequisite_patterns:
            if pattern_a in desc_a and pattern_b in desc_b:
                score += 0.4
        
        # Check for setup/execution patterns
        setup_words = ['create', 'make', 'mkdir', 'configure', 'initialize', 'setup']
        execution_words = ['write', 'execute', 'run', 'process', 'operate']
        
        a_is_setup = any(word in desc_a for word in setup_words)
        b_is_execution = any(word in desc_b for word in execution_words)
        
        if a_is_setup and b_is_execution:
            score += 0.3
        
        # File system prerequisites
        if 'directory' in desc_a and 'file' in desc_b:
            score += 0.2
        
        return min(score, 1.0)

    def _determine_complement_direction(self, tool_a: MCPToolMetadata, tool_b: MCPToolMetadata) -> str:
        """Determine if there's a directional relationship between complements."""
        desc_a = tool_a.description.lower()
        desc_b = tool_b.description.lower()
        
        # Check for producer/consumer relationships
        if any(word in desc_a for word in ['create', 'generate', 'produce']) and \
           any(word in desc_b for word in ['read', 'consume', 'process']):
            return f" (typically {tool_a.name} produces data for {tool_b.name})"
        
        if any(word in desc_b for word in ['create', 'generate', 'produce']) and \
           any(word in desc_a for word in ['read', 'consume', 'process']):
            return f" (typically {tool_b.name} produces data for {tool_a.name})"
        
        return ""

    def _tokenize_description(self, description: str) -> Set[str]:
        """Tokenize and normalize a description for comparison."""
        if not description:
            return set()
        
        # Remove punctuation and split into words
        words = re.findall(r'\b[a-zA-Z]+\b', description.lower())
        
        # Filter out common stop words
        stop_words = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'or', 'can', 'could', 'should', 'would'
        }
        
        return {word for word in words if word not in stop_words and len(word) > 2}