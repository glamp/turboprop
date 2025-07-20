#!/usr/bin/env python3
"""
task_analyzer.py: Task Analysis for Tool Recommendations

This module analyzes natural language task descriptions and extracts requirements,
constraints, and metadata to enable intelligent tool recommendations.
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from logging_config import get_logger

logger = get_logger(__name__)


class TaskComplexity(Enum):
    """Task complexity levels."""
    SIMPLE = "simple"
    MODERATE = "moderate" 
    COMPLEX = "complex"


@dataclass
class TaskAnalysis:
    """Comprehensive task analysis result."""
    task_description: str
    task_intent: str
    task_category: str  # 'file_operation', 'data_processing', 'analysis', etc.
    
    # Requirements
    required_capabilities: List[str]
    input_specifications: List[str]
    output_specifications: List[str]
    
    # Constraints
    performance_constraints: Dict[str, Any]
    quality_requirements: List[str]
    error_handling_needs: List[str]
    
    # Context
    complexity_level: str  # 'simple', 'moderate', 'complex'
    estimated_steps: int
    skill_level_required: str  # 'beginner', 'intermediate', 'advanced'
    
    # Metadata
    confidence: float
    analysis_notes: List[str]


@dataclass
class TaskRequirements:
    """Extracted task requirements."""
    functional_requirements: List[str]
    non_functional_requirements: List[str]
    input_types: List[str]
    output_types: List[str]
    performance_requirements: Dict[str, Any]
    reliability_requirements: List[str]
    usability_requirements: List[str]


class TaskNLPProcessor:
    """Natural language processing for task descriptions."""
    
    def __init__(self):
        """Initialize the NLP processor."""
        # Define common patterns for intent extraction
        self.action_patterns = {
            'read': r'\b(read|load|import|parse)\b',
            'write': r'\b(write|save|export|output)\b',
            'convert': r'\b(convert|transform|change)\b',
            'analyze': r'\b(analyze|analyse|examine|study)\b',
            'process': r'\b(process|handle|manage)\b',
            'download': r'\b(download|fetch|retrieve)\b',
            'upload': r'\b(upload|send|publish)\b',
            'resize': r'\b(resize|scale|adjust)\b',
            'clean': r'\b(clean|sanitize|normalize)\b'
        }
        
        self.capability_patterns = {
            'read': r'\b(read|load|import|parse)\b',
            'write': r'\b(write|save|export|output)\b',
            'analysis': r'\b(analyze|analyse|examine|study|identify)\b',
            'download': r'\b(download|fetch|retrieve)\b',
            'file_handling': r'\b(file|files|document|documents)\b',
            'data_processing': r'\b(data|dataset|database|records)\b',
            'image_processing': r'\b(image|images|photo|photos|picture|pictures|thumbnail|thumbnails)\b',
            'web_scraping': r'\b(scrape|scraping|website|webpage|web)\b',
            'scraping': r'\b(scrape|scraping)\b',
            'compression': r'\b(compress|compression|zip|archive)\b',
            'encryption': r'\b(encrypt|encryption|secure|security)\b',
            'machine_learning': r'\b(machine learning|ml|model|training|prediction)\b'
        }
    
    def extract_intent(self, task_description: str) -> str:
        """Extract the primary intent from task description."""
        description_lower = task_description.lower()
        
        for intent, pattern in self.action_patterns.items():
            if re.search(pattern, description_lower):
                return intent
                
        return "general"
    
    def extract_capabilities(self, task_description: str) -> List[str]:
        """Extract required capabilities from task description."""
        capabilities = []
        description_lower = task_description.lower()
        
        for capability, pattern in self.capability_patterns.items():
            if re.search(pattern, description_lower):
                capabilities.append(capability)
                
        return capabilities


class TaskPatternRecognizer:
    """Recognize common task patterns and workflows."""
    
    def __init__(self):
        """Initialize the pattern recognizer."""
        self.pattern_definitions = {
            'file_conversion': r'\b(convert|transform|change).*\b(to|into|as)\b',
            'data_format_transformation': r'\b(csv|json|xml|parquet|excel|pdf|text)\b.*\b(csv|json|xml|parquet|excel|pdf|text)\b',
            'data_cleaning': r'\b(clean|remove\s+duplicates|missing\s+values|normalize)\b',
            'data_preprocessing': r'\b(preprocess|prepare|clean|normalize|transform)\s+data',
            'database_backup': r'\b(backup|copy)\s+database\b',
            'web_scraping': r'\b(scrape|extract)\s+.*\bwebsite\b'
        }
    
    def recognize_patterns(self, task_description: str) -> List[str]:
        """Recognize patterns in the task description."""
        patterns = []
        description_lower = task_description.lower()
        
        for pattern_name, pattern_regex in self.pattern_definitions.items():
            if re.search(pattern_regex, description_lower):
                patterns.append(pattern_name)
                
        return patterns


class RequirementExtractor:
    """Extract specific requirements from task descriptions."""
    
    def __init__(self):
        """Initialize the requirement extractor."""
        self.functional_patterns = {
            'database_backup': r'\b(backup|copy).*\bdatabase\b',
            'file_conversion': r'\b(convert|transform|change)\b',
            'compression': r'\b(compress|compression|zip)\b',
            'encryption': r'\b(encrypt|encryption|secure)\b',
            'data_validation': r'\b(validate|validation|verify)\b'
        }
        
        self.file_type_patterns = {
            'csv': r'\bcsv\b',
            'json': r'\bjson\b',
            'xml': r'\bxml\b',
            'excel': r'\b(excel|xlsx|xls)\b',
            'pdf': r'\bpdf\b',
            'text': r'\b(text|txt)\b',
            'parquet': r'\bparquet\b',
            'database': r'\b(database|db)\b'
        }
        
        self.performance_patterns = {
            'processing_speed': r'\b(fast|quick|speed|performance|quickly)\b',
            'performance': r'\b(performance|efficiently|fast|quick|quickly)\b',
            'memory_efficiency': r'\b(low\s+memory|memory\s+efficient|memory\s+usage)\b'
        }
    
    def extract_functional_requirements(self, task_description: str) -> List[str]:
        """Extract functional requirements from task description."""
        requirements = []
        description_lower = task_description.lower()
        
        for req_name, pattern in self.functional_patterns.items():
            if re.search(pattern, description_lower):
                requirements.append(req_name)
                
        return requirements
    
    def extract_input_output_types(self, task_description: str) -> Tuple[List[str], List[str]]:
        """Extract input and output type specifications."""
        input_types = []
        output_types = []
        description_lower = task_description.lower()
        
        # Look for conversion patterns like "Convert PDF to text"
        for file_type, pattern in self.file_type_patterns.items():
            if re.search(pattern, description_lower):
                # Check if this appears before "to" (input) or after "to" (output)
                to_match = re.search(rf'{pattern}.*\bto\b', description_lower)
                from_match = re.search(rf'\bto\b.*{pattern}', description_lower)
                
                if to_match and not from_match:
                    input_types.append(file_type)
                elif from_match and not to_match:
                    output_types.append(file_type)
                elif not to_match and not from_match:
                    # No conversion context, could be either
                    input_types.append(file_type)
                    
        return input_types, output_types
    
    def extract_performance_requirements(self, task_description: str) -> Dict[str, str]:
        """Extract performance requirements from task description."""
        requirements = {}
        description_lower = task_description.lower()
        
        for perf_name, pattern in self.performance_patterns.items():
            if re.search(pattern, description_lower):
                requirements[perf_name] = "required"
                
        return requirements


class TaskAnalyzer:
    """Analyze development tasks for tool recommendation."""
    
    def __init__(self):
        """Initialize the task analyzer."""
        self.nlp_processor = TaskNLPProcessor()
        self.pattern_recognizer = TaskPatternRecognizer()
        self.requirement_extractor = RequirementExtractor()
        
        # Define category mappings (order matters - more specific first)
        self.category_patterns = {
            'web_scraping': [r'\b(scrape|scraping|website|web)\b'],
            'machine_learning': [r'\b(machine learning|ml|model|training)\b'],
            'data_extraction': [r'\b(extract|extraction)\b'],
            'data_processing': [r'\b(data|dataset|process|analyze|clean)\b'],
            'file_operation': [r'\b(file|files|read|write|load|save)\b']
        }
        
        # Complexity indicators
        self.simple_indicators = [
            r'\b(list|show|display|read|simple)\b'
        ]
        
        self.complex_indicators = [
            r'\b(pipeline|machine learning|deployment|monitoring|complex|advanced)\b',
            r'\b(analyze.*and.*generate)\b',
            r'\b(multiple|several|various)\b.*\b(steps|stages|processes)\b'
        ]
    
    def analyze_task(self, task_description: str) -> TaskAnalysis:
        """Comprehensive analysis of task description."""
        logger.info(f"Analyzing task: {task_description}")
        
        # Extract basic components
        task_intent = self.nlp_processor.extract_intent(task_description)
        capabilities = self.nlp_processor.extract_capabilities(task_description)
        patterns = self.pattern_recognizer.recognize_patterns(task_description)
        
        # Determine category
        category = self._categorize_task(task_description)
        
        # Extract requirements
        requirements = self.extract_task_requirements(task_description)
        
        # Estimate complexity and steps
        complexity_level = self._determine_complexity(task_description, patterns)
        estimated_steps = self._estimate_steps(task_description, complexity_level)
        skill_level = self._determine_skill_level(complexity_level, patterns)
        
        # Calculate confidence
        confidence = self._calculate_confidence(task_description, patterns, capabilities)
        
        # Create analysis
        analysis = TaskAnalysis(
            task_description=task_description,
            task_intent=task_intent,
            task_category=category,
            required_capabilities=capabilities,
            input_specifications=requirements.input_types,
            output_specifications=requirements.output_types,
            performance_constraints=requirements.performance_requirements,
            quality_requirements=requirements.usability_requirements,
            error_handling_needs=requirements.reliability_requirements,
            complexity_level=complexity_level,
            estimated_steps=estimated_steps,
            skill_level_required=skill_level,
            confidence=confidence,
            analysis_notes=[f"Detected patterns: {patterns}"]
        )
        
        logger.info(f"Task analysis complete: category={category}, complexity={complexity_level}")
        return analysis
    
    def extract_task_requirements(self, task_description: str) -> TaskRequirements:
        """Extract specific requirements from task description."""
        functional_reqs = self.requirement_extractor.extract_functional_requirements(task_description)
        input_types, output_types = self.requirement_extractor.extract_input_output_types(task_description)
        performance_reqs = self.requirement_extractor.extract_performance_requirements(task_description)
        
        # Extract reliability requirements from description
        reliability_reqs = []
        if re.search(r'\berror\s+handling\b', task_description.lower()):
            reliability_reqs.append("error_handling")
        if re.search(r'\bvalidation\b', task_description.lower()):
            reliability_reqs.append("validation")
            
        return TaskRequirements(
            functional_requirements=functional_reqs,
            non_functional_requirements=list(performance_reqs.keys()),
            input_types=input_types,
            output_types=output_types,
            performance_requirements=performance_reqs,
            reliability_requirements=reliability_reqs,
            usability_requirements=[]
        )
    
    def classify_task_complexity(self, task_analysis: TaskAnalysis) -> TaskComplexity:
        """Classify task complexity for appropriate tool matching."""
        # Check estimated steps
        if task_analysis.estimated_steps <= 2:
            return TaskComplexity.SIMPLE
        elif task_analysis.estimated_steps >= 8:
            return TaskComplexity.COMPLEX
        else:
            return TaskComplexity.MODERATE
    
    def _categorize_task(self, task_description: str) -> str:
        """Categorize the task based on description patterns."""
        description_lower = task_description.lower()
        
        for category, patterns in self.category_patterns.items():
            for pattern in patterns:
                if re.search(pattern, description_lower):
                    return category
                    
        return "general"
    
    def _determine_complexity(self, task_description: str, patterns: List[str]) -> str:
        """Determine task complexity based on description and patterns."""
        description_lower = task_description.lower()
        
        # Check for complex indicators
        for pattern in self.complex_indicators:
            if re.search(pattern, description_lower):
                return "complex"
                
        # Check for simple indicators  
        for pattern in self.simple_indicators:
            if re.search(pattern, description_lower):
                return "simple"
                
        # Check pattern complexity
        if len(patterns) >= 3:
            return "complex"
        elif len(patterns) <= 1:
            return "simple"
        else:
            return "moderate"
    
    def _estimate_steps(self, task_description: str, complexity_level: str) -> int:
        """Estimate number of steps required."""
        if complexity_level == "simple":
            return 1
        elif complexity_level == "moderate":
            return 3 + len(re.findall(r'\band\b', task_description.lower()))
        else:  # complex
            base_steps = 6
            # Add steps for each "and" conjunction
            additional_steps = len(re.findall(r'\band\b', task_description.lower()))
            return base_steps + additional_steps
    
    def _determine_skill_level(self, complexity_level: str, patterns: List[str]) -> str:
        """Determine required skill level."""
        if complexity_level == "simple":
            return "beginner"
        elif complexity_level == "complex" or any("machine_learning" in p for p in patterns):
            return "advanced"
        else:
            return "intermediate"
    
    def _calculate_confidence(self, task_description: str, patterns: List[str], capabilities: List[str]) -> float:
        """Calculate confidence in the analysis."""
        base_confidence = 0.7
        
        # Boost confidence if we detected patterns
        if len(patterns) > 0:
            base_confidence += 0.1
            
        # Boost confidence if we detected capabilities
        if len(capabilities) > 0:
            base_confidence += 0.1
            
        # Boost confidence for clear, specific descriptions
        if len(task_description.split()) > 5:
            base_confidence += 0.1
            
        return min(base_confidence, 1.0)