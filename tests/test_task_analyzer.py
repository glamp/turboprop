#!/usr/bin/env python3
"""
Tests for task_analyzer.py - Task Analysis for Tool Recommendations

This module tests the task analysis functionality that parses natural language
task descriptions and extracts requirements for tool recommendation.
"""

from dataclasses import asdict

import pytest
from task_analyzer import (
    RequirementExtractor,
    TaskAnalysis,
    TaskAnalyzer,
    TaskComplexity,
    TaskNLPProcessor,
    TaskPatternRecognizer,
    TaskRequirements,
)


class TestTaskAnalyzer:
    """Test cases for TaskAnalyzer class."""

    @pytest.fixture
    def task_analyzer(self):
        """Create a TaskAnalyzer instance for testing."""
        return TaskAnalyzer()

    def test_analyze_simple_file_operation_task(self, task_analyzer):
        """Test analysis of a simple file operation task."""
        task_description = "Read a CSV file and extract the column names"

        result = task_analyzer.analyze_task(task_description)

        assert isinstance(result, TaskAnalysis)
        assert result.task_description == task_description
        assert result.task_category in ["file_operation", "data_extraction"]
        assert result.complexity_level == "simple"
        assert "read" in result.required_capabilities
        assert "csv" in result.input_specifications
        assert len(result.analysis_notes) > 0
        assert result.confidence > 0.7

    def test_analyze_complex_data_processing_task(self, task_analyzer):
        """Test analysis of a complex data processing task."""
        task_description = (
            "Analyze large dataset to identify anomalies using machine learning "
            "and generate detailed report with visualizations"
        )

        result = task_analyzer.analyze_task(task_description)

        assert result.task_category in ["data_processing", "machine_learning"]
        assert result.complexity_level == "complex"
        assert result.skill_level_required == "advanced"
        assert "analysis" in result.required_capabilities
        assert "machine_learning" in result.required_capabilities
        assert result.estimated_steps >= 5

    def test_analyze_web_scraping_task(self, task_analyzer):
        """Test analysis of a web scraping task."""
        task_description = "Scrape product information from an e-commerce website and save to database"

        result = task_analyzer.analyze_task(task_description)

        assert result.task_category in ["web_scraping", "data_extraction"]
        assert "scraping" in result.required_capabilities
        assert "database" in result.output_specifications
        assert result.complexity_level in ["simple", "moderate", "complex"]

    def test_extract_task_requirements_file_task(self, task_analyzer):
        """Test requirement extraction for file operations."""
        task_description = "Convert Excel files to JSON format with error handling"

        requirements = task_analyzer.extract_task_requirements(task_description)

        assert isinstance(requirements, TaskRequirements)
        assert "file_conversion" in requirements.functional_requirements
        assert "excel" in requirements.input_types
        assert "json" in requirements.output_types
        assert "error_handling" in requirements.reliability_requirements

    def test_classify_task_complexity_simple(self, task_analyzer):
        """Test complexity classification for simple tasks."""
        # Create a simple task analysis
        task_analysis = TaskAnalysis(
            task_description="List files in directory",
            task_intent="file_listing",
            task_category="file_operation",
            required_capabilities=["list", "directory"],
            input_specifications=["directory_path"],
            output_specifications=["file_list"],
            performance_constraints={},
            quality_requirements=[],
            error_handling_needs=[],
            complexity_level="",  # Will be set by classifier
            estimated_steps=1,
            skill_level_required="",  # Will be set by classifier
            confidence=0.9,
            analysis_notes=[],
        )

        complexity = task_analyzer.classify_task_complexity(task_analysis)

        assert complexity == TaskComplexity.SIMPLE

    def test_classify_task_complexity_complex(self, task_analyzer):
        """Test complexity classification for complex tasks."""
        # Create a complex task analysis
        task_analysis = TaskAnalysis(
            task_description="Build ML pipeline with data preprocessing, model training, and deployment",
            task_intent="ml_pipeline_creation",
            task_category="machine_learning",
            required_capabilities=["data_preprocessing", "model_training", "deployment", "monitoring"],
            input_specifications=["raw_data", "requirements"],
            output_specifications=["trained_model", "api_endpoint", "monitoring_dashboard"],
            performance_constraints={"latency": "< 100ms", "throughput": "> 1000 rps"},
            quality_requirements=["accuracy > 95%", "reliability > 99.9%"],
            error_handling_needs=["data_validation", "model_fallback", "monitoring_alerts"],
            complexity_level="",  # Will be set by classifier
            estimated_steps=10,
            skill_level_required="",  # Will be set by classifier
            confidence=0.8,
            analysis_notes=[],
        )

        complexity = task_analyzer.classify_task_complexity(task_analysis)

        assert complexity == TaskComplexity.COMPLEX

    def test_task_analysis_dataclass_serialization(self):
        """Test that TaskAnalysis can be serialized to dict."""
        analysis = TaskAnalysis(
            task_description="test task",
            task_intent="testing",
            task_category="test",
            required_capabilities=["test_capability"],
            input_specifications=["test_input"],
            output_specifications=["test_output"],
            performance_constraints={"test": "constraint"},
            quality_requirements=["test_quality"],
            error_handling_needs=["test_error_handling"],
            complexity_level="simple",
            estimated_steps=1,
            skill_level_required="beginner",
            confidence=0.9,
            analysis_notes=["test_note"],
        )

        result_dict = asdict(analysis)
        assert result_dict["task_description"] == "test task"
        assert result_dict["confidence"] == 0.9


class TestTaskNLPProcessor:
    """Test cases for TaskNLPProcessor class."""

    @pytest.fixture
    def nlp_processor(self):
        """Create a TaskNLPProcessor instance for testing."""
        return TaskNLPProcessor()

    def test_extract_intent_file_operation(self, nlp_processor):
        """Test intent extraction for file operations."""
        task_description = "Read and parse JSON configuration file"

        intent = nlp_processor.extract_intent(task_description)

        assert "read" in intent or "parse" in intent

    def test_extract_capabilities_from_description(self, nlp_processor):
        """Test capability extraction from task descriptions."""
        task_description = "Download images from URLs and resize them for thumbnails"

        capabilities = nlp_processor.extract_capabilities(task_description)

        assert "download" in capabilities
        assert "image_processing" in capabilities or "resize" in capabilities


class TestTaskPatternRecognizer:
    """Test cases for TaskPatternRecognizer class."""

    @pytest.fixture
    def pattern_recognizer(self):
        """Create a TaskPatternRecognizer instance for testing."""
        return TaskPatternRecognizer()

    def test_recognize_file_operation_patterns(self, pattern_recognizer):
        """Test recognition of file operation patterns."""
        task_description = "Convert CSV files to Parquet format"

        patterns = pattern_recognizer.recognize_patterns(task_description)

        assert "file_conversion" in patterns
        assert "data_format_transformation" in patterns

    def test_recognize_data_processing_patterns(self, pattern_recognizer):
        """Test recognition of data processing patterns."""
        task_description = "Clean dataset by removing duplicates and handling missing values"

        patterns = pattern_recognizer.recognize_patterns(task_description)

        assert "data_cleaning" in patterns
        assert "data_preprocessing" in patterns


class TestRequirementExtractor:
    """Test cases for RequirementExtractor class."""

    @pytest.fixture
    def requirement_extractor(self):
        """Create a RequirementExtractor instance for testing."""
        return RequirementExtractor()

    def test_extract_functional_requirements(self, requirement_extractor):
        """Test extraction of functional requirements."""
        task_description = "Create backup of database with compression and encryption"

        requirements = requirement_extractor.extract_functional_requirements(task_description)

        assert "database_backup" in requirements
        assert "compression" in requirements
        assert "encryption" in requirements

    def test_extract_input_output_types(self, requirement_extractor):
        """Test extraction of input/output type specifications."""
        task_description = "Convert PDF documents to searchable text files"

        input_types, output_types = requirement_extractor.extract_input_output_types(task_description)

        assert "pdf" in input_types
        assert "text" in output_types

    def test_extract_performance_requirements(self, requirement_extractor):
        """Test extraction of performance requirements."""
        task_description = "Process large files quickly with low memory usage"

        requirements = requirement_extractor.extract_performance_requirements(task_description)

        assert "processing_speed" in requirements or "performance" in requirements
        assert "memory_efficiency" in requirements or "low_memory" in requirements
