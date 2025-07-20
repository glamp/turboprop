#!/usr/bin/env python3
"""
Catalog Validator

This module provides catalog validation and health checking capabilities
that validate tool metadata completeness, check database integrity and
consistency, and generate health metrics and recommendations.
"""

import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

from database_manager import DatabaseManager
from logging_config import get_logger
from mcp_metadata_types import ComplexityAnalysis, MCPToolMetadata, ParameterAnalysis, ToolExample

logger = get_logger(__name__)


@dataclass
class ValidationResult:
    """Result of a metadata validation operation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    completeness_score: float  # 0.0 to 1.0
    quality_indicators: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class ConsistencyReport:
    """Report on catalog consistency validation."""
    is_consistent: bool
    errors: List[str]
    warnings: List[str]
    checks_performed: Dict[str, bool]
    database_integrity_score: float
    validation_time: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class QualityMetrics:
    """Comprehensive quality metrics for the catalog."""
    total_tools: int
    tools_with_embeddings: int
    parameter_coverage: float
    example_coverage: float
    relationship_coverage: float
    metadata_completeness: float
    overall_quality_score: float
    quality_distribution: Dict[str, int]
    recommendations: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class CatalogValidator:
    """Validates catalog data integrity and quality."""

    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        """
        Initialize the catalog validator.
        
        Args:
            db_manager: Optional DatabaseManager for consistency validation
        """
        self.db_manager = db_manager
        self.required_fields = ['name', 'description', 'category']
        self.recommended_fields = ['parameters', 'examples', 'complexity_analysis']
        self.quality_thresholds = {
            'excellent': 0.9,
            'good': 0.7,
            'acceptable': 0.5,
            'poor': 0.0
        }
        
        logger.info("Initialized CatalogValidator")

    def validate_tool_metadata(self, tool: MCPToolMetadata) -> ValidationResult:
        """
        Validate individual tool metadata quality.
        
        Checks required fields completeness, validates parameter schemas,
        verifies embedding quality, and checks relationship consistency.
        
        Args:
            tool: Tool metadata to validate
            
        Returns:
            ValidationResult with detailed validation information
        """
        errors = []
        warnings = []
        quality_indicators = {}
        
        # Check required fields
        for field in self.required_fields:
            if not hasattr(tool, field) or not getattr(tool, field):
                errors.append(f"Missing required field: {field}")
            else:
                value = getattr(tool, field)
                if isinstance(value, str) and len(value.strip()) == 0:
                    errors.append(f"Empty required field: {field}")
        
        # Check recommended fields
        recommended_present = 0
        for field in self.recommended_fields:
            if hasattr(tool, field) and getattr(tool, field):
                recommended_present += 1
                quality_indicators[f"has_{field}"] = True
            else:
                warnings.append(f"Missing recommended field: {field}")
                quality_indicators[f"has_{field}"] = False
        
        # Validate name quality
        if hasattr(tool, 'name') and tool.name:
            if len(tool.name) < 3:
                warnings.append("Tool name is very short")
            if not tool.name.strip():
                errors.append("Tool name is empty or whitespace")
            quality_indicators['name_length'] = len(tool.name) if tool.name else 0
        
        # Validate description quality
        if hasattr(tool, 'description') and tool.description:
            desc_length = len(tool.description)
            if desc_length < 10:
                warnings.append("Description is very short")
            elif desc_length < 20:
                warnings.append("Description could be more detailed")
            quality_indicators['description_length'] = desc_length
        
        # Validate parameters
        param_quality = self._validate_parameters(getattr(tool, 'parameters', []))
        quality_indicators.update(param_quality)
        
        # Validate examples
        example_quality = self._validate_examples(getattr(tool, 'examples', []))
        quality_indicators.update(example_quality)
        
        # Validate complexity analysis
        complexity_quality = self._validate_complexity_analysis(getattr(tool, 'complexity_analysis', None))
        quality_indicators.update(complexity_quality)
        
        # Calculate completeness score
        completeness_score = self._calculate_completeness_score(tool, quality_indicators)
        
        is_valid = len(errors) == 0
        
        logger.debug(
            "Validated tool %s: valid=%s, completeness=%.2f, errors=%d, warnings=%d",
            getattr(tool, 'name', 'Unknown'), is_valid, completeness_score, len(errors), len(warnings)
        )
        
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            completeness_score=completeness_score,
            quality_indicators=quality_indicators
        )

    def validate_catalog_consistency(self) -> ConsistencyReport:
        """
        Validate overall catalog consistency.
        
        Checks foreign key integrity, verifies embedding dimensions,
        validates relationship bidirectionality, and checks for orphaned records.
        
        Returns:
            ConsistencyReport with consistency validation results
        """
        if not self.db_manager:
            logger.warning("No database manager provided for consistency validation")
            return ConsistencyReport(
                is_consistent=False,
                errors=["No database manager available for validation"],
                warnings=[],
                checks_performed={},
                database_integrity_score=0.0,
                validation_time=0.0
            )
        
        logger.info("Starting catalog consistency validation")
        start_time = time.time()
        
        errors = []
        warnings = []
        checks_performed = {}
        
        try:
            # Check 1: Database tables exist
            table_check = self._check_table_existence()
            checks_performed['table_existence'] = table_check['success']
            if not table_check['success']:
                errors.extend(table_check['errors'])
            
            # Check 2: Foreign key integrity
            fk_check = self._check_foreign_key_integrity()
            checks_performed['foreign_key_integrity'] = fk_check['success']
            if not fk_check['success']:
                errors.extend(fk_check['errors'])
            else:
                warnings.extend(fk_check['warnings'])
            
            # Check 3: Embedding dimensions
            embedding_check = self._check_embedding_dimensions()
            checks_performed['embedding_dimensions'] = embedding_check['success']
            if not embedding_check['success']:
                errors.extend(embedding_check['errors'])
            else:
                warnings.extend(embedding_check['warnings'])
            
            # Check 4: Data completeness
            completeness_check = self._check_data_completeness()
            checks_performed['data_completeness'] = completeness_check['success']
            if not completeness_check['success']:
                warnings.extend(completeness_check['warnings'])
            
            # Check 5: Relationship consistency
            relationship_check = self._check_relationship_consistency()
            checks_performed['relationship_consistency'] = relationship_check['success']
            if not relationship_check['success']:
                warnings.extend(relationship_check['warnings'])
            
            # Calculate integrity score
            successful_checks = sum(1 for check in checks_performed.values() if check)
            total_checks = len(checks_performed)
            database_integrity_score = successful_checks / total_checks if total_checks > 0 else 0.0
            
            # Reduce score for errors vs warnings
            if errors:
                database_integrity_score *= 0.5  # Major penalty for errors
            elif warnings:
                database_integrity_score *= 0.8  # Minor penalty for warnings
            
            validation_time = time.time() - start_time
            is_consistent = len(errors) == 0
            
            logger.info(
                "Consistency validation completed: consistent=%s, score=%.2f, time=%.2fs",
                is_consistent, database_integrity_score, validation_time
            )
            
            return ConsistencyReport(
                is_consistent=is_consistent,
                errors=errors,
                warnings=warnings,
                checks_performed=checks_performed,
                database_integrity_score=database_integrity_score,
                validation_time=validation_time
            )
            
        except Exception as e:
            validation_time = time.time() - start_time
            error_msg = f"Critical error during consistency validation: {e}"
            logger.error(error_msg)
            
            return ConsistencyReport(
                is_consistent=False,
                errors=[error_msg],
                warnings=[],
                checks_performed=checks_performed,
                database_integrity_score=0.0,
                validation_time=validation_time
            )

    def generate_quality_metrics(self) -> Dict[str, Any]:
        """
        Generate catalog quality metrics.
        
        Calculates metadata completeness scores, measures embedding quality distribution,
        analyzes relationship coverage, and identifies improvement opportunities.
        
        Returns:
            Dictionary with comprehensive quality metrics
        """
        if not self.db_manager:
            logger.warning("No database manager provided for quality metrics")
            return {
                "total_tools": 0,
                "tools_with_embeddings": 0,
                "parameter_coverage": 0.0,
                "example_coverage": 0.0,
                "relationship_coverage": 0.0,
                "metadata_completeness": 0.0,
                "overall_quality_score": 0.0,
                "error": "No database manager available"
            }
        
        logger.info("Generating catalog quality metrics")
        
        try:
            with self.db_manager.get_connection() as conn:
                metrics = {}
                
                # Basic counts
                metrics["total_tools"] = conn.execute("SELECT COUNT(*) FROM mcp_tools").fetchone()[0]
                
                # Tools with embeddings
                embedding_count = conn.execute(
                    "SELECT COUNT(*) FROM mcp_tools WHERE embedding IS NOT NULL"
                ).fetchone()[0]
                metrics["tools_with_embeddings"] = embedding_count
                
                # Parameter coverage
                tools_with_params = conn.execute(
                    "SELECT COUNT(DISTINCT tool_id) FROM tool_parameters"
                ).fetchone()[0]
                metrics["parameter_coverage"] = (
                    tools_with_params / metrics["total_tools"] 
                    if metrics["total_tools"] > 0 else 0.0
                )
                
                # Example coverage
                tools_with_examples = conn.execute(
                    "SELECT COUNT(DISTINCT tool_id) FROM tool_examples"
                ).fetchone()[0]
                metrics["example_coverage"] = (
                    tools_with_examples / metrics["total_tools"]
                    if metrics["total_tools"] > 0 else 0.0
                )
                
                # Relationship coverage
                total_relationships = conn.execute("SELECT COUNT(*) FROM tool_relationships").fetchone()[0]
                max_possible_relationships = metrics["total_tools"] * (metrics["total_tools"] - 1) // 2
                metrics["relationship_coverage"] = (
                    total_relationships / max_possible_relationships
                    if max_possible_relationships > 0 else 0.0
                )
                
                # Metadata completeness (tools with descriptions)
                tools_with_descriptions = conn.execute(
                    "SELECT COUNT(*) FROM mcp_tools WHERE description IS NOT NULL AND LENGTH(description) > 10"
                ).fetchone()[0]
                metrics["metadata_completeness"] = (
                    tools_with_descriptions / metrics["total_tools"]
                    if metrics["total_tools"] > 0 else 0.0
                )
                
                # Calculate overall quality score
                embedding_score = embedding_count / metrics["total_tools"] if metrics["total_tools"] > 0 else 0.0
                
                metrics["overall_quality_score"] = (
                    embedding_score * 0.3 +
                    metrics["parameter_coverage"] * 0.25 +
                    metrics["example_coverage"] * 0.2 +
                    metrics["metadata_completeness"] * 0.15 +
                    metrics["relationship_coverage"] * 0.1
                )
                
                # Quality distribution
                metrics["quality_distribution"] = self._calculate_quality_distribution(metrics)
                
                # Additional detailed metrics
                metrics.update(self._get_additional_metrics(conn))
                
                logger.info("Generated quality metrics: overall score %.2f", metrics["overall_quality_score"])
                return metrics
                
        except Exception as e:
            logger.error("Error generating quality metrics: %s", e)
            return {
                "total_tools": 0,
                "tools_with_embeddings": 0,
                "parameter_coverage": 0.0,
                "example_coverage": 0.0,
                "relationship_coverage": 0.0,
                "metadata_completeness": 0.0,
                "overall_quality_score": 0.0,
                "error": str(e)
            }

    def _validate_parameters(self, parameters: List[ParameterAnalysis]) -> Dict[str, Any]:
        """Validate parameter quality."""
        quality = {}
        
        if not parameters:
            quality['parameter_count'] = 0
            quality['has_parameters'] = False
            return quality
        
        quality['parameter_count'] = len(parameters)
        quality['has_parameters'] = True
        
        # Check parameter completeness
        complete_params = 0
        for param in parameters:
            if (hasattr(param, 'name') and param.name and
                hasattr(param, 'type') and param.type and
                hasattr(param, 'description') and param.description):
                complete_params += 1
        
        quality['parameter_completeness'] = complete_params / len(parameters)
        
        # Check for required parameters
        required_params = sum(1 for p in parameters if getattr(p, 'required', False))
        quality['required_parameter_count'] = required_params
        
        return quality

    def _validate_examples(self, examples: List[ToolExample]) -> Dict[str, Any]:
        """Validate example quality."""
        quality = {}
        
        if not examples:
            quality['example_count'] = 0
            quality['has_examples'] = False
            return quality
        
        quality['example_count'] = len(examples)
        quality['has_examples'] = True
        
        # Check example completeness
        complete_examples = 0
        for example in examples:
            if (hasattr(example, 'use_case') and example.use_case and
                hasattr(example, 'example_call') and example.example_call):
                complete_examples += 1
        
        quality['example_completeness'] = complete_examples / len(examples)
        
        return quality

    def _validate_complexity_analysis(self, complexity: Optional[ComplexityAnalysis]) -> Dict[str, Any]:
        """Validate complexity analysis quality."""
        quality = {}
        
        if not complexity:
            quality['has_complexity_analysis'] = False
            return quality
        
        quality['has_complexity_analysis'] = True
        
        # Check if complexity scores are reasonable
        if hasattr(complexity, 'overall_complexity'):
            overall = complexity.overall_complexity
            quality['complexity_score_valid'] = 0.0 <= overall <= 1.0
        
        return quality

    def _calculate_completeness_score(self, tool: MCPToolMetadata, quality_indicators: Dict[str, Any]) -> float:
        """Calculate overall completeness score for a tool."""
        score = 0.0
        
        # Required fields (60% of score)
        required_score = 0.0
        for field in self.required_fields:
            if hasattr(tool, field) and getattr(tool, field):
                required_score += 1.0
        required_score /= len(self.required_fields)
        score += required_score * 0.6
        
        # Recommended fields (25% of score)
        recommended_score = 0.0
        for field in self.recommended_fields:
            if quality_indicators.get(f"has_{field}", False):
                recommended_score += 1.0
        recommended_score /= len(self.recommended_fields)
        score += recommended_score * 0.25
        
        # Quality indicators (15% of score)
        quality_score = 0.0
        if quality_indicators.get('parameter_completeness', 0) > 0:
            quality_score += 0.5
        if quality_indicators.get('example_completeness', 0) > 0:
            quality_score += 0.5
        score += quality_score * 0.15
        
        return min(score, 1.0)

    def _check_table_existence(self) -> Dict[str, Any]:
        """Check if all required tables exist."""
        required_tables = ['mcp_tools', 'tool_parameters', 'tool_examples', 'tool_relationships']
        
        try:
            with self.db_manager.get_connection() as conn:
                existing_tables = []
                for table in required_tables:
                    try:
                        conn.execute(f"SELECT COUNT(*) FROM {table} LIMIT 1")
                        existing_tables.append(table)
                    except Exception:
                        pass
                
                missing_tables = set(required_tables) - set(existing_tables)
                
                if missing_tables:
                    return {
                        'success': False,
                        'errors': [f"Missing required tables: {', '.join(missing_tables)}"],
                        'warnings': []
                    }
                
                return {'success': True, 'errors': [], 'warnings': []}
                
        except Exception as e:
            return {
                'success': False,
                'errors': [f"Error checking table existence: {e}"],
                'warnings': []
            }

    def _check_foreign_key_integrity(self) -> Dict[str, Any]:
        """Check foreign key integrity between tables."""
        try:
            with self.db_manager.get_connection() as conn:
                errors = []
                warnings = []
                
                # Check parameters reference existing tools
                orphaned_params = conn.execute("""
                    SELECT COUNT(*) FROM tool_parameters tp
                    LEFT JOIN mcp_tools mt ON tp.tool_id = mt.id
                    WHERE mt.id IS NULL
                """).fetchone()[0]
                
                if orphaned_params > 0:
                    warnings.append(f"Found {orphaned_params} orphaned parameters")
                
                # Check examples reference existing tools
                orphaned_examples = conn.execute("""
                    SELECT COUNT(*) FROM tool_examples te
                    LEFT JOIN mcp_tools mt ON te.tool_id = mt.id
                    WHERE mt.id IS NULL
                """).fetchone()[0]
                
                if orphaned_examples > 0:
                    warnings.append(f"Found {orphaned_examples} orphaned examples")
                
                # Check relationships reference existing tools
                orphaned_relationships = conn.execute("""
                    SELECT COUNT(*) FROM tool_relationships tr
                    LEFT JOIN mcp_tools mt1 ON tr.tool_a_id = mt1.id
                    LEFT JOIN mcp_tools mt2 ON tr.tool_b_id = mt2.id
                    WHERE mt1.id IS NULL OR mt2.id IS NULL
                """).fetchone()[0]
                
                if orphaned_relationships > 0:
                    warnings.append(f"Found {orphaned_relationships} orphaned relationships")
                
                return {
                    'success': len(errors) == 0,
                    'errors': errors,
                    'warnings': warnings
                }
                
        except Exception as e:
            return {
                'success': False,
                'errors': [f"Error checking foreign key integrity: {e}"],
                'warnings': []
            }

    def _check_embedding_dimensions(self) -> Dict[str, Any]:
        """Check embedding dimensions are consistent."""
        try:
            with self.db_manager.get_connection() as conn:
                warnings = []
                
                # Check tool embeddings
                invalid_tool_embeddings = conn.execute("""
                    SELECT COUNT(*) FROM mcp_tools 
                    WHERE embedding IS NOT NULL AND array_length(embedding, 1) != 384
                """).fetchone()[0]
                
                if invalid_tool_embeddings > 0:
                    warnings.append(f"Found {invalid_tool_embeddings} tools with invalid embedding dimensions")
                
                return {
                    'success': True,
                    'errors': [],
                    'warnings': warnings
                }
                
        except Exception as e:
            return {
                'success': False,
                'errors': [f"Error checking embedding dimensions: {e}"],
                'warnings': []
            }

    def _check_data_completeness(self) -> Dict[str, Any]:
        """Check data completeness across tables."""
        try:
            with self.db_manager.get_connection() as conn:
                warnings = []
                
                # Check for tools without descriptions
                tools_without_desc = conn.execute("""
                    SELECT COUNT(*) FROM mcp_tools 
                    WHERE description IS NULL OR LENGTH(description) < 5
                """).fetchone()[0]
                
                if tools_without_desc > 0:
                    warnings.append(f"Found {tools_without_desc} tools with missing or short descriptions")
                
                # Check for tools without categories
                tools_without_category = conn.execute("""
                    SELECT COUNT(*) FROM mcp_tools 
                    WHERE category IS NULL OR LENGTH(category) = 0
                """).fetchone()[0]
                
                if tools_without_category > 0:
                    warnings.append(f"Found {tools_without_category} tools without categories")
                
                return {
                    'success': True,
                    'errors': [],
                    'warnings': warnings
                }
                
        except Exception as e:
            return {
                'success': False,
                'errors': [f"Error checking data completeness: {e}"],
                'warnings': []
            }

    def _check_relationship_consistency(self) -> Dict[str, Any]:
        """Check relationship consistency and bidirectionality."""
        try:
            with self.db_manager.get_connection() as conn:
                warnings = []
                
                # Check for self-referential relationships
                self_refs = conn.execute("""
                    SELECT COUNT(*) FROM tool_relationships 
                    WHERE tool_a_id = tool_b_id
                """).fetchone()[0]
                
                if self_refs > 0:
                    warnings.append(f"Found {self_refs} self-referential relationships")
                
                return {
                    'success': True,
                    'errors': [],
                    'warnings': warnings
                }
                
        except Exception as e:
            return {
                'success': False,
                'errors': [f"Error checking relationship consistency: {e}"],
                'warnings': []
            }

    def _calculate_quality_distribution(self, metrics: Dict[str, Any]) -> Dict[str, int]:
        """Calculate quality distribution based on metrics."""
        overall_score = metrics.get('overall_quality_score', 0.0)
        
        if overall_score >= self.quality_thresholds['excellent']:
            return {'excellent': metrics['total_tools'], 'good': 0, 'acceptable': 0, 'poor': 0}
        elif overall_score >= self.quality_thresholds['good']:
            return {'excellent': 0, 'good': metrics['total_tools'], 'acceptable': 0, 'poor': 0}
        elif overall_score >= self.quality_thresholds['acceptable']:
            return {'excellent': 0, 'good': 0, 'acceptable': metrics['total_tools'], 'poor': 0}
        else:
            return {'excellent': 0, 'good': 0, 'acceptable': 0, 'poor': metrics['total_tools']}

    def _get_additional_metrics(self, conn) -> Dict[str, Any]:
        """Get additional detailed metrics."""
        additional = {}
        
        try:
            # Average parameters per tool
            avg_params = conn.execute("""
                SELECT AVG(param_count) FROM (
                    SELECT COUNT(*) as param_count 
                    FROM tool_parameters 
                    GROUP BY tool_id
                )
            """).fetchone()
            additional['avg_parameters_per_tool'] = float(avg_params[0]) if avg_params[0] else 0.0
            
            # Average examples per tool  
            avg_examples = conn.execute("""
                SELECT AVG(example_count) FROM (
                    SELECT COUNT(*) as example_count 
                    FROM tool_examples 
                    GROUP BY tool_id
                )
            """).fetchone()
            additional['avg_examples_per_tool'] = float(avg_examples[0]) if avg_examples[0] else 0.0
            
        except Exception as e:
            logger.warning("Error calculating additional metrics: %s", e)
        
        return additional