#!/usr/bin/env python3
"""
Tool Catalog Manager

This module provides the central orchestrator for the comprehensive tool cataloging
system. It integrates discovery, metadata extraction, embedding generation, relationship
detection, and storage operations to provide complete catalog management functionality.
"""

import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

from catalog_validator import CatalogValidator
from database_manager import DatabaseManager
from embedding_helper import EmbeddingGenerator
from logging_config import get_logger
from mcp_metadata_extractor import MCPMetadataExtractor
from mcp_metadata_types import MCPToolMetadata
from mcp_tool_discovery import MCPToolDiscovery
from tool_embedding_pipeline import ToolEmbeddingPipeline
from tool_relationship_detector import ToolRelationshipDetector
from tool_storage_operations import ToolStorageOperations

logger = get_logger(__name__)

# Configuration constants for catalog management
DEFAULT_BATCH_SIZE = 50  # Default number of tools to process in batches
DEFAULT_MAX_RETRY_ATTEMPTS = 3  # Maximum retry attempts for failed operations
DEFAULT_HEALTH_CHECK_INTERVAL = 300  # Health check interval in seconds (5 minutes)
REBUILD_TIMEOUT_SECONDS = 3600  # Maximum time for full catalog rebuild (1 hour)
VALIDATION_TIMEOUT_SECONDS = 600  # Maximum time for catalog validation (10 minutes)


@dataclass
class CatalogRebuildResult:
    """Result of a full catalog rebuild operation."""

    success: bool
    tools_processed: int
    tools_stored: int
    tools_failed: int
    relationships_created: int
    embeddings_generated: int
    execution_time: float
    errors: List[str]
    phase_timings: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class CatalogUpdateResult:
    """Result of an incremental catalog update operation."""

    success: bool
    tools_updated: int
    tools_added: int
    tools_removed: int
    relationships_updated: int
    embeddings_updated: int
    execution_time: float
    errors: List[str]
    changes_detected: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class CatalogHealthReport:
    """Comprehensive health report for the catalog."""

    is_healthy: bool
    overall_health_score: float
    metrics: Dict[str, Any]
    checks: Dict[str, Dict[str, Any]]
    recommendations: List[str]
    critical_issues: List[str]
    validation_time: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class ToolCatalogManager:
    """Central manager for tool cataloging and storage operations."""

    def __init__(
        self,
        db_manager: DatabaseManager,
        discovery_engine: MCPToolDiscovery,
        metadata_extractor: MCPMetadataExtractor,
        embedding_generator: EmbeddingGenerator,
    ):
        """
        Initialize the tool catalog manager.

        Args:
            db_manager: DatabaseManager for storage operations
            discovery_engine: MCPToolDiscovery for tool discovery
            metadata_extractor: MCPMetadataExtractor for metadata extraction
            embedding_generator: EmbeddingGenerator for semantic embeddings
        """
        self.db_manager = db_manager
        self.discovery_engine = discovery_engine
        self.metadata_extractor = metadata_extractor
        self.embedding_generator = embedding_generator

        # Initialize specialized components
        self.storage_ops = ToolStorageOperations(db_manager)
        self.embedding_pipeline = ToolEmbeddingPipeline(embedding_generator)
        self.relationship_detector = ToolRelationshipDetector()
        self.validator = CatalogValidator(db_manager)

        # Configuration
        self.batch_size = DEFAULT_BATCH_SIZE
        self.max_retry_attempts = DEFAULT_MAX_RETRY_ATTEMPTS
        self.health_check_interval = DEFAULT_HEALTH_CHECK_INTERVAL

        logger.info("Initialized ToolCatalogManager with all components")

    def _execute_discovery_phase(self) -> tuple[List[Any], Dict[str, float], List[str]]:
        """
        Execute the discovery phase of catalog rebuild.
        
        Returns:
            Tuple of (discovered_tools, phase_timings, errors)
        """
        logger.info("Phase 1: Discovering tools")
        phase_start = time.time()
        errors = []
        
        try:
            discovered_tools = self.discovery_engine.discover_system_tools()
            custom_tools = self.discovery_engine.discover_custom_tools()
            all_discovered = discovered_tools + custom_tools
            
            phase_timing = time.time() - phase_start
            logger.info("Discovery completed: %d tools found in %.2fs", len(all_discovered), phase_timing)
            
            return all_discovered, {"discovery": phase_timing}, errors
            
        except Exception as e:
            error_msg = f"Discovery phase failed: {e}"
            errors.append(error_msg)
            logger.error(error_msg)
            phase_timing = time.time() - phase_start
            return [], {"discovery": phase_timing}, errors

    def _execute_analysis_phase(self, discovered_tools: List[Any]) -> tuple[List[Any], int, Dict[str, float], List[str]]:
        """
        Execute the metadata analysis phase.
        
        Args:
            discovered_tools: Tools to analyze
            
        Returns:
            Tuple of (analyzed_tools, tools_failed, phase_timings, errors)
        """
        logger.info("Phase 2: Analyzing metadata")
        phase_start = time.time()
        analyzed_tools = []
        tools_failed = 0
        errors = []
        
        try:
            for tool in discovered_tools:
                try:
                    # Extract comprehensive metadata
                    metadata = self.metadata_extractor.extract_from_tool_definition(
                        {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": [p.to_dict() for p in tool.parameters],
                        }
                    )
                    analyzed_tools.append(metadata)

                except Exception as e:
                    tools_failed += 1
                    error_msg = f"Metadata analysis failed for {tool.name}: {e}"
                    errors.append(error_msg)
                    logger.error(error_msg)

            phase_timing = time.time() - phase_start
            logger.info("Analysis completed: %d tools analyzed in %.2fs", len(analyzed_tools), phase_timing)
            
            return analyzed_tools, tools_failed, {"analysis": phase_timing}, errors

        except Exception as e:
            error_msg = f"Analysis phase failed: {e}"
            errors.append(error_msg)
            logger.error(error_msg)
            phase_timing = time.time() - phase_start
            return [], tools_failed, {"analysis": phase_timing}, errors

    def _execute_embedding_phase(self, analyzed_tools: List[Any]) -> tuple[int, Dict[str, float], List[str]]:
        """
        Execute the embedding generation phase.
        
        Args:
            analyzed_tools: Tools to generate embeddings for
            
        Returns:
            Tuple of (embeddings_generated, phase_timings, errors)
        """
        logger.info("Phase 3: Generating embeddings")
        phase_start = time.time()
        errors = []
        
        try:
            embedding_result = self.embedding_pipeline.generate_tool_embeddings(analyzed_tools)

            if not embedding_result.success:
                errors.extend(embedding_result.errors)
                logger.warning("Embedding generation had errors: %d", len(embedding_result.errors))

            embeddings_generated = len(embedding_result.embeddings)
            phase_timing = time.time() - phase_start
            logger.info("Embedding completed: %d embeddings in %.2fs", embeddings_generated, phase_timing)
            
            return embeddings_generated, {"embedding": phase_timing}, errors

        except Exception as e:
            error_msg = f"Embedding phase failed: {e}"
            errors.append(error_msg)
            logger.error(error_msg)
            phase_timing = time.time() - phase_start
            return 0, {"embedding": phase_timing}, errors

    def _execute_relationship_phase(self, analyzed_tools: List[Any]) -> tuple[List[Any], int, Dict[str, float], List[str]]:
        """
        Execute the relationship detection phase.
        
        Args:
            analyzed_tools: Tools to analyze relationships for
            
        Returns:
            Tuple of (relationships, relationships_count, phase_timings, errors)
        """
        logger.info("Phase 4: Detecting relationships")
        phase_start = time.time()
        errors = []
        
        try:
            relationship_result = self.relationship_detector.analyze_all_relationships(analyzed_tools)

            all_relationships = (
                relationship_result.alternatives
                + relationship_result.complements
                + relationship_result.prerequisites
            )
            relationships_created = len(all_relationships)

            phase_timing = time.time() - phase_start
            logger.info("Relationship detection completed: %d relationships in %.2fs", relationships_created, phase_timing)
            
            return all_relationships, relationships_created, {"relationships": phase_timing}, errors

        except Exception as e:
            error_msg = f"Relationship detection failed: {e}"
            errors.append(error_msg)
            logger.error(error_msg)
            # Continue without relationships - not critical for basic functionality
            phase_timing = time.time() - phase_start
            return [], 0, {"relationships": phase_timing}, errors

    def _execute_storage_phase(self, analyzed_tools: List[Any], relationships: List[Any]) -> tuple[int, int, Dict[str, float], List[str]]:
        """
        Execute the storage phase.
        
        Args:
            analyzed_tools: Tools to store
            relationships: Relationships to store
            
        Returns:
            Tuple of (tools_stored, tools_failed, phase_timings, errors)
        """
        logger.info("Phase 5: Storing catalog data")
        phase_start = time.time()
        errors = []
        tools_failed = 0
        
        try:
            # Store tools with metadata
            storage_result = self.storage_ops.store_tool_batch(analyzed_tools)

            if not storage_result.success:
                errors.extend(storage_result.errors)
                tools_failed += storage_result.tools_failed

            tools_stored = storage_result.tools_stored

            # Store relationships
            if relationships:
                try:
                    self.storage_ops.store_tool_relationships(relationships)
                    logger.info("Stored %d relationships", len(relationships))
                except Exception as e:
                    error_msg = f"Failed to store relationships: {e}"
                    errors.append(error_msg)
                    logger.error(error_msg)

            phase_timing = time.time() - phase_start
            logger.info("Storage completed: %d tools stored in %.2fs", tools_stored, phase_timing)
            
            return tools_stored, tools_failed, {"storage": phase_timing}, errors

        except Exception as e:
            error_msg = f"Storage phase failed: {e}"
            errors.append(error_msg)
            logger.error(error_msg)
            phase_timing = time.time() - phase_start
            return 0, tools_failed, {"storage": phase_timing}, errors

    def _execute_validation_phase(self) -> tuple[Dict[str, float], List[str]]:
        """
        Execute the validation phase.
        
        Returns:
            Tuple of (phase_timings, errors)
        """
        logger.info("Phase 6: Validating catalog")
        phase_start = time.time()
        errors = []
        
        try:
            consistency_report = self.validator.validate_catalog_consistency()

            if not consistency_report.is_consistent:
                errors.extend(consistency_report.errors)
                logger.warning("Catalog consistency issues found: %d errors", len(consistency_report.errors))

            phase_timing = time.time() - phase_start
            logger.info("Validation completed in %.2fs", phase_timing)
            
            return {"validation": phase_timing}, errors

        except Exception as e:
            error_msg = f"Validation phase failed: {e}"
            errors.append(error_msg)
            logger.error(error_msg)
            phase_timing = time.time() - phase_start
            return {"validation": phase_timing}, errors

    def full_catalog_rebuild(self) -> CatalogRebuildResult:
        """
        Perform complete catalog rebuild from scratch.

        Process:
        1. Discovery Phase: Enumerate all available MCP tools
        2. Analysis Phase: Extract comprehensive metadata for each tool
        3. Embedding Phase: Generate semantic embeddings for descriptions and parameters
        4. Relationship Phase: Detect tool relationships and dependencies
        5. Storage Phase: Store all data in database with proper relationships
        6. Validation Phase: Verify catalog integrity and completeness

        Returns:
            CatalogRebuildResult with detailed operation statistics
        """
        logger.info("Starting full catalog rebuild")
        start_time = time.time()

        # Initialize tracking variables
        phase_timings = {}
        all_errors = []

        try:
            # Ensure database tables exist
            try:
                self.db_manager.create_mcp_tool_tables()
                logger.info("Database tables verified/created")
            except Exception as e:
                error_msg = f"Failed to create database tables: {e}"
                all_errors.append(error_msg)
                logger.error(error_msg)
                return self._create_failed_rebuild_result(start_time, all_errors)

            # Execute Phase 1: Discovery
            discovered_tools, discovery_timings, discovery_errors = self._execute_discovery_phase()
            phase_timings.update(discovery_timings)
            all_errors.extend(discovery_errors)
            
            if discovery_errors:
                return self._create_failed_rebuild_result(start_time, all_errors, phase_timings)

            # Execute Phase 2: Analysis
            analyzed_tools, tools_failed_analysis, analysis_timings, analysis_errors = self._execute_analysis_phase(discovered_tools)
            phase_timings.update(analysis_timings)
            all_errors.extend(analysis_errors)
            
            if analysis_errors:
                return self._create_failed_rebuild_result(start_time, all_errors, phase_timings)

            # Execute Phase 3: Embeddings
            embeddings_generated, embedding_timings, embedding_errors = self._execute_embedding_phase(analyzed_tools)
            phase_timings.update(embedding_timings)
            all_errors.extend(embedding_errors)
            
            if embedding_errors:
                return self._create_failed_rebuild_result(start_time, all_errors, phase_timings)

            # Execute Phase 4: Relationships
            relationships, relationships_created, relationship_timings, relationship_errors = self._execute_relationship_phase(analyzed_tools)
            phase_timings.update(relationship_timings)
            all_errors.extend(relationship_errors)
            # Note: Relationship errors are not fatal, continue

            # Execute Phase 5: Storage
            tools_stored, tools_failed_storage, storage_timings, storage_errors = self._execute_storage_phase(analyzed_tools, relationships)
            phase_timings.update(storage_timings)
            all_errors.extend(storage_errors)
            
            if storage_errors:
                return self._create_failed_rebuild_result(start_time, all_errors, phase_timings)

            # Execute Phase 6: Validation
            validation_timings, validation_errors = self._execute_validation_phase()
            phase_timings.update(validation_timings)
            all_errors.extend(validation_errors)
            # Note: Validation errors are not fatal

            # Calculate final results
            execution_time = time.time() - start_time
            success = len(all_errors) == 0
            tools_processed = len(discovered_tools)
            total_tools_failed = tools_failed_analysis + tools_failed_storage

            result = CatalogRebuildResult(
                success=success,
                tools_processed=tools_processed,
                tools_stored=tools_stored,
                tools_failed=total_tools_failed,
                relationships_created=relationships_created,
                embeddings_generated=embeddings_generated,
                execution_time=execution_time,
                errors=all_errors,
                phase_timings=phase_timings,
            )

            logger.info(
                "Full catalog rebuild completed: success=%s, %d/%d tools stored, %.2fs total",
                success,
                tools_stored,
                tools_processed,
                execution_time,
            )

            return result

        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Critical error during catalog rebuild: {e}"
            logger.error(error_msg)
            return self._create_failed_rebuild_result(start_time, [error_msg], phase_timings)

    def incremental_catalog_update(self) -> CatalogUpdateResult:
        """
        Update catalog with new or changed tools.

        Process:
        1. Compare current tools with stored catalog
        2. Identify new, updated, and removed tools
        3. Process only changes for efficiency
        4. Update relationships affected by changes

        Returns:
            CatalogUpdateResult with update statistics
        """
        logger.info("Starting incremental catalog update")
        start_time = time.time()

        tools_updated = 0
        tools_added = 0
        tools_removed = 0
        relationships_updated = 0
        embeddings_updated = 0
        errors = []
        changes_detected = {}

        try:
            # Get current tools from discovery
            current_tools = self.discovery_engine.discover_system_tools()
            current_tools.extend(self.discovery_engine.discover_custom_tools())

            # Get existing tools from database
            existing_tools = self._get_existing_tools_from_db()

            # Compare and identify changes
            changes = self._compare_tool_sets(current_tools, existing_tools)
            changes_detected = {
                "new_tools": len(changes["new"]),
                "updated_tools": len(changes["updated"]),
                "removed_tools": len(changes["removed"]),
            }

            logger.info(
                "Changes detected: %d new, %d updated, %d removed",
                len(changes["new"]),
                len(changes["updated"]),
                len(changes["removed"]),
            )

            # Process new tools
            if changes["new"]:
                try:
                    new_metadata = []
                    for tool in changes["new"]:
                        metadata = self.metadata_extractor.extract_from_tool_definition(
                            {
                                "name": tool.name,
                                "description": tool.description,
                                "parameters": [p.to_dict() for p in tool.parameters],
                            }
                        )
                        new_metadata.append(metadata)

                    storage_result = self.storage_ops.store_tool_batch(new_metadata)
                    tools_added = storage_result.tools_stored

                    if not storage_result.success:
                        errors.extend(storage_result.errors)

                except Exception as e:
                    error_msg = f"Failed to process new tools: {e}"
                    errors.append(error_msg)
                    logger.error(error_msg)

            # Process updated tools (for now, treat as new - full update logic would be more complex)
            if changes["updated"]:
                logger.info("Updated tools detected but full update logic not implemented - treating as informational")
                # In a full implementation, we'd update changed tools
                tools_updated = len(changes["updated"])

            # Process removed tools (for now, just log - removal logic would need careful consideration)
            if changes["removed"]:
                logger.info("Removed tools detected but removal logic not implemented - treating as informational")
                tools_removed = len(changes["removed"])

            # Update relationships if tools changed
            if tools_added > 0 or tools_updated > 0:
                try:
                    # Get all current tool metadata for relationship analysis
                    all_tools = []
                    for tool in current_tools:
                        metadata = self.metadata_extractor.extract_from_tool_definition(
                            {
                                "name": tool.name,
                                "description": tool.description,
                                "parameters": [p.to_dict() for p in tool.parameters],
                            }
                        )
                        all_tools.append(metadata)

                    # Detect new relationships
                    relationship_result = self.relationship_detector.analyze_all_relationships(all_tools)
                    new_relationships = (
                        relationship_result.alternatives
                        + relationship_result.complements
                        + relationship_result.prerequisites
                    )

                    if new_relationships:
                        self.storage_ops.store_tool_relationships(new_relationships)
                        relationships_updated = len(new_relationships)
                        logger.info("Updated %d relationships", relationships_updated)

                except Exception as e:
                    error_msg = f"Failed to update relationships: {e}"
                    errors.append(error_msg)
                    logger.error(error_msg)

            execution_time = time.time() - start_time
            success = len(errors) == 0

            result = CatalogUpdateResult(
                success=success,
                tools_updated=tools_updated,
                tools_added=tools_added,
                tools_removed=tools_removed,
                relationships_updated=relationships_updated,
                embeddings_updated=embeddings_updated,
                execution_time=execution_time,
                errors=errors,
                changes_detected=changes_detected,
            )

            logger.info(
                "Incremental update completed: success=%s, %d added, %d updated in %.2fs",
                success,
                tools_added,
                tools_updated,
                execution_time,
            )

            return result

        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Critical error during incremental update: {e}"
            errors.append(error_msg)
            logger.error(error_msg)

            return CatalogUpdateResult(
                success=False,
                tools_updated=tools_updated,
                tools_added=tools_added,
                tools_removed=tools_removed,
                relationships_updated=relationships_updated,
                embeddings_updated=embeddings_updated,
                execution_time=execution_time,
                errors=errors,
                changes_detected=changes_detected,
            )

    def validate_catalog_health(self) -> CatalogHealthReport:
        """
        Comprehensive catalog validation and health check.

        Verifies:
        - Data integrity and consistency
        - Embedding quality and coverage
        - Relationship validity and completeness
        - Overall catalog health metrics

        Returns:
            CatalogHealthReport with comprehensive health analysis
        """
        logger.info("Starting catalog health validation")
        start_time = time.time()

        try:
            checks = {}
            recommendations = []
            critical_issues = []

            # Check 1: Data Consistency
            try:
                consistency_report = self.validator.validate_catalog_consistency()
                checks["data_consistency"] = {
                    "status": "pass" if consistency_report.is_consistent else "fail",
                    "score": consistency_report.database_integrity_score,
                    "errors": consistency_report.errors,
                    "warnings": consistency_report.warnings,
                }

                if not consistency_report.is_consistent:
                    critical_issues.extend(consistency_report.errors)

            except Exception as e:
                checks["data_consistency"] = {"status": "error", "score": 0.0, "errors": [str(e)], "warnings": []}
                critical_issues.append(f"Data consistency check failed: {e}")

            # Check 2: Embedding Quality
            try:
                quality_metrics = self.validator.generate_quality_metrics()
                embedding_coverage = quality_metrics.get("tools_with_embeddings", 0) / max(
                    quality_metrics.get("total_tools", 1), 1
                )

                checks["embedding_quality"] = {
                    "status": "pass" if embedding_coverage > 0.8 else "warning" if embedding_coverage > 0.5 else "fail",
                    "score": embedding_coverage,
                    "coverage": embedding_coverage,
                    "total_tools": quality_metrics.get("total_tools", 0),
                    "tools_with_embeddings": quality_metrics.get("tools_with_embeddings", 0),
                }

                if embedding_coverage < 0.5:
                    critical_issues.append(f"Low embedding coverage: {embedding_coverage:.1%}")
                elif embedding_coverage < 0.8:
                    recommendations.append(f"Improve embedding coverage from {embedding_coverage:.1%} to >80%")

            except Exception as e:
                checks["embedding_quality"] = {"status": "error", "score": 0.0, "errors": [str(e)]}
                critical_issues.append(f"Embedding quality check failed: {e}")

            # Check 3: Relationship Coverage
            try:
                with self.db_manager.get_connection() as conn:
                    tool_count = conn.execute("SELECT COUNT(*) FROM mcp_tools").fetchone()[0]
                    relationship_count = conn.execute("SELECT COUNT(*) FROM tool_relationships").fetchone()[0]

                    # Calculate relationship coverage (relationships per tool pair)
                    max_relationships = tool_count * (tool_count - 1) // 2
                    relationship_coverage = relationship_count / max(max_relationships, 1)

                    checks["relationship_coverage"] = {
                        "status": "pass"
                        if relationship_coverage > 0.1
                        else "warning"
                        if relationship_coverage > 0.05
                        else "fail",
                        "score": min(relationship_coverage * 10, 1.0),  # Scale up small percentages
                        "coverage": relationship_coverage,
                        "total_relationships": relationship_count,
                        "total_tools": tool_count,
                    }

                    if relationship_coverage < 0.05:
                        recommendations.append(
                            f"Consider running relationship detection - only {relationship_coverage:.2%} coverage"
                        )

            except Exception as e:
                checks["relationship_coverage"] = {"status": "error", "score": 0.0, "errors": [str(e)]}
                critical_issues.append(f"Relationship coverage check failed: {e}")

            # Calculate overall health metrics
            try:
                metrics = self.validator.generate_quality_metrics()
            except Exception as e:
                metrics = {"error": str(e)}
                critical_issues.append(f"Failed to generate quality metrics: {e}")

            # Calculate overall health score
            check_scores = [check.get("score", 0.0) for check in checks.values() if "score" in check]
            overall_health_score = sum(check_scores) / len(check_scores) if check_scores else 0.0

            # Determine if healthy
            is_healthy = overall_health_score > 0.7 and len(critical_issues) == 0

            # Add general recommendations
            if overall_health_score < 0.5:
                recommendations.append("Consider running a full catalog rebuild to improve overall health")

            validation_time = time.time() - start_time

            result = CatalogHealthReport(
                is_healthy=is_healthy,
                overall_health_score=overall_health_score,
                metrics=metrics,
                checks=checks,
                recommendations=recommendations,
                critical_issues=critical_issues,
                validation_time=validation_time,
            )

            logger.info(
                "Health validation completed: healthy=%s, score=%.2f, %d issues in %.2fs",
                is_healthy,
                overall_health_score,
                len(critical_issues),
                validation_time,
            )

            return result

        except Exception as e:
            validation_time = time.time() - start_time
            error_msg = f"Critical error during health validation: {e}"
            logger.error(error_msg)

            return CatalogHealthReport(
                is_healthy=False,
                overall_health_score=0.0,
                metrics={"error": str(e)},
                checks={"critical_error": {"status": "error", "score": 0.0, "errors": [str(e)]}},
                recommendations=["Fix critical health validation error"],
                critical_issues=[error_msg],
                validation_time=validation_time,
            )

    def get_catalog_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive catalog statistics.

        Returns:
            Dictionary with detailed catalog statistics
        """
        try:
            # Get quality metrics
            quality_metrics = self.validator.generate_quality_metrics()

            # Get storage statistics
            storage_stats = self.storage_ops.get_batch_storage_statistics()

            # Combine statistics
            return {
                "catalog_health": quality_metrics,
                "storage_statistics": storage_stats.to_dict() if hasattr(storage_stats, "to_dict") else storage_stats,
                "last_updated": time.time(),
            }

        except Exception as e:
            logger.error("Error getting catalog statistics: %s", e)
            return {"error": str(e), "last_updated": time.time()}

    def _create_failed_rebuild_result(
        self, start_time: float, errors: List[str], phase_timings: Optional[Dict[str, float]] = None
    ) -> CatalogRebuildResult:
        """Create a failed rebuild result with current state."""
        return CatalogRebuildResult(
            success=False,
            tools_processed=0,
            tools_stored=0,
            tools_failed=0,
            relationships_created=0,
            embeddings_generated=0,
            execution_time=time.time() - start_time,
            errors=errors,
            phase_timings=phase_timings or {},
        )

    def _get_existing_tools_from_db(self) -> List[Dict[str, Any]]:
        """Get existing tools from database for comparison."""
        try:
            with self.db_manager.get_connection() as conn:
                results = conn.execute("SELECT id, name, description FROM mcp_tools").fetchall()

                tools = []
                for row in results:
                    tools.append({"id": row[0], "name": row[1], "description": row[2]})

                return tools

        except Exception as e:
            logger.error("Error getting existing tools from database: %s", e)
            return []

    def _compare_tool_sets(self, current_tools, existing_tools) -> Dict[str, List]:
        """Compare current tools with existing tools to find changes."""
        # Create lookup sets
        existing_names = {tool["name"] for tool in existing_tools}
        current_names = {tool.name for tool in current_tools}

        # Find differences
        new_tool_names = current_names - existing_names
        removed_tool_names = existing_names - current_names
        potentially_updated_names = current_names & existing_names

        # Categorize tools
        new_tools = [tool for tool in current_tools if tool.name in new_tool_names]
        removed_tools = [tool for tool in existing_tools if tool["name"] in removed_tool_names]

        # For updated tools, we'd need more sophisticated comparison
        # For now, just identify potential updates
        updated_tools = [tool for tool in current_tools if tool.name in potentially_updated_names]

        return {
            "new": new_tools,
            "updated": updated_tools,  # Simplified - full implementation would check for actual changes
            "removed": removed_tools,
        }
