"""
MCP Tool Schema Definitions and Migration Utilities

This module defines the database schema for storing MCP tool metadata and provides
migration utilities for safe database schema evolution.
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple
from uuid import uuid4

import duckdb

from config import EmbeddingConfig
from database_manager import DatabaseManager
from exceptions import DatabaseError, DatabaseMigrationError
from logging_config import get_logger

logger = get_logger(__name__)


class MCPToolSchema:
    """Database schema definitions for MCP tool metadata."""

    # Schema version for tracking migrations
    CURRENT_SCHEMA_VERSION = 1

    @staticmethod
    def get_mcp_tools_table_sql() -> str:
        """Get the SQL for creating the mcp_tools table."""
        return f"""
        CREATE TABLE IF NOT EXISTS mcp_tools (
            id VARCHAR PRIMARY KEY,           -- Tool identifier (e.g., 'bash', 'read', 'custom_tool')
            name VARCHAR NOT NULL,            -- Display name
            description TEXT,                 -- Tool description
            tool_type VARCHAR,               -- 'system', 'custom', 'third_party'
            provider VARCHAR,                -- Tool provider/source
            version VARCHAR,                 -- Tool version if available
            category VARCHAR,                -- 'file_ops', 'web', 'analysis', etc.
            embedding DOUBLE[{EmbeddingConfig.DIMENSIONS}],           -- Semantic embedding of description
            metadata_json TEXT,              -- Additional metadata as JSON
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_active BOOLEAN DEFAULT TRUE
        )
        """

    @staticmethod
    def get_tool_parameters_table_sql() -> str:
        """Get the SQL for creating the tool_parameters table."""
        return f"""
        CREATE TABLE IF NOT EXISTS tool_parameters (
            id VARCHAR PRIMARY KEY,
            tool_id VARCHAR NOT NULL,
            parameter_name VARCHAR NOT NULL,
            parameter_type VARCHAR,          -- 'string', 'number', 'boolean', 'array', 'object'
            is_required BOOLEAN DEFAULT FALSE,
            description TEXT,
            default_value TEXT,
            schema_json TEXT,                -- Full JSON schema
            embedding DOUBLE[{EmbeddingConfig.DIMENSIONS}],           -- Embedding of parameter description
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """

    @staticmethod
    def get_tool_examples_table_sql() -> str:
        """Get the SQL for creating the tool_examples table."""
        return f"""
        CREATE TABLE IF NOT EXISTS tool_examples (
            id VARCHAR PRIMARY KEY,
            tool_id VARCHAR NOT NULL,
            use_case VARCHAR,                -- Brief description of the use case
            example_call TEXT,               -- Example tool invocation
            expected_output TEXT,            -- Expected response/output
            context TEXT,                    -- When to use this pattern
            embedding DOUBLE[{EmbeddingConfig.DIMENSIONS}],           -- Embedding of use case + context
            effectiveness_score FLOAT DEFAULT 0.0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """

    @staticmethod
    def get_tool_relationships_table_sql() -> str:
        """Get the SQL for creating the tool_relationships table."""
        return """
        CREATE TABLE IF NOT EXISTS tool_relationships (
            id VARCHAR PRIMARY KEY,
            tool_a_id VARCHAR NOT NULL,
            tool_b_id VARCHAR NOT NULL,
            relationship_type VARCHAR,      -- 'alternative', 'complement', 'prerequisite'
            strength FLOAT DEFAULT 0.0,    -- 0.0 to 1.0 relationship strength
            description TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(tool_a_id, tool_b_id, relationship_type)
        )
        """

    @staticmethod
    def get_schema_version_table_sql() -> str:
        """Get the SQL for creating the schema_version table for migration tracking."""
        return """
        CREATE TABLE IF NOT EXISTS schema_version (
            version INTEGER PRIMARY KEY,
            applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            description TEXT
        )
        """

    @staticmethod
    def get_index_definitions() -> List[str]:
        """Get all index creation SQL statements."""
        return [
            "CREATE INDEX IF NOT EXISTS idx_mcp_tools_category ON mcp_tools(category)",
            "CREATE INDEX IF NOT EXISTS idx_mcp_tools_type ON mcp_tools(tool_type)",
            "CREATE INDEX IF NOT EXISTS idx_mcp_tools_provider ON mcp_tools(provider)",
            "CREATE INDEX IF NOT EXISTS idx_mcp_tools_active ON mcp_tools(is_active)",
            "CREATE INDEX IF NOT EXISTS idx_tool_parameters_tool_id ON tool_parameters(tool_id)",
            "CREATE INDEX IF NOT EXISTS idx_tool_parameters_required ON tool_parameters(tool_id, is_required)",
            "CREATE INDEX IF NOT EXISTS idx_tool_parameters_type ON tool_parameters(parameter_type)",
            "CREATE INDEX IF NOT EXISTS idx_tool_examples_tool_id ON tool_examples(tool_id)",
            "CREATE INDEX IF NOT EXISTS idx_tool_examples_score ON tool_examples(effectiveness_score DESC)",
            "CREATE INDEX IF NOT EXISTS idx_tool_relationships_tools ON tool_relationships(tool_a_id, tool_b_id)",
            "CREATE INDEX IF NOT EXISTS idx_tool_relationships_type ON tool_relationships(relationship_type)",
        ]


class MCPToolMigration:
    """Handles database migrations for MCP tool schema."""

    def __init__(self, db_manager: DatabaseManager) -> None:
        """Initialize migration manager with database manager."""
        self.db_manager = db_manager
        self.migrations_dir = Path(__file__).parent / "migrations"
        self.migrations_dir.mkdir(exist_ok=True)

    def get_current_version(self) -> int:
        """Get the current schema version from the database."""
        try:
            with self.db_manager.get_connection() as conn:
                # Create schema version table if it doesn't exist
                conn.execute(MCPToolSchema.get_schema_version_table_sql())

                result = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()

                return result[0] if result and result[0] is not None else 0
        except (duckdb.Error, DatabaseError, DatabaseMigrationError) as e:
            logger.warning("Failed to get schema version, assuming version 0: %s", e)
            return 0

    def apply_migration(self, version: int, description: str, migration_sql: List[str]) -> None:
        """
        Apply a migration to the database.

        Args:
            version: Migration version number
            description: Description of the migration
            migration_sql: List of SQL statements to execute
        """
        logger.info("Applying migration version %d: %s", version, description)

        try:
            with self.db_manager.get_connection() as conn:
                # Start transaction
                conn.execute("BEGIN TRANSACTION")

                try:
                    # Ensure schema_version table exists
                    conn.execute(MCPToolSchema.get_schema_version_table_sql())

                    # Execute all migration statements
                    for sql in migration_sql:
                        conn.execute(sql)

                    # Record migration in schema_version table
                    conn.execute(
                        "INSERT INTO schema_version (version, description) VALUES (?, ?)", (version, description)
                    )

                    # Commit transaction
                    conn.execute("COMMIT")
                    logger.info("Successfully applied migration version %d", version)

                except Exception as e:
                    # Rollback on error
                    conn.execute("ROLLBACK")
                    raise DatabaseMigrationError(f"Migration version {version} failed: {e}") from e

        except Exception as e:
            logger.error("Failed to apply migration version %d: %s", version, e)
            raise

    def create_rollback_script(self, version: int, rollback_sql: List[str]) -> Path:
        """
        Create a rollback script for a migration.

        Args:
            version: Migration version
            rollback_sql: List of SQL statements for rollback

        Returns:
            Path to the created rollback script
        """
        rollback_file = self.migrations_dir / f"rollback_v{version:03d}.sql"

        with open(rollback_file, "w") as f:
            f.write(f"-- Rollback script for migration version {version}\n")
            f.write(f"-- Generated at {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("BEGIN TRANSACTION;\n\n")

            for sql in rollback_sql:
                f.write(f"{sql};\n\n")

            f.write("-- Remove migration record\n")
            f.write(f"DELETE FROM schema_version WHERE version = {version};\n\n")
            f.write("COMMIT;\n")

        logger.info("Created rollback script: %s", rollback_file)
        return rollback_file

    def migrate_to_latest(self) -> bool:
        """
        Migrate database to the latest schema version.

        Returns:
            True if migrations were applied, False if already up to date
        """
        current_version = self.get_current_version()
        target_version = MCPToolSchema.CURRENT_SCHEMA_VERSION

        if current_version >= target_version:
            logger.info("Database schema is up to date (version %d)", current_version)
            return False

        logger.info("Migrating from version %d to %d", current_version, target_version)

        # Apply migration version 1: Initial MCP tool schema
        if current_version < 1:
            self._apply_initial_mcp_schema_migration()

        logger.info("Database migration completed successfully")
        return True

    def _apply_initial_mcp_schema_migration(self) -> None:
        """Apply the initial MCP tool schema migration."""
        migration_sql = [
            MCPToolSchema.get_mcp_tools_table_sql(),
            MCPToolSchema.get_tool_parameters_table_sql(),
            MCPToolSchema.get_tool_examples_table_sql(),
            MCPToolSchema.get_tool_relationships_table_sql(),
        ]

        # Add all indexes
        migration_sql.extend(MCPToolSchema.get_index_definitions())

        # Create rollback script
        rollback_sql = [
            "DROP TABLE IF EXISTS tool_relationships",
            "DROP TABLE IF EXISTS tool_examples",
            "DROP TABLE IF EXISTS tool_parameters",
            "DROP TABLE IF EXISTS mcp_tools",
        ]

        self.create_rollback_script(1, rollback_sql)

        # Apply migration
        self.apply_migration(
            version=1,
            description="Initial MCP tool schema with tools, parameters, examples, and relationships tables",
            migration_sql=migration_sql,
        )

    def _validate_required_tables(self, conn, validation_results: Dict[str, Any]) -> None:
        """Validate that all required tables exist and get their row counts."""
        required_tables = ["mcp_tools", "tool_parameters", "tool_examples", "tool_relationships"]

        for table in required_tables:
            try:
                result = conn.execute(
                    "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?", (table,)
                ).fetchone()

                if not result or result[0] == 0:
                    validation_results["errors"].append(f"Required table {table} does not exist")
                    validation_results["valid"] = False
                else:
                    # Get row count for the table
                    count_result = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
                    validation_results["table_counts"][table] = count_result[0] if count_result else 0

            except Exception as e:
                validation_results["errors"].append(f"Error checking table {table}: {e}")
                validation_results["valid"] = False

    def _validate_orphaned_parameters(self, conn, validation_results: Dict[str, Any]) -> None:
        """Check for orphaned tool parameters that reference non-existent tools."""
        orphaned_params = conn.execute(
            """
            SELECT COUNT(*) FROM tool_parameters tp
            LEFT JOIN mcp_tools mt ON tp.tool_id = mt.id
            WHERE mt.id IS NULL
            """
        ).fetchone()

        if orphaned_params and orphaned_params[0] > 0:
            validation_results["warnings"].append(f"Found {orphaned_params[0]} orphaned tool parameters")

    def _validate_orphaned_examples(self, conn, validation_results: Dict[str, Any]) -> None:
        """Check for orphaned tool examples that reference non-existent tools."""
        orphaned_examples = conn.execute(
            """
            SELECT COUNT(*) FROM tool_examples te
            LEFT JOIN mcp_tools mt ON te.tool_id = mt.id
            WHERE mt.id IS NULL
            """
        ).fetchone()

        if orphaned_examples and orphaned_examples[0] > 0:
            validation_results["warnings"].append(f"Found {orphaned_examples[0]} orphaned tool examples")

    def _validate_orphaned_relationships(self, conn, validation_results: Dict[str, Any]) -> None:
        """Check for orphaned tool relationships that reference non-existent tools."""
        orphaned_relationships = conn.execute(
            """
            SELECT COUNT(*) FROM tool_relationships tr
            LEFT JOIN mcp_tools mta ON tr.tool_a_id = mta.id
            LEFT JOIN mcp_tools mtb ON tr.tool_b_id = mtb.id
            WHERE mta.id IS NULL OR mtb.id IS NULL
            """
        ).fetchone()

        if orphaned_relationships and orphaned_relationships[0] > 0:
            validation_results["warnings"].append(f"Found {orphaned_relationships[0]} orphaned tool relationships")

    def _validate_foreign_key_constraints(self, conn, validation_results: Dict[str, Any]) -> None:
        """Validate all foreign key constraints."""
        try:
            self._validate_orphaned_parameters(conn, validation_results)
            self._validate_orphaned_examples(conn, validation_results)
            self._validate_orphaned_relationships(conn, validation_results)
        except Exception as e:
            validation_results["errors"].append(f"Error checking foreign key constraints: {e}")
            validation_results["valid"] = False

    def validate_schema_integrity(self) -> Dict[str, Any]:
        """
        Validate the integrity of the MCP tool schema.

        Returns:
            Dictionary with validation results
        """
        validation_results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "table_counts": {},
        }

        try:
            with self.db_manager.get_connection() as conn:
                self._validate_required_tables(conn, validation_results)
                self._validate_foreign_key_constraints(conn, validation_results)

        except Exception as e:
            validation_results["errors"].append(f"General validation error: {e}")
            validation_results["valid"] = False

        return validation_results


def generate_tool_id() -> str:
    """Generate a unique ID for a tool."""
    return str(uuid4())


def generate_parameter_id() -> str:
    """Generate a unique ID for a tool parameter."""
    return str(uuid4())


def generate_example_id() -> str:
    """Generate a unique ID for a tool example."""
    return str(uuid4())


def generate_relationship_id() -> str:
    """Generate a unique ID for a tool relationship."""
    return str(uuid4())


def validate_tool_metadata(tool_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate tool metadata before storage.

    Args:
        tool_data: Dictionary containing tool metadata

    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []

    # Required fields
    required_fields = ["id", "name"]
    for field in required_fields:
        if field not in tool_data or not tool_data[field]:
            errors.append(f"Required field '{field}' is missing or empty")

    # Validate tool_type
    valid_tool_types = ["system", "custom", "third_party"]
    if "tool_type" in tool_data and tool_data["tool_type"] not in valid_tool_types:
        errors.append(f"Invalid tool_type: {tool_data['tool_type']}. Must be one of {valid_tool_types}")

    # Validate embedding dimensions if present
    if "embedding" in tool_data and tool_data["embedding"]:
        if not isinstance(tool_data["embedding"], list):
            errors.append("Embedding must be a list of numbers")
        elif len(tool_data["embedding"]) != EmbeddingConfig.DIMENSIONS:
            embedding_len = len(tool_data["embedding"])
            errors.append(f"Embedding must have exactly {EmbeddingConfig.DIMENSIONS} dimensions, got {embedding_len}")

    # Validate JSON fields
    if "metadata_json" in tool_data and tool_data["metadata_json"]:
        try:
            json.loads(tool_data["metadata_json"])
        except json.JSONDecodeError as e:
            errors.append(f"Invalid JSON in metadata_json: {e}")

    return len(errors) == 0, errors


def validate_parameter_metadata(param_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate parameter metadata before storage.

    Args:
        param_data: Dictionary containing parameter metadata

    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []

    # Required fields
    required_fields = ["id", "tool_id", "parameter_name"]
    for field in required_fields:
        if field not in param_data or not param_data[field]:
            errors.append(f"Required field '{field}' is missing or empty")

    # Validate parameter_type
    valid_parameter_types = ["string", "number", "boolean", "array", "object"]
    if "parameter_type" in param_data and param_data["parameter_type"]:
        if param_data["parameter_type"] not in valid_parameter_types:
            errors.append(
                f"Invalid parameter_type: {param_data['parameter_type']}. Must be one of {valid_parameter_types}"
            )

    # Validate schema_json if present
    if "schema_json" in param_data and param_data["schema_json"]:
        try:
            json.loads(param_data["schema_json"])
        except json.JSONDecodeError as e:
            errors.append(f"Invalid JSON in schema_json: {e}")

    # Validate embedding dimensions if present
    if "embedding" in param_data and param_data["embedding"]:
        if not isinstance(param_data["embedding"], list):
            errors.append("Embedding must be a list of numbers")
        elif len(param_data["embedding"]) != EmbeddingConfig.DIMENSIONS:
            embedding_len = len(param_data["embedding"])
            errors.append(f"Embedding must have exactly {EmbeddingConfig.DIMENSIONS} dimensions, got {embedding_len}")

    return len(errors) == 0, errors
