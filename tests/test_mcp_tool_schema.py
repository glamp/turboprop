"""
Tests for MCP Tool Schema and Migration functionality.

This module contains comprehensive tests for the MCP tool schema definitions,
migration utilities, and database operations.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import duckdb
import pytest

from database_manager import DatabaseManager
from exceptions import DatabaseError, DatabaseMigrationError
from mcp_tool_schema import (
    MCPToolMigration,
    MCPToolSchema,
    generate_example_id,
    generate_parameter_id,
    generate_relationship_id,
    generate_tool_id,
    validate_parameter_metadata,
    validate_tool_metadata,
)


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "test.duckdb"
        yield db_path


@pytest.fixture
def db_manager(temp_db):
    """Create a database manager with temporary database."""
    return DatabaseManager(temp_db)


@pytest.fixture
def migration_manager(db_manager):
    """Create a migration manager for testing."""
    return MCPToolMigration(db_manager)


class TestMCPToolSchema:
    """Test the MCP tool schema definitions."""

    def test_schema_table_sql_generation(self):
        """Test that all schema table SQL statements are valid."""
        # Test each table creation SQL
        tables = [
            MCPToolSchema.get_mcp_tools_table_sql(),
            MCPToolSchema.get_tool_parameters_table_sql(),
            MCPToolSchema.get_tool_examples_table_sql(),
            MCPToolSchema.get_tool_relationships_table_sql(),
            MCPToolSchema.get_schema_version_table_sql(),
        ]

        # All should be non-empty strings containing CREATE TABLE
        for sql in tables:
            assert isinstance(sql, str)
            assert len(sql) > 0
            assert "CREATE TABLE" in sql.upper()

    def test_index_definitions(self):
        """Test that index definitions are properly formatted."""
        indexes = MCPToolSchema.get_index_definitions()

        assert isinstance(indexes, list)
        assert len(indexes) > 0

        # All should be CREATE INDEX statements
        for index_sql in indexes:
            assert isinstance(index_sql, str)
            assert "CREATE INDEX" in index_sql.upper()

    def test_schema_version_tracking(self):
        """Test that schema version is properly defined."""
        assert hasattr(MCPToolSchema, "CURRENT_SCHEMA_VERSION")
        assert isinstance(MCPToolSchema.CURRENT_SCHEMA_VERSION, int)
        assert MCPToolSchema.CURRENT_SCHEMA_VERSION > 0


class TestMCPToolMigration:
    """Test the migration functionality."""

    def test_migration_manager_initialization(self, db_manager):
        """Test migration manager initialization."""
        migration = MCPToolMigration(db_manager)

        assert migration.db_manager is db_manager
        assert migration.migrations_dir.exists()
        assert migration.migrations_dir.name == "migrations"

    def test_get_current_version_new_database(self, migration_manager):
        """Test getting current version from a new database."""
        # New database should return version 0
        version = migration_manager.get_current_version()
        assert version == 0

    def test_get_current_version_with_existing_version(self, migration_manager):
        """Test getting current version when versions exist."""
        # First create schema_version table and insert a version
        with migration_manager.db_manager.get_connection() as conn:
            conn.execute(MCPToolSchema.get_schema_version_table_sql())
            conn.execute("INSERT INTO schema_version (version, description) VALUES (1, 'Test version')")

        version = migration_manager.get_current_version()
        assert version == 1

    def test_apply_migration_success(self, migration_manager):
        """Test successful migration application."""
        migration_sql = [
            "CREATE TABLE test_table (id INTEGER, name TEXT)",
            "INSERT INTO test_table (id, name) VALUES (1, 'test')",
        ]

        migration_manager.apply_migration(1, "Test migration", migration_sql)

        # Verify migration was recorded
        version = migration_manager.get_current_version()
        assert version == 1

        # Verify table was created and data inserted
        with migration_manager.db_manager.get_connection() as conn:
            result = conn.execute("SELECT COUNT(*) FROM test_table").fetchone()
            assert result[0] == 1

    def test_apply_migration_failure_rollback(self, migration_manager):
        """Test that failed migrations are rolled back."""
        migration_sql = [
            "CREATE TABLE test_table (id INTEGER, name TEXT)",
            "INVALID SQL STATEMENT",  # This should cause failure
        ]

        with pytest.raises(DatabaseMigrationError):
            migration_manager.apply_migration(1, "Failed migration", migration_sql)

        # Verify no migration was recorded
        version = migration_manager.get_current_version()
        assert version == 0

        # Verify test_table was not created (rollback successful)
        with migration_manager.db_manager.get_connection() as conn:
            with pytest.raises(duckdb.Error):
                conn.execute("SELECT COUNT(*) FROM test_table")

    def test_create_rollback_script(self, migration_manager):
        """Test rollback script creation."""
        rollback_sql = ["DROP TABLE test_table", "DELETE FROM some_table WHERE id = 1"]

        rollback_file = migration_manager.create_rollback_script(1, rollback_sql)

        assert rollback_file.exists()
        assert rollback_file.name == "rollback_v001.sql"

        # Verify file content
        content = rollback_file.read_text()
        assert "DROP TABLE test_table;" in content
        assert "DELETE FROM some_table WHERE id = 1;" in content
        assert "DELETE FROM schema_version WHERE version = 1;" in content
        assert "BEGIN TRANSACTION;" in content
        assert "COMMIT;" in content

    def test_migrate_to_latest_new_database(self, migration_manager):
        """Test migrating a new database to the latest version."""
        result = migration_manager.migrate_to_latest()

        assert result is True  # Migrations were applied
        assert migration_manager.get_current_version() == MCPToolSchema.CURRENT_SCHEMA_VERSION

        # Verify all MCP tool tables were created
        with migration_manager.db_manager.get_connection() as conn:
            tables = ["mcp_tools", "tool_parameters", "tool_examples", "tool_relationships"]
            for table in tables:
                result = conn.execute(
                    "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?", (table,)
                ).fetchone()
                assert result[0] == 1

    def test_migrate_to_latest_already_current(self, migration_manager):
        """Test migrating when database is already at current version."""
        # First migrate to latest
        migration_manager.migrate_to_latest()

        # Then try again
        result = migration_manager.migrate_to_latest()

        assert result is False  # No migrations needed

    def test_validate_schema_integrity_valid(self, migration_manager):
        """Test schema integrity validation on valid schema."""
        # First migrate to create tables
        migration_manager.migrate_to_latest()

        validation = migration_manager.validate_schema_integrity()

        assert validation["valid"] is True
        assert len(validation["errors"]) == 0
        assert "table_counts" in validation
        assert validation["table_counts"]["mcp_tools"] == 0

    def test_validate_schema_integrity_missing_tables(self, migration_manager):
        """Test schema integrity validation when tables are missing."""
        # Don't migrate, so tables won't exist
        validation = migration_manager.validate_schema_integrity()

        assert validation["valid"] is False
        assert len(validation["errors"]) > 0
        assert any("does not exist" in error for error in validation["errors"])

    def test_validate_schema_integrity_with_data(self, migration_manager, db_manager):
        """Test schema integrity validation with actual data."""
        # Migrate and create tables
        migration_manager.migrate_to_latest()

        # Add some test data
        db_manager.store_mcp_tool(
            tool_id="test_tool", name="Test Tool", description="A test tool", tool_type="system", category="testing"
        )

        validation = migration_manager.validate_schema_integrity()

        assert validation["valid"] is True
        assert validation["table_counts"]["mcp_tools"] == 1
        assert len(validation["warnings"]) == 0  # No orphaned records


class TestDatabaseManagerMCPOperations:
    """Test the MCP-specific database operations."""

    def setup_method(self):
        """Set up test database with MCP tool tables."""
        pass

    def test_create_mcp_tool_tables(self, db_manager):
        """Test creating MCP tool tables."""
        db_manager.create_mcp_tool_tables()

        # Verify all tables exist
        with db_manager.get_connection() as conn:
            tables = ["mcp_tools", "tool_parameters", "tool_examples", "tool_relationships"]
            for table in tables:
                result = conn.execute(
                    "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?", (table,)
                ).fetchone()
                assert result[0] == 1

    def test_store_and_get_mcp_tool(self, db_manager):
        """Test storing and retrieving an MCP tool."""
        db_manager.create_mcp_tool_tables()

        # Store a tool
        tool_data = {
            "tool_id": "test_tool",
            "name": "Test Tool",
            "description": "A tool for testing",
            "tool_type": "system",
            "provider": "test_provider",
            "version": "1.0.0",
            "category": "testing",
            "embedding": [0.1] * 384,
            "metadata_json": json.dumps({"key": "value"}),
            "is_active": True,
        }

        db_manager.store_mcp_tool(**tool_data)

        # Retrieve the tool
        retrieved_tool = db_manager.get_mcp_tool("test_tool")

        assert retrieved_tool is not None
        assert retrieved_tool["id"] == "test_tool"
        assert retrieved_tool["name"] == "Test Tool"
        assert retrieved_tool["tool_type"] == "system"
        assert retrieved_tool["is_active"] is True

    def test_store_and_get_tool_parameters(self, db_manager):
        """Test storing and retrieving tool parameters."""
        db_manager.create_mcp_tool_tables()

        # First store a tool
        db_manager.store_mcp_tool(tool_id="test_tool", name="Test Tool")

        # Store parameters
        param_data = {
            "parameter_id": "param_1",
            "tool_id": "test_tool",
            "parameter_name": "file_path",
            "parameter_type": "string",
            "is_required": True,
            "description": "Path to file",
            "embedding": [0.2] * 384,
        }

        db_manager.store_tool_parameter(**param_data)

        # Retrieve parameters
        parameters = db_manager.get_tool_parameters("test_tool")

        assert len(parameters) == 1
        assert parameters[0]["parameter_name"] == "file_path"
        assert parameters[0]["is_required"] is True

    def test_store_and_get_tool_examples(self, db_manager):
        """Test storing and retrieving tool examples."""
        db_manager.create_mcp_tool_tables()

        # First store a tool
        db_manager.store_mcp_tool(tool_id="test_tool", name="Test Tool")

        # Store example
        example_data = {
            "example_id": "example_1",
            "tool_id": "test_tool",
            "use_case": "Reading a file",
            "example_call": 'read_file("/path/to/file.txt")',
            "expected_output": "File contents",
            "context": "When you need to read file contents",
            "embedding": [0.3] * 384,
            "effectiveness_score": 0.9,
        }

        db_manager.store_tool_example(**example_data)

        # Retrieve examples
        examples = db_manager.get_tool_examples("test_tool")

        assert len(examples) == 1
        assert examples[0]["use_case"] == "Reading a file"
        assert abs(examples[0]["effectiveness_score"] - 0.9) < 1e-6

    def test_store_and_get_tool_relationships(self, db_manager):
        """Test storing and retrieving tool relationships."""
        db_manager.create_mcp_tool_tables()

        # Store two tools
        db_manager.store_mcp_tool(tool_id="tool_a", name="Tool A")
        db_manager.store_mcp_tool(tool_id="tool_b", name="Tool B")

        # Store relationship
        relationship_data = {
            "relationship_id": "rel_1",
            "tool_a_id": "tool_a",
            "tool_b_id": "tool_b",
            "relationship_type": "complement",
            "strength": 0.8,
            "description": "These tools work well together",
        }

        db_manager.store_tool_relationship(**relationship_data)

        # Retrieve relationships
        related_tools = db_manager.get_related_tools("tool_a")

        assert len(related_tools) == 1
        assert related_tools[0]["name"] == "Tool B"
        assert related_tools[0]["relationship_type"] == "complement"

    def test_search_mcp_tools_by_embedding(self, db_manager):
        """Test semantic search of MCP tools."""
        db_manager.create_mcp_tool_tables()

        # Store tools with embeddings
        tools = [
            ("tool_1", "File Reader", [0.1] * 384, "file_ops", "system"),
            ("tool_2", "Web Scraper", [0.2] * 384, "web", "custom"),
            ("tool_3", "Data Analyzer", [0.3] * 384, "analysis", "third_party"),
        ]

        for tool_id, name, embedding, category, tool_type in tools:
            db_manager.store_mcp_tool(
                tool_id=tool_id, name=name, embedding=embedding, category=category, tool_type=tool_type
            )

        # Search by embedding
        query_embedding = [0.15] * 384
        results = db_manager.search_mcp_tools_by_embedding(query_embedding, limit=2)

        assert len(results) <= 2
        assert all("similarity_score" in result for result in results)

        # Test with category filter
        filtered_results = db_manager.search_mcp_tools_by_embedding(query_embedding, limit=10, category="web")

        assert len(filtered_results) == 1
        assert filtered_results[0]["name"] == "Web Scraper"

    def test_get_mcp_tool_statistics(self, db_manager):
        """Test getting MCP tool statistics."""
        db_manager.create_mcp_tool_tables()

        # Initially empty
        stats = db_manager.get_mcp_tool_statistics()
        assert stats["total_tools"] == 0

        # Add some tools
        tools_data = [
            ("tool_1", "Tool 1", "system", "file_ops", [0.1] * 384),
            ("tool_2", "Tool 2", "custom", "web", None),  # No embedding
            ("tool_3", "Tool 3", "system", "file_ops", [0.3] * 384),
        ]

        for tool_id, name, tool_type, category, embedding in tools_data:
            db_manager.store_mcp_tool(
                tool_id=tool_id, name=name, tool_type=tool_type, category=category, embedding=embedding
            )

        # Check statistics
        stats = db_manager.get_mcp_tool_statistics()

        assert stats["total_tools"] == 3
        assert stats["active_tools"] == 3
        assert stats["tools_by_type"]["system"] == 2
        assert stats["tools_by_type"]["custom"] == 1
        assert stats["tools_by_category"]["file_ops"] == 2
        assert stats["tools_by_category"]["web"] == 1
        assert stats["tools_with_embeddings"] == 2

    def test_remove_mcp_tool(self, db_manager):
        """Test removing an MCP tool."""
        db_manager.create_mcp_tool_tables()

        # Store a tool with parameters and examples
        db_manager.store_mcp_tool(tool_id="test_tool", name="Test Tool")
        db_manager.store_tool_parameter(parameter_id="param_1", tool_id="test_tool", parameter_name="test_param")
        db_manager.store_tool_example(example_id="example_1", tool_id="test_tool", use_case="Test case")

        # Verify tool exists
        assert db_manager.get_mcp_tool("test_tool") is not None
        assert len(db_manager.get_tool_parameters("test_tool")) == 1
        assert len(db_manager.get_tool_examples("test_tool")) == 1

        # Remove tool
        result = db_manager.remove_mcp_tool("test_tool")
        assert result is True

        # Verify tool and related data are gone (cascade delete)
        assert db_manager.get_mcp_tool("test_tool") is None
        assert len(db_manager.get_tool_parameters("test_tool")) == 0
        assert len(db_manager.get_tool_examples("test_tool")) == 0

    def test_remove_nonexistent_tool(self, db_manager):
        """Test removing a tool that doesn't exist."""
        db_manager.create_mcp_tool_tables()

        result = db_manager.remove_mcp_tool("nonexistent_tool")
        assert result is False


class TestUtilityFunctions:
    """Test utility functions for MCP tool schema."""

    def test_id_generation_functions(self):
        """Test all ID generation functions."""
        generators = [generate_tool_id, generate_parameter_id, generate_example_id, generate_relationship_id]

        for generator in generators:
            id1 = generator()
            id2 = generator()

            # IDs should be strings
            assert isinstance(id1, str)
            assert isinstance(id2, str)

            # IDs should be unique
            assert id1 != id2

            # IDs should be non-empty
            assert len(id1) > 0
            assert len(id2) > 0

    def test_validate_tool_metadata_valid(self):
        """Test validating valid tool metadata."""
        valid_data = {
            "id": "test_tool",
            "name": "Test Tool",
            "description": "A test tool",
            "tool_type": "system",
            "embedding": [0.1] * 384,
            "metadata_json": json.dumps({"key": "value"}),
        }

        is_valid, errors = validate_tool_metadata(valid_data)

        assert is_valid is True
        assert len(errors) == 0

    def test_validate_tool_metadata_missing_required(self):
        """Test validating tool metadata with missing required fields."""
        invalid_data = {"description": "Missing id and name"}

        is_valid, errors = validate_tool_metadata(invalid_data)

        assert is_valid is False
        assert len(errors) >= 2  # Missing id and name
        assert any("id" in error for error in errors)
        assert any("name" in error for error in errors)

    def test_validate_tool_metadata_invalid_type(self):
        """Test validating tool metadata with invalid tool_type."""
        invalid_data = {"id": "test_tool", "name": "Test Tool", "tool_type": "invalid_type"}

        is_valid, errors = validate_tool_metadata(invalid_data)

        assert is_valid is False
        assert any("Invalid tool_type" in error for error in errors)

    def test_validate_tool_metadata_invalid_embedding(self):
        """Test validating tool metadata with invalid embedding."""
        invalid_data = {"id": "test_tool", "name": "Test Tool", "embedding": [0.1] * 100}  # Wrong dimension count

        is_valid, errors = validate_tool_metadata(invalid_data)

        assert is_valid is False
        assert any("384 dimensions" in error for error in errors)

    def test_validate_tool_metadata_invalid_json(self):
        """Test validating tool metadata with invalid JSON."""
        invalid_data = {"id": "test_tool", "name": "Test Tool", "metadata_json": '{"invalid": json}'}  # Invalid JSON

        is_valid, errors = validate_tool_metadata(invalid_data)

        assert is_valid is False
        assert any("Invalid JSON" in error for error in errors)

    def test_validate_parameter_metadata_valid(self):
        """Test validating valid parameter metadata."""
        valid_data = {
            "id": "param_1",
            "tool_id": "test_tool",
            "parameter_name": "file_path",
            "parameter_type": "string",
            "embedding": [0.1] * 384,
            "schema_json": json.dumps({"type": "string"}),
        }

        is_valid, errors = validate_parameter_metadata(valid_data)

        assert is_valid is True
        assert len(errors) == 0

    def test_validate_parameter_metadata_missing_required(self):
        """Test validating parameter metadata with missing required fields."""
        invalid_data = {"description": "Missing required fields"}

        is_valid, errors = validate_parameter_metadata(invalid_data)

        assert is_valid is False
        assert len(errors) >= 3  # Missing id, tool_id, parameter_name

    def test_validate_parameter_metadata_invalid_type(self):
        """Test validating parameter metadata with invalid parameter_type."""
        invalid_data = {
            "id": "param_1",
            "tool_id": "test_tool",
            "parameter_name": "test_param",
            "parameter_type": "invalid_type",
        }

        is_valid, errors = validate_parameter_metadata(invalid_data)

        assert is_valid is False
        assert any("Invalid parameter_type" in error for error in errors)


class TestErrorHandling:
    """Test error handling in MCP tool schema operations."""

    def test_database_error_handling(self, db_manager):
        """Test that database errors are properly handled."""
        # Try to store tool without creating tables first
        with pytest.raises(DatabaseError):
            db_manager.store_mcp_tool(tool_id="test", name="Test")

    def test_migration_error_handling(self, migration_manager):
        """Test that migration errors are properly handled."""
        # Try to apply migration with invalid SQL
        invalid_sql = ["COMPLETELY INVALID SQL"]

        with pytest.raises(DatabaseMigrationError):
            migration_manager.apply_migration(1, "Invalid migration", invalid_sql)

    @patch("duckdb.connect")
    def test_connection_error_handling(self, mock_connect, temp_db):
        """Test handling of database connection errors."""
        mock_connect.side_effect = Exception("Connection failed")

        with pytest.raises(DatabaseError):
            db_manager = DatabaseManager(temp_db)
            db_manager.create_mcp_tool_tables()


# Integration test fixtures and helpers
@pytest.fixture
def populated_db(db_manager):
    """Create a database populated with test MCP tool data."""
    # Create tables
    db_manager.create_mcp_tool_tables()

    # Add sample tools
    tools = [
        {
            "tool_id": "bash",
            "name": "Bash",
            "description": "Execute bash commands",
            "tool_type": "system",
            "category": "file_ops",
            "embedding": [0.1] * 384,
        },
        {
            "tool_id": "read",
            "name": "Read",
            "description": "Read file contents",
            "tool_type": "system",
            "category": "file_ops",
            "embedding": [0.2] * 384,
        },
        {
            "tool_id": "web_fetch",
            "name": "WebFetch",
            "description": "Fetch web content",
            "tool_type": "system",
            "category": "web",
            "embedding": [0.3] * 384,
        },
    ]

    for tool in tools:
        db_manager.store_mcp_tool(**tool)

    # Add parameters for bash tool
    bash_params = [
        {
            "parameter_id": "bash_command",
            "tool_id": "bash",
            "parameter_name": "command",
            "parameter_type": "string",
            "is_required": True,
            "description": "Command to execute",
            "embedding": [0.11] * 384,
        },
        {
            "parameter_id": "bash_timeout",
            "tool_id": "bash",
            "parameter_name": "timeout",
            "parameter_type": "number",
            "is_required": False,
            "description": "Timeout in seconds",
            "embedding": [0.12] * 384,
        },
    ]

    for param in bash_params:
        db_manager.store_tool_parameter(**param)

    # Add examples
    examples = [
        {
            "example_id": "bash_ls",
            "tool_id": "bash",
            "use_case": "List files",
            "example_call": 'bash("ls -la")',
            "expected_output": "File listing",
            "context": "When you need to see directory contents",
            "embedding": [0.13] * 384,
            "effectiveness_score": 0.9,
        }
    ]

    for example in examples:
        db_manager.store_tool_example(**example)

    # Add relationships
    relationships = [
        {
            "relationship_id": "bash_read_complement",
            "tool_a_id": "bash",
            "tool_b_id": "read",
            "relationship_type": "complement",
            "strength": 0.7,
            "description": "Bash can create files that Read can access",
        }
    ]

    for rel in relationships:
        db_manager.store_tool_relationship(**rel)

    return db_manager


class TestIntegration:
    """Integration tests for the complete MCP tool schema system."""

    def test_full_migration_and_operations(self, db_manager):
        """Test complete migration and basic operations."""
        # Start with fresh database
        migration = MCPToolMigration(db_manager)

        # Migrate to latest
        result = migration.migrate_to_latest()
        assert result is True

        # Validate schema
        validation = migration.validate_schema_integrity()
        assert validation["valid"] is True

        # Perform basic operations
        db_manager.store_mcp_tool(tool_id="test", name="Test Tool")
        tool = db_manager.get_mcp_tool("test")
        assert tool is not None

        stats = db_manager.get_mcp_tool_statistics()
        assert stats["total_tools"] == 1

    def test_search_functionality(self, populated_db):
        """Test search functionality with populated database."""
        # Search for file operations
        file_embedding = [0.15] * 384  # Similar to bash/read embeddings
        results = populated_db.search_mcp_tools_by_embedding(file_embedding, category="file_ops")

        assert len(results) == 2  # bash and read
        assert all(tool["category"] == "file_ops" for tool in results)

        # Search parameters
        param_results = populated_db.search_tool_parameters_by_embedding(file_embedding, tool_id="bash")

        assert len(param_results) == 2  # command and timeout
        assert all(param["tool_id"] == "bash" for param in param_results)

    def test_relationship_queries(self, populated_db):
        """Test relationship queries with populated database."""
        # Get tools related to bash
        related = populated_db.get_related_tools("bash")

        assert len(related) == 1
        assert related[0]["name"] == "Read"
        assert related[0]["relationship_type"] == "complement"

        # Filter by relationship type
        complements = populated_db.get_related_tools("bash", "complement")
        assert len(complements) == 1

        alternatives = populated_db.get_related_tools("bash", "alternative")
        assert len(alternatives) == 0

    def test_cascade_deletion(self, populated_db):
        """Test that cascade deletion works properly."""
        # Verify bash tool has parameters and examples
        assert len(populated_db.get_tool_parameters("bash")) == 2
        assert len(populated_db.get_tool_examples("bash")) == 1

        # Remove bash tool
        result = populated_db.remove_mcp_tool("bash")
        assert result is True

        # Verify cascaded deletion
        assert populated_db.get_mcp_tool("bash") is None
        assert len(populated_db.get_tool_parameters("bash")) == 0
        assert len(populated_db.get_tool_examples("bash")) == 0

        # Verify relationships are also deleted
        related = populated_db.get_related_tools("read")
        assert len(related) == 0  # bash relationship should be gone
