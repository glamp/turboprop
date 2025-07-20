# Database Migrations

This directory contains database migration scripts for the MCP Tool Search System.

## Migration Files

Migration files are automatically generated when schema migrations are applied. Each migration creates:

1. **Migration record**: Stored in the `schema_version` table
2. **Rollback script**: File named `rollback_v{version:03d}.sql`

## Running Migrations

Migrations are run automatically when initializing the MCP tool schema:

```python
from database_manager import DatabaseManager
from mcp_tool_schema import MCPToolMigration

# Initialize database manager
db_manager = DatabaseManager(db_path)

# Create migration manager and run migrations
migration = MCPToolMigration(db_manager)
migration.migrate_to_latest()
```

## Manual Rollback

If you need to rollback a migration, you can run the corresponding rollback script:

```bash
# Example: Rollback migration version 1
sqlite3 /path/to/database.db < migrations/rollback_v001.sql
```

## Migration Versions

- **Version 0**: Base schema (original code_files table)
- **Version 1**: Initial MCP tool schema (mcp_tools, tool_parameters, tool_examples, tool_relationships)

## Safety Notes

- Always backup your database before running migrations
- Test migrations on a copy of production data first
- Rollback scripts are provided but use with caution
- Migration failures will rollback automatically within the transaction