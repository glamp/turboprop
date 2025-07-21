-- Rollback script for migration version 1
-- Generated at 2025-07-21 08:07:53

BEGIN TRANSACTION;

DROP TABLE IF EXISTS tool_relationships;

DROP TABLE IF EXISTS tool_examples;

DROP TABLE IF EXISTS tool_parameters;

DROP TABLE IF EXISTS mcp_tools;

-- Remove migration record
DELETE FROM schema_version WHERE version = 1;

COMMIT;
