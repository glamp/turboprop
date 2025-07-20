#!/usr/bin/env python3
"""
MCP Tool Discovery Framework

This module provides comprehensive tool discovery capabilities for Claude Code's
built-in system tools and custom MCP tools. It automatically identifies, catalogs,
and generates semantic embeddings for all available tools.
"""

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from database_manager import DatabaseManager
from embedding_helper import EmbeddingGenerator
from logging_config import get_logger
from mcp_tool_schema import generate_tool_id
from tool_metadata_extractor import MCPToolMetadata, ToolMetadataExtractor
from tool_registry import ToolRegistry

logger = get_logger(__name__)


@dataclass
class MCPTool:
    """Represents a discovered MCP tool with comprehensive metadata."""

    id: str
    name: str
    description: str
    tool_type: str  # 'system', 'custom', 'third_party'
    provider: str
    category: str
    parameters: List["ParameterInfo"] = field(default_factory=list)
    examples: List["ToolExample"] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: Optional[str] = None
    fingerprint: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class ParameterInfo:
    """Represents a tool parameter with validation schema."""

    name: str
    type: str
    required: bool
    description: str
    default_value: Optional[Any] = None
    schema: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class ToolExample:
    """Represents a tool usage example."""

    use_case: str
    example_call: str
    expected_output: str
    context: str = ""
    effectiveness_score: float = 0.8

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class CatalogResult:
    """Result of cataloging tools to the database."""

    success: bool
    tools_stored: int
    tools_failed: int
    errors: List[str] = field(default_factory=list)
    execution_time: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class MCPToolDiscovery:
    """
    Discover and catalog all available MCP tools.

    This class provides comprehensive tool discovery for Claude Code's built-in
    system tools and extensible framework for custom tool discovery.
    """

    # Known Claude Code system tools with metadata
    SYSTEM_TOOLS_CATALOG = {
        "bash": {
            "name": "Bash",
            "description": (
                "Executes bash commands in a persistent shell session " "with timeout and security measures"
            ),
            "category": "execution",
            "parameters": [
                {"name": "command", "type": "string", "required": True, "description": "The command to execute"},
                {
                    "name": "timeout",
                    "type": "number",
                    "required": False,
                    "description": "Optional timeout in milliseconds (up to 600000ms / 10 minutes)",
                },
            ],
        },
        "read": {
            "name": "Read",
            "description": ("Reads a file from the local filesystem with line " "offset and limit support"),
            "category": "file_ops",
            "parameters": [
                {
                    "name": "file_path",
                    "type": "string",
                    "required": True,
                    "description": "The absolute path to the file to read",
                },
                {"name": "limit", "type": "number", "required": False, "description": "The number of lines to read"},
                {
                    "name": "offset",
                    "type": "number",
                    "required": False,
                    "description": "The line number to start reading from",
                },
            ],
        },
        "write": {
            "name": "Write",
            "description": ("Writes content to a file on the local filesystem, " "overwriting existing content"),
            "category": "file_ops",
            "parameters": [
                {
                    "name": "file_path",
                    "type": "string",
                    "required": True,
                    "description": "The absolute path to the file to write",
                },
                {
                    "name": "content",
                    "type": "string",
                    "required": True,
                    "description": "The content to write to the file",
                },
            ],
        },
        "edit": {
            "name": "Edit",
            "description": "Performs exact string replacements in files with precise matching",
            "category": "file_ops",
            "parameters": [
                {
                    "name": "file_path",
                    "type": "string",
                    "required": True,
                    "description": "The absolute path to the file to modify",
                },
                {"name": "old_string", "type": "string", "required": True, "description": "The text to replace"},
                {
                    "name": "new_string",
                    "type": "string",
                    "required": True,
                    "description": "The text to replace it with",
                },
                {
                    "name": "replace_all",
                    "type": "boolean",
                    "required": False,
                    "description": "Replace all occurrences of old_string",
                },
            ],
        },
        "multiedit": {
            "name": "MultiEdit",
            "description": "Makes multiple edits to a single file in one atomic operation",
            "category": "file_ops",
            "parameters": [
                {
                    "name": "file_path",
                    "type": "string",
                    "required": True,
                    "description": "The absolute path to the file to modify",
                },
                {
                    "name": "edits",
                    "type": "array",
                    "required": True,
                    "description": "Array of edit operations to perform sequentially",
                },
            ],
        },
        "grep": {
            "name": "Grep",
            "description": "Powerful search tool built on ripgrep with regex and filtering support",
            "category": "search",
            "parameters": [
                {
                    "name": "pattern",
                    "type": "string",
                    "required": True,
                    "description": "The regular expression pattern to search for",
                },
                {"name": "path", "type": "string", "required": False, "description": "File or directory to search in"},
                {"name": "glob", "type": "string", "required": False, "description": "Glob pattern to filter files"},
            ],
        },
        "glob": {
            "name": "Glob",
            "description": (
                "Fast file pattern matching tool supporting glob patterns " "and modification time sorting"
            ),
            "category": "search",
            "parameters": [
                {
                    "name": "pattern",
                    "type": "string",
                    "required": True,
                    "description": "The glob pattern to match files against",
                },
                {"name": "path", "type": "string", "required": False, "description": "The directory to search in"},
            ],
        },
        "ls": {
            "name": "LS",
            "description": "Lists files and directories with glob pattern ignore support",
            "category": "file_ops",
            "parameters": [
                {
                    "name": "path",
                    "type": "string",
                    "required": True,
                    "description": "The absolute path to the directory to list",
                },
                {
                    "name": "ignore",
                    "type": "array",
                    "required": False,
                    "description": "List of glob patterns to ignore",
                },
            ],
        },
        "task": {
            "name": "Task",
            "description": ("Launches a new agent with access to tools for " "autonomous task completion"),
            "category": "development",
            "parameters": [
                {
                    "name": "description",
                    "type": "string",
                    "required": True,
                    "description": "A short description of the task",
                },
                {
                    "name": "prompt",
                    "type": "string",
                    "required": True,
                    "description": "The task for the agent to perform",
                },
            ],
        },
        "todowrite": {
            "name": "TodoWrite",
            "description": ("Creates and manages a structured task list for " "tracking coding session progress"),
            "category": "development",
            "parameters": [
                {
                    "name": "todos",
                    "type": "array",
                    "required": True,
                    "description": "The updated todo list with status tracking",
                }
            ],
        },
        "webfetch": {
            "name": "WebFetch",
            "description": "Fetches content from URLs and processes it using AI models",
            "category": "web",
            "parameters": [
                {"name": "url", "type": "string", "required": True, "description": "The URL to fetch content from"},
                {
                    "name": "prompt",
                    "type": "string",
                    "required": True,
                    "description": "The prompt to run on the fetched content",
                },
            ],
        },
        "websearch": {
            "name": "WebSearch",
            "description": "Searches the web and returns formatted search results",
            "category": "web",
            "parameters": [
                {"name": "query", "type": "string", "required": True, "description": "The search query to use"},
                {
                    "name": "allowed_domains",
                    "type": "array",
                    "required": False,
                    "description": "Only include search results from these domains",
                },
            ],
        },
        "notebookread": {
            "name": "NotebookRead",
            "description": "Reads Jupyter notebook files and returns cells with outputs",
            "category": "notebook",
            "parameters": [
                {
                    "name": "notebook_path",
                    "type": "string",
                    "required": True,
                    "description": "The absolute path to the Jupyter notebook file",
                },
                {
                    "name": "cell_id",
                    "type": "string",
                    "required": False,
                    "description": "The ID of a specific cell to read",
                },
            ],
        },
        "notebokedit": {
            "name": "NotebookEdit",
            "description": ("Edits Jupyter notebook cells with support for insert, " "replace, and delete operations"),
            "category": "notebook",
            "parameters": [
                {
                    "name": "notebook_path",
                    "type": "string",
                    "required": True,
                    "description": "The absolute path to the Jupyter notebook file",
                },
                {
                    "name": "new_source",
                    "type": "string",
                    "required": True,
                    "description": "The new source for the cell",
                },
                {"name": "cell_id", "type": "string", "required": False, "description": "The ID of the cell to edit"},
            ],
        },
        "exit_plan_mode": {
            "name": "ExitPlanMode",
            "description": "Exits plan mode and prompts user approval for implementation steps",
            "category": "workflow",
            "parameters": [
                {
                    "name": "plan",
                    "type": "string",
                    "required": True,
                    "description": "The plan to present to the user for approval",
                }
            ],
        },
    }

    def __init__(self, db_manager: DatabaseManager, embedding_generator: EmbeddingGenerator):
        """Initialize the tool discovery engine."""
        self.db_manager = db_manager
        self.embedding_generator = embedding_generator
        self.tool_registry = ToolRegistry()
        self.metadata_extractor = ToolMetadataExtractor()

        logger.info("Initialized MCP Tool Discovery engine")

    def discover_system_tools(self) -> List[MCPTool]:
        """
        Discover Claude Code's built-in system tools.

        Uses the hardcoded system tools catalog to ensure complete and accurate
        discovery of all known system tools.

        Returns:
            List of discovered MCPTool instances
        """
        logger.info("Discovering Claude Code system tools")
        discovered_tools = []

        for tool_id, tool_def in self.SYSTEM_TOOLS_CATALOG.items():
            try:
                # Convert parameters to ParameterInfo objects
                parameters = []
                for param_def in tool_def.get("parameters", []):
                    param = ParameterInfo(
                        name=param_def["name"],
                        type=param_def["type"],
                        required=param_def["required"],
                        description=param_def["description"],
                        default_value=param_def.get("default_value"),
                        schema=param_def.get("schema", {}),
                    )
                    parameters.append(param)

                # Create MCPTool instance
                tool = MCPTool(
                    id=tool_id,
                    name=tool_def["name"],
                    description=tool_def["description"],
                    tool_type="system",
                    provider="claude-code",
                    category=tool_def["category"],
                    parameters=parameters,
                    examples=[],  # Could be populated later
                    metadata={"discovered_at": time.time()},
                    fingerprint=self._generate_tool_fingerprint(tool_def),
                )

                discovered_tools.append(tool)
                logger.debug("Discovered system tool: %s", tool.name)

            except Exception as e:
                logger.error("Failed to process system tool %s: %s", tool_id, e)
                continue

        logger.info("Discovered %d system tools", len(discovered_tools))
        return discovered_tools

    def discover_custom_tools(self) -> List[MCPTool]:
        """
        Discover custom MCP tools in the environment.

        This is a placeholder for future implementation of custom tool discovery.

        Returns:
            List of discovered custom MCPTool instances
        """
        logger.info("Custom tool discovery not yet implemented")
        return []

    def extract_tool_metadata(self, tool_def: Any) -> MCPToolMetadata:
        """
        Extract comprehensive metadata from tool definition.

        Args:
            tool_def: Tool definition object

        Returns:
            Extracted metadata object
        """
        return self.metadata_extractor.extract_tool_metadata(tool_def)

    def catalog_tools(self, tools: List[MCPTool]) -> CatalogResult:
        """
        Store discovered tools in database with embeddings.

        Args:
            tools: List of discovered tools to catalog

        Returns:
            CatalogResult with operation statistics
        """
        logger.info("Cataloging %d tools to database", len(tools))
        start_time = time.time()

        tools_stored = 0
        tools_failed = 0
        errors = []

        try:
            for tool in tools:
                try:
                    # Generate embedding for tool description
                    embedding = self.embedding_generator.encode(tool.description)

                    # Store tool in database
                    self.db_manager.store_mcp_tool(
                        tool_id=tool.id,
                        name=tool.name,
                        description=tool.description,
                        tool_type=tool.tool_type,
                        provider=tool.provider,
                        version=tool.version,
                        category=tool.category,
                        embedding=embedding.tolist(),
                        metadata_json=json.dumps(tool.metadata),
                        is_active=True,
                    )

                    # Store tool parameters
                    for param in tool.parameters:
                        param_embedding = self.embedding_generator.encode(param.description)
                        self.db_manager.store_tool_parameter(
                            parameter_id=generate_tool_id(),
                            tool_id=tool.id,
                            parameter_name=param.name,
                            parameter_type=param.type,
                            is_required=param.required,
                            description=param.description,
                            default_value=str(param.default_value) if param.default_value else None,
                            schema_json=json.dumps(param.schema),
                            embedding=param_embedding.tolist(),
                        )

                    # Store tool examples
                    for example in tool.examples:
                        example_text = f"{example.use_case} {example.context}"
                        example_embedding = self.embedding_generator.encode(example_text)
                        self.db_manager.store_tool_example(
                            example_id=generate_tool_id(),
                            tool_id=tool.id,
                            use_case=example.use_case,
                            example_call=example.example_call,
                            expected_output=example.expected_output,
                            context=example.context,
                            embedding=example_embedding.tolist(),
                            effectiveness_score=example.effectiveness_score,
                        )

                    # Register with tool registry
                    self.tool_registry.register_tool(tool)

                    tools_stored += 1
                    logger.debug("Cataloged tool: %s", tool.name)

                except Exception as e:
                    tools_failed += 1
                    error_msg = f"Failed to catalog tool {tool.name}: {e}"
                    errors.append(error_msg)
                    logger.error(error_msg)
                    continue

            execution_time = time.time() - start_time

            result = CatalogResult(
                success=tools_failed == 0,
                tools_stored=tools_stored,
                tools_failed=tools_failed,
                errors=errors,
                execution_time=execution_time,
            )

            logger.info(
                "Cataloging complete: %d stored, %d failed in %.2fs", tools_stored, tools_failed, execution_time
            )

            return result

        except Exception as e:
            logger.error("Critical error during tool cataloging: %s", e)
            return CatalogResult(
                success=False,
                tools_stored=tools_stored,
                tools_failed=len(tools) - tools_stored,
                errors=[str(e)],
                execution_time=time.time() - start_time,
            )

    def _generate_tool_fingerprint(self, tool_def: Dict[str, Any]) -> str:
        """
        Generate a fingerprint for tool definition to detect changes.

        Args:
            tool_def: Tool definition dictionary

        Returns:
            SHA-256 hash fingerprint
        """
        # Create a deterministic string representation
        fingerprint_data = json.dumps(tool_def, sort_keys=True)
        return hashlib.sha256(fingerprint_data.encode()).hexdigest()

    def discover_and_catalog_all(self) -> Dict[str, Any]:
        """
        Complete discovery and cataloging workflow.

        Discovers all available tools and catalogs them to the database.

        Returns:
            Dictionary with discovery and cataloging results
        """
        logger.info("Starting complete tool discovery and cataloging")
        start_time = time.time()

        # Discover system tools
        system_tools = self.discover_system_tools()

        # Discover custom tools (placeholder)
        custom_tools = self.discover_custom_tools()

        all_tools = system_tools + custom_tools

        # Catalog all tools
        catalog_result = self.catalog_tools(all_tools)

        execution_time = time.time() - start_time

        results = {
            "system_tools_found": len(system_tools),
            "custom_tools_found": len(custom_tools),
            "total_tools_discovered": len(all_tools),
            "catalog_result": catalog_result.to_dict(),
            "total_execution_time": execution_time,
        }

        logger.info("Discovery and cataloging complete: %d tools in %.2fs", len(all_tools), execution_time)

        return results
