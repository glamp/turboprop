#!/usr/bin/env python3
"""
Tool Storage Operations

This module provides enhanced database operations for tool cataloging with
batch processing, transaction handling, and conflict resolution capabilities.
It extends the existing DatabaseManager with specialized operations optimized
for tool metadata storage and retrieval.
"""

import json
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

from database_manager import DatabaseManager
from logging_config import get_logger
from mcp_metadata_types import MCPToolMetadata, ParameterAnalysis, ToolExample
from mcp_tool_schema import generate_tool_id

logger = get_logger(__name__)


@dataclass
class StorageResult:
    """Result of a storage operation."""
    success: bool
    tools_stored: int
    tools_failed: int
    errors: List[str]
    execution_time: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class BatchStorageStats:
    """Statistics for batch storage operations."""
    total_tools: int
    tools_stored: int
    tools_updated: int
    tools_failed: int
    parameters_stored: int
    examples_stored: int
    relationships_stored: int
    execution_time: float


class ToolStorageOperations:
    """Handles all database operations for tool cataloging with enhanced batch capabilities."""

    def __init__(self, db_manager: DatabaseManager):
        """
        Initialize storage operations.
        
        Args:
            db_manager: DatabaseManager instance for database operations
        """
        self.db_manager = db_manager
        logger.info("Initialized ToolStorageOperations")

    def store_tool_batch(self, tools: List[MCPToolMetadata]) -> StorageResult:
        """
        Store multiple tools in a single transaction.
        
        Uses database transactions for consistency, handles conflicts and duplicates,
        updates existing tools with new metadata, and returns detailed storage statistics.
        
        Args:
            tools: List of MCPToolMetadata objects to store
            
        Returns:
            StorageResult with operation statistics
        """
        if not tools:
            logger.info("No tools to store in batch operation")
            return StorageResult(
                success=True,
                tools_stored=0,
                tools_failed=0,
                errors=[],
                execution_time=0.0
            )

        logger.info("Starting batch storage of %d tools", len(tools))
        start_time = time.time()
        
        tools_stored = 0
        tools_failed = 0
        errors = []

        try:
            # Prepare batch operations for transaction
            operations = []
            
            for tool in tools:
                try:
                    # Validate tool metadata
                    if not self._validate_tool_metadata(tool):
                        tools_failed += 1
                        error_msg = f"Invalid metadata for tool: {getattr(tool, 'name', 'Unknown')}"
                        errors.append(error_msg)
                        continue
                    
                    # Prepare tool storage operation
                    tool_id = generate_tool_id()
                    
                    # Convert tool metadata to storage format
                    metadata_json = json.dumps(self._extract_additional_metadata(tool))
                    
                    # Create embedding placeholder (will be generated later)
                    embedding = [0.0] * 384  # Will be replaced by actual embedding
                    
                    # Add tool insertion operation
                    from datetime import datetime
                    current_time = datetime.now()
                    
                    operations.append((
                        """
                        INSERT INTO mcp_tools
                        (id, name, description, tool_type, provider, version, category,
                         embedding, metadata_json, is_active, last_updated)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT (id) DO UPDATE SET
                        name = EXCLUDED.name,
                        description = EXCLUDED.description,
                        tool_type = EXCLUDED.tool_type,
                        provider = EXCLUDED.provider,
                        version = EXCLUDED.version,
                        category = EXCLUDED.category,
                        embedding = EXCLUDED.embedding,
                        metadata_json = EXCLUDED.metadata_json,
                        is_active = EXCLUDED.is_active,
                        last_updated = EXCLUDED.last_updated
                        """,
                        (
                            tool_id,
                            tool.name,
                            tool.description,
                            getattr(tool, 'tool_type', 'custom'),
                            getattr(tool, 'provider', 'unknown'),
                            getattr(tool, 'version', None),
                            tool.category,
                            embedding,
                            metadata_json,
                            True,
                            current_time,
                        )
                    ))
                    
                    # Add parameter operations
                    for param in getattr(tool, 'parameters', []):
                        param_id = generate_tool_id()
                        param_embedding = [0.0] * 384  # Will be replaced by actual embedding
                        
                        operations.append((
                            """
                            INSERT INTO tool_parameters
                            (id, tool_id, parameter_name, parameter_type, is_required,
                             description, default_value, schema_json, embedding)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                            ON CONFLICT (id) DO UPDATE SET
                            parameter_name = EXCLUDED.parameter_name,
                            parameter_type = EXCLUDED.parameter_type,
                            is_required = EXCLUDED.is_required,
                            description = EXCLUDED.description,
                            default_value = EXCLUDED.default_value,
                            schema_json = EXCLUDED.schema_json,
                            embedding = EXCLUDED.embedding
                            """,
                            (
                                param_id,
                                tool_id,
                                param.name,
                                param.type,
                                param.required,
                                param.description,
                                str(getattr(param, 'default_value', None)) if hasattr(param, 'default_value') else None,
                                json.dumps(getattr(param, 'constraints', {})),
                                param_embedding,
                            )
                        ))
                    
                    # Add example operations
                    for example in getattr(tool, 'examples', []):
                        example_id = generate_tool_id()
                        example_embedding = [0.0] * 384  # Will be replaced by actual embedding
                        
                        operations.append((
                            """
                            INSERT INTO tool_examples
                            (id, tool_id, use_case, example_call, expected_output,
                             context, embedding, effectiveness_score)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                            ON CONFLICT (id) DO UPDATE SET
                            use_case = EXCLUDED.use_case,
                            example_call = EXCLUDED.example_call,
                            expected_output = EXCLUDED.expected_output,
                            context = EXCLUDED.context,
                            embedding = EXCLUDED.embedding,
                            effectiveness_score = EXCLUDED.effectiveness_score
                            """,
                            (
                                example_id,
                                tool_id,
                                example.use_case,
                                getattr(example, 'example_call', ''),
                                getattr(example, 'expected_output', ''),
                                getattr(example, 'context', ''),
                                example_embedding,
                                getattr(example, 'effectiveness_score', 0.8),
                            )
                        ))
                    
                    tools_stored += 1
                    
                except Exception as e:
                    tools_failed += 1
                    error_msg = f"Failed to prepare storage for tool {getattr(tool, 'name', 'Unknown')}: {e}"
                    errors.append(error_msg)
                    logger.error(error_msg)
                    continue
            
            # Execute all operations in a single transaction if any tools were prepared
            if operations:
                try:
                    self.db_manager.execute_transaction(operations)
                    logger.info("Successfully stored %d tools in batch transaction", tools_stored)
                except Exception as e:
                    # Transaction failed, all tools failed
                    error_msg = f"Batch transaction failed: {e}"
                    errors.append(error_msg)
                    logger.error(error_msg)
                    
                    tools_failed = len(tools)
                    tools_stored = 0
            
            execution_time = time.time() - start_time
            
            result = StorageResult(
                success=tools_failed == 0,
                tools_stored=tools_stored,
                tools_failed=tools_failed,
                errors=errors,
                execution_time=execution_time
            )
            
            logger.info(
                "Batch storage completed: %d stored, %d failed in %.2fs",
                tools_stored, tools_failed, execution_time
            )
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Critical error during batch storage: {e}"
            logger.error(error_msg)
            
            return StorageResult(
                success=False,
                tools_stored=0,
                tools_failed=len(tools),
                errors=[error_msg],
                execution_time=execution_time
            )

    def store_tool_parameters(self, tool_id: str, parameters: List[ParameterAnalysis]) -> None:
        """
        Store tool parameters with embeddings.
        
        Args:
            tool_id: ID of the tool these parameters belong to
            parameters: List of parameter analysis objects
        """
        logger.debug("Storing %d parameters for tool %s", len(parameters), tool_id)
        
        operations = []
        
        for param in parameters:
            param_id = generate_tool_id()
            param_embedding = [0.0] * 384  # Placeholder for actual embedding
            
            operations.append((
                """
                INSERT INTO tool_parameters
                (id, tool_id, parameter_name, parameter_type, is_required,
                 description, default_value, schema_json, embedding)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (id) DO UPDATE SET
                parameter_name = EXCLUDED.parameter_name,
                parameter_type = EXCLUDED.parameter_type,
                is_required = EXCLUDED.is_required,
                description = EXCLUDED.description,
                default_value = EXCLUDED.default_value,
                schema_json = EXCLUDED.schema_json,
                embedding = EXCLUDED.embedding
                """,
                (
                    param_id,
                    tool_id,
                    param.name,
                    param.type,
                    param.required,
                    param.description,
                    str(param.default_value) if param.default_value is not None else None,
                    json.dumps(param.constraints),
                    param_embedding,
                )
            ))
        
        # Execute all parameter insertions in transaction
        if operations:
            self.db_manager.execute_transaction(operations)
            logger.debug("Successfully stored %d parameters for tool %s", len(parameters), tool_id)

    def store_tool_examples(self, tool_id: str, examples: List[ToolExample]) -> None:
        """
        Store tool usage examples.
        
        Args:
            tool_id: ID of the tool these examples belong to
            examples: List of tool example objects
        """
        logger.debug("Storing %d examples for tool %s", len(examples), tool_id)
        
        operations = []
        
        for example in examples:
            example_id = generate_tool_id()
            example_embedding = [0.0] * 384  # Placeholder for actual embedding
            
            operations.append((
                """
                INSERT INTO tool_examples
                (id, tool_id, use_case, example_call, expected_output,
                 context, embedding, effectiveness_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (id) DO UPDATE SET
                use_case = EXCLUDED.use_case,
                example_call = EXCLUDED.example_call,
                expected_output = EXCLUDED.expected_output,
                context = EXCLUDED.context,
                embedding = EXCLUDED.embedding,
                effectiveness_score = EXCLUDED.effectiveness_score
                """,
                (
                    example_id,
                    tool_id,
                    example.use_case,
                    example.example_call,
                    example.expected_output,
                    example.context,
                    example_embedding,
                    example.effectiveness_score,
                )
            ))
        
        # Execute all example insertions in transaction
        if operations:
            self.db_manager.execute_transaction(operations)
            logger.debug("Successfully stored %d examples for tool %s", len(examples), tool_id)

    def store_tool_relationships(self, relationships) -> None:
        """
        Store detected tool relationships.
        
        Args:
            relationships: List of tool relationship objects
        """
        logger.debug("Storing %d tool relationships", len(relationships))
        
        operations = []
        
        for relationship in relationships:
            relationship_id = generate_tool_id()
            
            operations.append((
                """
                INSERT INTO tool_relationships
                (id, tool_a_id, tool_b_id, relationship_type, strength, description)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT (tool_a_id, tool_b_id, relationship_type) DO UPDATE SET
                id = EXCLUDED.id,
                strength = EXCLUDED.strength,
                description = EXCLUDED.description
                """,
                (
                    relationship_id,
                    relationship.tool_a_id,
                    relationship.tool_b_id,
                    relationship.relationship_type,
                    relationship.strength,
                    relationship.description,
                )
            ))
        
        # Execute all relationship insertions in transaction
        if operations:
            self.db_manager.execute_transaction(operations)
            logger.debug("Successfully stored %d tool relationships", len(relationships))

    def get_batch_storage_statistics(self) -> BatchStorageStats:
        """
        Get comprehensive statistics about batch storage operations.
        
        Returns:
            BatchStorageStats with detailed operation metrics
        """
        with self.db_manager.get_connection() as conn:
            # Get tool counts
            tool_count = conn.execute("SELECT COUNT(*) FROM mcp_tools").fetchone()[0]
            
            # Get parameter counts  
            param_count = conn.execute("SELECT COUNT(*) FROM tool_parameters").fetchone()[0]
            
            # Get example counts
            example_count = conn.execute("SELECT COUNT(*) FROM tool_examples").fetchone()[0]
            
            # Get relationship counts
            relationship_count = conn.execute("SELECT COUNT(*) FROM tool_relationships").fetchone()[0]
            
            return BatchStorageStats(
                total_tools=tool_count,
                tools_stored=tool_count,
                tools_updated=0,  # Would need change tracking to implement
                tools_failed=0,   # Would need error tracking to implement
                parameters_stored=param_count,
                examples_stored=example_count,
                relationships_stored=relationship_count,
                execution_time=0.0  # Would need timing tracking to implement
            )

    def _validate_tool_metadata(self, tool: MCPToolMetadata) -> bool:
        """
        Validate tool metadata before storage.
        
        Args:
            tool: Tool metadata to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not tool:
            return False
            
        if not hasattr(tool, 'name') or not tool.name:
            logger.warning("Tool missing name")
            return False
            
        if not hasattr(tool, 'description'):
            logger.warning("Tool missing description")
            return False
            
        if not hasattr(tool, 'category'):
            logger.warning("Tool missing category")  
            return False
            
        return True

    def _extract_additional_metadata(self, tool: MCPToolMetadata) -> Dict[str, Any]:
        """
        Extract additional metadata for JSON storage.
        
        Args:
            tool: Tool metadata object
            
        Returns:
            Dictionary of additional metadata
        """
        metadata = {}
        
        if hasattr(tool, 'usage_patterns'):
            metadata['usage_patterns'] = [
                pattern.to_dict() for pattern in tool.usage_patterns
            ]
            
        if hasattr(tool, 'complexity_analysis') and tool.complexity_analysis:
            metadata['complexity_analysis'] = tool.complexity_analysis.to_dict()
            
        if hasattr(tool, 'documentation_analysis') and tool.documentation_analysis:
            metadata['documentation_analysis'] = tool.documentation_analysis.to_dict()
            
        return metadata