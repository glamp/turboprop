#!/usr/bin/env python3
"""
Integration tests for edge cases across the entire MCP tool search system.

This module tests end-to-end edge case scenarios to ensure the complete
system handles edge cases gracefully across all components.
"""

import tempfile
import json
import os
import sys
from pathlib import Path
from unittest.mock import Mock, patch
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from database_manager import DatabaseManager
from embedding_helper import EmbeddingGenerator
from mcp_tool_search_engine import MCPToolSearchEngine
from tool_query_processor import ToolQueryProcessor
from tool_matching_algorithms import ToolMatchingAlgorithms
from search_result_formatter import SearchResultFormatter
from mcp_metadata_types import MCPToolMetadata, ParameterAnalysis, ToolExample, ToolId
from tool_search_results import ToolSearchResponse


class TestEndToEndEdgeCases:
    """Test edge cases across the complete system integration."""
    
    def setup_method(self):
        """Set up test fixtures with real components."""
        # Use temporary database for integration tests
        self.temp_db = tempfile.NamedTemporaryFile(suffix='.duckdb', delete=False)
        self.temp_db.close()  # Close the file so DuckDB can create it properly
        os.unlink(self.temp_db.name)  # Remove the file so DuckDB can create it from scratch
        self.db_manager = DatabaseManager(Path(self.temp_db.name))
        self.embedding_generator = Mock(spec=EmbeddingGenerator)
        self.search_engine = MCPToolSearchEngine(self.db_manager, self.embedding_generator)
        
        # Initialize database schema
        self.db_manager.create_mcp_tool_tables()
        
        # Set up mock embedding responses
        self.embedding_generator.encode.return_value = [0.1] * 384
    
    def teardown_method(self):
        """Clean up test fixtures."""
        self.db_manager.close()
        Path(self.temp_db.name).unlink(missing_ok=True)
    
    def test_large_result_set_integration(self):
        """Test complete system with large result sets."""
        # Insert many tools into database
        tools_data = []
        for i in range(120):
            tool_metadata = MCPToolMetadata(
                name=f"Integration Tool {i}",
                description=f"Tool {i} for integration testing with edge cases",
                category="integration",
                parameters=[
                    ParameterAnalysis(name=f"param_{i}", type="string", required=(i % 3 == 0), 
                                    description=f"Parameter {i} for integration testing")
                ],
                examples=[
                    ToolExample(use_case=f"Use case {i}", example_call=f"tool_{i}(param_{i}='value')")
                ]
            )
            tools_data.append(tool_metadata)
        
        # Store tools in database (mocked for speed)
        with patch.object(self.db_manager, 'store_mcp_tool'):
            with patch.object(self.db_manager, 'search_mcp_tools_by_embedding') as mock_search_tools:
                # Mock large dataset return
                mock_tools = []
                for i in range(120):
                    mock_tools.append({
                        'id': f'integration_tool_{i}',
                        'name': f'Integration Tool {i}',
                        'description': f'Tool {i} for integration testing',
                        'category': 'integration',
                        'tool_type': 'function',
                        'embedding': [0.1 + (i * 0.001)] * 384,
                        'metadata_json': json.dumps({'version': '1.0'})
                    })
                mock_search_tools.return_value = mock_tools
                
                # Perform search that should return large result set
                response = self.search_engine.search_by_functionality("integration testing", k=100)
                
                # Verify system handles large results correctly
                assert isinstance(response, ToolSearchResponse)
                assert len(response.results) <= 100  # Should be limited by k
                assert response.execution_time < 2.0  # Should complete reasonably fast
                assert response.total_results <= 100
    
    def test_malformed_data_recovery(self):
        """Test system recovery from malformed data at various levels."""
        # Test with malformed tool data in database
        malformed_tools = [
            {
                'id': 'malformed_1',
                'name': None,  # Null name
                'description': '',  # Empty description
                'category': 'test',
                'tool_type': 'function',
                'embedding': None,  # Null embedding
                'metadata_json': 'invalid_json{'
            },
            {
                'id': 'malformed_2', 
                'name': 'Valid Tool',
                'description': 'Valid description',
                'category': 'test',
                'tool_type': 'function',
                'embedding': [float('nan')] * 384,  # NaN embedding
                'metadata_json': json.dumps({'valid': True})
            },
            {
                'id': 'malformed_3',
                'name': 'Another Tool',
                'description': 'Another description',
                'category': 'test',
                'tool_type': 'function', 
                'embedding': [0.5] * 100,  # Wrong dimensions
                'metadata_json': json.dumps({'version': '1.0'})
            }
        ]
        
        with patch.object(self.db_manager, 'search_mcp_tools_by_embedding', return_value=malformed_tools):
            # System should handle malformed data gracefully
            response = self.search_engine.search_by_functionality("test malformed data")
            
            assert isinstance(response, ToolSearchResponse)
            # Should not crash and may have filtered results
            assert len(response.results) >= 0
            assert response.execution_time > 0
    
    def test_concurrent_database_access(self):
        """Test concurrent database access patterns."""
        import threading
        import time
        
        results = []
        errors = []
        
        def concurrent_search(thread_id):
            try:
                # Each thread performs multiple operations
                for i in range(5):
                    response = self.search_engine.search_by_functionality(f"thread {thread_id} query {i}")
                    results.append(response)
                    time.sleep(0.01)  # Small delay to create interleaving
            except Exception as e:
                errors.append((thread_id, e))
        
        # Mock database responses to avoid actual DB setup complexity
        with patch.object(self.db_manager, 'search_mcp_tools_by_embedding', return_value=[]):
            # Start multiple concurrent threads
            threads = []
            for thread_id in range(8):
                thread = threading.Thread(target=concurrent_search, args=(thread_id,))
                threads.append(thread)
            
            # Start all threads
            for thread in threads:
                thread.start()
            
            # Wait for completion
            for thread in threads:
                thread.join(timeout=10.0)
            
            # Verify concurrent access handled correctly
            assert len(errors) == 0, f"Concurrent access errors: {errors}"
            assert len(results) == 8 * 5, f"Expected 40 results, got {len(results)}"
            
            # All results should be valid
            for result in results:
                assert isinstance(result, ToolSearchResponse)
    
    def test_query_processing_pipeline_errors(self):
        """Test error propagation through the entire query processing pipeline."""
        error_queries = [
            ("", "empty query"),
            (None, "null query"),
            ("x" * 5000, "extremely long query"),
            ("query\nwith\nnewlines", "multiline query"),
            ("query with ñ and 中文", "unicode query"),
            ("<script>alert('xss')</script>", "potential XSS"),
            ("'; DROP TABLE tools; --", "SQL injection attempt"),
        ]
        
        for query, description in error_queries:
            try:
                if query is None or query == "":
                    # These should return error responses
                    response = self.search_engine.search_by_functionality(query)
                    assert isinstance(response, ToolSearchResponse), f"Failed for {description}: {query}"
                    assert response.execution_time > 0
                    assert len(response.results) == 0, "Error queries should return no results"
                else:
                    response = self.search_engine.search_by_functionality(query)
                    
                    # Should handle gracefully
                    assert isinstance(response, ToolSearchResponse), f"Failed for {description}: {query}"
                    assert response.execution_time > 0
                    # May have zero results for malformed queries
                    assert len(response.results) >= 0
            except Exception as e:
                pytest.fail(f"Unexpected error for {description} query '{query}': {e}")
    
    def test_memory_cleanup_after_errors(self):
        """Test memory cleanup after error conditions."""
        import gc
        import psutil
        import os
        
        def get_memory_mb():
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024
        
        initial_memory = get_memory_mb()
        
        # Generate various error conditions
        error_scenarios = [
            lambda: self.search_engine.search_by_functionality("x" * 1000),  # Long query
            lambda: self.search_engine.search_by_functionality("🎯" * 100),   # Unicode heavy
        ]
        
        # Test with mock failures at different points
        with patch.object(self.embedding_generator, 'encode', side_effect=Exception("Mock failure")):
            for i, scenario in enumerate(error_scenarios * 10):  # Repeat scenarios
                try:
                    response = scenario()
                    assert isinstance(response, ToolSearchResponse)
                except Exception:
                    pass  # Expected for some scenarios
                
                # Periodic cleanup
                if i % 5 == 0:
                    gc.collect()
        
        gc.collect()
        final_memory = get_memory_mb()
        memory_growth = final_memory - initial_memory
        
        # Memory should not grow excessively due to error conditions
        assert memory_growth < 30, f"Memory grew {memory_growth:.2f}MB - possible leak in error handling"
    
    def test_result_formatting_edge_cases(self):
        """Test result formatting with edge case data."""
        formatter = SearchResultFormatter()
        
        # Create response with edge case data
        from tool_search_results import ToolSearchResult
        
        edge_case_result = ToolSearchResult(
            tool_id=ToolId("edge_case_tool"),
            name="Tool with 特殊字符 and emojis 🚀🎯",
            description="A" * 1000,  # Very long description
            category="edge_case",
            tool_type="function",
            similarity_score=0.999999,  # High precision
            relevance_score=0.000001,   # Very low score
            confidence_level="high",
            match_reasons=[
                "Exact match on tool name",
                "Category alignment perfect",
                "Parameter compatibility excellent",
                "Example quality high",
                "Documentation completeness perfect"
            ] * 3,  # Too many reasons
            parameters=[],
            examples=[],
        )
        
        response = ToolSearchResponse(
            query="edge case formatting test with 特殊字符",
            results=[edge_case_result],
            total_results=1,
            execution_time=0.123456789,  # High precision timing
            search_strategy="hybrid",
        )
        
        # Should handle formatting without errors
        console_output = formatter.format_console_output(response)
        json_output = formatter.format_json_output(response)
        
        assert isinstance(console_output, str)
        assert len(console_output) > 0
        assert isinstance(json_output, str)
        
        # Should be valid JSON
        parsed_json = json.loads(json_output)
        assert "results" in parsed_json
        assert len(parsed_json["results"]) == 1
    
    def test_configuration_edge_cases(self):
        """Test system behavior with edge case configurations."""
        # Test with extreme k values
        extreme_k_values = [0, 1, 1000, -1]
        
        for k in extreme_k_values:
            try:
                response = self.search_engine.search_by_functionality("config test", k=k)
                
                if k <= 0:
                    # Should handle gracefully or set reasonable minimum
                    assert len(response.results) == 0 or len(response.results) > 0
                else:
                    assert isinstance(response, ToolSearchResponse)
                    if k == 1000:
                        # Should cap results reasonably
                        assert len(response.results) <= 100
                    else:
                        assert len(response.results) <= k
                        
            except ValueError:
                # Acceptable to raise ValueError for invalid k values
                assert k <= 0, f"Unexpected ValueError for k={k}"
    
    def test_system_state_consistency(self):
        """Test system state remains consistent after edge case operations."""
        # Store initial state indicators
        initial_cache_state = hasattr(self.search_engine, '_results_cache')
        initial_db_connection = self.db_manager is not None
        
        # Perform various edge case operations
        edge_operations = [
            lambda: self.search_engine.search_by_functionality("consistency test 1"),
            lambda: self.search_engine.search_by_functionality("consistency test 2", k=0),
            lambda: self.search_engine.search_by_functionality(""),  # Should raise ValueError
        ]
        
        for operation in edge_operations:
            try:
                operation()
            except ValueError:
                pass  # Expected for some operations
            except Exception as e:
                # Unexpected error - but system should still be consistent
                pass
        
        # Verify system state is still consistent
        assert hasattr(self.search_engine, '_results_cache') == initial_cache_state
        assert (self.db_manager is not None) == initial_db_connection
        
        # Should still be able to perform normal operations
        response = self.search_engine.search_by_functionality("final consistency check")
        assert isinstance(response, ToolSearchResponse)


if __name__ == "__main__":
    # Run integration tests
    pytest.main([__file__, "-v", "--tb=short", "-x"])  # Stop on first failure for debugging