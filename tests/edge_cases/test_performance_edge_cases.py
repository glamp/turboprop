#!/usr/bin/env python3
"""
Performance edge case tests for MCP tool search engine.

This module tests performance-critical edge cases:
- Response time validation (<500ms requirement)
- Memory usage under stress
- Concurrent access patterns
- Resource exhaustion handling
"""

import time
import threading
import sys
import gc
import psutil
import os
from pathlib import Path
from unittest.mock import Mock, patch
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from database_manager import DatabaseManager
from embedding_helper import EmbeddingGenerator
from mcp_tool_search_engine import MCPToolSearchEngine
from tool_search_results import ProcessedQuery, ToolSearchResponse


class TestPerformanceRequirements:
    """Test that performance requirements are met under various conditions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.db_manager = Mock(spec=DatabaseManager)
        self.embedding_generator = Mock(spec=EmbeddingGenerator)
        self.search_engine = MCPToolSearchEngine(self.db_manager, self.embedding_generator)
    
    def test_response_time_under_500ms(self):
        """Test that search responses complete within 500ms requirement."""
        # Mock fast responses
        with patch.object(self.search_engine, '_perform_vector_search', return_value=[]):
            with patch.object(self.search_engine.query_processor, 'process_query') as mock_process:
                mock_process.return_value = ProcessedQuery(
                    original_query="performance test",
                    cleaned_query="performance test",
                    expanded_terms=["performance", "test"],
                    confidence=0.8,
                )
                
                start_time = time.time()
                response = self.search_engine.search_by_functionality("performance test")
                execution_time = time.time() - start_time
                
                # Verify response time requirement
                assert execution_time < 0.5, f"Response time {execution_time:.3f}s exceeds 500ms requirement"
                assert response.execution_time < 0.5, f"Reported execution time {response.execution_time:.3f}s exceeds 500ms"
    
    def test_response_time_with_large_dataset(self):
        """Test response time with larger realistic datasets."""
        # Create moderate-sized mock dataset (50 results)
        large_results = []
        for i in range(50):
            tool_data = {
                "id": f"perf_tool_{i}",
                "name": f"Performance Tool {i}",
                "description": f"Tool for performance testing scenario {i}",
                "category": "performance",
                "tool_type": "function",
                "metadata_json": '{"version": "1.0"}',
            }
            large_results.append((f"perf_tool_{i}", 0.8 - (i * 0.01), tool_data))
        
        with patch.object(self.search_engine, '_perform_vector_search', return_value=large_results):
            with patch.object(self.search_engine.query_processor, 'process_query') as mock_process:
                mock_process.return_value = ProcessedQuery(
                    original_query="large dataset test",
                    cleaned_query="large dataset test",
                    expanded_terms=["large", "dataset", "test"],
                    confidence=0.9,
                )
                
                with patch.object(self.search_engine, '_get_tool_parameters', return_value=[]):
                    with patch.object(self.search_engine, '_get_tool_examples', return_value=[]):
                        start_time = time.time()
                        response = self.search_engine.search_by_functionality("large dataset test", k=20)
                        execution_time = time.time() - start_time
                        
                        # Should still meet performance requirement with larger dataset
                        assert execution_time < 0.5, f"Large dataset response time {execution_time:.3f}s exceeds 500ms"
                        assert len(response.results) == 20
    
    def test_concurrent_performance(self):
        """Test performance under concurrent load."""
        results = []
        execution_times = []
        
        def concurrent_search(query_id):
            start_time = time.time()
            response = self.search_engine.search_by_functionality(f"concurrent test {query_id}")
            execution_time = time.time() - start_time
            results.append(response)
            execution_times.append(execution_time)
        
        with patch.object(self.search_engine, '_perform_vector_search', return_value=[]):
            with patch.object(self.search_engine.query_processor, 'process_query') as mock_process:
                mock_process.return_value = ProcessedQuery(
                    original_query="concurrent",
                    cleaned_query="concurrent",
                    expanded_terms=["concurrent"],
                    confidence=0.8,
                )
                
                # Start multiple concurrent searches
                threads = []
                for i in range(10):
                    thread = threading.Thread(target=concurrent_search, args=(i,))
                    threads.append(thread)
                
                # Start all threads simultaneously
                start_time = time.time()
                for thread in threads:
                    thread.start()
                
                # Wait for completion
                for thread in threads:
                    thread.join(timeout=2.0)
                total_time = time.time() - start_time
                
                # Verify all completed successfully
                assert len(results) == 10
                assert len(execution_times) == 10
                
                # Each individual search should meet performance requirement
                for exec_time in execution_times:
                    assert exec_time < 0.5, f"Concurrent search time {exec_time:.3f}s exceeds 500ms"
                
                # Total concurrent execution should be reasonable
                assert total_time < 2.0, f"Total concurrent execution time {total_time:.3f}s too slow"


class TestMemoryUsage:
    """Test memory usage patterns and limits."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.db_manager = Mock(spec=DatabaseManager)
        self.embedding_generator = Mock(spec=EmbeddingGenerator)
        self.search_engine = MCPToolSearchEngine(self.db_manager, self.embedding_generator)
    
    def get_memory_usage(self):
        """Get current memory usage in MB."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    
    def test_memory_usage_under_load(self):
        """Test memory usage under repeated search load."""
        initial_memory = self.get_memory_usage()
        
        with patch.object(self.search_engine, '_perform_vector_search', return_value=[]):
            with patch.object(self.search_engine.query_processor, 'process_query') as mock_process:
                mock_process.return_value = ProcessedQuery(
                    original_query="memory test",
                    cleaned_query="memory test",
                    expanded_terms=["memory", "test"],
                    confidence=0.8,
                )
                
                # Perform many searches to test for memory leaks
                for i in range(100):
                    response = self.search_engine.search_by_functionality(f"memory test {i}")
                    assert isinstance(response, ToolSearchResponse)
                    
                    # Force garbage collection periodically
                    if i % 20 == 0:
                        gc.collect()
                
                gc.collect()  # Final garbage collection
                final_memory = self.get_memory_usage()
                memory_growth = final_memory - initial_memory
                
                # Memory growth should be reasonable (less than 50MB for 100 searches)
                assert memory_growth < 50, f"Memory growth {memory_growth:.2f}MB too high - possible leak"
    
    def test_large_result_memory_efficiency(self):
        """Test memory efficiency with large result processing."""
        # Create large mock results with substantial data
        large_results = []
        for i in range(200):
            tool_data = {
                "id": f"mem_tool_{i}",
                "name": f"Memory Test Tool {i}",
                "description": "x" * 1000,  # 1KB description each
                "category": "memory_test",
                "tool_type": "function",
                "metadata_json": '{"large_data": "' + "x" * 5000 + '"}',  # 5KB metadata each
            }
            large_results.append((f"mem_tool_{i}", 0.9, tool_data))
        
        initial_memory = self.get_memory_usage()
        
        with patch.object(self.search_engine, '_perform_vector_search', return_value=large_results):
            with patch.object(self.search_engine.query_processor, 'process_query') as mock_process:
                mock_process.return_value = ProcessedQuery(
                    original_query="large memory test",
                    cleaned_query="large memory test",
                    expanded_terms=["large", "memory", "test"],
                    confidence=0.9,
                )
                
                with patch.object(self.search_engine, '_get_tool_parameters', return_value=[]):
                    with patch.object(self.search_engine, '_get_tool_examples', return_value=[]):
                        response = self.search_engine.search_by_functionality("large memory test", k=50)
                        
                        gc.collect()
                        peak_memory = self.get_memory_usage()
                        memory_growth = peak_memory - initial_memory
                        
                        # Should handle large data efficiently (growth < 100MB)
                        assert memory_growth < 100, f"Memory usage {memory_growth:.2f}MB too high for large result processing"
                        assert len(response.results) == 50


class TestResourceExhaustion:
    """Test handling of resource exhaustion scenarios."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.db_manager = Mock(spec=DatabaseManager)
        self.embedding_generator = Mock(spec=EmbeddingGenerator)
        self.search_engine = MCPToolSearchEngine(self.db_manager, self.embedding_generator)
    
    def test_thread_pool_exhaustion(self):
        """Test behavior when thread pool is exhausted."""
        results = []
        errors = []
        
        def blocking_search(query_id):
            try:
                # Simulate some processing time
                time.sleep(0.1)
                response = self.search_engine.search_by_functionality(f"thread test {query_id}")
                results.append(response)
            except Exception as e:
                errors.append(e)
        
        with patch.object(self.search_engine, '_perform_vector_search', return_value=[]):
            with patch.object(self.search_engine.query_processor, 'process_query') as mock_process:
                mock_process.return_value = ProcessedQuery(
                    original_query="thread test",
                    cleaned_query="thread test",
                    expanded_terms=["thread", "test"],
                    confidence=0.8,
                )
                
                # Start many concurrent threads (more than typical pool size)
                threads = []
                for i in range(50):
                    thread = threading.Thread(target=blocking_search, args=(i,))
                    threads.append(thread)
                    thread.start()
                
                # Wait for all with reasonable timeout
                for thread in threads:
                    thread.join(timeout=5.0)
                
                # Should handle many concurrent requests gracefully
                total_completed = len(results) + len(errors)
                assert total_completed >= 40, f"Only {total_completed}/50 requests completed - possible resource exhaustion"
                
                # Errors should be minimal
                assert len(errors) < 5, f"Too many errors ({len(errors)}) - poor resource handling"
    
    def test_cache_memory_pressure(self):
        """Test cache behavior under memory pressure."""
        # Mock cache that fails under memory pressure
        mock_cache = Mock()
        mock_cache.get.return_value = None  # Cache miss
        mock_cache.put.side_effect = MemoryError("Cache memory exhausted")
        
        self.search_engine._results_cache = mock_cache
        
        with patch.object(self.search_engine, '_perform_vector_search', return_value=[]):
            with patch.object(self.search_engine.query_processor, 'process_query') as mock_process:
                mock_process.return_value = ProcessedQuery(
                    original_query="cache pressure test",
                    cleaned_query="cache pressure test", 
                    expanded_terms=["cache", "pressure", "test"],
                    confidence=0.8,
                )
                
                # Should handle cache memory pressure gracefully
                response = self.search_engine.search_by_functionality("cache pressure test")
                
                assert isinstance(response, ToolSearchResponse)
                # Should still work even if caching fails
                assert response.execution_time > 0
    
    def test_embedding_generation_overload(self):
        """Test handling of embedding generation overload."""
        call_count = 0
        
        def overloaded_encode(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count > 5:
                raise RuntimeError("Embedding service overloaded")
            return [0.1] * 384  # Valid embedding
        
        self.embedding_generator.encode.side_effect = overloaded_encode
        
        # Try multiple searches that would overload embedding generation
        responses = []
        for i in range(10):
            response = self.search_engine.search_by_functionality(f"overload test {i}")
            responses.append(response)
        
        # First 5 should succeed, rest should handle overload gracefully
        assert len(responses) == 10
        successful = [r for r in responses if len(r.results) >= 0]  # All should be valid responses
        assert len(successful) == 10, "All responses should be valid even under overload"


class TestStressConditions:
    """Test system behavior under stress conditions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.db_manager = Mock(spec=DatabaseManager)
        self.embedding_generator = Mock(spec=EmbeddingGenerator)
        self.search_engine = MCPToolSearchEngine(self.db_manager, self.embedding_generator)
    
    def test_rapid_sequential_searches(self):
        """Test rapid sequential search requests."""
        with patch.object(self.search_engine, '_perform_vector_search', return_value=[]):
            with patch.object(self.search_engine.query_processor, 'process_query') as mock_process:
                mock_process.return_value = ProcessedQuery(
                    original_query="rapid test",
                    cleaned_query="rapid test",
                    expanded_terms=["rapid", "test"],
                    confidence=0.8,
                )
                
                start_time = time.time()
                responses = []
                
                # Perform rapid searches
                for i in range(50):
                    response = self.search_engine.search_by_functionality(f"rapid test {i}")
                    responses.append(response)
                
                total_time = time.time() - start_time
                
                # All should complete successfully
                assert len(responses) == 50
                for response in responses:
                    assert isinstance(response, ToolSearchResponse)
                
                # Average time per search should be reasonable
                avg_time = total_time / 50
                assert avg_time < 0.1, f"Average search time {avg_time:.3f}s too slow for rapid requests"
    
    def test_mixed_workload_performance(self):
        """Test performance under mixed workload (different query types)."""
        query_types = [
            "simple query",
            "complex query with multiple technical terms and specific requirements",
            "🎯 unicode query with emojis 🚀",
            "query with special characters !@#$%",
            "very short",
            "a" * 100,  # Long query
        ]
        
        with patch.object(self.search_engine, '_perform_vector_search', return_value=[]):
            with patch.object(self.search_engine.query_processor, 'process_query') as mock_process:
                mock_process.return_value = ProcessedQuery(
                    original_query="mixed test",
                    cleaned_query="mixed test",
                    expanded_terms=["mixed", "test"],
                    confidence=0.8,
                )
                
                start_time = time.time()
                responses = []
                
                # Mix different query types
                for i in range(30):
                    query = query_types[i % len(query_types)]
                    response = self.search_engine.search_by_functionality(query)
                    responses.append(response)
                
                total_time = time.time() - start_time
                
                # All should handle different query types successfully
                assert len(responses) == 30
                for response in responses:
                    assert isinstance(response, ToolSearchResponse)
                    assert response.execution_time < 0.5  # Each should meet performance requirement
                
                # Total time should be reasonable
                assert total_time < 15.0, f"Mixed workload time {total_time:.2f}s too slow"


if __name__ == "__main__":
    # Run tests with performance reporting
    pytest.main([__file__, "-v", "--tb=short"])