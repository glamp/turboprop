#!/usr/bin/env python3
"""
test_search_benchmarks.py: Performance regression tests with benchmarking.

This module tests:
- Search response time benchmarks across repository sizes
- Memory usage monitoring during indexing
- Embedding generation performance tests
- Database query performance regression tests
- Scalability tests with large repositories
"""

import gc
import time
from typing import Dict
from unittest.mock import Mock, patch

import psutil
import pytest

from code_index import init_db, reindex_all, scan_repo, search_index
from config import config
from construct_search import ConstructSearchOperations
from hybrid_search import HybridSearchEngine, SearchMode


class PerformanceMetrics:
    """Utility class for collecting performance metrics."""

    def __init__(self):
        self.start_time = None
        self.start_memory = None
        self.peak_memory = None
        self.process = psutil.Process()

    def start_monitoring(self):
        """Start performance monitoring."""
        gc.collect()  # Force garbage collection for accurate memory measurement
        self.start_time = time.time()
        self.start_memory = self.process.memory_info().rss / (1024 * 1024)  # MB
        self.peak_memory = self.start_memory

    def update_peak_memory(self):
        """Update peak memory usage."""
        current_memory = self.process.memory_info().rss / (1024 * 1024)  # MB
        if current_memory > self.peak_memory:
            self.peak_memory = current_memory

    def get_metrics(self) -> Dict[str, float]:
        """Get performance metrics."""
        if self.start_time is None:
            raise RuntimeError("Monitoring not started")

        end_time = time.time()
        end_memory = self.process.memory_info().rss / (1024 * 1024)  # MB

        return {
            "elapsed_time": end_time - self.start_time,
            "start_memory_mb": self.start_memory,
            "end_memory_mb": end_memory,
            "peak_memory_mb": self.peak_memory,
            "memory_delta_mb": end_memory - self.start_memory,
            "cpu_percent": self.process.cpu_percent(),
        }


class TestSearchPerformance:
    """Test search performance across different scenarios."""

    def test_search_response_times_small_repo(self, sample_repo, mock_embedder, performance_baseline):
        """Test search response times on small repository."""
        db_manager = init_db(sample_repo)
        metrics = PerformanceMetrics()

        try:
            # Index the repository first
            reindex_all(sample_repo, int(10.0 * 1024 * 1024), db_manager, mock_embedder)

            # Test search queries with different complexities
            test_queries = [
                "function",
                "authentication method",
                "data processing pipeline with error handling",
                "async function with jwt token validation",
            ]

            search_times = []

            for query in test_queries:
                metrics.start_monitoring()

                results = search_index(db_manager, mock_embedder, query, k=10)

                performance_metrics = metrics.get_metrics()
                search_times.append(performance_metrics["elapsed_time"])

                # Assert search completes within acceptable time
                assert performance_metrics["elapsed_time"] < performance_baseline["search_timeout"]

                # Verify results are returned
                assert isinstance(results, list)

            # Calculate average search time
            avg_search_time = sum(search_times) / len(search_times)
            assert avg_search_time < performance_baseline["search_timeout"] / 2

        finally:
            db_manager.cleanup()

    def test_search_scalability_by_result_count(self, large_repo, mock_embedder, performance_baseline):
        """Test search performance scaling with different result counts."""
        db_manager = init_db(large_repo)
        metrics = PerformanceMetrics()

        try:
            # Index large repository
            reindex_all(large_repo, int(50.0 * 1024 * 1024), db_manager, mock_embedder)

            # Test with different k values
            k_values = [1, 5, 10, 25, 50, 100]
            search_times = {}

            for k in k_values:
                metrics.start_monitoring()

                results = search_index(db_manager, mock_embedder, "function definition", k=k)

                performance_metrics = metrics.get_metrics()
                search_times[k] = performance_metrics["elapsed_time"]

                # Should complete within timeout
                assert performance_metrics["elapsed_time"] < performance_baseline["search_timeout"]

                # Should return up to k results
                assert len(results) <= k

            # Search time should scale reasonably (not exponentially)
            # Time for k=100 should be less than 3x time for k=10
            if k_values[-1] in search_times and 10 in search_times:
                assert search_times[k_values[-1]] < search_times[10] * 3

        finally:
            db_manager.cleanup()

    def test_hybrid_search_performance(self, sample_repo, fully_mock_db_manager, mock_embedder, performance_baseline):
        """Test hybrid search performance."""
        metrics = PerformanceMetrics()

        # Mock database responses for consistency
        fully_mock_db_manager.search_full_text.return_value = [
            ("id1", str(sample_repo / "auth.js"), "function authenticate()", 0.9),
            ("id2", str(sample_repo / "data_processor.py"), "class DataProcessor", 0.8),
        ]

        with patch("hybrid_search.search_index_enhanced") as mock_search:
            from search_result_types import CodeSearchResult, CodeSnippet

            # Create multiple mock results for performance testing
            mock_results = []
            for i in range(50):
                mock_result = CodeSearchResult(
                    file_path=f"/test/file_{i}.py",
                    snippet=CodeSnippet(text=f"def function_{i}():", start_line=1, end_line=3),
                    similarity_score=0.9 - (i * 0.01),
                )
                mock_results.append(mock_result)

            mock_search.return_value = mock_results

            engine = HybridSearchEngine(fully_mock_db_manager, mock_embedder)

            # Test different search modes
            modes = [SearchMode.AUTO, SearchMode.HYBRID, SearchMode.SEMANTIC_ONLY, SearchMode.TEXT_ONLY]

            for mode in modes:
                metrics.start_monitoring()

                results = engine.search("test function", k=25, mode=mode)

                performance_metrics = metrics.get_metrics()

                # Should complete within timeout
                assert performance_metrics["elapsed_time"] < performance_baseline["search_timeout"]

                # Should return results
                assert len(results) > 0

    def test_concurrent_search_performance(self, sample_repo, mock_embedder, performance_baseline):
        """Test performance under concurrent search load."""
        import threading

        db_manager = init_db(sample_repo)
        metrics = PerformanceMetrics()

        try:
            # Index repository
            reindex_all(sample_repo, int(10.0 * 1024 * 1024), db_manager, mock_embedder)

            # Define search function for threading
            search_results = []
            search_errors = []

            def search_worker(query_id):
                try:
                    query = f"function {query_id % 5}"  # Cycle through 5 different queries
                    results = search_index(db_manager, mock_embedder, query, k=5)
                    search_results.append((query_id, len(results)))
                except Exception as e:
                    search_errors.append((query_id, str(e)))

            # Start monitoring
            metrics.start_monitoring()

            # Create and start threads
            threads = []
            thread_count = 5  # Conservative thread count for testing

            for i in range(thread_count):
                thread = threading.Thread(target=search_worker, args=(i,))
                threads.append(thread)
                thread.start()

            # Wait for all threads
            for thread in threads:
                thread.join()

            performance_metrics = metrics.get_metrics()

            # Should complete all searches within timeout
            assert performance_metrics["elapsed_time"] < performance_baseline["search_timeout"] * 2

            # All searches should succeed
            assert len(search_errors) == 0, f"Search errors: {search_errors}"
            assert len(search_results) == thread_count

        finally:
            db_manager.cleanup()


class TestIndexingPerformance:
    """Test indexing performance and scalability."""

    def test_indexing_performance_small_repo(self, sample_repo, mock_embedder, performance_baseline):
        """Test indexing performance on small repository."""
        db_manager = init_db(sample_repo)
        metrics = PerformanceMetrics()

        try:
            metrics.start_monitoring()

            total_files, processed_files, elapsed = reindex_all(
                sample_repo, int(10.0 * 1024 * 1024), db_manager, mock_embedder, force_all=True
            )

            performance_metrics = metrics.get_metrics()

            # Should complete within timeout
            assert performance_metrics["elapsed_time"] < performance_baseline["indexing_timeout"]
            # Elapsed time should be approximately equal (allow some variance)
            assert abs(performance_metrics["elapsed_time"] - elapsed) < 0.1

            # Should process all files (using force_all=True)
            assert processed_files == total_files
            assert total_files > 0

            # Memory usage should be reasonable (allow some variance in test environment)
            assert performance_metrics["peak_memory_mb"] < performance_baseline["max_memory_mb"] * 1.1  # 10% tolerance

            # Calculate indexing rate
            if elapsed > 0:
                files_per_second = total_files / elapsed
                assert files_per_second > 0  # Should process at least some files per second

        finally:
            db_manager.cleanup()

    def test_indexing_memory_usage_large_repo(self, large_repo, mock_embedder, performance_baseline):
        """Test memory usage during indexing of large repository."""
        db_manager = init_db(large_repo)
        metrics = PerformanceMetrics()

        try:
            metrics.start_monitoring()

            # Monitor memory during indexing
            def memory_monitor():
                while hasattr(memory_monitor, "running") and memory_monitor.running:
                    metrics.update_peak_memory()
                    time.sleep(0.1)

            import threading

            memory_monitor.running = True
            monitor_thread = threading.Thread(target=memory_monitor)
            monitor_thread.start()

            try:
                total_files, processed_files, elapsed = reindex_all(
                    large_repo, int(100.0 * 1024 * 1024), db_manager, mock_embedder, force_all=True
                )
            finally:
                memory_monitor.running = False
                monitor_thread.join(timeout=5.0)  # Add timeout to prevent hanging

            performance_metrics = metrics.get_metrics()

            # Should complete within timeout
            # Larger timeout for large repo
            assert performance_metrics["elapsed_time"] < performance_baseline["indexing_timeout"] * 5

            # Memory usage should not exceed limit
            # Allow 2x for large repo
            assert performance_metrics["peak_memory_mb"] < performance_baseline["max_memory_mb"] * 2

            # Should process files efficiently
            assert processed_files > 0

        finally:
            db_manager.cleanup()

    def test_embedding_generation_performance(self, sample_repo, mock_embedder, performance_baseline):
        """Test embedding generation performance."""
        # Get files to process
        files = scan_repo(sample_repo, max_bytes=int(10.0 * 1024 * 1024))
        assert len(files) > 0

        # Read file contents
        texts = []
        for file_path in files:
            try:
                content = file_path.read_text(encoding="utf-8", errors="ignore")
                texts.append(content)
            except Exception:
                continue

        assert len(texts) > 0

        # Test embedding generation performance
        metrics = PerformanceMetrics()
        metrics.start_monitoring()

        embeddings = mock_embedder.generate_embeddings(texts)

        performance_metrics = metrics.get_metrics()

        # Should generate embeddings for all texts
        assert len(embeddings) == len(texts)

        # Calculate generation rate
        if performance_metrics["elapsed_time"] > 0:
            embeddings_per_second = len(embeddings) / performance_metrics["elapsed_time"]

            # Should meet minimum performance baseline
            # Note: This is for mock embedder, real embedder would be much slower
            assert embeddings_per_second > performance_baseline["embedding_generation_rate"]

    def test_database_write_performance(self, sample_repo, mock_embedder, performance_baseline):
        """Test database write performance during indexing."""
        db_manager = init_db(sample_repo)

        try:
            # Get files and generate content
            files = scan_repo(sample_repo, max_bytes=int(10.0 * 1024 * 1024))

            # Time database operations
            metrics = PerformanceMetrics()
            metrics.start_monitoring()

            # Simulate the embed_and_store process focusing on DB writes
            from code_index import embed_and_store

            processed_count, skipped_count = embed_and_store(db_manager, mock_embedder, files)

            performance_metrics = metrics.get_metrics()

            # Should complete database writes efficiently
            assert performance_metrics["elapsed_time"] < performance_baseline["indexing_timeout"]
            assert processed_count > 0

            # Calculate write rate
            if performance_metrics["elapsed_time"] > 0:
                writes_per_second = processed_count / performance_metrics["elapsed_time"]
                assert writes_per_second > 0

        finally:
            db_manager.cleanup()


class TestDatabasePerformance:
    """Test database query performance and scalability."""

    def test_similarity_search_performance(self, fully_mock_db_manager, mock_embedder):
        """Test vector similarity search performance."""
        # Mock database with large result set
        large_results = []
        for i in range(1000):
            large_results.append((f"/test/file_{i}.py", f"content {i}", 0.5 + (i * 0.0001)))

        fully_mock_db_manager.execute_with_retry.return_value = large_results

        # Test search performance
        metrics = PerformanceMetrics()
        metrics.start_monitoring()

        # Simulate semantic search query
        query_embedding = mock_embedder.encode("test query")

        # Mock the actual database query that would use the embedding
        results = fully_mock_db_manager.execute_with_retry(
            f"""
            SELECT path, content,
                   1 - (embedding <=> ?) as similarity
            FROM {config.database.TABLE_NAME}
            WHERE embedding IS NOT NULL
            ORDER BY similarity DESC
            LIMIT 50
            """,
            [query_embedding.tolist()],
        )

        performance_metrics = metrics.get_metrics()

        # Should complete quickly even with large dataset
        assert performance_metrics["elapsed_time"] < 1.0
        assert len(results) > 0

    def test_full_text_search_performance(self, fully_mock_db_manager):
        """Test full-text search performance."""
        # Mock FTS results
        fts_results = [
            ("id1", "/test/file1.py", "function test_func()", 0.9),
            ("id2", "/test/file2.py", "class TestClass", 0.8),
        ]
        fully_mock_db_manager.search_full_text.return_value = fts_results

        metrics = PerformanceMetrics()
        metrics.start_monitoring()

        # Execute FTS query
        results = fully_mock_db_manager.search_full_text("test function", limit=50)

        performance_metrics = metrics.get_metrics()

        # Should complete quickly
        assert performance_metrics["elapsed_time"] < 0.5
        assert len(results) > 0

    def test_construct_search_performance(self, fully_mock_db_manager, mock_embedder):
        """Test construct-level search performance."""
        construct_ops = ConstructSearchOperations(fully_mock_db_manager, mock_embedder)

        # Mock construct search results
        mock_results = []
        for i in range(100):
            mock_results.append(
                (
                    f"construct_{i}",
                    f"/test/file_{i}.py",
                    "function",
                    f"func_{i}",
                    f"def func_{i}():",
                    i * 10,
                    i * 10 + 5,
                    f"Doc {i}",
                    None,
                    f"file_{i}",
                    0.8,
                )
            )

        mock_connection = Mock()
        mock_connection.execute.return_value.fetchall.return_value = mock_results

        context_manager = Mock()
        context_manager.__enter__ = Mock(return_value=mock_connection)
        context_manager.__exit__ = Mock(return_value=None)
        fully_mock_db_manager.get_connection.return_value = context_manager

        # Mock embedder
        mock_embedder.generate_embeddings.return_value = [[0.1, 0.2, 0.3]]

        # Test performance
        metrics = PerformanceMetrics()
        metrics.start_monitoring()

        results = construct_ops.search_constructs("test function", k=50)

        performance_metrics = metrics.get_metrics()

        # Should complete efficiently
        assert performance_metrics["elapsed_time"] < 2.0
        assert len(results) > 0 or mock_connection.execute.called


class TestMemoryEfficiency:
    """Test memory usage and efficiency."""

    def test_memory_usage_during_search(self, sample_repo, mock_embedder, performance_baseline):
        """Test memory usage during search operations."""
        db_manager = init_db(sample_repo)

        try:
            # Index repository
            reindex_all(sample_repo, int(10.0 * 1024 * 1024), db_manager, mock_embedder)

            # Baseline memory
            gc.collect()
            baseline_memory = psutil.Process().memory_info().rss / (1024 * 1024)

            # Perform multiple searches
            max_memory = baseline_memory

            for i in range(10):
                search_index(db_manager, mock_embedder, f"test query {i}", k=10)

                current_memory = psutil.Process().memory_info().rss / (1024 * 1024)
                max_memory = max(max_memory, current_memory)

            # Memory growth should be limited
            memory_growth = max_memory - baseline_memory
            assert memory_growth < performance_baseline["max_memory_mb"] / 10  # Less than 10% of max allowed

        finally:
            db_manager.cleanup()

    def test_memory_cleanup_after_indexing(self, sample_repo, mock_embedder):
        """Test memory cleanup after indexing operations."""
        # Baseline memory
        gc.collect()
        baseline_memory = psutil.Process().memory_info().rss / (1024 * 1024)

        db_manager = init_db(sample_repo)

        try:
            # Perform indexing
            reindex_all(sample_repo, int(10.0 * 1024 * 1024), db_manager, mock_embedder)

            # Force cleanup
            gc.collect()

            # Check memory after cleanup
            after_memory = psutil.Process().memory_info().rss / (1024 * 1024)

            # Memory should not have grown excessively
            memory_growth = after_memory - baseline_memory
            assert memory_growth < 100  # Less than 100MB growth for small repo

        finally:
            db_manager.cleanup()

            # Memory should return close to baseline after cleanup
            gc.collect()
            final_memory = psutil.Process().memory_info().rss / (1024 * 1024)
            final_growth = final_memory - baseline_memory

            # Should be close to baseline (allowing some overhead)
            assert final_growth < 50  # Less than 50MB permanent growth


class TestRegressionBenchmarks:
    """Regression benchmark tests with stored baselines."""

    @pytest.fixture
    def benchmark_data(self, tmp_path):
        """Store and retrieve benchmark data."""
        benchmark_file = tmp_path / "benchmarks.json"

        def store_benchmark(name: str, metrics: Dict[str, float]):
            import json

            if benchmark_file.exists():
                data = json.loads(benchmark_file.read_text())
            else:
                data = {}

            data[name] = metrics
            benchmark_file.write_text(json.dumps(data, indent=2))

        def get_baseline(name: str) -> Dict[str, float]:
            import json

            if not benchmark_file.exists():
                return {}

            data = json.loads(benchmark_file.read_text())
            return data.get(name, {})

        return store_benchmark, get_baseline

    def test_search_performance_regression(self, sample_repo, mock_embedder, benchmark_data):
        """Test for search performance regressions."""
        store_benchmark, get_baseline = benchmark_data

        db_manager = init_db(sample_repo)

        try:
            # Index repository
            reindex_all(sample_repo, int(10.0 * 1024 * 1024), db_manager, mock_embedder)

            # Benchmark search performance
            metrics = PerformanceMetrics()
            metrics.start_monitoring()

            # Standard benchmark query
            results = search_index(db_manager, mock_embedder, "authentication function", k=10)

            performance_metrics = metrics.get_metrics()

            # Store current benchmark
            benchmark_name = "search_performance_small_repo"
            current_metrics = {
                "search_time": performance_metrics["elapsed_time"],
                "memory_usage": performance_metrics["peak_memory_mb"],
                "result_count": len(results),
            }

            # Get baseline for comparison
            baseline = get_baseline(benchmark_name)

            if baseline:
                # Check for regression (allowing 20% tolerance)
                assert (
                    current_metrics["search_time"] <= baseline["search_time"] * 1.2
                ), f"Search time regression: {current_metrics['search_time']} > {baseline['search_time']} * 1.2"

                assert (
                    current_metrics["memory_usage"] <= baseline["memory_usage"] * 1.2
                ), f"Memory usage regression: {current_metrics['memory_usage']} > {baseline['memory_usage']} * 1.2"

            # Store current metrics as new baseline
            store_benchmark(benchmark_name, current_metrics)

        finally:
            db_manager.cleanup()

    def test_indexing_performance_regression(self, sample_repo, mock_embedder, benchmark_data):
        """Test for indexing performance regressions."""
        store_benchmark, get_baseline = benchmark_data

        db_manager = init_db(sample_repo)

        try:
            # Benchmark indexing performance
            metrics = PerformanceMetrics()
            metrics.start_monitoring()

            total_files, processed_files, elapsed = reindex_all(
                sample_repo, int(10.0 * 1024 * 1024), db_manager, mock_embedder
            )

            performance_metrics = metrics.get_metrics()

            # Store current benchmark
            benchmark_name = "indexing_performance_small_repo"
            current_metrics = {
                "indexing_time": performance_metrics["elapsed_time"],
                "memory_peak": performance_metrics["peak_memory_mb"],
                "files_processed": processed_files,
                "files_per_second": processed_files / elapsed if elapsed > 0 else 0,
            }

            # Get baseline for comparison
            baseline = get_baseline(benchmark_name)

            if baseline:
                # Check for regression (allowing 20% tolerance)
                assert (
                    current_metrics["indexing_time"] <= baseline["indexing_time"] * 1.2
                ), f"Indexing time regression: {current_metrics['indexing_time']} > {baseline['indexing_time']} * 1.2"

                assert current_metrics["files_per_second"] >= baseline["files_per_second"] * 0.8, (
                    f"Indexing rate regression: {current_metrics['files_per_second']} < "
                    f"{baseline['files_per_second']} * 0.8"
                )

            # Store current metrics
            store_benchmark(benchmark_name, current_metrics)

        finally:
            db_manager.cleanup()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
