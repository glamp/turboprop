import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class CacheEntry:
    """Cache entry with metadata"""

    response_data: Dict[str, Any]
    timestamp: float
    last_accessed: float
    access_count: int


class CacheStats:
    """Track cache performance statistics"""

    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.total_response_time_saved = 0.0
        self.start_time = time.time()

    def record_hit(self, response_time_saved: float = 0.5):
        """Record a cache hit"""
        self.hits += 1
        self.total_response_time_saved += response_time_saved

    def record_miss(self):
        """Record a cache miss"""
        self.misses += 1

    def hit_rate(self) -> float:
        """Calculate cache hit rate"""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def total_requests(self) -> int:
        """Get total cache requests"""
        return self.hits + self.misses

    def avg_time_saved(self) -> float:
        """Calculate average response time saved per hit"""
        return self.total_response_time_saved / self.hits if self.hits > 0 else 0.0


class ToolSearchResponseCache:
    """Intelligent caching system for tool search responses"""

    def __init__(self, cache_size: int = 1000, cache_ttl: int = 3600):  # 1 hour TTL
        self.cache_size = cache_size
        self.cache_ttl = cache_ttl
        self.cache_data: Dict[str, CacheEntry] = {}
        self.cache_stats = CacheStats()
        self.cache_by_tool: Dict[str, set] = defaultdict(set)

    def get(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get response from cache if valid"""
        if cache_key not in self.cache_data:
            self.cache_stats.record_miss()
            return None

        entry = self.cache_data[cache_key]

        # Check if expired
        if time.time() - entry.timestamp > self.cache_ttl:
            self._remove_entry(cache_key)
            self.cache_stats.record_miss()
            return None

        # Update access time and return data
        entry.last_accessed = time.time()
        entry.access_count += 1

        # Estimate time saved (average response time for tool operations)
        estimated_time_saved = self._estimate_response_time_saved(entry.response_data)
        self.cache_stats.record_hit(estimated_time_saved)

        return entry.response_data.copy()

    def set(self, cache_key: str, response_data: Dict[str, Any]) -> None:
        """Store response in cache"""

        # Ensure cache size limits
        if len(self.cache_data) >= self.cache_size:
            self._evict_lru_entries()

        # Track by tool for targeted invalidation
        tool_name = response_data.get("tool", "unknown")
        self.cache_by_tool[tool_name].add(cache_key)

        # Store entry
        self.cache_data[cache_key] = CacheEntry(
            response_data=response_data.copy(), timestamp=time.time(), last_accessed=time.time(), access_count=1
        )

    def invalidate_tool_cache(self, tool_name: str) -> int:
        """Invalidate all cache entries for a specific tool"""
        invalidated_count = 0

        if tool_name in self.cache_by_tool:
            cache_keys = list(self.cache_by_tool[tool_name])

            for cache_key in cache_keys:
                if cache_key in self.cache_data:
                    del self.cache_data[cache_key]
                    invalidated_count += 1

            del self.cache_by_tool[tool_name]

        return invalidated_count

    def clear_cache(self) -> None:
        """Clear all cache entries"""
        self.cache_data.clear()
        self.cache_by_tool.clear()

    def _remove_entry(self, cache_key: str) -> None:
        """Remove single cache entry and update tool tracking"""
        if cache_key in self.cache_data:
            entry = self.cache_data[cache_key]
            tool_name = entry.response_data.get("tool", "unknown")

            # Remove from cache data
            del self.cache_data[cache_key]

            # Remove from tool tracking
            if tool_name in self.cache_by_tool:
                self.cache_by_tool[tool_name].discard(cache_key)
                if not self.cache_by_tool[tool_name]:
                    del self.cache_by_tool[tool_name]

    def _evict_lru_entries(self) -> None:
        """Evict least recently used entries"""
        # Sort by last accessed time and remove oldest 20%
        sorted_entries = sorted(self.cache_data.items(), key=lambda x: x[1].last_accessed)

        evict_count = max(1, len(sorted_entries) // 5)
        for cache_key, _ in sorted_entries[:evict_count]:
            self._remove_entry(cache_key)

    def _estimate_response_time_saved(self, response_data: Dict[str, Any]) -> float:
        """Estimate response time saved by cache hit"""
        tool_name = response_data.get("tool", "")

        # Base estimates for different tool types
        time_estimates = {
            "search_mcp_tools": 1.0,  # Search operations
            "get_tool_details": 0.5,  # Detail retrieval
            "recommend_tools_for_task": 1.5,  # AI processing
            "compare_mcp_tools": 2.0,  # Complex comparison
            "analyze_task_requirements": 1.0,  # Analysis operations
        }

        base_time = time_estimates.get(tool_name, 0.8)

        # Adjust based on result complexity
        result_count = self._count_cached_results(response_data)
        if result_count > 10:
            base_time *= 1.2
        elif result_count > 20:
            base_time *= 1.5

        return base_time

    def _count_cached_results(self, response_data: Dict[str, Any]) -> int:
        """Count results in cached response"""
        if "results" in response_data:
            return len(response_data["results"])
        elif "recommendations" in response_data:
            return len(response_data["recommendations"])
        elif "comparison_result" in response_data:
            comparison = response_data["comparison_result"]
            if isinstance(comparison, dict) and "tools" in comparison:
                return len(comparison["tools"])
        return 0

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        return {
            "hit_rate": self.cache_stats.hit_rate(),
            "total_requests": self.cache_stats.total_requests(),
            "cache_size": len(self.cache_data),
            "cache_capacity": self.cache_size,
            "avg_response_time_saved": self.cache_stats.avg_time_saved(),
            "total_time_saved": self.cache_stats.total_response_time_saved,
            "cache_by_tool_counts": {tool: len(keys) for tool, keys in self.cache_by_tool.items()},
            "uptime_hours": (time.time() - self.cache_stats.start_time) / 3600,
        }

    def get_cache_health(self) -> Dict[str, Any]:
        """Get cache health metrics"""
        now = time.time()

        # Analyze cache age distribution
        ages = [now - entry.timestamp for entry in self.cache_data.values()]

        health_metrics = {
            "status": "healthy",
            "total_entries": len(self.cache_data),
            "capacity_usage": len(self.cache_data) / self.cache_size,
            "average_entry_age_minutes": sum(ages) / len(ages) / 60 if ages else 0,
            "expired_entries": sum(1 for age in ages if age > self.cache_ttl),
            "recommendations": [],
        }

        # Add health recommendations
        if health_metrics["capacity_usage"] > 0.9:
            health_metrics["status"] = "warning"
            health_metrics["recommendations"].append("Cache near capacity - consider increasing cache_size")

        if health_metrics["expired_entries"] > len(ages) * 0.3:
            health_metrics["recommendations"].append("Many expired entries - consider running cleanup")

        if self.cache_stats.hit_rate() < 0.3:
            health_metrics["recommendations"].append("Low hit rate - review caching strategy or increase TTL")

        return health_metrics

    def cleanup_expired_entries(self) -> int:
        """Remove all expired entries and return count"""
        now = time.time()
        expired_keys = [
            cache_key for cache_key, entry in self.cache_data.items() if now - entry.timestamp > self.cache_ttl
        ]

        for cache_key in expired_keys:
            self._remove_entry(cache_key)

        return len(expired_keys)

    def get_top_accessed_entries(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get most frequently accessed cache entries"""
        sorted_entries = sorted(self.cache_data.items(), key=lambda x: x[1].access_count, reverse=True)

        return [
            {
                "cache_key": cache_key[:16] + "...",  # Truncate for readability
                "tool": entry.response_data.get("tool", "unknown"),
                "access_count": entry.access_count,
                "age_minutes": (time.time() - entry.timestamp) / 60,
            }
            for cache_key, entry in sorted_entries[:limit]
        ]
