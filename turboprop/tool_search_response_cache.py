import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .mcp_response_config import (
    CACHE_CAPACITY_WARNING_THRESHOLD,
    CACHE_EVICTION_PERCENTAGE,
    CACHE_EXPIRED_ENTRIES_WARNING_RATIO,
    CACHE_HIT_RATE_WARNING_THRESHOLD,
    CACHE_KEY_DISPLAY_LENGTH,
    DEFAULT_CACHE_SIZE,
    DEFAULT_CACHE_TTL,
    DEFAULT_TOOL_RESPONSE_TIME,
    RESULT_COMPLEXITY_MULTIPLIER_HIGH,
    RESULT_COMPLEXITY_MULTIPLIER_MEDIUM,
    TOOL_RESPONSE_TIME_ESTIMATES,
)

logger = logging.getLogger(__name__)


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

    def __init__(self, cache_size: int = DEFAULT_CACHE_SIZE, cache_ttl: int = DEFAULT_CACHE_TTL):
        self.cache_size = cache_size
        self.cache_ttl = cache_ttl
        self.cache_data: Dict[str, CacheEntry] = {}
        self.cache_stats = CacheStats()
        self.cache_by_tool: Dict[str, set] = defaultdict(set)

    def get(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get response from cache if valid"""
        try:
            if cache_key not in self.cache_data:
                self.cache_stats.record_miss()
                return None

            entry = self.cache_data[cache_key]

            # Validate cache entry integrity
            if not self._validate_cache_entry(entry):
                logger.warning(f"Corrupted cache entry detected for key: {cache_key[:CACHE_KEY_DISPLAY_LENGTH]}...")
                self._remove_entry(cache_key)
                self.cache_stats.record_miss()
                return None

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

            # Safely copy response data
            try:
                return entry.response_data.copy()
            except (AttributeError, TypeError) as e:
                logger.warning(f"Failed to copy cached response data: {e}")
                self._remove_entry(cache_key)
                self.cache_stats.record_miss()
                return None

        except (KeyError, AttributeError, TypeError) as e:
            logger.error(f"Cache corruption detected during get operation: {e}")
            self._handle_cache_corruption(cache_key)
            self.cache_stats.record_miss()
            return None

    def set(self, cache_key: str, response_data: Dict[str, Any]) -> None:
        """Store response in cache"""
        try:
            # Validate input data integrity before caching
            if not self._validate_response_data(response_data):
                logger.warning(f"Invalid response data provided for cache key: {cache_key[:16]}...")
                return

            # Validate that data is JSON serializable
            try:
                json.dumps(response_data)
            except (TypeError, ValueError, RecursionError) as e:
                logger.warning(f"Response data not JSON serializable, skipping cache: {e}")
                return

            # Ensure cache size limits
            if len(self.cache_data) >= self.cache_size:
                self._evict_lru_entries()

            # Track by tool for targeted invalidation
            tool_name = response_data.get("tool", "unknown")
            if not isinstance(tool_name, str):
                logger.warning(f"Invalid tool name type: {type(tool_name)}, using 'unknown'")
                tool_name = "unknown"

            try:
                self.cache_by_tool[tool_name].add(cache_key)
            except (TypeError, AttributeError) as e:
                logger.error(f"Failed to update tool tracking: {e}")
                # Recreate the tool tracking if corrupted
                self.cache_by_tool[tool_name] = {cache_key}

            # Store entry with safe data copy
            try:
                safe_data = response_data.copy()
                self.cache_data[cache_key] = CacheEntry(
                    response_data=safe_data, timestamp=time.time(), last_accessed=time.time(), access_count=1
                )
            except (TypeError, AttributeError) as e:
                logger.error(f"Failed to create cache entry: {e}")
                return

        except Exception as e:
            logger.error(f"Unexpected error during cache set operation: {e}")
            return

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
        try:
            if cache_key in self.cache_data:
                try:
                    entry = self.cache_data[cache_key]
                    tool_name = (
                        entry.response_data.get("tool", "unknown") if hasattr(entry, "response_data") else "unknown"
                    )
                except (AttributeError, TypeError) as e:
                    logger.warning(f"Corrupted cache entry detected during removal: {e}")
                    tool_name = "unknown"

                # Remove from cache data
                try:
                    del self.cache_data[cache_key]
                except KeyError:
                    logger.warning(f"Cache key {cache_key[:CACHE_KEY_DISPLAY_LENGTH]}... already removed")

                # Remove from tool tracking
                try:
                    if tool_name in self.cache_by_tool:
                        self.cache_by_tool[tool_name].discard(cache_key)
                        if not self.cache_by_tool[tool_name]:
                            del self.cache_by_tool[tool_name]
                except (TypeError, AttributeError) as e:
                    logger.warning(f"Error updating tool tracking during entry removal: {e}")
                    # Attempt to repair tool tracking
                    self._repair_tool_tracking()

        except Exception as e:
            logger.error(f"Failed to remove cache entry {cache_key[:CACHE_KEY_DISPLAY_LENGTH]}...: {e}")
            # Force removal if possible
            try:
                if cache_key in self.cache_data:
                    del self.cache_data[cache_key]
            except Exception:
                pass

    def _evict_lru_entries(self) -> None:
        """Evict least recently used entries"""
        # Sort by last accessed time and remove oldest entries
        sorted_entries = sorted(self.cache_data.items(), key=lambda x: x[1].last_accessed)

        evict_count = max(1, len(sorted_entries) * CACHE_EVICTION_PERCENTAGE // 100)
        for cache_key, _ in sorted_entries[:evict_count]:
            self._remove_entry(cache_key)

    def _estimate_response_time_saved(self, response_data: Dict[str, Any]) -> float:
        """Estimate response time saved by cache hit"""
        tool_name = response_data.get("tool", "")

        # Use configured time estimates
        base_time = TOOL_RESPONSE_TIME_ESTIMATES.get(tool_name, DEFAULT_TOOL_RESPONSE_TIME)

        # Adjust based on result complexity
        result_count = self._count_cached_results(response_data)
        if result_count > 10:
            base_time *= RESULT_COMPLEXITY_MULTIPLIER_MEDIUM
        elif result_count > 20:
            base_time *= RESULT_COMPLEXITY_MULTIPLIER_HIGH

        return base_time

    def _count_cached_results(self, response_data: Dict[str, Any]) -> int:
        """Count results in cached response"""
        try:
            if not isinstance(response_data, dict):
                return 0

            if "results" in response_data:
                results = response_data["results"]
                return len(results) if isinstance(results, list) else 0
            elif "recommendations" in response_data:
                recommendations = response_data["recommendations"]
                return len(recommendations) if isinstance(recommendations, list) else 0
            elif "comparison_result" in response_data:
                comparison = response_data["comparison_result"]
                if isinstance(comparison, dict) and "tools" in comparison:
                    tools = comparison["tools"]
                    return len(tools) if isinstance(tools, list) else 0
            return 0
        except (TypeError, AttributeError) as e:
            logger.warning(f"Error counting cached results: {e}")
            return 0

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        try:
            # Safely calculate tool counts
            cache_by_tool_counts = {}
            try:
                cache_by_tool_counts = {
                    tool: len(keys) for tool, keys in self.cache_by_tool.items() if isinstance(keys, set)
                }
            except (TypeError, AttributeError) as e:
                logger.warning(f"Error calculating tool counts: {e}")
                cache_by_tool_counts = {}

            return {
                "hit_rate": self.cache_stats.hit_rate(),
                "total_requests": self.cache_stats.total_requests(),
                "cache_size": len(self.cache_data) if self.cache_data else 0,
                "cache_capacity": self.cache_size,
                "avg_response_time_saved": self.cache_stats.avg_time_saved(),
                "total_time_saved": self.cache_stats.total_response_time_saved,
                "cache_by_tool_counts": cache_by_tool_counts,
                "uptime_hours": (time.time() - self.cache_stats.start_time) / 3600,
            }
        except Exception as e:
            logger.error(f"Error generating cache statistics: {e}")
            return {
                "hit_rate": 0.0,
                "total_requests": 0,
                "cache_size": 0,
                "cache_capacity": self.cache_size,
                "avg_response_time_saved": 0.0,
                "total_time_saved": 0.0,
                "cache_by_tool_counts": {},
                "uptime_hours": 0.0,
                "error": "Statistics unavailable due to cache corruption",
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
        if health_metrics["capacity_usage"] > CACHE_CAPACITY_WARNING_THRESHOLD:
            health_metrics["status"] = "warning"
            health_metrics["recommendations"].append("Cache near capacity - consider increasing cache_size")

        if health_metrics["expired_entries"] > len(ages) * CACHE_EXPIRED_ENTRIES_WARNING_RATIO:
            health_metrics["recommendations"].append("Many expired entries - consider running cleanup")

        if self.cache_stats.hit_rate() < CACHE_HIT_RATE_WARNING_THRESHOLD:
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

        try:
            return [
                {
                    "cache_key": cache_key[:CACHE_KEY_DISPLAY_LENGTH] + "...",  # Truncate for readability
                    "tool": entry.response_data.get("tool", "unknown"),
                    "access_count": entry.access_count,
                    "age_minutes": (time.time() - entry.timestamp) / 60,
                }
                for cache_key, entry in sorted_entries[:limit]
                if self._validate_cache_entry(entry)
            ]
        except (AttributeError, TypeError) as e:
            logger.error(f"Error accessing top entries due to cache corruption: {e}")
            return []

    def _validate_cache_entry(self, entry: CacheEntry) -> bool:
        """Validate cache entry integrity"""
        try:
            # Check entry has required attributes
            if not hasattr(entry, "response_data") or not hasattr(entry, "timestamp"):
                return False

            # Check data types
            if not isinstance(entry.response_data, dict):
                return False

            if not isinstance(entry.timestamp, (int, float)) or entry.timestamp <= 0:
                return False

            # Check if response data is valid
            return self._validate_response_data(entry.response_data)

        except Exception as e:
            logger.warning(f"Cache entry validation failed: {e}")
            return False

    def _validate_response_data(self, response_data: Dict[str, Any]) -> bool:
        """Validate response data structure and content"""
        try:
            # Must be a dictionary
            if not isinstance(response_data, dict):
                return False

            # Check for basic required structure
            if not response_data:  # Empty dict
                return False

            # Validate data doesn't contain problematic types
            return self._validate_data_types(response_data)

        except Exception as e:
            logger.warning(f"Response data validation failed: {e}")
            return False

    def _validate_data_types(self, data: Any) -> bool:
        """Recursively validate data types are JSON-serializable"""
        try:
            if isinstance(data, dict):
                for key, value in data.items():
                    if not isinstance(key, str):
                        return False
                    if not self._validate_data_types(value):
                        return False
            elif isinstance(data, list):
                for item in data:
                    if not self._validate_data_types(item):
                        return False
            elif data is not None and not isinstance(data, (str, int, float, bool)):
                # Only allow JSON-serializable types
                return False

            return True

        except Exception:
            return False

    def _handle_cache_corruption(self, cache_key: str) -> None:
        """Handle cache corruption by safely removing problematic entries"""
        try:
            # Safely remove the corrupted entry
            if cache_key in self.cache_data:
                try:
                    entry = self.cache_data[cache_key]
                    tool_name = entry.response_data.get("tool", "unknown")

                    # Remove from main cache
                    del self.cache_data[cache_key]

                    # Remove from tool tracking
                    if tool_name in self.cache_by_tool:
                        self.cache_by_tool[tool_name].discard(cache_key)
                        if not self.cache_by_tool[tool_name]:
                            del self.cache_by_tool[tool_name]

                except Exception as e:
                    # Force removal if normal removal fails
                    logger.error(f"Failed to cleanly remove corrupted entry, forcing removal: {e}")
                    if cache_key in self.cache_data:
                        del self.cache_data[cache_key]

            # Check for broader corruption
            self._check_cache_integrity()

        except Exception as e:
            logger.error(f"Failed to handle cache corruption: {e}")
            # Last resort: clear entire cache if corruption handling fails
            logger.warning("Clearing entire cache due to corruption handling failure")
            self.clear_cache()

    def _check_cache_integrity(self) -> None:
        """Check and repair cache integrity issues"""
        try:
            corrupted_keys = []

            # Check main cache entries
            for cache_key, entry in list(self.cache_data.items()):
                if not self._validate_cache_entry(entry):
                    corrupted_keys.append(cache_key)

            # Remove corrupted entries
            for cache_key in corrupted_keys:
                logger.warning(f"Removing corrupted cache entry: {cache_key[:16]}...")
                self._remove_entry(cache_key)

            # Verify tool tracking consistency
            self._repair_tool_tracking()

            if corrupted_keys:
                logger.info(f"Cache integrity check complete: removed {len(corrupted_keys)} corrupted entries")

        except Exception as e:
            logger.error(f"Cache integrity check failed: {e}")

    def _repair_tool_tracking(self) -> None:
        """Repair tool tracking data structure if corrupted"""
        try:
            # Rebuild tool tracking from valid cache entries
            new_cache_by_tool = defaultdict(set)

            for cache_key, entry in self.cache_data.items():
                try:
                    tool_name = entry.response_data.get("tool", "unknown")
                    new_cache_by_tool[tool_name].add(cache_key)
                except Exception:
                    logger.warning(f"Skipping entry with corrupted tool tracking: {cache_key[:16]}...")

            self.cache_by_tool = new_cache_by_tool
            logger.info("Tool tracking data structure repaired")

        except Exception as e:
            logger.error(f"Failed to repair tool tracking: {e}")
            self.cache_by_tool = defaultdict(set)
