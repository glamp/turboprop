import logging
import statistics
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetric:
    """Individual performance metric data point."""

    timestamp: float
    tool_name: str
    response_time: float
    cache_hit: bool
    result_count: int
    response_size: int
    error_occurred: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp,
            "tool_name": self.tool_name,
            "response_time": self.response_time,
            "cache_hit": self.cache_hit,
            "result_count": self.result_count,
            "response_size": self.response_size,
            "error_occurred": self.error_occurred,
        }


@dataclass
class ToolPerformanceStats:
    """Aggregated performance statistics for a specific tool."""

    tool_name: str
    total_requests: int = 0
    total_errors: int = 0
    cache_hits: int = 0
    response_times: deque = field(default_factory=lambda: deque(maxlen=1000))
    response_sizes: deque = field(default_factory=lambda: deque(maxlen=1000))
    result_counts: deque = field(default_factory=lambda: deque(maxlen=1000))
    last_updated: float = field(default_factory=time.time)

    def add_metric(self, metric: PerformanceMetric) -> None:
        """Add a new performance metric."""
        self.total_requests += 1
        if metric.error_occurred:
            self.total_errors += 1
        if metric.cache_hit:
            self.cache_hits += 1

        self.response_times.append(metric.response_time)
        self.response_sizes.append(metric.response_size)
        self.result_counts.append(metric.result_count)
        self.last_updated = time.time()

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        cache_hit_rate = self.cache_hits / self.total_requests if self.total_requests > 0 else 0
        error_rate = self.total_errors / self.total_requests if self.total_requests > 0 else 0

        response_times_list = list(self.response_times)
        avg_response_time = statistics.mean(response_times_list) if response_times_list else 0

        response_sizes_list = list(self.response_sizes)
        avg_response_size = statistics.mean(response_sizes_list) if response_sizes_list else 0

        result_counts_list = list(self.result_counts)
        avg_result_count = statistics.mean(result_counts_list) if result_counts_list else 0

        return {
            "tool_name": self.tool_name,
            "total_requests": self.total_requests,
            "error_rate": error_rate,
            "cache_hit_rate": cache_hit_rate,
            "avg_response_time": avg_response_time,
            "avg_response_size": avg_response_size,
            "avg_result_count": avg_result_count,
            "last_updated": self.last_updated,
        }


class PerformanceAlerts:
    """Handles performance alerting and threshold monitoring."""

    def __init__(self):
        self.thresholds = {
            "response_time": 5.0,  # seconds
            "error_rate": 0.1,  # 10%
            "cache_hit_rate_min": 0.3,  # 30%
            "response_size": 100000,  # 100KB
        }
        self.alert_cooldown = 300  # 5 minutes
        self.last_alerts = {}

    def check_thresholds(self, stats: ToolPerformanceStats) -> List[str]:
        """Check if any thresholds are exceeded and return alerts."""
        alerts = []
        now = time.time()

        summary = stats.get_summary()

        # Response time alert
        if summary["avg_response_time"] > self.thresholds["response_time"]:
            alert_key = f"{stats.tool_name}_response_time"
            if self._should_alert(alert_key, now):
                alerts.append(
                    f"High response time for {stats.tool_name}: "
                    f"{summary['avg_response_time']:.2f}s (threshold: {self.thresholds['response_time']}s)"
                )
                self.last_alerts[alert_key] = now

        # Error rate alert
        if summary["error_rate"] > self.thresholds["error_rate"]:
            alert_key = f"{stats.tool_name}_error_rate"
            if self._should_alert(alert_key, now):
                alerts.append(
                    f"High error rate for {stats.tool_name}: "
                    f"{summary['error_rate']:.1%} (threshold: {self.thresholds['error_rate']:.1%})"
                )
                self.last_alerts[alert_key] = now

        # Cache hit rate alert
        if summary["cache_hit_rate"] < self.thresholds["cache_hit_rate_min"]:
            alert_key = f"{stats.tool_name}_cache_hit_rate"
            if self._should_alert(alert_key, now):
                alerts.append(
                    f"Low cache hit rate for {stats.tool_name}: "
                    f"{summary['cache_hit_rate']:.1%} (threshold: {self.thresholds['cache_hit_rate_min']:.1%})"
                )
                self.last_alerts[alert_key] = now

        # Response size alert
        if summary["avg_response_size"] > self.thresholds["response_size"]:
            alert_key = f"{stats.tool_name}_response_size"
            if self._should_alert(alert_key, now):
                alerts.append(
                    f"Large response size for {stats.tool_name}: "
                    f"{summary['avg_response_size']:.0f} bytes (threshold: {self.thresholds['response_size']} bytes)"
                )
                self.last_alerts[alert_key] = now

        return alerts

    def _should_alert(self, alert_key: str, now: float) -> bool:
        """Check if enough time has passed since last alert."""
        if alert_key not in self.last_alerts:
            return True
        return now - self.last_alerts[alert_key] > self.alert_cooldown


class MCPToolPerformanceMonitor:
    """Monitor performance of MCP tool responses with comprehensive analytics."""

    def __init__(self, enable_alerts: bool = True, metrics_retention_hours: int = 24):
        self.metrics_retention_hours = metrics_retention_hours
        self.metrics: List[PerformanceMetric] = []
        self.tool_stats: Dict[str, ToolPerformanceStats] = {}
        self.alerts = PerformanceAlerts() if enable_alerts else None
        self.lock = threading.RLock()
        self.start_time = time.time()

        logger.info("MCP Tool Performance Monitor initialized")

    def record_response_time(
        self,
        tool_name: str,
        response_time: float,
        cache_hit: bool = False,
        result_count: int = 0,
        response_size: int = 0,
        error_occurred: bool = False,
    ) -> None:
        """Record response time and related metrics for performance analysis."""
        with self.lock:
            # Create metric
            metric = PerformanceMetric(
                timestamp=time.time(),
                tool_name=tool_name,
                response_time=response_time,
                cache_hit=cache_hit,
                result_count=result_count,
                response_size=response_size,
                error_occurred=error_occurred,
            )

            # Store metric
            self.metrics.append(metric)

            # Update tool-specific stats
            if tool_name not in self.tool_stats:
                self.tool_stats[tool_name] = ToolPerformanceStats(tool_name=tool_name)

            self.tool_stats[tool_name].add_metric(metric)

            # Check for alerts
            if self.alerts:
                alerts = self.alerts.check_thresholds(self.tool_stats[tool_name])
                for alert in alerts:
                    logger.warning(f"Performance Alert: {alert}")

            # Clean up old metrics
            self._cleanup_old_metrics()

    def record_cache_performance(self, tool_name: str, cache_hit: bool) -> None:
        """Record cache hit/miss for analysis."""
        # This is typically called as part of record_response_time
        # but can be used separately for cache-only metrics
        pass

    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance analysis report."""
        with self.lock:
            report = {
                "generated_at": time.time(),
                "monitoring_duration_hours": (time.time() - self.start_time) / 3600,
                "total_metrics": len(self.metrics),
                "tools_monitored": len(self.tool_stats),
                "tool_statistics": {},
                "overall_statistics": self._get_overall_statistics(),
                "performance_trends": self._get_performance_trends(),
                "recommendations": self._get_optimization_recommendations(),
            }

            # Add per-tool statistics
            for tool_name, stats in self.tool_stats.items():
                report["tool_statistics"][tool_name] = stats.get_summary()

            return report

    def get_tool_performance(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get performance data for a specific tool."""
        with self.lock:
            if tool_name not in self.tool_stats:
                return None

            stats = self.tool_stats[tool_name]
            summary = stats.get_summary()

            # Add recent metrics
            recent_metrics = [m for m in self.metrics if m.tool_name == tool_name and time.time() - m.timestamp < 3600]

            summary["recent_metrics_count"] = len(recent_metrics)
            summary["recent_metrics"] = [m.to_dict() for m in recent_metrics[-10:]]  # Last 10

            return summary

    def get_top_performing_tools(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get top performing tools by response time."""
        with self.lock:
            tool_summaries = [stats.get_summary() for stats in self.tool_stats.values()]

            # Sort by average response time (lower is better)
            sorted_tools = sorted(tool_summaries, key=lambda x: x["avg_response_time"])

            return sorted_tools[:limit]

    def get_problematic_tools(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get tools with performance issues."""
        with self.lock:
            problematic = []

            for tool_name, stats in self.tool_stats.items():
                summary = stats.get_summary()
                issues = []

                # Check for issues
                if summary["avg_response_time"] > 3.0:
                    issues.append(f"Slow response time: {summary['avg_response_time']:.2f}s")

                if summary["error_rate"] > 0.05:  # 5%
                    issues.append(f"High error rate: {summary['error_rate']:.1%}")

                if summary["cache_hit_rate"] < 0.2:  # 20%
                    issues.append(f"Low cache hit rate: {summary['cache_hit_rate']:.1%}")

                if issues:
                    problematic.append(
                        {**summary, "issues": issues, "severity_score": self._calculate_severity_score(summary)}
                    )

            # Sort by severity score (higher is worse)
            problematic.sort(key=lambda x: x["severity_score"], reverse=True)

            return problematic[:limit]

    def get_cache_effectiveness_report(self) -> Dict[str, Any]:
        """Generate cache effectiveness analysis."""
        with self.lock:
            cache_stats = {}
            overall_hits = 0
            overall_total = 0

            for tool_name, stats in self.tool_stats.items():
                summary = stats.get_summary()
                cache_stats[tool_name] = {
                    "hit_rate": summary["cache_hit_rate"],
                    "total_requests": summary["total_requests"],
                    "cache_hits": stats.cache_hits,
                }

                overall_hits += stats.cache_hits
                overall_total += stats.total_requests

            overall_hit_rate = overall_hits / overall_total if overall_total > 0 else 0

            return {
                "overall_hit_rate": overall_hit_rate,
                "total_requests": overall_total,
                "total_cache_hits": overall_hits,
                "by_tool": cache_stats,
                "recommendations": self._get_cache_recommendations(cache_stats),
            }

    def _cleanup_old_metrics(self) -> None:
        """Remove metrics older than retention period."""
        cutoff_time = time.time() - (self.metrics_retention_hours * 3600)
        self.metrics = [m for m in self.metrics if m.timestamp > cutoff_time]

    def _get_overall_statistics(self) -> Dict[str, Any]:
        """Calculate overall system performance statistics."""
        if not self.metrics:
            return {}

        total_requests = len(self.metrics)
        total_errors = sum(1 for m in self.metrics if m.error_occurred)
        total_cache_hits = sum(1 for m in self.metrics if m.cache_hit)

        response_times = [m.response_time for m in self.metrics]
        response_sizes = [m.response_size for m in self.metrics]

        return {
            "total_requests": total_requests,
            "error_rate": total_errors / total_requests if total_requests > 0 else 0,
            "cache_hit_rate": total_cache_hits / total_requests if total_requests > 0 else 0,
            "avg_response_time": statistics.mean(response_times) if response_times else 0,
            "median_response_time": statistics.median(response_times) if response_times else 0,
            "p95_response_time": self._percentile(response_times, 0.95) if response_times else 0,
            "avg_response_size": statistics.mean(response_sizes) if response_sizes else 0,
            "requests_per_hour": self._calculate_requests_per_hour(),
        }

    def _get_performance_trends(self) -> Dict[str, Any]:
        """Analyze performance trends over time."""
        if len(self.metrics) < 10:
            return {"insufficient_data": True}

        # Get metrics from last hour vs previous hour
        now = time.time()
        one_hour_ago = now - 3600
        two_hours_ago = now - 7200

        recent_metrics = [m for m in self.metrics if m.timestamp > one_hour_ago]
        previous_metrics = [m for m in self.metrics if two_hours_ago < m.timestamp <= one_hour_ago]

        if not recent_metrics or not previous_metrics:
            return {"insufficient_data": True}

        recent_avg_time = statistics.mean([m.response_time for m in recent_metrics])
        previous_avg_time = statistics.mean([m.response_time for m in previous_metrics])

        recent_error_rate = sum(1 for m in recent_metrics if m.error_occurred) / len(recent_metrics)
        previous_error_rate = sum(1 for m in previous_metrics if m.error_occurred) / len(previous_metrics)

        return {
            "response_time_trend": "improving" if recent_avg_time < previous_avg_time else "degrading",
            "response_time_change_pct": ((recent_avg_time - previous_avg_time) / previous_avg_time * 100)
            if previous_avg_time > 0
            else 0,
            "error_rate_trend": "improving" if recent_error_rate < previous_error_rate else "degrading",
            "error_rate_change": recent_error_rate - previous_error_rate,
            "recent_period_requests": len(recent_metrics),
            "previous_period_requests": len(previous_metrics),
        }

    def _get_optimization_recommendations(self) -> List[str]:
        """Generate optimization recommendations based on metrics."""
        recommendations = []
        overall_stats = self._get_overall_statistics()

        if not overall_stats:
            return recommendations

        # Response time recommendations
        if overall_stats["avg_response_time"] > 2.0:
            recommendations.append("Consider optimizing slow-performing tools or implementing response caching")

        if overall_stats["p95_response_time"] > 5.0:
            recommendations.append(
                "Some requests are very slow - investigate outliers and optimize worst-case performance"
            )

        # Cache recommendations
        if overall_stats["cache_hit_rate"] < 0.4:
            recommendations.append("Low cache hit rate - review caching strategy and increase TTL if appropriate")

        # Error rate recommendations
        if overall_stats["error_rate"] > 0.05:
            recommendations.append("High error rate detected - investigate error patterns and improve error handling")

        # Response size recommendations
        if overall_stats["avg_response_size"] > 50000:
            recommendations.append("Large response sizes - consider response compression or pagination")

        return recommendations

    def _get_cache_recommendations(self, cache_stats: Dict[str, Any]) -> List[str]:
        """Generate cache-specific recommendations."""
        recommendations = []

        # Find tools with low hit rates
        low_hit_rate_tools = [
            tool for tool, stats in cache_stats.items() if stats["hit_rate"] < 0.3 and stats["total_requests"] > 10
        ]

        if low_hit_rate_tools:
            recommendations.append(f"Tools with low cache hit rates: {', '.join(low_hit_rate_tools)}")
            recommendations.append("Consider increasing cache TTL or reviewing cache invalidation strategy")

        return recommendations

    def _calculate_severity_score(self, summary: Dict[str, Any]) -> float:
        """Calculate severity score for problematic tools."""
        score = 0.0

        # Response time impact
        if summary["avg_response_time"] > 5.0:
            score += 3.0
        elif summary["avg_response_time"] > 2.0:
            score += 1.0

        # Error rate impact
        score += summary["error_rate"] * 10.0  # 10% error rate = 1.0 point

        # Request volume impact
        if summary["total_requests"] > 100:
            score *= 1.5  # Higher impact for high-volume tools

        return score

    def _calculate_requests_per_hour(self) -> float:
        """Calculate current requests per hour rate."""
        one_hour_ago = time.time() - 3600
        recent_metrics = [m for m in self.metrics if m.timestamp > one_hour_ago]
        return len(recent_metrics)

    def _percentile(self, data: List[float], percentile: float) -> float:
        """Calculate percentile of a dataset."""
        if not data:
            return 0.0

        sorted_data = sorted(data)
        index = int(percentile * len(sorted_data))
        if index >= len(sorted_data):
            index = len(sorted_data) - 1

        return sorted_data[index]


# Global monitor instance
_performance_monitor: Optional[MCPToolPerformanceMonitor] = None


def initialize_performance_monitor(enable_alerts: bool = True) -> None:
    """Initialize the global performance monitor."""
    global _performance_monitor
    _performance_monitor = MCPToolPerformanceMonitor(enable_alerts=enable_alerts)
    logger.info("MCP Tool Performance Monitor initialized globally")


def get_performance_monitor() -> Optional[MCPToolPerformanceMonitor]:
    """Get the global performance monitor instance."""
    return _performance_monitor


def record_tool_performance(
    tool_name: str,
    response_time: float,
    cache_hit: bool = False,
    result_count: int = 0,
    response_size: int = 0,
    error_occurred: bool = False,
) -> None:
    """Record performance metrics for a tool (convenience function)."""
    if _performance_monitor:
        _performance_monitor.record_response_time(
            tool_name=tool_name,
            response_time=response_time,
            cache_hit=cache_hit,
            result_count=result_count,
            response_size=response_size,
            error_occurred=error_occurred,
        )
