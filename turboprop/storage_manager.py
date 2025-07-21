#!/usr/bin/env python3
"""
Centralized Storage Manager - Path Management and File Operations

This module provides centralized storage path management for all automatic tool
selection components. It ensures consistent storage locations and handles
directory creation, file operations, and path validation.
"""

import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

from .automatic_selection_config import CONFIG
from .logging_config import get_logger

logger = get_logger(__name__)


class StorageManager:
    """Centralized storage manager for automatic selection system."""

    def __init__(self, base_directory: Optional[str] = None):
        """Initialize storage manager with base directory."""
        self.config = CONFIG["storage_config"]
        self.base_dir = Path(base_directory or self.config["default_storage_dir"])
        self._ensure_base_directory()
        self._lock = threading.Lock()

        # Track file access for optimization
        self._file_access_times = {}

        # Batching for optimized I/O
        self._pending_writes = {}
        self._batch_lock = threading.Lock()
        self._last_batch_flush = time.time()
        self._batch_timeout = 5.0  # Flush pending writes after 5 seconds

        logger.info("Initialized Storage Manager with base directory: %s", self.base_dir)

    def _ensure_base_directory(self) -> None:
        """Ensure the base storage directory exists."""
        try:
            self.base_dir.mkdir(parents=True, exist_ok=True)
            logger.debug("Ensured storage directory exists: %s", self.base_dir)
        except Exception as e:
            logger.error("Failed to create storage directory %s: %s", self.base_dir, e)
            raise

    def get_preferences_path(self) -> Path:
        """Get the path for user preferences file."""
        return self.base_dir / self.config["preferences_file"]

    def get_learning_model_path(self) -> Path:
        """Get the path for learning model data file."""
        return self.base_dir / self.config["learning_model_file"]

    def get_effectiveness_data_path(self) -> Path:
        """Get the path for effectiveness tracking data file."""
        return self.base_dir / self.config["effectiveness_data_file"]

    def get_custom_file_path(self, filename: str) -> Path:
        """Get path for a custom file within the storage directory."""
        return self.base_dir / filename

    def save_json_data(self, file_path: Path, data: Dict[str, Any]) -> bool:
        """Save data to JSON file with thread safety."""
        try:
            with self._lock:
                # Create parent directories if needed
                file_path.parent.mkdir(parents=True, exist_ok=True)

                # Write to temporary file first for atomic write
                temp_path = file_path.with_suffix(f"{file_path.suffix}.tmp")
                with open(temp_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)

                # Atomic move to final location
                temp_path.replace(file_path)

                # Track access time
                self._file_access_times[str(file_path)] = time.time()

                logger.debug("Saved JSON data to: %s", file_path)
                return True

        except Exception as e:
            logger.error("Failed to save JSON data to %s: %s", file_path, e)
            return False

    def load_json_data(self, file_path: Path, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Load data from JSON file with error handling."""
        try:
            if not file_path.exists():
                logger.debug("File does not exist, returning default: %s", file_path)
                return default or {}

            with self._lock:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Track access time
                self._file_access_times[str(file_path)] = time.time()

                logger.debug("Loaded JSON data from: %s", file_path)
                return data

        except Exception as e:
            logger.error("Failed to load JSON data from %s: %s", file_path, e)
            return default or {}

    def file_exists(self, file_path: Path) -> bool:
        """Check if a file exists."""
        return file_path.exists()

    def get_file_size(self, file_path: Path) -> int:
        """Get file size in bytes."""
        try:
            return file_path.stat().st_size if file_path.exists() else 0
        except Exception as e:
            logger.error("Failed to get file size for %s: %s", file_path, e)
            return 0

    def delete_file(self, file_path: Path) -> bool:
        """Delete a file safely."""
        try:
            if file_path.exists():
                file_path.unlink()
                # Remove from access tracking
                self._file_access_times.pop(str(file_path), None)
                logger.debug("Deleted file: %s", file_path)
                return True
            return False
        except Exception as e:
            logger.error("Failed to delete file %s: %s", file_path, e)
            return False

    def cleanup_old_files(self, max_age_days: int = 30) -> int:
        """Clean up old files based on age."""
        cleaned_count = 0
        cutoff_time = time.time() - (max_age_days * 24 * 60 * 60)

        try:
            for file_path in self.base_dir.glob("*.json"):
                try:
                    if file_path.stat().st_mtime < cutoff_time:
                        if self.delete_file(file_path):
                            cleaned_count += 1
                except Exception as e:
                    logger.warning("Error checking file age for %s: %s", file_path, e)

            logger.info("Cleaned up %d old files", cleaned_count)
            return cleaned_count

        except Exception as e:
            logger.error("Error during cleanup: %s", e)
            return cleaned_count

    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage directory statistics."""
        try:
            files = list(self.base_dir.glob("*"))
            total_size = sum(f.stat().st_size for f in files if f.is_file())

            return {
                "base_directory": str(self.base_dir),
                "total_files": len([f for f in files if f.is_file()]),
                "total_directories": len([f for f in files if f.is_directory()]),
                "total_size_bytes": total_size,
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "recent_access_count": len(self._file_access_times),
            }
        except Exception as e:
            logger.error("Error getting storage stats: %s", e)
            return {}

    def queue_write(self, file_path: Path, data: Dict[str, Any], priority: str = "normal") -> bool:
        """Queue data for batched write operation."""
        try:
            with self._batch_lock:
                self._pending_writes[str(file_path)] = {"data": data, "priority": priority, "queued_at": time.time()}

                logger.debug("Queued write for %s (priority: %s)", file_path, priority)

                # Auto-flush if timeout exceeded or high priority
                current_time = time.time()
                time_since_last_flush = current_time - self._last_batch_flush

                if time_since_last_flush >= self._batch_timeout or priority == "high":
                    return self.flush_pending_writes()

                return True

        except Exception as e:
            logger.error("Failed to queue write for %s: %s", file_path, e)
            return False

    def flush_pending_writes(self) -> bool:
        """Flush all pending writes to disk."""
        try:
            with self._batch_lock:
                if not self._pending_writes:
                    return True

                pending_count = len(self._pending_writes)
                success_count = 0

                # Sort by priority and age
                sorted_writes = sorted(
                    self._pending_writes.items(), key=lambda x: (x[1]["priority"] == "high", -x[1]["queued_at"])
                )

                for file_path_str, write_info in sorted_writes:
                    file_path = Path(file_path_str)
                    if self.save_json_data(file_path, write_info["data"]):
                        success_count += 1
                    else:
                        logger.error("Failed to write queued data to %s", file_path)

                # Clear completed writes
                self._pending_writes.clear()
                self._last_batch_flush = time.time()

                logger.info("Flushed %d/%d pending writes", success_count, pending_count)
                return success_count == pending_count

        except Exception as e:
            logger.error("Error flushing pending writes: %s", e)
            return False

    def save_consolidated_data(self, data_map: Dict[Path, Dict[str, Any]]) -> bool:
        """Save multiple files in a single consolidated operation."""
        try:
            success_count = 0
            total_files = len(data_map)

            # Process in order of file path to ensure consistency
            for file_path, data in sorted(data_map.items()):
                if self.save_json_data(file_path, data):
                    success_count += 1
                else:
                    logger.error("Failed to save consolidated data to %s", file_path)

            logger.info("Consolidated save: %d/%d files succeeded", success_count, total_files)
            return success_count == total_files

        except Exception as e:
            logger.error("Error in consolidated save: %s", e)
            return False

    def get_batch_stats(self) -> Dict[str, Any]:
        """Get statistics about batched operations."""
        with self._batch_lock:
            current_time = time.time()
            return {
                "pending_writes": len(self._pending_writes),
                "time_since_last_flush": round(current_time - self._last_batch_flush, 2),
                "batch_timeout": self._batch_timeout,
                "oldest_queued_write": min(
                    (current_time - write_info["queued_at"] for write_info in self._pending_writes.values()), default=0
                ),
            }


# Global storage manager instance
_storage_manager = None
_storage_manager_lock = threading.Lock()


def get_storage_manager(base_directory: Optional[str] = None) -> StorageManager:
    """Get the global storage manager instance (singleton pattern)."""
    global _storage_manager

    with _storage_manager_lock:
        if _storage_manager is None or (base_directory and str(_storage_manager.base_dir) != base_directory):
            _storage_manager = StorageManager(base_directory)

        return _storage_manager


def reset_storage_manager() -> None:
    """Reset the global storage manager (mainly for testing)."""
    global _storage_manager

    with _storage_manager_lock:
        _storage_manager = None
