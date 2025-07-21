"""
Comprehensive tests for the configuration system.

This module tests all validation functions, configuration classes,
environment variable handling, and error scenarios for the config module.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

# Import the config module and all its components
from config import (
    Config,
    ConfigValidationError,
    DatabaseConfig,
    EmbeddingConfig,
    FileProcessingConfig,
    LoggingConfig,
    MCPConfig,
    SearchConfig,
    ServerConfig,
    validate_boolean,
    validate_device,
    validate_log_level,
    validate_memory_limit,
    validate_non_negative_float,
    validate_positive_float,
    validate_positive_int,
    validate_range_float,
)


class TestValidationFunctions:
    """Test all validation functions with valid and invalid inputs."""

    def test_validate_positive_int_valid(self):
        """Test validate_positive_int with valid inputs."""
        assert validate_positive_int("5", "TEST_VAR", 1) == 5
        assert validate_positive_int("100", "TEST_VAR", 1) == 100
        assert validate_positive_int("1", "TEST_VAR", 1) == 1

    def test_validate_positive_int_invalid(self):
        """Test validate_positive_int with invalid inputs."""
        with pytest.raises(ConfigValidationError, match="TEST_VAR must be positive"):
            validate_positive_int("0", "TEST_VAR", 1)

        with pytest.raises(ConfigValidationError, match="TEST_VAR must be positive"):
            validate_positive_int("-1", "TEST_VAR", 1)

        with pytest.raises(ConfigValidationError, match="TEST_VAR must be a valid integer"):
            validate_positive_int("abc", "TEST_VAR", 1)

        with pytest.raises(ConfigValidationError, match="TEST_VAR must be a valid integer"):
            validate_positive_int("1.5", "TEST_VAR", 1)

    def test_validate_positive_float_valid(self):
        """Test validate_positive_float with valid inputs."""
        assert validate_positive_float("5.0", "TEST_VAR", 1.0) == 5.0
        assert validate_positive_float("0.1", "TEST_VAR", 1.0) == 0.1
        assert validate_positive_float("100.5", "TEST_VAR", 1.0) == 100.5
        assert validate_positive_float("1", "TEST_VAR", 1.0) == 1.0

    def test_validate_positive_float_invalid(self):
        """Test validate_positive_float with invalid inputs."""
        with pytest.raises(ConfigValidationError, match="TEST_VAR must be positive"):
            validate_positive_float("0", "TEST_VAR", 1.0)

        with pytest.raises(ConfigValidationError, match="TEST_VAR must be positive"):
            validate_positive_float("-1.5", "TEST_VAR", 1.0)

        with pytest.raises(ConfigValidationError, match="TEST_VAR must be a valid float"):
            validate_positive_float("abc", "TEST_VAR", 1.0)

    def test_validate_non_negative_float_valid(self):
        """Test validate_non_negative_float with valid inputs."""
        assert validate_non_negative_float("0.0", "TEST_VAR", 1.0) == 0.0
        assert validate_non_negative_float("5.5", "TEST_VAR", 1.0) == 5.5
        assert validate_non_negative_float("0", "TEST_VAR", 1.0) == 0.0

    def test_validate_non_negative_float_invalid(self):
        """Test validate_non_negative_float with invalid inputs."""
        with pytest.raises(ConfigValidationError, match="TEST_VAR must be non-negative"):
            validate_non_negative_float("-1.0", "TEST_VAR", 1.0)

        with pytest.raises(ConfigValidationError, match="TEST_VAR must be a valid float"):
            validate_non_negative_float("abc", "TEST_VAR", 1.0)

    def test_validate_memory_limit_valid(self):
        """Test validate_memory_limit with valid inputs."""
        assert validate_memory_limit("1GB", "TEST_VAR") == "1GB"
        assert validate_memory_limit("512MB", "TEST_VAR") == "512MB"
        assert validate_memory_limit("1024KB", "TEST_VAR") == "1024KB"
        assert validate_memory_limit("500000B", "TEST_VAR") == "500000B"
        assert validate_memory_limit("1.5GB", "TEST_VAR") == "1.5GB"
        assert validate_memory_limit("1gb", "TEST_VAR") == "1gb"  # Should work with lowercase

    def test_validate_memory_limit_invalid(self):
        """Test validate_memory_limit with invalid inputs."""
        with pytest.raises(ConfigValidationError, match="TEST_VAR must be in format like"):
            validate_memory_limit("1", "TEST_VAR")

        with pytest.raises(ConfigValidationError, match="TEST_VAR must be in format like"):
            validate_memory_limit("1TB", "TEST_VAR")  # TB not supported

        with pytest.raises(ConfigValidationError, match="TEST_VAR must be in format like"):
            validate_memory_limit("abc", "TEST_VAR")

    def test_validate_boolean_valid(self):
        """Test validate_boolean with valid inputs."""
        # True values
        assert validate_boolean("true", "TEST_VAR") is True
        assert validate_boolean("TRUE", "TEST_VAR") is True
        assert validate_boolean("1", "TEST_VAR") is True
        assert validate_boolean("yes", "TEST_VAR") is True
        assert validate_boolean("YES", "TEST_VAR") is True
        assert validate_boolean("on", "TEST_VAR") is True
        assert validate_boolean("ON", "TEST_VAR") is True

        # False values
        assert validate_boolean("false", "TEST_VAR") is False
        assert validate_boolean("FALSE", "TEST_VAR") is False
        assert validate_boolean("0", "TEST_VAR") is False
        assert validate_boolean("no", "TEST_VAR") is False
        assert validate_boolean("NO", "TEST_VAR") is False
        assert validate_boolean("off", "TEST_VAR") is False
        assert validate_boolean("OFF", "TEST_VAR") is False

    def test_validate_boolean_invalid(self):
        """Test validate_boolean with invalid inputs."""
        with pytest.raises(ConfigValidationError, match="TEST_VAR must be"):
            validate_boolean("maybe", "TEST_VAR")

        with pytest.raises(ConfigValidationError, match="TEST_VAR must be"):
            validate_boolean("2", "TEST_VAR")

    def test_validate_log_level_valid(self):
        """Test validate_log_level with valid inputs."""
        assert validate_log_level("DEBUG", "TEST_VAR") == "DEBUG"
        assert validate_log_level("debug", "TEST_VAR") == "DEBUG"
        assert validate_log_level("INFO", "TEST_VAR") == "INFO"
        assert validate_log_level("info", "TEST_VAR") == "INFO"
        assert validate_log_level("WARNING", "TEST_VAR") == "WARNING"
        assert validate_log_level("ERROR", "TEST_VAR") == "ERROR"
        assert validate_log_level("CRITICAL", "TEST_VAR") == "CRITICAL"

    def test_validate_log_level_invalid(self):
        """Test validate_log_level with invalid inputs."""
        with pytest.raises(ConfigValidationError, match="TEST_VAR must be one of"):
            validate_log_level("INVALID", "TEST_VAR")

        with pytest.raises(ConfigValidationError, match="TEST_VAR must be one of"):
            validate_log_level("TRACE", "TEST_VAR")

    def test_validate_device_valid(self):
        """Test validate_device with valid inputs."""
        assert validate_device("cpu", "TEST_VAR") == "cpu"
        assert validate_device("CPU", "TEST_VAR") == "cpu"
        assert validate_device("cuda", "TEST_VAR") == "cuda"
        assert validate_device("CUDA", "TEST_VAR") == "cuda"
        assert validate_device("mps", "TEST_VAR") == "mps"
        assert validate_device("MPS", "TEST_VAR") == "mps"

    def test_validate_device_invalid(self):
        """Test validate_device with invalid inputs."""
        with pytest.raises(ConfigValidationError, match="TEST_VAR must be one of"):
            validate_device("gpu", "TEST_VAR")

        with pytest.raises(ConfigValidationError, match="TEST_VAR must be one of"):
            validate_device("tpu", "TEST_VAR")

    def test_validate_range_float_valid(self):
        """Test validate_range_float with valid inputs."""
        assert validate_range_float("0.5", "TEST_VAR", 0.0, 1.0) == 0.5
        assert validate_range_float("0.0", "TEST_VAR", 0.0, 1.0) == 0.0
        assert validate_range_float("1.0", "TEST_VAR", 0.0, 1.0) == 1.0
        assert validate_range_float("5", "TEST_VAR", 1, 10) == 5.0

    def test_validate_range_float_invalid(self):
        """Test validate_range_float with invalid inputs."""
        with pytest.raises(ConfigValidationError, match="TEST_VAR must be between"):
            validate_range_float("1.5", "TEST_VAR", 0.0, 1.0)

        with pytest.raises(ConfigValidationError, match="TEST_VAR must be between"):
            validate_range_float("-0.5", "TEST_VAR", 0.0, 1.0)

        with pytest.raises(ConfigValidationError, match="TEST_VAR must be a valid float"):
            validate_range_float("abc", "TEST_VAR", 0.0, 1.0)


class TestConfigValidationError:
    """Test ConfigValidationError exception."""

    def test_config_validation_error_inheritance(self):
        """Test that ConfigValidationError inherits from ValueError."""
        assert issubclass(ConfigValidationError, ValueError)

    def test_config_validation_error_message(self):
        """Test that ConfigValidationError preserves message."""
        error = ConfigValidationError("Test error message")
        assert str(error) == "Test error message"


class TestDatabaseConfig:
    """Test DatabaseConfig class."""

    def test_database_config_defaults(self):
        """Test DatabaseConfig with default values."""
        # Test that defaults are set correctly
        db_config = DatabaseConfig()

        # Check that values are reasonable defaults
        assert db_config.MEMORY_LIMIT == "1GB"
        assert db_config.THREADS == 4
        assert db_config.MAX_RETRIES == 3
        assert db_config.RETRY_DELAY == 0.1
        assert db_config.DEFAULT_DB_NAME == "code_index.duckdb"
        assert db_config.DEFAULT_DB_DIR == ".turboprop"

    def test_get_db_path(self):
        """Test get_db_path method."""
        db_config = DatabaseConfig()

        # Test with default (current working directory)
        path = db_config.get_db_path()
        assert path == Path.cwd() / ".turboprop" / "code_index.duckdb"

        # Test with custom repo path
        custom_path = Path("/tmp/test_repo")
        path = db_config.get_db_path(custom_path)
        assert path == custom_path / ".turboprop" / "code_index.duckdb"

    @patch.dict(os.environ, {"TURBOPROP_DB_MEMORY_LIMIT": "2GB"})
    def test_database_config_env_override(self):
        """Test DatabaseConfig with environment variable override."""
        # Need to reload the module to pick up new env vars
        import importlib

        import config

        importlib.reload(config)

        # Check that environment variable is used
        assert config.DatabaseConfig.MEMORY_LIMIT == "2GB"

    @patch.dict(os.environ, {"TURBOPROP_DB_THREADS": "invalid"})
    def test_database_config_invalid_env_var(self):
        """Test DatabaseConfig with invalid environment variable."""
        # This should fail during class initialization
        import importlib

        import config

        # Manually check for the exception
        try:
            importlib.reload(config)
            # If we get here, the test should fail
            assert False, "Expected ConfigValidationError to be raised"
        except Exception as e:
            # Check that it's the right type of exception by name
            # (to avoid isinstance issues with reloaded modules)
            assert (
                type(e).__name__ == "ConfigValidationError"
            ), f"Expected ConfigValidationError, got {type(e).__name__}"
            assert "TURBOPROP_DB_THREADS must be a valid integer" in str(e)


class TestFileProcessingConfig:
    """Test FileProcessingConfig class."""

    def test_file_processing_config_defaults(self):
        """Test FileProcessingConfig with default values."""
        file_config = FileProcessingConfig()

        assert file_config.MAX_FILE_SIZE_MB == 1.0
        assert file_config.DEBOUNCE_SECONDS == 5.0
        assert file_config.PREVIEW_LENGTH == 200
        assert file_config.SNIPPET_LENGTH == 300
        assert file_config.BATCH_SIZE == 100


class TestSearchConfig:
    """Test SearchConfig class."""

    def test_search_config_defaults(self):
        """Test SearchConfig with default values."""
        search_config = SearchConfig()

        assert search_config.DEFAULT_MAX_RESULTS == 5
        assert search_config.MAX_RESULTS_LIMIT == 20
        assert search_config.MIN_SIMILARITY_THRESHOLD == 0.1
        assert search_config.SEPARATOR_LENGTH == 50


class TestEmbeddingConfig:
    """Test EmbeddingConfig class."""

    def test_embedding_config_defaults(self):
        """Test EmbeddingConfig with default values."""
        embedding_config = EmbeddingConfig()

        assert embedding_config.EMBED_MODEL == "all-MiniLM-L6-v2"
        assert embedding_config.DIMENSIONS == 384
        assert embedding_config.DEVICE == "cpu"
        assert embedding_config.BATCH_SIZE == 32
        assert embedding_config.MAX_RETRIES == 3
        assert embedding_config.RETRY_BASE_DELAY == 1.0


class TestServerConfig:
    """Test ServerConfig class."""

    def test_server_config_defaults(self):
        """Test ServerConfig with default values."""
        server_config = ServerConfig()

        assert server_config.HOST == "0.0.0.0"
        assert server_config.PORT == 8000
        assert server_config.WATCH_DIRECTORY == "."
        assert server_config.WATCH_MAX_FILE_SIZE_MB == 1.0
        assert server_config.WATCH_DEBOUNCE_SECONDS == 5.0


class TestLoggingConfig:
    """Test LoggingConfig class."""

    def test_logging_config_defaults(self):
        """Test LoggingConfig with default values."""
        logging_config = LoggingConfig()

        assert logging_config.LOG_LEVEL == "INFO"
        assert logging_config.LOG_FORMAT == "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        assert logging_config.LOG_FILE is None
        assert logging_config.LOG_MAX_SIZE == 10485760  # 10MB
        assert logging_config.LOG_BACKUP_COUNT == 5


class TestMCPConfig:
    """Test MCPConfig class."""

    def test_mcp_config_defaults(self):
        """Test MCPConfig with default values."""
        mcp_config = MCPConfig()

        assert mcp_config.DEFAULT_MAX_FILE_SIZE_MB == 1.0
        assert mcp_config.DEFAULT_DEBOUNCE_SECONDS == 5.0
        assert mcp_config.MAX_FILES_LIST == 100
        assert mcp_config.MAX_SEARCH_RESULTS == 20


class TestMainConfig:
    """Test the main Config class."""

    def test_config_initialization(self):
        """Test that Config class initializes all sub-configs."""
        main_config = Config()

        assert isinstance(main_config.database, DatabaseConfig)
        assert isinstance(main_config.file_processing, FileProcessingConfig)
        assert isinstance(main_config.search, SearchConfig)
        assert isinstance(main_config.embedding, EmbeddingConfig)
        assert isinstance(main_config.server, ServerConfig)
        assert isinstance(main_config.logging, LoggingConfig)
        assert isinstance(main_config.mcp, MCPConfig)

    def test_config_validate_success(self):
        """Test Config.validate() method with valid configuration."""
        result = Config.validate()
        assert result is True

    def test_config_get_validation_status_success(self):
        """Test Config.get_validation_status() method with valid configuration."""
        status = Config.get_validation_status()
        assert status == "✅ All configuration values are valid"

    def test_config_get_summary(self):
        """Test Config.get_summary() method."""
        summary = Config.get_summary()

        # Check that summary contains expected sections
        assert "Turboprop Configuration Summary" in summary
        assert "Database:" in summary
        assert "File Processing:" in summary
        assert "Search:" in summary
        assert "Embedding:" in summary
        assert "Server:" in summary
        assert "Logging:" in summary
        assert "Validation Status:" in summary

        # Check that some key values are included
        assert "Memory Limit:" in summary
        assert "Max File Size:" in summary
        assert "Model:" in summary

    def test_global_config_instance(self):
        """Test that global config instance is properly initialized."""
        from config import config as global_config

        # Check by class name to avoid isinstance issues with reloaded modules
        assert type(global_config).__name__ == "Config", f"Expected Config, got {type(global_config).__name__}"
        assert global_config.validate() is True


class TestEnvironmentVariableHandling:
    """Test environment variable parsing and handling."""

    def test_environment_variable_parsing(self):
        """Test that environment variables are properly parsed."""
        # Test with custom environment variables
        env_vars = {
            "TURBOPROP_DB_THREADS": "8",
            "TURBOPROP_MAX_FILE_SIZE_MB": "2.5",
            "TURBOPROP_LOG_LEVEL": "DEBUG",
            "TURBOPROP_DEVICE": "cuda",
            "TURBOPROP_DB_AUTO_VACUUM": "false",
        }

        with patch.dict(os.environ, env_vars):
            # Need to reload the module to pick up new env vars
            import importlib

            import config

            importlib.reload(config)

            # Check that environment variables are used
            assert config.DatabaseConfig.THREADS == 8
            assert config.FileProcessingConfig.MAX_FILE_SIZE_MB == 2.5
            assert config.LoggingConfig.LOG_LEVEL == "DEBUG"
            assert config.EmbeddingConfig.DEVICE == "cuda"
            assert config.DatabaseConfig.AUTO_VACUUM is False

    def test_invalid_environment_variables(self):
        """Test handling of invalid environment variables."""
        test_cases = [
            ("TURBOPROP_DB_THREADS", "invalid"),
            ("TURBOPROP_MAX_FILE_SIZE_MB", "not_a_number"),
            ("TURBOPROP_LOG_LEVEL", "INVALID_LEVEL"),
            ("TURBOPROP_DEVICE", "invalid_device"),
            ("TURBOPROP_DB_AUTO_VACUUM", "maybe"),
        ]

        for env_var, invalid_value in test_cases:
            with patch.dict(os.environ, {env_var: invalid_value}):
                # This should fail during module reload
                import importlib

                import config

                try:
                    importlib.reload(config)
                    # If we get here, the test should fail
                    assert False, f"Expected ConfigValidationError to be raised " f"for {env_var}={invalid_value}"
                except Exception as e:
                    # Check that it's the right type of exception by name
                    # (to avoid isinstance issues with reloaded modules)
                    assert type(e).__name__ == "ConfigValidationError", (
                        f"Expected ConfigValidationError, got {type(e).__name__} " f"for {env_var}={invalid_value}"
                    )


class TestBoundaryConditions:
    """Test boundary conditions and edge cases."""

    def test_boundary_values(self):
        """Test boundary values for various validation functions."""
        # Test edge cases for positive integers
        assert validate_positive_int("1", "TEST", 1) == 1

        # Test edge cases for floats
        assert validate_positive_float("0.000001", "TEST", 1.0) == 0.000001
        assert validate_non_negative_float("0.0", "TEST", 1.0) == 0.0

        # Test edge cases for ranges
        assert validate_range_float("0.0", "TEST", 0.0, 1.0) == 0.0
        assert validate_range_float("1.0", "TEST", 0.0, 1.0) == 1.0

    def test_whitespace_handling(self):
        """Test handling of whitespace in inputs."""
        # Memory limit should handle whitespace
        assert validate_memory_limit("1 GB", "TEST") == "1 GB"

        # Other validators should handle leading/trailing whitespace
        with patch.dict(os.environ, {"TURBOPROP_DB_THREADS": " 4 "}):
            import importlib

            import config

            importlib.reload(config)
            assert config.DatabaseConfig.THREADS == 4

    def test_very_large_numbers(self):
        """Test handling of very large numbers."""
        # Test large but valid numbers
        assert validate_positive_int("999999", "TEST", 1) == 999999
        assert validate_positive_float("999999.999", "TEST", 1.0) == 999999.999

        # Test memory limits with large numbers
        assert validate_memory_limit("999GB", "TEST") == "999GB"


class TestConfigurationIntegration:
    """Test configuration integration with other components."""

    def test_config_with_temporary_directory(self):
        """Test configuration with temporary directory paths."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Test database path generation
            db_path = DatabaseConfig.get_db_path(temp_path)
            expected_path = temp_path / ".turboprop" / "code_index.duckdb"
            assert db_path == expected_path

    def test_config_summary_formatting(self):
        """Test that configuration summary is properly formatted."""
        summary = Config.get_summary()

        # Check that summary is well-formatted
        lines = summary.split("\n")
        assert len(lines) > 10  # Should have multiple lines

        # Check for proper section headers
        section_headers = [line for line in lines if line.endswith(":") and not line.startswith("  ")]
        assert len(section_headers) >= 6  # Should have at least 6 main sections

    def test_config_validation_with_mock_errors(self):
        """Test configuration validation with mocked validation errors."""
        with patch.object(Config, "validate") as mock_validate:
            mock_validate.side_effect = ConfigValidationError("Mocked error")

            # Should propagate the validation error
            with pytest.raises(ConfigValidationError, match="Mocked error"):
                Config.validate()

    def test_config_get_validation_status_with_errors(self):
        """Test get_validation_status with validation errors."""
        # Import ConfigValidationError from config module to avoid namespace issues
        from config import ConfigValidationError as ConfigError

        with patch.object(Config, "validate") as mock_validate:
            mock_validate.side_effect = ConfigError("Test error")

            status = Config.get_validation_status()
            assert "❌ Configuration validation failed: Test error" in status
