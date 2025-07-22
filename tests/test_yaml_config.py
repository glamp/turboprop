"""
Unit tests for the yaml_config module.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from turboprop.yaml_config import (
    YAMLConfigError,
    _get_nested_value,
    create_sample_config,
    find_config_file,
    get_config_file_path,
    get_config_value,
    load_yaml_config,
    merge_configs,
    validate_yaml_structure,
)


class TestFindConfigFile:
    """Test the find_config_file function."""

    def test_find_config_file_exists(self):
        """Test finding config file when it exists."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = Path(temp_dir) / ".turboprop.yml"
            config_file.write_text("test: value")

            result = find_config_file(temp_dir)
            assert result == config_file

    def test_find_config_file_not_exists(self):
        """Test finding config file when it doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            result = find_config_file(temp_dir)
            assert result is None

    def test_find_config_file_default_directory(self):
        """Test finding config file in current directory."""
        with patch("turboprop.yaml_config.Path.cwd") as mock_cwd:
            mock_cwd.return_value = Path("/fake/path")
            with patch("pathlib.Path.exists", return_value=False):
                result = find_config_file()
                assert result is None


class TestLoadYAMLConfig:
    """Test the load_yaml_config function."""

    def test_load_valid_yaml(self):
        """Test loading valid YAML configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = Path(temp_dir) / ".turboprop.yml"
            config_content = """
database:
  threads: 8
  memory_limit: "2GB"
search:
  default_max_results: 10
            """
            config_file.write_text(config_content)

            result = load_yaml_config(temp_dir)

            assert result["database"]["threads"] == 8
            assert result["database"]["memory_limit"] == "2GB"
            assert result["search"]["default_max_results"] == 10

    def test_load_no_config_file(self):
        """Test loading when no config file exists."""
        with tempfile.TemporaryDirectory() as temp_dir:
            result = load_yaml_config(temp_dir)
            assert result == {}

    def test_load_invalid_yaml(self):
        """Test loading invalid YAML raises error."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = Path(temp_dir) / ".turboprop.yml"
            config_file.write_text("invalid: yaml: content: [")

            with pytest.raises(YAMLConfigError, match="Failed to parse YAML"):
                load_yaml_config(temp_dir)

    def test_load_empty_yaml(self):
        """Test loading empty YAML file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = Path(temp_dir) / ".turboprop.yml"
            config_file.write_text("")

            result = load_yaml_config(temp_dir)
            assert result == {}


class TestGetConfigValue:
    """Test the get_config_value function."""

    def test_get_nested_value_exists(self):
        """Test getting nested value that exists."""
        config = {"database": {"threads": 8, "settings": {"timeout": 30}}}

        assert get_config_value(config, "database.threads", None) == 8
        assert get_config_value(config, "database.settings.timeout", None) == 30

    def test_get_nested_value_not_exists(self):
        """Test getting nested value that doesn't exist."""
        config = {"database": {"threads": 8}}

        assert get_config_value(config, "database.memory", None) is None
        assert get_config_value(config, "search.max_results", None) is None

    def test_get_value_with_env_override(self):
        """Test getting value with environment variable override."""
        config = {"key": "yaml_value"}

        with patch.dict(os.environ, {"TEST_VAR": "env_value"}):
            result = get_config_value(config, "key", "default", "TEST_VAR")
            assert result == "env_value"  # env should override YAML

    def test_get_value_with_yaml_fallback(self):
        """Test getting value from YAML when no environment variable."""
        config = {"key": "yaml_value"}

        result = get_config_value(config, "key", "default", "UNSET_VAR")
        assert result == "yaml_value"  # should use YAML when env not set

    def test_get_value_with_default_fallback(self):
        """Test getting value with default fallback."""
        config = {}

        result = get_config_value(config, "missing.key", "default_value")
        assert result == "default_value"


class TestGetNestedValue:
    """Test the _get_nested_value helper function."""

    def test_empty_key_path(self):
        """Test with empty key path."""
        assert _get_nested_value({"key": "value"}, "") is None

    def test_single_key(self):
        """Test with single key."""
        data = {"key": "value"}
        assert _get_nested_value(data, "key") == "value"

    def test_nested_keys(self):
        """Test with nested keys."""
        data = {"level1": {"level2": {"level3": "deep_value"}}}
        assert _get_nested_value(data, "level1.level2.level3") == "deep_value"

    def test_missing_intermediate_key(self):
        """Test with missing intermediate key."""
        data = {"level1": {"level2": "value"}}
        assert _get_nested_value(data, "level1.missing.level3") is None

    def test_non_dict_intermediate_value(self):
        """Test with non-dict intermediate value."""
        data = {"level1": "not_a_dict"}
        assert _get_nested_value(data, "level1.level2") is None


class TestValidateYAMLStructure:
    """Test the validate_yaml_structure function."""

    def test_valid_structure(self):
        """Test validating valid YAML structure."""
        config = {"database": {"threads": 4}, "search": {"max_results": 5}, "embedding": {"model": "test"}}

        assert validate_yaml_structure(config) is True

    def test_invalid_root_type(self):
        """Test validating invalid root type."""
        assert validate_yaml_structure("not_a_dict") is False
        assert validate_yaml_structure(["list", "of", "items"]) is False

    def test_invalid_section_type(self):
        """Test validating invalid section type."""
        config = {"database": "not_a_dict", "search": {"max_results": 5}}

        assert validate_yaml_structure(config) is False

    def test_unknown_section_warning(self):
        """Test that unknown sections generate warnings but don't fail validation."""
        config = {"database": {"threads": 4}, "unknown_section": {"key": "value"}}

        # Should return True but log a warning
        assert validate_yaml_structure(config) is True


class TestMergeConfigs:
    """Test the merge_configs function."""

    def test_merge_simple_configs(self):
        """Test merging simple configurations."""
        yaml_config = {"key1": "yaml_value", "key2": "yaml_only"}
        env_config = {"key1": "env_value", "key3": "env_only"}

        result = merge_configs(yaml_config, env_config)

        # Environment should override YAML
        assert result["key1"] == "env_value"
        assert result["key2"] == "yaml_only"
        assert result["key3"] == "env_only"

    def test_merge_nested_configs(self):
        """Test merging nested configurations."""
        yaml_config = {"database": {"threads": 4, "memory": "1GB"}, "search": {"max_results": 5}}
        env_config = {"database": {"threads": 8}, "server": {"port": 8080}}

        result = merge_configs(yaml_config, env_config)

        # Should have merged database section
        assert result["database"]["threads"] == 8  # env override
        assert result["database"]["memory"] == "1GB"  # from yaml
        assert result["search"]["max_results"] == 5  # from yaml
        assert result["server"]["port"] == 8080  # from env


class TestCreateSampleConfig:
    """Test the create_sample_config function."""

    def test_create_sample_returns_string(self):
        """Test that create_sample_config returns a string."""
        result = create_sample_config()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_sample_contains_expected_sections(self):
        """Test that sample config contains expected sections."""
        result = create_sample_config()

        expected_sections = ["database:", "file_processing:", "search:", "embedding:", "server:", "logging:", "mcp:"]

        for section in expected_sections:
            assert section in result


class TestGetConfigFilePath:
    """Test the get_config_file_path function."""

    def test_default_directory(self):
        """Test getting config file path in default directory."""
        with patch("turboprop.yaml_config.Path.cwd") as mock_cwd:
            mock_cwd.return_value = Path("/current/dir")

            result = get_config_file_path()
            assert result == Path("/current/dir/.turboprop.yml")

    def test_specific_directory(self):
        """Test getting config file path in specific directory."""
        result = get_config_file_path("/custom/path")
        assert result == Path("/custom/path/.turboprop.yml")

    def test_path_object_directory(self):
        """Test getting config file path with Path object."""
        custom_path = Path("/custom/path")
        result = get_config_file_path(custom_path)
        assert result == Path("/custom/path/.turboprop.yml")


class TestYAMLConfigError:
    """Test the YAMLConfigError exception."""

    def test_yaml_config_error_inheritance(self):
        """Test that YAMLConfigError inherits from Exception."""
        error = YAMLConfigError("test message")
        assert isinstance(error, Exception)
        assert str(error) == "test message"


class TestIntegration:
    """Integration tests for the yaml_config module."""

    def test_full_config_loading_workflow(self):
        """Test the complete configuration loading workflow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a test config file
            config_file = Path(temp_dir) / ".turboprop.yml"
            config_content = """
database:
  threads: 6
  memory_limit: "1.5GB"
  auto_vacuum: true
search:
  default_max_results: 15
  min_similarity: 0.2
embedding:
  model: "custom-model"
  batch_size: 64
            """
            config_file.write_text(config_content)

            # Load the configuration
            config = load_yaml_config(temp_dir)

            # Test various access patterns
            assert get_config_value(config, "database.threads", 4) == 6
            assert get_config_value(config, "database.memory_limit", "1GB") == "1.5GB"
            assert get_config_value(config, "database.auto_vacuum", False) is True
            assert get_config_value(config, "search.default_max_results", 5) == 15
            assert get_config_value(config, "search.min_similarity", 0.1) == 0.2
            assert get_config_value(config, "embedding.model", "default") == "custom-model"
            assert get_config_value(config, "embedding.batch_size", 32) == 64

            # Test non-existent keys
            assert get_config_value(config, "nonexistent.key", "default") == "default"
            assert get_config_value(config, "database.nonexistent", None) is None
