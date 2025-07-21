import json
import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class ResponseValidationError(Exception):
    """Custom exception for response validation errors"""

    pass


class MCPResponseValidator:
    """Validate MCP response formats and content"""

    def __init__(self):
        self.schemas = self._load_response_schemas()
        self.validation_rules = self._load_validation_rules()

    def validate_response(self, response: Dict[str, Any], tool_name: str) -> Dict[str, Any]:
        """Validate response format and content"""

        validation_errors = []
        validation_warnings = []

        # Check required fields
        required_fields = self._get_required_fields(tool_name)
        missing_fields = []
        for field in required_fields:
            if field not in response:
                missing_fields.append(field)
                validation_errors.append(f"Missing required field: {field}")

        # Validate field types
        type_errors = self._validate_field_types(response, tool_name)
        validation_errors.extend(type_errors)

        # Validate field values
        value_errors = self._validate_field_values(response, tool_name)
        validation_errors.extend(value_errors)

        # Check response structure
        structure_errors = self._validate_response_structure(response, tool_name)
        validation_errors.extend(structure_errors)

        # Validate JSON serialization
        serialization_errors = self._validate_json_serialization(response)
        validation_errors.extend(serialization_errors)

        # Performance and size validations (warnings only)
        performance_warnings = self._validate_performance_metrics(response)
        validation_warnings.extend(performance_warnings)

        # Content quality validations
        content_warnings = self._validate_content_quality(response, tool_name)
        validation_warnings.extend(content_warnings)

        # Add validation metadata
        response["validation"] = {
            "valid": len(validation_errors) == 0,
            "errors": validation_errors,
            "warnings": validation_warnings,
            "error_count": len(validation_errors),
            "warning_count": len(validation_warnings),
            "validated_at": time.strftime("%Y-%m-%d %H:%M:%S UTC"),
            "validator_version": "1.0",
            "tool_name": tool_name,
            "validation_duration_ms": 0,  # Will be filled by timing wrapper
        }

        # Log validation issues
        if validation_errors:
            logger.error(f"Response validation errors for {tool_name}: {validation_errors}")
        if validation_warnings:
            logger.warning(f"Response validation warnings for {tool_name}: {validation_warnings}")

        return response

    def _get_required_fields(self, tool_name: str) -> List[str]:
        """Get required fields for tool response"""
        base_fields = ["success", "tool", "timestamp", "version", "response_id"]

        tool_specific = {
            "search_mcp_tools": ["query", "results", "total_results"],
            "get_tool_details": ["tool_id", "tool_details"],
            "recommend_tools_for_task": ["task_description", "recommendations"],
            "compare_mcp_tools": ["tool_ids", "comparison_result"],
            "analyze_task_requirements": ["task_description", "analysis_result"],
        }

        return base_fields + tool_specific.get(tool_name, [])

    def _validate_field_types(self, response: Dict[str, Any], tool_name: str) -> List[str]:
        """Validate field data types"""
        errors = []

        # Standard field type validations
        type_rules = {
            "success": bool,
            "tool": str,
            "timestamp": str,
            "version": str,
            "response_id": str,
            "total_results": int,
            "query": str,
            "task_description": str,
        }

        for field, expected_type in type_rules.items():
            if field in response:
                value = response[field]
                if not isinstance(value, expected_type):
                    errors.append(f"Field '{field}' should be {expected_type.__name__}, got {type(value).__name__}")

        # Validate list fields
        list_fields = ["results", "recommendations", "tool_ids"]
        for field in list_fields:
            if field in response:
                value = response[field]
                if not isinstance(value, list):
                    errors.append(f"Field '{field}' should be a list, got {type(value).__name__}")

        # Validate dict fields
        dict_fields = ["tool_details", "comparison_result", "analysis_result", "performance"]
        for field in dict_fields:
            if field in response:
                value = response[field]
                if not isinstance(value, dict):
                    errors.append(f"Field '{field}' should be a dict, got {type(value).__name__}")

        return errors

    def _validate_field_values(self, response: Dict[str, Any], tool_name: str) -> List[str]:
        """Validate field value constraints"""
        errors = []

        # Validate success field
        if "success" in response:
            success = response["success"]
            if success not in [True, False]:
                errors.append("Field 'success' must be boolean True or False")

        # Validate timestamp format
        if "timestamp" in response:
            timestamp = response["timestamp"]
            if not self._is_valid_timestamp(timestamp):
                errors.append("Field 'timestamp' must be in valid UTC format")

        # Validate version format
        if "version" in response:
            version = response["version"]
            if not self._is_valid_version(version):
                errors.append("Field 'version' must be in semantic version format (e.g., '1.0')")

        # Validate tool name
        if "tool" in response:
            tool = response["tool"]
            if tool != tool_name:
                errors.append(f"Field 'tool' value '{tool}' doesn't match expected '{tool_name}'")

        # Validate result counts
        if "total_results" in response:
            total = response["total_results"]
            if total < 0:
                errors.append("Field 'total_results' cannot be negative")

            # Check consistency with actual results
            if "results" in response:
                actual_count = len(response["results"])
                if total != actual_count:
                    errors.append(
                        f"Field 'total_results' ({total}) doesn't match actual results count ({actual_count})"
                    )

        return errors

    def _validate_response_structure(self, response: Dict[str, Any], tool_name: str) -> List[str]:
        """Validate response structure and nested objects"""
        errors = []

        # Validate results structure
        if "results" in response:
            results = response["results"]
            for i, result in enumerate(results):
                if not isinstance(result, dict):
                    errors.append(f"Result item {i} must be a dictionary")
                    continue

                # Check for required result fields
                required_result_fields = self._get_required_result_fields(tool_name)
                for field in required_result_fields:
                    if field not in result:
                        errors.append(f"Result item {i} missing required field: {field}")

        # Validate recommendations structure
        if "recommendations" in response:
            recommendations = response["recommendations"]
            for i, rec in enumerate(recommendations):
                if not isinstance(rec, dict):
                    errors.append(f"Recommendation item {i} must be a dictionary")
                    continue

                # Check for required recommendation fields
                required_rec_fields = ["tool_id", "confidence_score", "reasoning"]
                for field in required_rec_fields:
                    if field not in rec:
                        errors.append(f"Recommendation item {i} missing required field: {field}")

        # Validate comparison structure
        if "comparison_result" in response:
            comparison = response["comparison_result"]
            required_comp_fields = ["tools", "comparison_matrix", "summary"]
            for field in required_comp_fields:
                if field not in comparison:
                    errors.append(f"Comparison result missing required field: {field}")

        return errors

    def _validate_json_serialization(self, response: Dict[str, Any]) -> List[str]:
        """Validate that response can be JSON serialized"""
        errors = []

        try:
            json.dumps(response)
        except TypeError as e:
            errors.append(f"JSON serialization error: {str(e)}")
        except ValueError as e:
            errors.append(f"JSON value error: {str(e)}")
        except Exception as e:
            errors.append(f"Unexpected serialization error: {str(e)}")

        return errors

    def _validate_performance_metrics(self, response: Dict[str, Any]) -> List[str]:
        """Validate performance-related metrics (generates warnings)"""
        warnings = []

        # Check response size
        try:
            response_size = len(json.dumps(response))
            if response_size > 100000:  # 100KB
                warnings.append(f"Response size ({response_size} bytes) is very large and may impact performance")
            elif response_size > 50000:  # 50KB
                warnings.append(f"Response size ({response_size} bytes) is large")
        except:
            warnings.append("Could not calculate response size")

        # Check execution time if available
        if "performance" in response and "execution_time" in response["performance"]:
            exec_time = response["performance"]["execution_time"]
            if isinstance(exec_time, (int, float)):
                if exec_time > 5.0:
                    warnings.append(f"Execution time ({exec_time:.2f}s) exceeds recommended 5 second threshold")
                elif exec_time > 2.0:
                    warnings.append(f"Execution time ({exec_time:.2f}s) is slower than optimal")

        # Check result count efficiency
        if "results" in response:
            result_count = len(response["results"])
            if result_count > 100:
                warnings.append(f"Large result count ({result_count}) may impact processing performance")

        return warnings

    def _validate_content_quality(self, response: Dict[str, Any], tool_name: str) -> List[str]:
        """Validate content quality and completeness (generates warnings)"""
        warnings = []

        # Check for empty results
        if "results" in response:
            results = response["results"]
            if len(results) == 0 and response.get("success", True):
                warnings.append("Successful response has no results")

            # Check for incomplete result objects
            for i, result in enumerate(results):
                if isinstance(result, dict):
                    if "description" in result and not result["description"]:
                        warnings.append(f"Result item {i} has empty description")
                    if "name" in result and not result["name"]:
                        warnings.append(f"Result item {i} has empty name")

        # Check for missing navigation hints
        if "navigation" not in response:
            warnings.append("Response missing navigation hints for better user experience")

        # Check for missing performance metadata
        if "performance" not in response:
            warnings.append("Response missing performance metadata")

        return warnings

    def _get_required_result_fields(self, tool_name: str) -> List[str]:
        """Get required fields for individual result items"""
        base_fields = ["id", "name"]

        tool_specific = {
            "search_mcp_tools": ["description", "category"],
            "get_tool_details": ["version", "description"],
            "recommend_tools_for_task": ["confidence_score", "reasoning"],
            "compare_mcp_tools": ["category", "features"],
        }

        return base_fields + tool_specific.get(tool_name, [])

    def _is_valid_timestamp(self, timestamp: str) -> bool:
        """Check if timestamp is in valid UTC format"""
        try:
            datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S UTC")
            return True
        except ValueError:
            return False

    def _is_valid_version(self, version: str) -> bool:
        """Check if version string is valid"""
        if not isinstance(version, str):
            return False
        parts = version.split(".")
        return len(parts) >= 1 and all(part.isdigit() for part in parts)

    def _load_response_schemas(self) -> Dict[str, Any]:
        """Load response validation schemas"""
        # This would load from configuration files in a real implementation
        return {
            "base_schema": {
                "required": ["success", "tool", "timestamp", "version", "response_id"],
                "properties": {
                    "success": {"type": "boolean"},
                    "tool": {"type": "string"},
                    "timestamp": {"type": "string"},
                    "version": {"type": "string"},
                    "response_id": {"type": "string"},
                },
            }
        }

    def _load_validation_rules(self) -> Dict[str, Any]:
        """Load validation rules configuration"""
        return {
            "max_response_size": 100000,  # 100KB
            "max_execution_time": 5.0,  # 5 seconds
            "max_results": 100,
            "required_metadata": ["performance", "navigation"],
        }


def validate_mcp_response(func):
    """Decorator to automatically validate MCP tool responses"""

    def wrapper(*args, **kwargs):
        start_time = time.time()

        # Execute function and get response
        response = func(*args, **kwargs)

        # Validate response
        validator = MCPResponseValidator()
        validated_response = validator.validate_response(response, func.__name__)

        # Add validation timing
        validation_duration = (time.time() - start_time) * 1000  # Convert to milliseconds
        if "validation" in validated_response:
            validated_response["validation"]["validation_duration_ms"] = validation_duration

        return validated_response

    return wrapper
