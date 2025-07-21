# Docker Test Suite for uvx MCP Installation

This directory contains a comprehensive Docker test suite that validates the `uvx turboprop@latest mcp --repository . --auto-index` command in an isolated environment.

## Purpose

The test suite ensures that the exact MCP configuration command used in production works correctly:

```json
"turboprop": {
  "command": "uvx",
  "args": ["turboprop@latest", "mcp", "--repository", ".", "--auto-index"],
  "env": {}
}
```

## Test Environment

- **Base Image:** Python 3.12-slim
- **Package Manager:** uvx (installed via pipx)
- **Test Repository:** `example-codebases/poker` (Next.js/TypeScript/React project)
- **Isolation:** Completely fresh Docker container with no pre-installed packages

## Files Structure

```
tests/docker/
├── Dockerfile                 # Test environment definition
├── docker-compose.yml         # Container orchestration
├── test_uvx_mcp.py           # Main installation tests
├── validate_mcp.py           # Functionality validation tests
├── run_tests.sh              # Test runner script
├── README.md                 # This documentation
└── test-results/             # Generated test outputs (created during runs)
    ├── test_report.json      # Installation test results
    ├── validation_report.json # Functionality test results
    ├── summary_report.md     # Combined test summary
    └── *.log                 # Various log files
```

## Test Components

### 1. Installation Tests (`test_uvx_mcp.py`)

Validates the core installation flow:

- ✅ **Repository Structure**: Verifies poker codebase is properly set up
- ✅ **uvx Availability**: Confirms uvx is installed and working
- ✅ **Turboprop Installation**: Tests `uvx turboprop@latest --version` (caches package)
- ✅ **MCP Server Startup**: Verifies `uvx turboprop mcp --repository . --auto-index` starts correctly

### 2. Functionality Validation (`validate_mcp.py`)

Tests actual functionality after installation:

- 🔍 **Index Creation**: Tests `uvx turboprop index . --max-mb 1.0`
- 🔍 **Search Functionality**: Tests `uvx turboprop search "React component" --k 3`
- 📖 **Help Command**: Tests `uvx turboprop mcp --help`
- 🗄️ **Database Integrity**: Verifies search works with created index
- 📁 **File Coverage**: Confirms expected files are indexed (TypeScript, components, etc.)

## Performance Optimizations

The test suite includes several key optimizations:

### uvx Caching Strategy
```python
# First run: uvx turboprop@latest --version (downloads and caches)
# Subsequent runs: uvx turboprop <command> (uses cached version)
```

This prevents re-downloading the ~94MB of ML dependencies (PyTorch, transformers, etc.) for each command.

### Timeout Configuration
- **Installation**: 300s (for initial ML dependency download)
- **Subsequent Operations**: 30-120s (using cached dependencies)
- **MCP Server Startup**: 180s (for model loading and auto-indexing)

## Usage

### Run Complete Test Suite
```bash
cd tests/docker
./run_tests.sh
```

### Run Individual Tests
```bash
# Installation tests only
docker-compose run --rm uvx-mcp-test

# Functionality validation only
docker-compose run --rm validate-mcp

# Build and run manually
docker-compose build
docker-compose run --rm uvx-mcp-test python /app/test_uvx_mcp.py
```

### View Results
```bash
# Summary report
cat test-results/summary_report.md

# Detailed results
cat test-results/test_report.json
cat test-results/validation_report.json

# Logs
cat test-results/test_run.log
cat test-results/mcp_stderr.log
```

## Expected Behavior

### Successful Run Output
```
🚀 Starting Turboprop uvx MCP Installation Test Suite
======================================================

✅ Installation tests PASSED
✅ Functionality validation PASSED

🎉 All tests passed! uvx MCP installation flow is working correctly.
```

### Test Results Interpretation

**Installation Tests (4 tests)**
- `repository_structure` → Poker codebase properly mounted
- `uvx_available` → uvx command accessible  
- `turboprop_installation` → Package installs without errors
- `mcp_server_startup` → MCP server starts and auto-indexes

**Functionality Validation (5 tests)**
- `index_creation` → CLI indexing creates proper DuckDB files
- `search_functionality` → Semantic search returns relevant results
- `mcp_help_command` → Help documentation accessible
- `database_integrity` → Created index is queryable
- `file_coverage` → Expected files (*.tsx, package.json) are indexed

## Troubleshooting

### Common Issues

**Timeout Errors**
- Increase timeout values in test scripts
- Check internet connectivity for package downloads
- Verify system has sufficient resources

**Docker Build Failures**  
- Ensure Docker daemon is running
- Clear Docker build cache: `docker system prune`
- Check disk space for image downloads

**uvx Command Not Found**
- Verify Dockerfile installs pipx correctly
- Check PATH includes `/root/.local/bin`
- Test uvx installation with `docker run python:3.12-slim uvx --version`

### Debugging Commands

```bash
# Interactive container debugging
docker-compose run --rm uvx-mcp-test bash

# Check uvx installation
docker-compose run --rm uvx-mcp-test uvx --version

# Manual turboprop test
docker-compose run --rm uvx-mcp-test uvx turboprop@latest --version

# Check repository contents
docker-compose run --rm uvx-mcp-test ls -la /test-repo
```

## Integration with CI/CD

This test suite is designed to be integrated into automated testing pipelines:

```yaml
# Example GitHub Actions workflow
- name: Test uvx MCP Installation
  run: |
    cd tests/docker
    ./run_tests.sh
  timeout-minutes: 15
```

## Development Notes

- **Test Isolation**: Each run uses a completely fresh container
- **Reproducibility**: Pinned to Python 3.12-slim for consistency  
- **Error Handling**: Comprehensive error collection and reporting
- **Performance**: Optimized for CI/CD with smart caching
- **Documentation**: Self-documenting test outputs and reports

The test suite provides confidence that the uvx MCP installation flow works correctly in production environments, especially when users configure turboprop as an MCP server in their Claude applications.