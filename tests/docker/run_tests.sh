#!/bin/bash
# Test runner script for Docker-based uvx MCP installation testing.
#
# This script orchestrates the complete test suite for validating that:
# uvx turboprop@latest mcp --repository . --auto-index
#
# works correctly in an isolated environment.

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/test-results"
LOG_FILE="${RESULTS_DIR}/test_run.log"

echo_color() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

log_message() {
    local message=$1
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] $message" | tee -a "$LOG_FILE"
}

setup_test_environment() {
    echo_color $BLUE "Setting up test environment..."
    
    # Create results directory
    mkdir -p "$RESULTS_DIR"
    
    # Initialize log file
    echo "=== Turboprop uvx MCP Test Run ===" > "$LOG_FILE"
    echo "Started: $(date)" >> "$LOG_FILE"
    echo "" >> "$LOG_FILE"
    
    # Clean up any existing containers
    echo_color $YELLOW "Cleaning up existing containers..."
    docker-compose -f "$SCRIPT_DIR/docker-compose.yml" down --remove-orphans 2>/dev/null || true
    
    # Clean up any dangling images
    docker system prune -f >/dev/null 2>&1 || true
    
    log_message "Test environment setup complete"
}

run_installation_tests() {
    echo_color $BLUE "Running uvx MCP installation tests..."
    
    log_message "Building test container..."
    if ! docker-compose -f "$SCRIPT_DIR/docker-compose.yml" build uvx-mcp-test; then
        echo_color $RED "Failed to build test container"
        return 1
    fi
    
    log_message "Running installation tests..."
    if docker-compose -f "$SCRIPT_DIR/docker-compose.yml" run --rm uvx-mcp-test; then
        echo_color $GREEN "✅ Installation tests PASSED"
        return 0
    else
        echo_color $RED "❌ Installation tests FAILED"
        return 1
    fi
}

run_functionality_validation() {
    echo_color $BLUE "Running MCP functionality validation..."
    
    log_message "Running functionality validation..."
    if docker-compose -f "$SCRIPT_DIR/docker-compose.yml" run --rm validate-mcp; then
        echo_color $GREEN "✅ Functionality validation PASSED"
        return 0
    else
        echo_color $RED "❌ Functionality validation FAILED"
        return 1
    fi
}

cleanup_test_environment() {
    echo_color $YELLOW "Cleaning up test environment..."
    
    # Stop and remove containers
    docker-compose -f "$SCRIPT_DIR/docker-compose.yml" down --remove-orphans 2>/dev/null || true
    
    log_message "Test environment cleanup complete"
}

generate_summary_report() {
    local installation_result=$1
    local validation_result=$2
    
    echo_color $BLUE "Generating summary report..."
    
    # Create summary report
    local summary_file="${RESULTS_DIR}/summary_report.md"
    
    cat > "$summary_file" << EOF
# Turboprop uvx MCP Installation Test Summary

**Test Run Date:** $(date)

**Command Tested:** \`uvx turboprop@latest mcp --repository . --auto-index\`

## Test Results

### Installation Tests
Status: $([ $installation_result -eq 0 ] && echo "✅ PASSED" || echo "❌ FAILED")

Tests performed:
- uvx availability and functionality
- turboprop@latest package installation
- MCP server startup with auto-index
- Repository structure validation

### Functionality Validation
Status: $([ $validation_result -eq 0 ] && echo "✅ PASSED" || echo "❌ FAILED")

Validations performed:
- Index creation via CLI
- Search functionality testing
- MCP help command validation
- Database integrity checking
- File coverage verification

## Overall Result
$(if [ $installation_result -eq 0 ] && [ $validation_result -eq 0 ]; then
    echo "🎉 **ALL TESTS PASSED** - uvx MCP installation flow is working correctly"
elif [ $installation_result -eq 0 ]; then
    echo "⚠️ **PARTIAL SUCCESS** - Installation works, but functionality validation failed"
elif [ $validation_result -eq 0 ]; then
    echo "⚠️ **PARTIAL SUCCESS** - Validation passed, but installation tests failed"
else
    echo "🚨 **ALL TESTS FAILED** - uvx MCP installation flow has issues"
fi)

## Files Generated
- \`test_report.json\` - Detailed installation test results
- \`validation_report.json\` - Detailed functionality validation results
- \`mcp_stderr.log\` - MCP server output (if available)
- \`mcp_stdout.log\` - MCP server stdout (if available)
- \`test_run.log\` - Complete test execution log

## Test Environment
- Docker container with Python 3.12-slim
- Fresh uvx installation via pipx
- Poker example codebase (Next.js/TypeScript/React)
- Isolated network environment

---
*Generated by Turboprop Docker Test Suite*
EOF

    echo_color $GREEN "Summary report generated: $summary_file"
}

show_results() {
    echo_color $BLUE "Test execution complete. Results:"
    echo
    
    # Show summary if available
    if [ -f "${RESULTS_DIR}/summary_report.md" ]; then
        echo_color $YELLOW "Summary Report:"
        echo_color $YELLOW "=============="
        cat "${RESULTS_DIR}/summary_report.md"
        echo
    fi
    
    # Show available result files
    echo_color $YELLOW "Generated Files:"
    echo_color $YELLOW "==============="
    ls -la "${RESULTS_DIR}/" | grep -v "^total" | while read -r line; do
        echo "  $line"
    done
    
    echo
    echo_color $BLUE "To view detailed results:"
    echo "  cat ${RESULTS_DIR}/test_report.json"
    echo "  cat ${RESULTS_DIR}/validation_report.json"
    echo "  cat ${RESULTS_DIR}/test_run.log"
}

main() {
    echo_color $GREEN "🚀 Starting Turboprop uvx MCP Installation Test Suite"
    echo_color $GREEN "======================================================"
    echo
    
    # Check dependencies
    if ! command -v docker &> /dev/null; then
        echo_color $RED "Docker is required but not installed"
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        echo_color $RED "docker-compose is required but not installed"
        exit 1
    fi
    
    # Change to script directory for relative paths
    cd "$SCRIPT_DIR"
    
    local installation_result=1
    local validation_result=1
    
    # Setup
    setup_test_environment
    
    # Run installation tests
    if run_installation_tests; then
        installation_result=0
    fi
    
    # Run functionality validation (independent of installation test result)
    if run_functionality_validation; then
        validation_result=0
    fi
    
    # Generate summary
    generate_summary_report $installation_result $validation_result
    
    # Cleanup
    cleanup_test_environment
    
    # Show results
    show_results
    
    # Final exit status
    if [ $installation_result -eq 0 ] && [ $validation_result -eq 0 ]; then
        echo_color $GREEN "🎉 All tests passed! uvx MCP installation flow is working correctly."
        exit 0
    else
        echo_color $RED "❌ Some tests failed. Check the detailed reports for more information."
        exit 1
    fi
}

# Handle script interruption
trap cleanup_test_environment EXIT

# Run main function
main "$@"