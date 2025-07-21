#!/bin/bash

# run-unit-tests.sh - Run all unit tests for the turboprop project
#
# This script runs the complete test suite with appropriate options
# for continuous integration and development workflows.

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🔍 Starting turboprop unit tests...${NC}"

# Ensure we're in the project root
cd "$(dirname "$0")/.."

# Check if pytest is available
if ! command -v pytest &> /dev/null; then
    echo -e "${RED}Error: pytest is not installed or not in PATH${NC}"
    echo "Please install pytest: pip install pytest"
    exit 1
fi

# Parse command line arguments
VERBOSE=""
COVERAGE=""
STOP_ON_FIRST_FAILURE=""
TEST_PATTERN="tests/"

while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--verbose)
            VERBOSE="-v"
            shift
            ;;
        -c|--coverage)
            COVERAGE="--cov=. --cov-report=html --cov-report=term"
            shift
            ;;
        -x|--stop-on-first)
            STOP_ON_FIRST_FAILURE="-x"
            shift
            ;;
        -p|--pattern)
            TEST_PATTERN="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  -v, --verbose        Verbose output"
            echo "  -c, --coverage       Run with coverage reporting"
            echo "  -x, --stop-on-first  Stop on first failure"
            echo "  -p, --pattern PATH   Run specific test pattern (default: tests/)"
            echo "  -h, --help           Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Run the tests
echo -e "${YELLOW}Running pytest with pattern: ${TEST_PATTERN}${NC}"

if [[ -n "$COVERAGE" ]]; then
    echo -e "${YELLOW}Running with coverage analysis...${NC}"
fi

# Build the pytest command
PYTEST_CMD="pytest $TEST_PATTERN $VERBOSE $COVERAGE $STOP_ON_FIRST_FAILURE --tb=short --disable-warnings"

echo -e "${YELLOW}Command: $PYTEST_CMD${NC}"
echo

# Run tests and capture exit code
$PYTEST_CMD
exit_code=$?

echo
if [ $exit_code -eq 0 ]; then
    echo -e "${GREEN}✅ All tests passed successfully!${NC}"
    if [[ -n "$COVERAGE" ]]; then
        echo -e "${BLUE}📊 Coverage report generated in htmlcov/index.html${NC}"
    fi
else
    echo -e "${RED}❌ Some tests failed (exit code: $exit_code)${NC}"
    echo -e "${YELLOW}💡 Tips for debugging:${NC}"
    echo "  • Run with -v for verbose output"
    echo "  • Run with -x to stop on first failure"
    echo "  • Run specific test pattern with -p tests/test_specific.py"
fi

echo -e "${BLUE}🏁 Test run completed.${NC}"
exit $exit_code