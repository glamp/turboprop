#!/bin/bash
# Test local installation and uvx functionality

set -e

echo "🧪 Testing local turboprop installation"
echo "======================================="

# Test direct module execution
echo "📍 Testing python -m turboprop..."
python -m turboprop --version

echo "📍 Testing python -m turboprop mcp..."
python -m turboprop mcp --version

# Test entry point scripts (if installed)
if command -v turboprop &> /dev/null; then
    echo "📍 Testing turboprop command..."
    turboprop --version
fi

if command -v turboprop-mcp &> /dev/null; then
    echo "📍 Testing turboprop-mcp command..."
    turboprop-mcp --version
fi

# Test a quick index operation
echo "📍 Testing basic functionality..."
python -m turboprop index example-codebases/bashplotlib --max-mb 0.5
python -m turboprop search "histogram" --k 3

echo "✅ All tests passed!"
echo "Ready for uvx usage:"
echo "  uvx turboprop-mcp --repository /path/to/repo"