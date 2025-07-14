#!/bin/bash
# Build and publish turboprop to PyPI

set -e  # Exit on any error

echo "🚀 Building and publishing Turboprop to PyPI"
echo "============================================="

# Clean previous builds
echo "🧹 Cleaning previous builds..."
rm -rf dist/ build/ *.egg-info/

# Check if we're using uv or regular pip
if command -v uv &> /dev/null; then
    echo "📦 Using uv for build tools..."
    uv add --dev build twine
    BUILD_CMD="uv run python -m build"
    TWINE_CMD="uv run python -m twine"
else
    # Install build dependencies if needed
    echo "📦 Ensuring build tools are available..."
    python -m pip install --upgrade build twine
    BUILD_CMD="python -m build"
    TWINE_CMD="python -m twine"
fi

# Build the package
echo "🔨 Building package..."
$BUILD_CMD

# Check the built package
echo "🔍 Checking package..."
$TWINE_CMD check dist/*

# Upload to PyPI (will prompt for credentials)
echo "📤 Uploading to PyPI..."
echo "NOTE: You'll need to enter your PyPI credentials"
$TWINE_CMD upload dist/*

echo "✅ Package published successfully!"
echo "Users can now install with:"
echo "  pip install turboprop"
echo "  uvx turboprop-mcp --help"