#!/bin/bash

# Build script for turboprop
# This script handles the basic build process including dependency installation and testing

set -e  # Exit on any error

echo "🔧 Building turboprop..."

# Clean previous build artifacts
echo "📁 Cleaning previous build artifacts..."
rm -rf build/ dist/ *.egg-info/

# Install dependencies directly
echo "📦 Installing dependencies..."
pip install duckdb sentence-transformers watchdog fastapi uvicorn mcp transformers

# Install development dependencies
echo "📦 Installing development dependencies..."
pip install pytest

# Run tests without coverage since pytest-cov has auth issues
echo "🧪 Running tests..."
pytest tests/

echo "✅ Build completed successfully!"