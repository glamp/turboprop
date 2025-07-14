#!/bin/bash
# Simple build script that works around CodeArtifact authentication issues

set -e

echo "🚀 Building Turboprop package (simplified)"
echo "=========================================="

# Clean previous builds
echo "🧹 Cleaning previous builds..."
rm -rf dist/ build/ *.egg-info/

# Create basic setup.py for fallback
echo "📦 Creating distribution..."

# Create the distribution using setuptools directly (not isolated)
python -c "
import setuptools
import os
import sys
from pathlib import Path

# Read version
try:
    with open('turboprop/_version.py') as f:
        exec(f.read())
except:
    __version__ = '0.1.0'

# Read long description
try:
    with open('README.md') as f:
        long_description = f.read()
except:
    long_description = 'Lightning-fast semantic code search and indexing with DuckDB vector operations'

setuptools.setup(
    name='turboprop',
    version=__version__,
    description='Lightning-fast semantic code search and indexing with DuckDB vector operations',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Greg Lamp',
    author_email='greg@example.com',
    url='https://github.com/glamp/turboprop',
    packages=['turboprop'],
    py_modules=['code_index', 'server', 'mcp_server'],
    python_requires='>=3.12',
    install_requires=[
        'duckdb>=1.3.2',
        'fastapi>=0.116.1', 
        'uvicorn>=0.35.0',
        'sentence-transformers>=5.0.0',
        'transformers>=4.53.2',
        'watchdog>=6.0.0',
        'mcp>=1.0.0',
    ],
    extras_require={
        'dev': ['pytest>=7.0.0', 'pytest-cov>=4.0.0'],
    },
    entry_points={
        'console_scripts': [
            'turboprop=code_index:main',
            'turboprop-mcp=mcp_server:main',
        ],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
        'Topic :: Software Development :: Libraries',
        'Topic :: Text Processing :: Indexing',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    keywords=['code', 'search', 'semantic', 'embedding', 'mcp', 'ai', 'indexing'],
)
"

# Build distributions
echo "🔨 Creating wheel and source distribution..."
python setup.py sdist bdist_wheel

echo "✅ Build complete!"
echo "📁 Files created in dist/:"
ls -la dist/

echo ""
echo "🚀 To publish to PyPI:"
echo "  pip install twine"
echo "  twine upload dist/*"