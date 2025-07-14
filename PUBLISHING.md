# Publishing Turboprop to PyPI

This guide explains how to publish Turboprop to PyPI for uvx/pip installation.

## Prerequisites

1. **PyPI Account**: Create account at https://pypi.org
2. **Clean Environment**: Temporarily disable CodeArtifact if using it
3. **API Token**: Generate PyPI API token for secure uploads

## Quick Publish (When CodeArtifact is disabled)

```bash
# Method 1: Use the publish script
./scripts/publish.sh

# Method 2: Manual steps
pip install build twine
python -m build
twine upload dist/*
```

## Alternative: Manual Build & Upload

If automated build fails due to environment issues:

```bash
# 1. Clean previous builds
rm -rf dist/ build/ *.egg-info/

# 2. Create setup.py temporarily
cat > setup.py << 'EOF'
from setuptools import setup, find_packages

# Read version
exec(open('turboprop/_version.py').read())

# Read long description
with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
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
EOF

# 3. Build and upload
python setup.py sdist bdist_wheel
pip install twine
twine upload dist/*

# 4. Cleanup
rm setup.py
```

## After Publishing

Once published to PyPI, users can:

```bash
# Install and use immediately (like npx)
uvx turboprop-mcp --repository /path/to/repo --auto-index

# Install globally
pip install turboprop
turboprop index .
turboprop-mcp --help

# Use in Claude Desktop MCP config
{
  "mcpServers": {
    "turboprop": {
      "command": "uvx",
      "args": ["turboprop-mcp", "--repository", "/absolute/path/to/repo", "--auto-index"]
    }
  }
}
```

## Testing Before Publishing

```bash
# Test local functionality
./scripts/test-local.sh

# Test package installation locally
pip install -e .
turboprop --version
turboprop-mcp --version
```

## Package Structure ✅

The package is correctly structured for PyPI:

- ✅ `pyproject.toml` with proper metadata
- ✅ `turboprop/` package with `__init__.py` and `__main__.py`
- ✅ Version management via `_version.py`
- ✅ Console scripts for both `turboprop` and `turboprop-mcp`
- ✅ Module execution support (`python -m turboprop`)
- ✅ MCP server routing via `python -m turboprop mcp`

The only blocker is the CodeArtifact authentication. Disable it temporarily to publish to public PyPI.