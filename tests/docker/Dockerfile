# Dockerfile for testing uvx turboprop MCP installation in isolated environment
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uvx using the official method (pipx-based installer)
RUN pip install --user pipx && \
    python -m pipx ensurepath && \
    export PATH="$PATH:/root/.local/bin" && \
    pipx install uv

# Ensure uvx is in PATH for all sessions
ENV PATH="/root/.local/bin:$PATH"

# Create test repository directory
RUN mkdir -p /test-repo

# Copy the poker example codebase to test directory
COPY example-codebases/poker/ /test-repo/

# Set working directory to test repo for MCP command
WORKDIR /test-repo

# Initialize git repo (required for turboprop)
RUN git init && \
    git config user.name "Test User" && \
    git config user.email "test@example.com" && \
    git add . && \
    git commit -m "Initial commit"

# Copy test scripts
COPY tests/docker/test_uvx_mcp.py /app/
COPY tests/docker/validate_mcp.py /app/

# Default command runs the test
CMD ["python", "/app/test_uvx_mcp.py"]