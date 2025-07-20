#!/usr/bin/env python3
"""
conftest.py: Shared pytest fixtures and test utilities for comprehensive testing.

This module provides:
- Database fixtures with temporary DuckDB instances
- Mock embedder fixtures with consistent test vectors
- Sample repository fixtures with diverse code
- Performance baseline fixtures and utilities
- Cleanup utilities for test isolation
"""

import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import List
from unittest.mock import Mock

import numpy as np
import pytest

# Set environment variable to avoid tokenizers warning
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from config import config
from database_manager import DatabaseManager
from embedding_helper import EmbeddingGenerator

# Test data constants
SAMPLE_PYTHON_CODE = """
import json
from typing import List, Dict, Optional

class DataProcessor:
    \"\"\"Process and transform data.\"\"\"

    def __init__(self, config: Dict):
        self.config = config
        self.cache = {}

    def process_data(self, data: List[Dict]) -> List[Dict]:
        \"\"\"Process raw data into structured format.\"\"\"
        results = []
        for item in data:
            processed = self._transform_item(item)
            if processed:
                results.append(processed)
        return results

    def _transform_item(self, item: Dict) -> Optional[Dict]:
        \"\"\"Transform a single data item.\"\"\"
        if not self._validate_item(item):
            return None

        return {
            'id': item.get('id'),
            'name': item.get('name', '').strip(),
            'value': float(item.get('value', 0))
        }

    def _validate_item(self, item: Dict) -> bool:
        \"\"\"Validate data item structure.\"\"\"
        required_fields = ['id', 'name']
        return all(field in item for field in required_fields)

def parse_json_file(filepath: str) -> List[Dict]:
    \"\"\"Parse JSON file and return data.\"\"\"
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error parsing {filepath}: {e}")
        return []
"""

SAMPLE_JAVASCRIPT_CODE = """
// Authentication utility functions
const bcrypt = require('bcrypt');
const jwt = require('jsonwebtoken');

class AuthService {
    constructor(secretKey) {
        this.secretKey = secretKey;
        this.saltRounds = 10;
    }

    /**
     * Hash password with bcrypt
     * @param {string} password - Plain text password
     * @returns {Promise<string>} Hashed password
     */
    async hashPassword(password) {
        try {
            return await bcrypt.hash(password, this.saltRounds);
        } catch (error) {
            throw new Error(`Password hashing failed: ${error.message}`);
        }
    }

    /**
     * Verify password against hash
     * @param {string} password - Plain text password
     * @param {string} hash - Hashed password
     * @returns {Promise<boolean>} Verification result
     */
    async verifyPassword(password, hash) {
        try {
            return await bcrypt.compare(password, hash);
        } catch (error) {
            return false;
        }
    }

    /**
     * Generate JWT token
     * @param {Object} payload - Token payload
     * @param {string} expiresIn - Token expiration
     * @returns {string} JWT token
     */
    generateToken(payload, expiresIn = '24h') {
        return jwt.sign(payload, this.secretKey, { expiresIn });
    }

    /**
     * Verify JWT token
     * @param {string} token - JWT token to verify
     * @returns {Object|null} Decoded payload or null
     */
    verifyToken(token) {
        try {
            return jwt.verify(token, this.secretKey);
        } catch (error) {
            return null;
        }
    }
}

module.exports = AuthService;
"""

SAMPLE_GO_CODE = """
package main

import (
    "encoding/json"
    "fmt"
    "log"
    "net/http"
    "strconv"
    "github.com/gorilla/mux"
)

// User represents a user in the system
type User struct {
    ID    int    `json:"id"`
    Name  string `json:"name"`
    Email string `json:"email"`
}

// UserService handles user operations
type UserService struct {
    users map[int]User
    nextID int
}

// NewUserService creates a new user service
func NewUserService() *UserService {
    return &UserService{
        users:  make(map[int]User),
        nextID: 1,
    }
}

// CreateUser creates a new user
func (s *UserService) CreateUser(name, email string) User {
    user := User{
        ID:    s.nextID,
        Name:  name,
        Email: email,
    }
    s.users[user.ID] = user
    s.nextID++
    return user
}

// GetUser retrieves a user by ID
func (s *UserService) GetUser(id int) (User, bool) {
    user, exists := s.users[id]
    return user, exists
}

// GetAllUsers returns all users
func (s *UserService) GetAllUsers() []User {
    users := make([]User, 0, len(s.users))
    for _, user := range s.users {
        users = append(users, user)
    }
    return users
}

// HTTP handlers
func (s *UserService) handleCreateUser(w http.ResponseWriter, r *http.Request) {
    var req struct {
        Name  string `json:"name"`
        Email string `json:"email"`
    }

    if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
        http.Error(w, "Invalid JSON", http.StatusBadRequest)
        return
    }

    user := s.CreateUser(req.Name, req.Email)

    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(user)
}

func main() {
    service := NewUserService()

    r := mux.NewRouter()
    r.HandleFunc("/users", service.handleCreateUser).Methods("POST")

    fmt.Println("Server starting on :8080")
    log.Fatal(http.ListenAndServe(":8080", r))
}
"""


@pytest.fixture(scope="session")
def temp_root_dir():
    """Create a temporary root directory for all tests."""
    temp_dir = tempfile.mkdtemp(prefix="turboprop_test_")
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def temp_db_path(temp_root_dir):
    """Create a temporary database path."""
    db_path = temp_root_dir / "test.duckdb"
    yield db_path
    # Cleanup
    if db_path.exists():
        db_path.unlink()
    # Also cleanup WAL files
    wal_path = Path(str(db_path) + ".wal")
    if wal_path.exists():
        wal_path.unlink()


@pytest.fixture
def mock_db_manager(temp_db_path):
    """Create a real DatabaseManager with temporary database."""
    db_manager = DatabaseManager(temp_db_path)

    # Create the main table with full schema
    db_manager.execute_with_retry(
        f"""
        CREATE TABLE IF NOT EXISTS {config.database.TABLE_NAME} (
            id VARCHAR PRIMARY KEY,
            path VARCHAR,
            content TEXT,
            embedding DOUBLE[384],
            last_modified TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            file_mtime TIMESTAMP,
            file_type VARCHAR,
            language VARCHAR,
            size_bytes INTEGER,
            line_count INTEGER,
            category VARCHAR
        )
    """
    )

    # Create constructs table if needed
    db_manager.execute_with_retry(
        """
        CREATE TABLE IF NOT EXISTS code_constructs (
            id VARCHAR PRIMARY KEY,
            file_id VARCHAR,
            parent_construct_id VARCHAR,
            construct_type VARCHAR,
            name VARCHAR,
            signature TEXT,
            start_line INTEGER,
            end_line INTEGER,
            docstring TEXT,
            embedding DOUBLE[384],
            FOREIGN KEY (file_id) REFERENCES code_files(id)
        )
    """
    )

    yield db_manager
    db_manager.cleanup()


@pytest.fixture
def mock_embedder():
    """Create a mock EmbeddingGenerator with consistent test vectors that understand semantic similarity."""
    embedder = Mock(spec=EmbeddingGenerator)

    # Generate consistent embeddings based on input with semantic understanding
    def mock_encode(text):
        if isinstance(text, str):
            # Create embeddings that are similar for similar content
            # This creates semantic relationships for test scenarios
            import hashlib

            # Base embedding from text hash
            text_hash = int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16)
            np.random.seed(text_hash % 2**32)
            base_embedding = np.random.random(384)

            # Create semantic clusters for related terms
            text_lower = text.lower()

            # Define semantic groups that should have similar embeddings
            semantic_groups = {
                "functionality": ["functionality", "function", "feature", "capability", "method"],
                "new": ["new", "newly", "added", "recent", "latest"],
                "data": ["data", "processing", "process", "processor", "transform"],
                "auth": ["auth", "authentication", "login", "password", "security"],
                "search": ["search", "find", "query", "lookup", "discover"],
            }

            # Find which semantic group(s) this text belongs to
            group_weights = {}
            for group, keywords in semantic_groups.items():
                weight = sum(1.0 for keyword in keywords if keyword in text_lower)
                if weight > 0:
                    group_weights[group] = weight

            if group_weights:
                # Create embedding that's influenced by semantic groups
                # Use group-specific seeds to ensure similar embeddings for related content
                group_embedding = np.zeros(384)
                total_weight = sum(group_weights.values())

                for group, weight in group_weights.items():
                    group_hash = int(hashlib.md5(group.encode("utf-8")).hexdigest(), 16)
                    np.random.seed(group_hash % 2**32)
                    group_specific = np.random.random(384)
                    group_embedding += (weight / total_weight) * group_specific

                # Blend base embedding with semantic group embedding
                embedding = 0.3 * base_embedding + 0.7 * group_embedding
            else:
                embedding = base_embedding

            # Normalize to unit vector
            embedding = embedding / np.linalg.norm(embedding)
            return embedding  # Return numpy array, let the caller handle .tolist()
        elif isinstance(text, list):
            return [mock_encode(t) for t in text]
        return np.random.random(384)  # Return numpy array

    def mock_generate_embeddings(texts):
        return [mock_encode(text).tolist() for text in texts]

    embedder.encode.side_effect = mock_encode
    embedder.generate_embeddings.side_effect = mock_generate_embeddings

    return embedder


@pytest.fixture
def sample_repo(temp_root_dir):
    """Create a sample Git repository with diverse code files."""
    repo_path = temp_root_dir / "sample_repo"
    repo_path.mkdir(exist_ok=True)

    # Initialize git repository
    subprocess.run(["git", "init"], cwd=repo_path, capture_output=True, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo_path, capture_output=True, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo_path, capture_output=True, check=True)

    # Create Python file
    (repo_path / "data_processor.py").write_text(SAMPLE_PYTHON_CODE)

    # Create JavaScript file
    (repo_path / "auth.js").write_text(SAMPLE_JAVASCRIPT_CODE)

    # Create Go file
    (repo_path / "server.go").write_text(SAMPLE_GO_CODE)

    # Create README
    (repo_path / "README.md").write_text(
        """# Sample Repository

This is a sample repository for testing turboprop functionality.

## Features
- Data processing in Python
- Authentication service in JavaScript
- HTTP server in Go

## Usage
See individual files for usage examples.
"""
    )

    # Create configuration file
    (repo_path / "config.json").write_text(
        """{
    "database": {
        "host": "localhost",
        "port": 5432,
        "name": "testdb"
    },
    "logging": {
        "level": "info",
        "file": "app.log"
    }
}"""
    )

    # Create .gitignore
    (repo_path / ".gitignore").write_text(
        """
*.log
__pycache__/
node_modules/
.env
*.tmp
.turboprop/
*.duckdb
*.duckdb.wal
"""
    )

    # Add all files to git
    subprocess.run(["git", "add", "."], cwd=repo_path, capture_output=True, check=True)

    # Check if there's anything to commit
    status_result = subprocess.run(["git", "status", "--porcelain"], cwd=repo_path, capture_output=True, text=True)
    if status_result.returncode == 0 and status_result.stdout.strip():
        try:
            commit_result = subprocess.run(
                ["git", "commit", "-m", "Initial commit"], cwd=repo_path, capture_output=True, text=True
            )
            if commit_result.returncode != 0:
                # If commit fails, create a simple file and try again
                (repo_path / "README.md").write_text("# Test repository")
                subprocess.run(["git", "add", "README.md"], cwd=repo_path, capture_output=True)
                subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=repo_path, capture_output=True)
        except subprocess.CalledProcessError:
            # Create a minimal valid commit
            (repo_path / "README.md").write_text("# Test repository")
            subprocess.run(["git", "add", "README.md"], cwd=repo_path, capture_output=True)
            subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=repo_path, capture_output=True)
    else:
        # If no changes to commit, create at least one valid file
        (repo_path / "README.md").write_text("# Sample repository placeholder")
        subprocess.run(["git", "add", "README.md"], cwd=repo_path, capture_output=True)
        subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=repo_path, capture_output=True)

    yield repo_path


@pytest.fixture
def large_repo(temp_root_dir):
    """Create a larger repository for performance testing."""
    repo_path = temp_root_dir / "large_repo"
    repo_path.mkdir(exist_ok=True)

    # Initialize git repository
    subprocess.run(["git", "init"], cwd=repo_path, capture_output=True, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo_path, capture_output=True, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo_path, capture_output=True, check=True)

    # Create multiple directories with many files
    for dir_name in ["src", "tests", "utils", "api", "models"]:
        dir_path = repo_path / dir_name
        dir_path.mkdir(exist_ok=True)

        # Create 10 files per directory
        for i in range(10):
            file_content = f"""
# File {i} in {dir_name}
# This is a test file for performance testing

def function_{i}_in_{dir_name}():
    \"\"\"Function {i} in {dir_name}.\"\"\"
    result = []
    for j in range(100):
        result.append(f"item_{{j}}")
    return result

class Class{i}In{dir_name.capitalize()}:
    \"\"\"Class {i} in {dir_name}.\"\"\"

    def __init__(self):
        self.data = function_{i}_in_{dir_name}()

    def process(self, input_data):
        \"\"\"Process input data.\"\"\"
        processed = []
        for item in input_data:
            if isinstance(item, str):
                processed.append(item.upper())
        return processed

    def validate(self, data):
        \"\"\"Validate data format.\"\"\"
        return isinstance(data, list) and len(data) > 0

# Constants
CONSTANT_{str(i).upper()} = "value_{i}"
MAX_ITEMS_{str(i).upper()} = {i * 10}

# Helper functions
def helper_{i}(param):
    \"\"\"Helper function {i}.\"\"\"
    return param * 2 + {i}
"""
            (dir_path / f"file_{i}.py").write_text(file_content)

    # Create .gitignore
    (repo_path / ".gitignore").write_text(
        """
*.log
__pycache__/
.turboprop/
*.duckdb
*.duckdb.wal
"""
    )

    # Add all files to git
    subprocess.run(["git", "add", "."], cwd=repo_path, capture_output=True, check=True)

    # Check if there's anything to commit
    status_result = subprocess.run(["git", "status", "--porcelain"], cwd=repo_path, capture_output=True, text=True)
    if status_result.returncode == 0 and status_result.stdout.strip():
        try:
            commit_result = subprocess.run(
                ["git", "commit", "-m", "Initial large repo"], cwd=repo_path, capture_output=True, text=True
            )
            if commit_result.returncode != 0:
                # If commit fails, create a simple file and try again
                (repo_path / "README.md").write_text("# Large test repository")
                subprocess.run(["git", "add", "README.md"], cwd=repo_path, capture_output=True)
                subprocess.run(["git", "commit", "-m", "Initial large repo"], cwd=repo_path, capture_output=True)
        except subprocess.CalledProcessError:
            # Create a minimal valid commit
            (repo_path / "README.md").write_text("# Large test repository")
            subprocess.run(["git", "add", "README.md"], cwd=repo_path, capture_output=True)
            subprocess.run(["git", "commit", "-m", "Initial large repo"], cwd=repo_path, capture_output=True)
    else:
        # If no changes to commit, create at least one valid file
        (repo_path / "README.md").write_text("# Large repository placeholder")
        subprocess.run(["git", "add", "README.md"], cwd=repo_path, capture_output=True)
        subprocess.run(["git", "commit", "-m", "Initial large repo"], cwd=repo_path, capture_output=True)

    yield repo_path


@pytest.fixture
def corrupted_repo(temp_root_dir):
    """Create a repository with corrupted/problematic files."""
    repo_path = temp_root_dir / "corrupted_repo"
    repo_path.mkdir(exist_ok=True)

    # Initialize git repository
    subprocess.run(["git", "init"], cwd=repo_path, capture_output=True, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo_path, capture_output=True, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo_path, capture_output=True, check=True)

    # Create file with invalid UTF-8
    with open(repo_path / "binary.dat", "wb") as f:
        f.write(b"\x80\x81\x82\x83\x84\x85")  # Invalid UTF-8 bytes

    # Create extremely large text file (for memory testing)
    large_content = "x" * (1024 * 1024)  # 1MB of 'x'
    (repo_path / "large.txt").write_text(large_content)

    # Create file with unusual encoding
    with open(repo_path / "encoding_test.py", "w", encoding="latin1") as f:
        f.write("# -*- coding: latin1 -*-\n")
        f.write("# File with special characters: àáâãäåæçèéêë\n")
        f.write("def test_function():\n    return 'special chars: àáâãäåæçèéêë'\n")

    # Create file with very long lines
    long_line = "# " + "a" * 5000 + "\n"
    (repo_path / "long_lines.py").write_text(long_line + "def test(): pass\n")

    # Create empty file
    (repo_path / "empty.py").touch()

    # Create .gitignore
    (repo_path / ".gitignore").write_text(
        """
*.log
__pycache__/
.turboprop/
*.duckdb
*.duckdb.wal
"""
    )

    # Add files to git
    subprocess.run(["git", "add", "."], cwd=repo_path, capture_output=True, check=True)

    # Check if there's anything to commit
    status_result = subprocess.run(["git", "status", "--porcelain"], cwd=repo_path, capture_output=True, text=True)
    if status_result.returncode == 0 and status_result.stdout.strip():
        try:
            commit_result = subprocess.run(
                ["git", "commit", "-m", "Corrupted repo"], cwd=repo_path, capture_output=True, text=True
            )
            if commit_result.returncode != 0:
                # If commit fails, create a simple file and try again
                (repo_path / "README.md").write_text("# Test repository with problematic files")
                subprocess.run(["git", "add", "README.md"], cwd=repo_path, capture_output=True)
                subprocess.run(["git", "commit", "-m", "Corrupted repo"], cwd=repo_path, capture_output=True)
        except subprocess.CalledProcessError:
            # Create a minimal valid commit
            (repo_path / "README.md").write_text("# Test repository with problematic files")
            subprocess.run(["git", "add", "README.md"], cwd=repo_path, capture_output=True)
            subprocess.run(["git", "commit", "-m", "Corrupted repo"], cwd=repo_path, capture_output=True)
    else:
        # If no changes to commit, create at least one valid file
        (repo_path / "README.md").write_text("# Test repository with problematic files")
        subprocess.run(["git", "add", "README.md"], cwd=repo_path, capture_output=True)
        subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=repo_path, capture_output=True)

    yield repo_path


@pytest.fixture
def performance_baseline():
    """Performance baseline configuration and utilities."""
    return {
        "search_timeout": 5.0,  # seconds
        "indexing_timeout": 30.0,  # seconds
        "max_memory_mb": 900,  # MB (increased for test environment with additional features)
        "min_search_accuracy": 0.7,  # minimum relevance for search results
        "embedding_generation_rate": 100,  # embeddings per second minimum
    }


@pytest.fixture
def test_queries():
    """Standard test queries for consistent testing."""
    return {
        "simple": [
            "function",
            "class definition",
            "import statement",
            "error handling",
        ],
        "natural_language": [
            "how to authenticate users",
            "parse json data",
            "create http server",
            "handle database connections",
        ],
        "technical": [
            "async def authenticate",
            "class UserService",
            "import json",
            "try catch exception",
        ],
        "complex": [
            "authentication with jwt tokens and password hashing",
            "data processing pipeline with error handling",
            "http server with user management endpoints",
            "json parsing with validation and error recovery",
        ],
    }


@pytest.fixture
def sample_search_results():
    """Generate sample search results for testing."""
    from search_result_types import CodeSearchResult, CodeSnippet

    def create_results(count=5):
        results = []
        for i in range(count):
            snippet = CodeSnippet(
                text=f"def test_function_{i}():\n    return {i}", start_line=i * 10 + 1, end_line=i * 10 + 2
            )

            result = CodeSearchResult(
                file_path=f"/test/file_{i}.py",
                snippet=snippet,
                similarity_score=0.9 - (i * 0.1),
                confidence_level="high" if i < 2 else "medium",
            )
            results.append(result)

        return results

    return create_results


@pytest.fixture(autouse=True)
def cleanup_test_files():
    """Automatically cleanup test files after each test."""
    yield
    # Reset global database manager to ensure clean state between tests
    try:
        from code_index import reset_db

        reset_db()
    except ImportError:
        pass

    # Remove any temporary turboprop directories
    current_dir = Path.cwd()
    turboprop_dirs = list(current_dir.glob(".turboprop*"))
    for turboprop_dir in turboprop_dirs:
        if turboprop_dir.is_dir():
            shutil.rmtree(turboprop_dir, ignore_errors=True)


@pytest.fixture
def fully_mock_db_manager():
    """Create a fully mocked DatabaseManager for tests that need complete mocking."""
    mock_manager = Mock(spec=DatabaseManager)

    # Set up default mock behaviors
    mock_manager.execute_with_retry.return_value = []
    mock_manager.search_full_text.return_value = []
    mock_manager.get_connection.return_value = Mock()
    mock_manager.cleanup.return_value = None
    mock_manager.create_fts_index.return_value = None  # For hybrid search

    return mock_manager


@pytest.fixture
def mock_mcp_server():
    """Mock MCP server for integration testing."""
    from mcp_server import TurbopropServer

    server = Mock(spec=TurbopropServer)
    server.index_repository = Mock(return_value={"status": "success", "files": 10})
    server.search_code = Mock(return_value={"results": [], "total": 0})
    server.search_constructs = Mock(return_value={"results": [], "total": 0})

    return server


@pytest.fixture
def mock_ide_integration():
    """Mock IDE integration for testing."""
    from ide_integration import IDEIntegration

    integration = Mock(spec=IDEIntegration)
    integration.get_current_file = Mock(return_value="/test/current.py")
    integration.get_cursor_position = Mock(return_value=(10, 5))
    integration.navigate_to_location = Mock(return_value=True)

    return integration


class TestDataGenerator:
    """Utility class for generating test data."""

    @staticmethod
    def create_test_embedding(dimension=384, seed=None):
        """Create a test embedding vector."""
        if seed is not None:
            np.random.seed(seed)
        embedding = np.random.random(dimension)
        return embedding / np.linalg.norm(embedding)  # Normalize

    @staticmethod
    def create_test_files(repo_path: Path, languages: List[str], count_per_lang: int = 3):
        """Create test files for different programming languages."""
        templates = {
            "python": """
def {name}_function(param):
    \"\"\"Function for {name} operations.\"\"\"
    return param * 2

class {name}Class:
    \"\"\"Class for {name} operations.\"\"\"
    pass
""",
            "javascript": """
function {name}Function(param) {{
    // Function for {name} operations
    return param * 2;
}}

class {name}Class {{
    // Class for {name} operations
    constructor() {{
        this.value = 0;
    }}
}}
""",
            "go": """
package main

func {name}Function(param int) int {{
    // Function for {name} operations
    return param * 2
}}

type {name}Struct struct {{
    Value int
}}
""",
        }

        files = []
        for lang in languages:
            if lang in templates:
                for i in range(count_per_lang):
                    name = f"{lang}_{i}"
                    content = templates[lang].format(name=name.replace("_", "").title())

                    ext = {"python": ".py", "javascript": ".js", "go": ".go"}[lang]
                    file_path = repo_path / f"{name}{ext}"
                    file_path.write_text(content)
                    files.append(file_path)

        return files


@pytest.fixture
def test_data_generator():
    """Provide TestDataGenerator utility."""
    return TestDataGenerator


# Performance monitoring utilities
class PerformanceMonitor:
    """Utility class for monitoring test performance."""

    def __init__(self):
        self.start_time = None
        self.memory_start = None

    def start(self):
        """Start monitoring."""
        self.start_time = time.time()
        # Note: psutil would be needed for memory monitoring
        # For now, we'll use time-based monitoring only

    def stop(self):
        """Stop monitoring and return metrics."""
        if self.start_time is None:
            raise RuntimeError("Monitor not started")

        elapsed = time.time() - self.start_time
        return {
            "elapsed_time": elapsed,
            "memory_usage": None,  # Would need psutil
        }


@pytest.fixture
def performance_monitor():
    """Provide PerformanceMonitor utility."""
    return PerformanceMonitor()
