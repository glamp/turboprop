#!/usr/bin/env python3
"""
Git integration and repository context extraction for Turboprop.

This module provides functionality to extract git repository information,
detect project types, parse dependency files, and create comprehensive
repository context for better code understanding.
"""

import json
import hashlib
import re
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class RepositoryContext:
    """
    Repository context information including git data and project metadata.
    
    This class contains comprehensive information about a code repository
    that helps AI agents better understand the project structure and dependencies.
    """
    repository_id: str
    repository_path: str
    git_branch: Optional[str] = None
    git_commit: Optional[str] = None
    git_remote_url: Optional[str] = None
    project_type: Optional[str] = None
    dependencies: List[Dict[str, Any]] = field(default_factory=list)
    package_managers: List[str] = field(default_factory=list)
    indexed_at: Optional[datetime] = None
    created_at: Optional[datetime] = None

    def __post_init__(self):
        """Initialize timestamps if not provided."""
        if self.indexed_at is None:
            self.indexed_at = datetime.now()
        if self.created_at is None:
            self.created_at = datetime.now()

    @classmethod
    def compute_repository_id(cls, repository_path: str) -> str:
        """
        Compute a unique repository ID from the repository path.
        
        Args:
            repository_path: Absolute path to the repository
            
        Returns:
            SHA-256 hash of the repository path
        """
        return hashlib.sha256(repository_path.encode('utf-8')).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert repository context to dictionary for database storage.
        
        Returns:
            Dictionary representation suitable for database insertion
        """
        return {
            'repository_id': self.repository_id,
            'repository_path': self.repository_path,
            'git_branch': self.git_branch,
            'git_commit': self.git_commit,
            'git_remote_url': self.git_remote_url,
            'project_type': self.project_type,
            'dependencies': json.dumps(self.dependencies) if self.dependencies else None,
            'package_managers': json.dumps(self.package_managers) if self.package_managers else None,
            'indexed_at': self.indexed_at.isoformat() if self.indexed_at else None,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


class GitRepository:
    """
    Git repository information extraction utility.
    
    This class provides methods to extract git-specific information from
    a repository using subprocess calls to git commands.
    """
    
    def __init__(self, repo_path: Path):
        """
        Initialize GitRepository for the given path.
        
        Args:
            repo_path: Path to the repository (doesn't need to be git root)
        """
        self.repo_path = repo_path

    def is_git_repository(self) -> bool:
        """
        Check if the current path is within a git repository.
        
        Returns:
            True if this is a git repository, False otherwise
        """
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--is-inside-work-tree"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.returncode == 0 and result.stdout.strip() == "true"
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
            return False

    def get_repository_root(self) -> Optional[Path]:
        """
        Get the root directory of the git repository.
        
        Returns:
            Path to repository root, or None if not a git repository
        """
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=10,
                check=True
            )
            return Path(result.stdout.strip())
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError) as error:
            logger.debug("Failed to get git repository root: %s", error)
            return None

    def get_current_branch(self) -> Optional[str]:
        """
        Get the current git branch name.
        
        Returns:
            Current branch name, or None if extraction fails
        """
        try:
            result = subprocess.run(
                ["git", "branch", "--show-current"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=10,
                check=True
            )
            branch = result.stdout.strip()
            return branch if branch else None
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError) as error:
            logger.debug("Failed to get git branch: %s", error)
            return None

    def get_current_commit(self) -> Optional[str]:
        """
        Get the current git commit hash.
        
        Returns:
            Current commit hash (full SHA), or None if extraction fails
        """
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=10,
                check=True
            )
            return result.stdout.strip()
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError) as error:
            logger.debug("Failed to get git commit: %s", error)
            return None

    def get_remote_urls(self) -> Dict[str, str]:
        """
        Get git remote URLs.
        
        Returns:
            Dictionary mapping remote names to URLs
        """
        try:
            result = subprocess.run(
                ["git", "remote", "-v"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=10,
                check=True
            )
            
            remotes = {}
            for line in result.stdout.strip().split('\n'):
                if line and '\t' in line:
                    # Format: "remote_name\turl (fetch|push)"
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        remote_name = parts[0]
                        url_with_type = parts[1]
                        # Extract just the URL part (before the space and parentheses)
                        url = url_with_type.split(' ')[0]
                        # Only store fetch URLs to avoid duplicates
                        if '(fetch)' in url_with_type and remote_name not in remotes:
                            remotes[remote_name] = url
            
            return remotes
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError) as error:
            logger.debug("Failed to get git remotes: %s", error)
            return {}


class ProjectDetector:
    """
    Project type and dependency detection utility.
    
    This class analyzes a directory to detect the project type and extract
    dependency information from various package manager files.
    """
    
    # Project type detection patterns
    PROJECT_PATTERNS = {
        'python': [
            'pyproject.toml', 'setup.py', 'requirements.txt', 'Pipfile', 'setup.cfg'
        ],
        'javascript': [
            'package.json', 'package-lock.json', 'yarn.lock', '.nvmrc'
        ],
        'typescript': [
            'tsconfig.json', 'package.json'  # TypeScript often has package.json too
        ],
        'java': [
            'pom.xml', 'build.gradle', 'build.gradle.kts', 'gradle.properties'
        ],
        'go': [
            'go.mod', 'go.sum', 'Gopkg.toml', 'Gopkg.lock'
        ],
        'rust': [
            'Cargo.toml', 'Cargo.lock'
        ],
        'ruby': [
            'Gemfile', 'Gemfile.lock', '.ruby-version'
        ],
        'php': [
            'composer.json', 'composer.lock'
        ],
        'csharp': [
            '*.csproj', '*.sln', 'project.json', 'packages.config'
        ],
        'cpp': [
            'CMakeLists.txt', 'conanfile.txt', 'vcpkg.json'
        ]
    }

    def __init__(self, repo_path: Path):
        """
        Initialize ProjectDetector for the given path.
        
        Args:
            repo_path: Path to the repository to analyze
        """
        self.repo_path = repo_path

    def detect_project_type(self) -> Optional[str]:
        """
        Detect the primary project type based on files present.
        
        Returns:
            Detected project type string, or None if no clear type found
        """
        detected_types = []
        
        for project_type, patterns in self.PROJECT_PATTERNS.items():
            for pattern in patterns:
                if '*' in pattern:
                    # Handle glob patterns like *.csproj
                    matching_files = list(self.repo_path.glob(pattern))
                    if matching_files:
                        detected_types.append(project_type)
                        break
                else:
                    # Handle direct file matches
                    if (self.repo_path / pattern).exists():
                        detected_types.append(project_type)
                        break

        if not detected_types:
            return None
        
        # Return the first detected type (could be enhanced with scoring)
        return detected_types[0]

    def extract_dependencies(self) -> List[Dict[str, Any]]:
        """
        Extract dependency information from various package manager files.
        
        Returns:
            List of dependency dictionaries with name, version, source, and type
        """
        dependencies = []
        
        # Python dependencies
        dependencies.extend(self._extract_python_dependencies())
        
        # JavaScript dependencies
        dependencies.extend(self._extract_javascript_dependencies())
        
        # Java dependencies
        dependencies.extend(self._extract_java_dependencies())
        
        # Go dependencies
        dependencies.extend(self._extract_go_dependencies())
        
        # Rust dependencies
        dependencies.extend(self._extract_rust_dependencies())
        
        return dependencies

    def _extract_python_dependencies(self) -> List[Dict[str, Any]]:
        """Extract Python dependencies from various files."""
        dependencies = []
        
        # pyproject.toml
        pyproject_file = self.repo_path / "pyproject.toml"
        if pyproject_file.exists():
            try:
                import tomllib  # Python 3.11+
                with open(pyproject_file, 'rb') as f:
                    data = tomllib.load(f)
                
                # Extract from [project.dependencies]
                if 'project' in data and 'dependencies' in data['project']:
                    for dep_spec in data['project']['dependencies']:
                        parsed = self._parse_python_requirement(dep_spec)
                        parsed['source'] = 'pyproject.toml'
                        dependencies.append(parsed)
                        
            except (ImportError, OSError, ValueError) as error:
                logger.debug("Failed to parse pyproject.toml: %s", error)
                # Fallback to tomli for older Python versions
                try:
                    import tomli
                    with open(pyproject_file, 'rb') as f:
                        data = tomli.load(f)
                    
                    if 'project' in data and 'dependencies' in data['project']:
                        for dep_spec in data['project']['dependencies']:
                            parsed = self._parse_python_requirement(dep_spec)
                            parsed['source'] = 'pyproject.toml'
                            dependencies.append(parsed)
                except (ImportError, OSError, ValueError) as fallback_error:
                    logger.debug("Failed to parse pyproject.toml with tomli: %s", fallback_error)
        
        # requirements.txt
        requirements_file = self.repo_path / "requirements.txt"
        if requirements_file.exists():
            try:
                with open(requirements_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            parsed = self._parse_python_requirement(line)
                            parsed['source'] = 'requirements.txt'
                            dependencies.append(parsed)
            except OSError as error:
                logger.debug("Failed to read requirements.txt: %s", error)
        
        return dependencies

    def _parse_python_requirement(self, requirement: str) -> Dict[str, Any]:
        """Parse a Python requirement string into components."""
        # Simple parsing - could be enhanced with proper requirement parsing
        match = re.match(r'^([a-zA-Z0-9\-_.]+)(.*)$', requirement.strip())
        if match:
            name = match.group(1)
            version_spec = match.group(2).strip() if match.group(2) else None
            return {
                'name': name,
                'version': version_spec if version_spec else None
            }
        else:
            return {
                'name': requirement.strip(),
                'version': None
            }

    def _extract_javascript_dependencies(self) -> List[Dict[str, Any]]:
        """Extract JavaScript dependencies from package.json."""
        dependencies = []
        
        package_json = self.repo_path / "package.json"
        if package_json.exists():
            try:
                with open(package_json, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Production dependencies
                if 'dependencies' in data:
                    for name, version in data['dependencies'].items():
                        dependencies.append({
                            'name': name,
                            'version': version,
                            'source': 'package.json',
                            'type': 'production'
                        })
                
                # Development dependencies
                if 'devDependencies' in data:
                    for name, version in data['devDependencies'].items():
                        dependencies.append({
                            'name': name,
                            'version': version,
                            'source': 'package.json',
                            'type': 'development'
                        })
                        
            except (OSError, json.JSONDecodeError) as error:
                logger.debug("Failed to parse package.json: %s", error)
        
        return dependencies

    def _extract_java_dependencies(self) -> List[Dict[str, Any]]:
        """Extract Java dependencies from pom.xml or build.gradle."""
        dependencies = []
        
        # Maven pom.xml
        pom_file = self.repo_path / "pom.xml"
        if pom_file.exists():
            try:
                tree = ET.parse(pom_file)
                root = tree.getroot()
                
                # Maven XML namespace handling
                ns = {'maven': 'http://maven.apache.org/POM/4.0.0'}
                deps = root.findall('.//maven:dependency', ns) or root.findall('.//dependency')
                
                for dep in deps:
                    group_id = dep.find('groupId') or dep.find('maven:groupId', ns)
                    artifact_id = dep.find('artifactId') or dep.find('maven:artifactId', ns)
                    version = dep.find('version') or dep.find('maven:version', ns)
                    
                    if group_id is not None and artifact_id is not None:
                        dependencies.append({
                            'name': f"{group_id.text}:{artifact_id.text}",
                            'version': version.text if version is not None else None,
                            'source': 'pom.xml'
                        })
                        
            except (OSError, ET.ParseError) as error:
                logger.debug("Failed to parse pom.xml: %s", error)
        
        # Gradle build.gradle (basic text parsing)
        gradle_file = self.repo_path / "build.gradle"
        if gradle_file.exists():
            try:
                with open(gradle_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Basic regex to find dependencies
                dep_pattern = r'implementation\s+[\'"]([^:\'"]+):([^:\'"]+):([^\'"]+)[\'"]'
                matches = re.findall(dep_pattern, content)
                
                for group_id, artifact_id, version in matches:
                    dependencies.append({
                        'name': f"{group_id}:{artifact_id}",
                        'version': version,
                        'source': 'build.gradle'
                    })
                    
            except OSError as error:
                logger.debug("Failed to read build.gradle: %s", error)
        
        return dependencies

    def _extract_go_dependencies(self) -> List[Dict[str, Any]]:
        """Extract Go dependencies from go.mod."""
        dependencies = []
        
        go_mod = self.repo_path / "go.mod"
        if go_mod.exists():
            try:
                with open(go_mod, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Parse require block
                in_require = False
                for line in content.split('\n'):
                    line = line.strip()
                    
                    if line.startswith('require ('):
                        in_require = True
                        continue
                    elif line == ')' and in_require:
                        in_require = False
                        continue
                    
                    if in_require or line.startswith('require '):
                        # Handle single line or block format
                        if line.startswith('require '):
                            dep_line = line[8:]  # Remove 'require '
                        else:
                            dep_line = line
                        
                        parts = dep_line.split()
                        if len(parts) >= 2:
                            name = parts[0]
                            version = parts[1]
                            dependencies.append({
                                'name': name,
                                'version': version,
                                'source': 'go.mod'
                            })
                            
            except OSError as error:
                logger.debug("Failed to read go.mod: %s", error)
        
        return dependencies

    def _extract_rust_dependencies(self) -> List[Dict[str, Any]]:
        """Extract Rust dependencies from Cargo.toml."""
        dependencies = []
        
        cargo_toml = self.repo_path / "Cargo.toml"
        if cargo_toml.exists():
            try:
                import tomllib  # Python 3.11+
                with open(cargo_toml, 'rb') as f:
                    data = tomllib.load(f)
                
                # Extract dependencies
                if 'dependencies' in data:
                    for name, spec in data['dependencies'].items():
                        if isinstance(spec, str):
                            # Simple version specification
                            dependencies.append({
                                'name': name,
                                'version': spec,
                                'source': 'Cargo.toml'
                            })
                        elif isinstance(spec, dict):
                            # Complex specification with version key
                            version = spec.get('version')
                            dependencies.append({
                                'name': name,
                                'version': version,
                                'source': 'Cargo.toml'
                            })
                            
            except (ImportError, OSError, ValueError) as error:
                logger.debug("Failed to parse Cargo.toml: %s", error)
                # Fallback to tomli
                try:
                    import tomli
                    with open(cargo_toml, 'rb') as f:
                        data = tomli.load(f)
                    
                    if 'dependencies' in data:
                        for name, spec in data['dependencies'].items():
                            if isinstance(spec, str):
                                dependencies.append({
                                    'name': name,
                                    'version': spec,
                                    'source': 'Cargo.toml'
                                })
                            elif isinstance(spec, dict):
                                version = spec.get('version')
                                dependencies.append({
                                    'name': name,
                                    'version': version,
                                    'source': 'Cargo.toml'
                                })
                except (ImportError, OSError, ValueError) as fallback_error:
                    logger.debug("Failed to parse Cargo.toml with tomli: %s", fallback_error)
        
        return dependencies

    def get_package_managers(self) -> List[str]:
        """
        Detect package managers used in the project.
        
        Returns:
            List of detected package manager names
        """
        managers = []
        
        # Package manager file patterns
        manager_patterns = {
            'pip': ['requirements.txt', 'pyproject.toml', 'setup.py', 'Pipfile'],
            'npm': ['package.json', 'package-lock.json'],
            'yarn': ['yarn.lock'],
            'maven': ['pom.xml'],
            'gradle': ['build.gradle', 'build.gradle.kts'],
            'go': ['go.mod', 'go.sum'],
            'cargo': ['Cargo.toml', 'Cargo.lock'],
            'composer': ['composer.json', 'composer.lock'],
            'bundler': ['Gemfile', 'Gemfile.lock']
        }
        
        for manager, patterns in manager_patterns.items():
            for pattern in patterns:
                if (self.repo_path / pattern).exists():
                    managers.append(manager)
                    break
        
        return managers


class RepositoryContextExtractor:
    """
    Main class for extracting comprehensive repository context.
    
    This class orchestrates the extraction of git information and project
    metadata to create a complete repository context.
    """
    
    def extract_context(self, repo_path: Path) -> RepositoryContext:
        """
        Extract comprehensive repository context from the given path.
        
        Args:
            repo_path: Path to the repository to analyze
            
        Returns:
            RepositoryContext with all extracted information
        """
        repo_path = repo_path.resolve()  # Ensure absolute path
        repository_id = RepositoryContext.compute_repository_id(str(repo_path))
        
        # Initialize context with basic information
        context = RepositoryContext(
            repository_id=repository_id,
            repository_path=str(repo_path)
        )
        
        # Extract git information
        git_repo = GitRepository(repo_path)
        if git_repo.is_git_repository():
            # Update repo_path to git root if we're in a subdirectory
            git_root = git_repo.get_repository_root()
            if git_root:
                repo_path = git_root
                context.repository_path = str(repo_path)
                context.repository_id = RepositoryContext.compute_repository_id(str(repo_path))
                
                # Re-create GitRepository for the root path
                git_repo = GitRepository(repo_path)
            
            context.git_branch = git_repo.get_current_branch()
            context.git_commit = git_repo.get_current_commit()
            
            # Get primary remote URL (prefer origin)
            remotes = git_repo.get_remote_urls()
            if 'origin' in remotes:
                context.git_remote_url = remotes['origin']
            elif remotes:
                context.git_remote_url = next(iter(remotes.values()))
        
        # Extract project information
        detector = ProjectDetector(repo_path)
        context.project_type = detector.detect_project_type()
        context.dependencies = detector.extract_dependencies()
        context.package_managers = detector.get_package_managers()
        
        logger.debug(
            "Extracted repository context: type=%s, deps=%d, managers=%s",
            context.project_type, len(context.dependencies), context.package_managers
        )
        
        return context