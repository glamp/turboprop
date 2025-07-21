#!/usr/bin/env python3
"""
Documentation Validation Script

This script validates the consistency and completeness of the tool search documentation,
checking for broken links, consistent examples, and proper formatting.
"""

import re
import sys
from pathlib import Path


class DocumentationValidator:
    """Validates tool search documentation for consistency and completeness."""

    def __init__(self, docs_root: str):
        self.docs_root = Path(docs_root)
        self.errors = []
        self.warnings = []
        self.stats = {}

    def validate_all(self) -> bool:
        """Run all validation checks."""
        print("🔍 Validating Tool Search Documentation")
        print("=" * 50)

        # Check file structure
        self._validate_file_structure()

        # Check internal links
        self._validate_internal_links()

        # Check code examples
        self._validate_code_examples()

        # Check consistency
        self._validate_consistency()

        # Check completeness
        self._validate_completeness()

        # Report results
        self._report_results()

        return len(self.errors) == 0

    def _validate_file_structure(self):
        """Validate that all required files exist."""
        print("📁 Checking file structure...")

        required_files = [
            "README.md",
            "user_guide.md",
            "api_reference.md",
            "architecture.md",
            "migration_guide.md",
            "integration_guide.md",
            "examples/basic_usage.py",
            "examples/advanced_workflows.py",
            "examples/custom_integrations.py",
            "reference/mcp_tools_reference.md",
            "reference/search_algorithms.md",
            "reference/data_structures.md",
        ]

        missing_files = []
        for file_path in required_files:
            full_path = self.docs_root / file_path
            if not full_path.exists():
                missing_files.append(file_path)

        if missing_files:
            self.errors.append(f"Missing required files: {', '.join(missing_files)}")
        else:
            print("✅ All required files present")

        self.stats["total_files"] = len(required_files)
        self.stats["missing_files"] = len(missing_files)

    def _validate_internal_links(self):
        """Check for broken internal links."""
        print("🔗 Checking internal links...")

        md_files = list(self.docs_root.glob("**/*.md"))
        broken_links = []

        for md_file in md_files:
            content = md_file.read_text(encoding="utf-8")

            # Find markdown links
            link_pattern = r"\[([^\]]+)\]\(([^)]+)\)"
            links = re.findall(link_pattern, content)

            for link_text, link_url in links:
                if link_url.startswith(("#", "http", "https", "mailto")):
                    continue  # Skip anchors and external links

                # Resolve relative path
                target_path = md_file.parent / link_url
                target_path = target_path.resolve()

                if not target_path.exists():
                    broken_links.append(f"{md_file.name}: {link_text} -> {link_url}")

        if broken_links:
            self.errors.extend([f"Broken internal link: {link}" for link in broken_links])
        else:
            print("✅ All internal links valid")

        self.stats["broken_links"] = len(broken_links)

    def _validate_code_examples(self):
        """Validate Python code examples for syntax."""
        print("🐍 Checking code examples...")

        py_files = list(self.docs_root.glob("**/*.py"))
        syntax_errors = []

        for py_file in py_files:
            try:
                with open(py_file, "r", encoding="utf-8") as f:
                    content = f.read()

                # Basic syntax check (compile without execution)
                compile(content, str(py_file), "exec")

            except SyntaxError as e:
                syntax_errors.append(f"{py_file.name}: {e.msg} (line {e.lineno})")
            except Exception as e:
                self.warnings.append(f"{py_file.name}: Could not validate - {str(e)}")

        if syntax_errors:
            self.errors.extend([f"Syntax error: {error}" for error in syntax_errors])
        else:
            print("✅ All Python examples have valid syntax")

        # Check for consistent function signatures in examples
        self._check_function_signatures()

        self.stats["code_files"] = len(py_files)
        self.stats["syntax_errors"] = len(syntax_errors)

    def _check_function_signatures(self):
        """Check for consistent function signatures across examples."""
        print("📝 Checking function signature consistency...")

        # Expected function signatures (from API reference)
        expected_signatures = {
            "search_mcp_tools": ["query", "category", "tool_type", "max_results", "include_examples", "search_mode"],
            "recommend_tools_for_task": [
                "task_description",
                "context",
                "max_recommendations",
                "include_alternatives",
                "complexity_preference",
                "explain_reasoning",
            ],
            "compare_mcp_tools": [
                "tool_ids",
                "comparison_criteria",
                "include_decision_guidance",
                "comparison_context",
                "detail_level",
            ],
        }

        # Find function calls in Python examples
        py_files = list(self.docs_root.glob("**/*.py"))
        inconsistencies = []

        for py_file in py_files:
            content = py_file.read_text(encoding="utf-8")

            for func_name, expected_params in expected_signatures.items():
                # Find function calls (simplified pattern)
                pattern = rf"{func_name}\s*\([^)]*\)"
                calls = re.findall(pattern, content, re.DOTALL)

                for call in calls:
                    # Extract parameters (simplified)
                    param_pattern = r"(\w+)\s*="
                    used_params = re.findall(param_pattern, call)

                    # Check for unexpected parameters
                    unexpected = set(used_params) - set(expected_params)
                    if unexpected:
                        inconsistencies.append(f"{py_file.name}: {func_name} uses unexpected parameters: {unexpected}")

        if inconsistencies:
            self.warnings.extend(inconsistencies)
        else:
            print("✅ Function signatures consistent")

    def _validate_consistency(self):
        """Check for consistency across documentation."""
        print("🔄 Checking consistency...")

        # Check for consistent terminology
        self._check_terminology_consistency()

        # Check for consistent examples
        self._check_example_consistency()

    def _check_terminology_consistency(self):
        """Check for consistent use of terminology."""
        # Define canonical terms
        canonical_terms = {
            "MCP Tool Search System": ["tool search system", "Tool Search System", "mcp tool search"],
            "Claude Code": ["claude code", "Claude-Code", "claude-code"],
            "semantic search": ["Semantic Search", "semantic-search"],
            "hybrid search": ["Hybrid Search", "hybrid-search"],
        }

        md_files = list(self.docs_root.glob("**/*.md"))
        terminology_issues = []

        for md_file in md_files:
            content = md_file.read_text(encoding="utf-8")

            for canonical, variations in canonical_terms.items():
                for variation in variations:
                    if variation in content:
                        terminology_issues.append(f"{md_file.name}: Use '{canonical}' instead of '{variation}'")

        if terminology_issues:
            self.warnings.extend(terminology_issues)

        self.stats["terminology_issues"] = len(terminology_issues)

    def _check_example_consistency(self):
        """Check that examples are consistent across documents."""
        # Extract code examples from markdown files
        md_files = list(self.docs_root.glob("**/*.md"))
        code_examples = {}

        for md_file in md_files:
            content = md_file.read_text(encoding="utf-8")

            # Find code blocks
            code_pattern = r"```(?:python|bash|json)\n(.*?)\n```"
            blocks = re.findall(code_pattern, content, re.DOTALL)

            for block in blocks:
                if "search_mcp_tools" in block:
                    if "search_mcp_tools" not in code_examples:
                        code_examples["search_mcp_tools"] = []
                    code_examples["search_mcp_tools"].append((md_file.name, block))

        # Check for consistency in function calls
        inconsistencies = []
        for func_name, examples in code_examples.items():
            if len(examples) > 1:
                # Compare parameter usage patterns
                param_patterns = []
                for file_name, example in examples:
                    params = re.findall(r"(\w+)=", example)
                    param_patterns.append((file_name, set(params)))

                # Find inconsistencies
                if len(set(tuple(p[1]) for p in param_patterns)) > 1:
                    inconsistencies.append(f"{func_name} examples have inconsistent parameter usage")

        if inconsistencies:
            self.warnings.extend(inconsistencies)

    def _validate_completeness(self):
        """Check documentation completeness."""
        print("📋 Checking completeness...")

        # Check that all MCP tools mentioned in code have documentation
        self._check_tool_documentation_coverage()

        # Check for missing sections
        self._check_required_sections()

    def _check_tool_documentation_coverage(self):
        """Ensure all tools mentioned in examples are documented."""
        # Extract tool mentions from examples
        py_files = list(self.docs_root.glob("**/*.py"))
        mentioned_tools = set()

        for py_file in py_files:
            content = py_file.read_text(encoding="utf-8")

            # Find function calls
            tool_pattern = r"(search_mcp_tools|recommend_tools_for_task|compare_mcp_tools|get_tool_details)"
            if re.search(tool_pattern, content):
                mentioned_tools.add("search_tools")

        # Check if tools are documented in reference
        reference_file = self.docs_root / "reference" / "mcp_tools_reference.md"
        if reference_file.exists():
            reference_content = reference_file.read_text(encoding="utf-8")

            missing_docs = []
            for tool in mentioned_tools:
                if tool not in reference_content.lower():
                    missing_docs.append(tool)

            if missing_docs:
                self.warnings.extend([f"Tool missing from reference docs: {tool}" for tool in missing_docs])

    def _check_required_sections(self):
        """Check that required sections exist in main documents."""
        required_sections = {
            "user_guide.md": ["Overview", "Key Features", "Search Strategies", "Best Practices", "Troubleshooting"],
            "api_reference.md": [
                "Core MCP Tools",
                "search_mcp_tools",
                "recommend_tools_for_task",
                "Error Handling",
                "Performance Considerations",
            ],
            "architecture.md": ["Overview", "Core Components", "Data Flow", "Performance"],
        }

        missing_sections = []
        for file_name, sections in required_sections.items():
            file_path = self.docs_root / file_name
            if file_path.exists():
                content = file_path.read_text(encoding="utf-8")

                for section in sections:
                    if section not in content:
                        missing_sections.append(f"{file_name}: Missing section '{section}'")

        if missing_sections:
            self.warnings.extend(missing_sections)

        self.stats["missing_sections"] = len(missing_sections)

    def _report_results(self):
        """Report validation results."""
        print("\n" + "=" * 50)
        print("📊 Validation Results")
        print("=" * 50)

        # Summary statistics
        print(f"📁 Files checked: {self.stats.get('total_files', 0)}")
        print(f"🔗 Internal links: {self.stats.get('broken_links', 0)} broken")
        print(f"🐍 Code files: {self.stats.get('code_files', 0)}")
        print(f"⚠️  Terminology issues: {self.stats.get('terminology_issues', 0)}")

        # Errors
        if self.errors:
            print(f"\n❌ Errors ({len(self.errors)}):")
            for error in self.errors:
                print(f"   • {error}")

        # Warnings
        if self.warnings:
            print(f"\n⚠️  Warnings ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"   • {warning}")

        # Final status
        if not self.errors and not self.warnings:
            print("\n✅ Documentation validation passed with no issues!")
        elif not self.errors:
            print(f"\n⚠️  Documentation validation passed with {len(self.warnings)} warnings")
        else:
            print(
                f"\n❌ Documentation validation failed with {len(self.errors)} errors and {len(self.warnings)} warnings"
            )


def main():
    """Main validation function."""
    # Get documentation root directory
    script_dir = Path(__file__).parent
    docs_root = script_dir

    if not docs_root.exists():
        print(f"❌ Documentation directory not found: {docs_root}")
        sys.exit(1)

    # Run validation
    validator = DocumentationValidator(str(docs_root))
    success = validator.validate_all()

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
