#!/usr/bin/env python3
"""
Entry point for running turboprop as a module with python -m turboprop.

This enables users to run:
- python -m turboprop index /path/to/repo
- python -m turboprop search "query"
- python -m turboprop watch /path/to/repo

Also supports running the MCP server with:
- python -m turboprop mcp --repository /path/to/repo
"""

import sys
import argparse
from pathlib import Path

def main():
    """Main entry point for module execution."""
    # Check if first argument is 'mcp' to route to MCP server
    if len(sys.argv) > 1 and sys.argv[1] == 'mcp':
        # Remove 'mcp' from args and run MCP server
        sys.argv.pop(1)
        from mcp_server import main as mcp_main
        mcp_main()
    else:
        # Run regular CLI
        from code_index import main as cli_main
        cli_main()

if __name__ == '__main__':
    main()