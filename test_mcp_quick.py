#!/usr/bin/env python3
"""Quick test to verify MCP server is working with the correct model."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mcp_server import get_embedder, EMBED_MODEL, DIMENSIONS

def test_embedder():
    print(f"Testing embedder with model: {EMBED_MODEL}")
    print(f"Expected dimensions: {DIMENSIONS}")
    
    embedder = get_embedder()
    print(f"Loaded model successfully")
    
    # Test embedding
    test_text = "def hello_world():"
    embedding = embedder.encode([test_text])
    print(f"Embedding shape: {embedding.shape}")
    print(f"Embedding dimensions: {embedding.shape[1]}")
    
    if embedding.shape[1] == DIMENSIONS:
        print("✅ Embedder is working correctly!")
    else:
        print("❌ Dimension mismatch!")

if __name__ == "__main__":
    test_embedder()