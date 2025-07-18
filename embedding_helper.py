#!/usr/bin/env python3
"""
embedding_helper.py: A simple, reliable module for generating embeddings that works on Apple Silicon

This module provides a simple interface for generating embeddings using SentenceTransformers
with proper handling of Apple Silicon MPS tensor issues.
"""

import os
import platform
import sys

# Force CPU usage for PyTorch on Apple Silicon
is_apple_silicon = platform.processor() == "arm" or platform.machine() == "arm64"

if is_apple_silicon:
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
    os.environ["PYTORCH_DISABLE_MPS"] = "1"
    os.environ["PYTORCH_DEVICE"] = "cpu"

# Imports must be placed after environment variables are set for Apple Silicon MPS compatibility
# This ensures PyTorch uses CPU backend instead of problematic MPS backend
import numpy as np  # noqa: E402
import torch  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402

# Force PyTorch to use CPU backend on Apple Silicon
if is_apple_silicon:
    torch.backends.mps.is_available = lambda: False
    torch.backends.mps.is_built = lambda: False
    torch.set_default_device("cpu")


class EmbeddingGenerator:
    """Simple, reliable embedding generator for Apple Silicon"""

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """Initialize the embedding generator with CPU-only device"""
        self.model_name = model_name
        self.model = None
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the SentenceTransformer model with CPU device"""
        try:
            # Always use CPU device
            self.model = SentenceTransformer(self.model_name, device="cpu")
            # Ensure model is on CPU
            self.model = self.model.to("cpu")
            print(f"✅ Initialized {self.model_name} on CPU device", file=sys.stderr)
        except Exception as e:
            print(f"❌ Failed to initialize model: {e}", file=sys.stderr)
            raise

    def encode(self, text):
        """Generate embedding for text"""
        if self.model is None:
            raise ValueError("Model not initialized")

        try:
            # Generate embedding
            embedding = self.model.encode(text, show_progress_bar=False)
            # Ensure it's a numpy array and convert to float32
            embedding = np.array(embedding).astype(np.float32)
            return embedding
        except Exception as e:
            print(f"❌ Failed to encode text: {e}", file=sys.stderr)
            raise

    def encode_batch(self, texts):
        """Generate embeddings for a batch of texts"""
        if self.model is None:
            raise ValueError("Model not initialized")

        try:
            # Generate embeddings for all texts
            embeddings = self.model.encode(texts, show_progress_bar=False)
            # Ensure it's a numpy array and convert to float32
            embeddings = np.array(embeddings).astype(np.float32)
            return embeddings
        except Exception as e:
            print(f"❌ Failed to encode batch: {e}", file=sys.stderr)
            raise


def test_embedding_generator():
    """Test the embedding generator with various inputs"""
    print("🧪 Testing EmbeddingGenerator...")

    # Initialize generator
    generator = EmbeddingGenerator()

    # Test single text
    test_text = "This is a test sentence for embedding generation."
    print(f"📝 Testing single text: {test_text}")

    try:
        embedding = generator.encode(test_text)
        print(f"✅ Single text embedding shape: {embedding.shape}")
        print(f"✅ Single text embedding dtype: {embedding.dtype}")
        print(f"✅ Single text embedding first 5 values: {embedding[:5]}")
    except Exception as e:
        print(f"❌ Single text test failed: {e}")
        return False

    # Test batch of texts
    test_texts = [
        "This is the first test sentence.",
        "This is the second test sentence.",
        "This is the third test sentence.",
    ]
    print(f"📝 Testing batch of {len(test_texts)} texts")

    try:
        embeddings = generator.encode_batch(test_texts)
        print(f"✅ Batch embeddings shape: {embeddings.shape}")
        print(f"✅ Batch embeddings dtype: {embeddings.dtype}")
        print(f"✅ Batch embeddings first row first 5 values: {embeddings[0][:5]}")
    except Exception as e:
        print(f"❌ Batch test failed: {e}")
        return False

    print("🎉 All tests passed!")
    return True


if __name__ == "__main__":
    test_embedding_generator()
