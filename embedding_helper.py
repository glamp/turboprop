#!/usr/bin/env python3
"""
embedding_helper.py: A simple, reliable module for generating embeddings that works
on Apple Silicon

This module provides a simple interface for generating embeddings using
SentenceTransformers
with proper handling of Apple Silicon MPS tensor issues.
"""

import os
import platform
import sys
import time

# Force CPU usage for PyTorch on Apple Silicon
is_apple_silicon = platform.processor() == "arm" or platform.machine() == "arm64"

if is_apple_silicon:
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
    os.environ["PYTORCH_DISABLE_MPS"] = "1"
    os.environ["PYTORCH_DEVICE"] = "cpu"

# Imports must be placed after environment variables are set for Apple Silicon MPS
# compatibility
# This ensures PyTorch uses CPU backend instead of problematic MPS backend
import numpy as np  # noqa: E402
import torch  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402

from config import config  # noqa: E402

# Force PyTorch to use CPU backend on Apple Silicon
if is_apple_silicon:
    torch.backends.mps.is_available = lambda: False
    torch.backends.mps.is_built = lambda: False
    torch.set_default_device("cpu")


class EmbeddingGenerator:
    """Simple, reliable embedding generator for Apple Silicon"""

    def __init__(self, model_name=None):
        """Initialize the embedding generator with CPU-only device"""
        self.model_name = model_name or config.embedding.EMBED_MODEL
        self.model = None
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the SentenceTransformer model with CPU device"""
        try:
            # Use configured device, but force CPU on Apple Silicon
            device = "cpu" if is_apple_silicon else config.embedding.DEVICE
            self.model = SentenceTransformer(self.model_name, device=device)
            # Ensure model is on the correct device
            self.model = self.model.to(device)
            print(f"✅ Initialized {self.model_name} on {device} device", file=sys.stderr)
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

    def encode_batch(self, texts, batch_size=None, show_progress=False):
        """Generate embeddings for a batch of texts with configurable batch size"""
        if self.model is None:
            raise ValueError("Model not initialized")

        # Use configured batch size if not provided
        effective_batch_size = batch_size or config.embedding.BATCH_SIZE

        # If input is small enough, process all at once
        if len(texts) <= effective_batch_size:
            return self._encode_single_batch(texts, show_progress)

        # Process in chunks for large collections
        return self._encode_large_batch(texts, effective_batch_size, show_progress)

    def _encode_single_batch(self, texts, show_progress=False):
        """Process a single batch of texts"""
        try:
            # Generate embeddings for all texts
            embeddings = self.model.encode(texts, show_progress_bar=show_progress)
            # Ensure it's a numpy array and convert to float32
            embeddings = np.array(embeddings).astype(np.float32)
            return embeddings
        except Exception as e:
            print(f"❌ Failed to encode batch: {e}", file=sys.stderr)
            raise

    def _encode_large_batch(self, texts, batch_size, show_progress=False):
        """Process a large collection of texts in chunks"""
        all_embeddings = []
        total_batches = (len(texts) + batch_size - 1) // batch_size

        try:
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i : i + batch_size]
                batch_num = (i // batch_size) + 1

                if show_progress:
                    print(f"Processing batch {batch_num}/{total_batches} ({len(batch_texts)} texts)", file=sys.stderr)

                # Process this batch
                batch_embeddings = self.model.encode(batch_texts, show_progress_bar=False)
                batch_embeddings = np.array(batch_embeddings).astype(np.float32)
                all_embeddings.append(batch_embeddings)

            # Concatenate all batch results
            final_embeddings = np.concatenate(all_embeddings, axis=0)

            if show_progress:
                print(f"✅ Completed processing {len(texts)} texts in {total_batches} batches", file=sys.stderr)

            return final_embeddings

        except Exception as e:
            print(f"❌ Failed to encode large batch: {e}", file=sys.stderr)
            raise

    def encode_batch_with_retry(self, texts, batch_size=None, max_retries=3, show_progress=False):
        """Generate embeddings with retry logic for better reliability"""
        if self.model is None:
            raise ValueError("Model not initialized")

        effective_batch_size = batch_size or config.embedding.BATCH_SIZE

        for attempt in range(max_retries):
            try:
                return self.encode_batch(texts, effective_batch_size, show_progress)
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"❌ Failed to encode batch after {max_retries} attempts: {e}", file=sys.stderr)
                    raise
                else:
                    print(f"⚠️ Batch encoding attempt {attempt + 1} failed, retrying: {e}", file=sys.stderr)
                    time.sleep(1.0 * (2**attempt))  # Exponential backoff

        # This should never be reached, but just in case
        raise RuntimeError(f"Failed to encode batch after {max_retries} attempts")


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
