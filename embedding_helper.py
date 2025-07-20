#!/usr/bin/env python3
"""
embedding_helper.py: A simple, reliable module for generating embeddings that works
on Apple Silicon

This module provides a simple interface for generating embeddings using
SentenceTransformers
with proper handling of Apple Silicon MPS tensor issues.
"""

import sys
import time

from apple_silicon_compat import is_apple_silicon_device, setup_apple_silicon_compatibility

# Force CPU usage for PyTorch on Apple Silicon
setup_apple_silicon_compatibility()

# Imports must be placed after environment variables are set for Apple Silicon MPS
# compatibility
# This ensures PyTorch uses CPU backend instead of problematic MPS backend
import numpy as np  # noqa: E402

# torch import removed - not directly used in this module
from sentence_transformers import SentenceTransformer  # noqa: E402

# Force PyTorch to use CPU backend on Apple Silicon
from apple_silicon_compat import configure_torch_backend
from config import config  # noqa: E402

configure_torch_backend()


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
            device = "cpu" if is_apple_silicon_device() else config.embedding.DEVICE
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
                    print(
                        f"Processing batch {batch_num}/{total_batches} ({len(batch_texts)} texts)",
                        file=sys.stderr
                    )

                # Process this batch
                batch_embeddings = self.model.encode(batch_texts, show_progress_bar=False)
                batch_embeddings = np.array(batch_embeddings).astype(np.float32)
                all_embeddings.append(batch_embeddings)

            # Concatenate all batch results
            final_embeddings = np.concatenate(all_embeddings, axis=0)

            if show_progress:
                print(
                    f"✅ Completed processing {len(texts)} texts in {total_batches} batches",
                    file=sys.stderr
                )

            return final_embeddings

        except Exception as e:
            print(f"❌ Failed to encode large batch: {e}", file=sys.stderr)
            raise

    def encode_batch_with_retry(
        self, texts, batch_size=None, max_retries=None, show_progress=False
    ):
        """Generate embeddings with retry logic for better reliability"""
        if self.model is None:
            raise ValueError("Model not initialized")

        effective_batch_size = batch_size or config.embedding.BATCH_SIZE
        effective_max_retries = max_retries or config.embedding.MAX_RETRIES

        for attempt in range(effective_max_retries):
            try:
                return self.encode_batch(texts, effective_batch_size, show_progress)
            except Exception as e:
                if attempt == effective_max_retries - 1:
                    print(
                        f"❌ Failed to encode batch after {effective_max_retries} attempts: {e}",
                        file=sys.stderr
                    )
                    raise
                else:
                    print(
                        f"⚠️ Batch encoding attempt {attempt + 1} failed, retrying: {e}",
                        file=sys.stderr
                    )
                    # Exponential backoff
                    time.sleep(config.embedding.RETRY_BASE_DELAY * (2**attempt))

        # This should never be reached, but just in case
        raise RuntimeError(f"Failed to encode batch after {effective_max_retries} attempts")

    def generate_embeddings(self, texts, batch_size=None, show_progress=False):
        """
        Generate embeddings for a list of texts.
        
        This method provides API compatibility for components expecting a generate_embeddings method.
        
        Args:
            texts: List of text strings to encode
            batch_size: Optional batch size for processing
            show_progress: Whether to show progress for large batches
            
        Returns:
            List of numpy arrays, one embedding per input text
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # Use encode_batch to process the texts
        embeddings = self.encode_batch(texts, batch_size=batch_size, show_progress=show_progress)
        
        # Convert to list of individual embeddings for API compatibility
        return [embedding for embedding in embeddings]


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
