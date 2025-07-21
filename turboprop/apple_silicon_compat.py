"""
Apple Silicon compatibility module for PyTorch MPS handling.

This module provides centralized configuration for Apple Silicon devices
to ensure proper PyTorch operation by forcing CPU usage when necessary.
"""

import os
import platform


def setup_apple_silicon_compatibility() -> None:
    """
    Configure environment variables for Apple Silicon PyTorch compatibility.

    This function sets up the necessary environment variables to force CPU usage
    for PyTorch on Apple Silicon devices to avoid MPS (Metal Performance Shaders)
    backend issues that can cause tensor compatibility problems.

    Must be called before any PyTorch imports to be effective.
    """
    is_apple_silicon = platform.processor() == "arm" or platform.machine() == "arm64"

    if is_apple_silicon:
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
        os.environ["PYTORCH_DISABLE_MPS"] = "1"
        os.environ["PYTORCH_DEVICE"] = "cpu"


def configure_torch_backend() -> None:
    """
    Configure PyTorch backend for Apple Silicon compatibility.

    This function should be called after PyTorch is imported to ensure
    proper backend configuration for Apple Silicon devices.
    """
    if is_apple_silicon_device():
        try:
            import torch

            torch.backends.mps.is_available = lambda: False
            torch.backends.mps.is_built = lambda: False
            torch.set_default_device("cpu")
        except ImportError:
            # PyTorch not available, skip configuration
            pass


def is_apple_silicon_device() -> bool:
    """
    Check if the current device is Apple Silicon.

    Returns:
        bool: True if running on Apple Silicon (ARM64), False otherwise.
    """
    return platform.processor() == "arm" or platform.machine() == "arm64"
