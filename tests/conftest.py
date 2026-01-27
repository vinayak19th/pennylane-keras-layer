"""Configuration file for pytest."""

import pytest
import sys
import os


def pytest_addoption(parser):
    """Add custom command line options to pytest."""
    parser.addoption(
        "--backend",
        action="store",
        default=None,
        choices=["tensorflow", "torch", "jax"],
        help="Select Keras backend for testing: tensorflow, torch, or jax"
    )


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "backend(name): mark test to run with specific backend"
    )
    
    # Set the Keras backend based on command line option
    backend = config.getoption("--backend")
    if backend:
        os.environ["KERAS_BACKEND"] = backend
        print(f"\n=== Setting Keras backend to: {backend} ===")
        
        # Validate that the backend is available
        try:
            if backend == "tensorflow":
                import tensorflow  # noqa: F401
            elif backend == "torch":
                import torch  # noqa: F401
            elif backend == "jax":
                import jax  # noqa: F401
        except ImportError:
            pytest.exit(
                f"Backend '{backend}' selected but required package not installed. "
                f"Install it with: pip install {backend}"
            )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to handle missing dependencies gracefully."""
    for item in items:
        # Add marker for tests that require specific dependencies
        if "test_layer" in str(item.fspath) or "test_integration" in str(item.fspath):
            item.add_marker(pytest.mark.requires_deps)
