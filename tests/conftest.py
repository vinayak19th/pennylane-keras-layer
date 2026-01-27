"""Configuration file for pytest."""

import pytest
import sys


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


def pytest_collection_modifyitems(config, items):
    """Modify test collection to handle missing dependencies gracefully."""
    for item in items:
        # Add marker for tests that require specific dependencies
        if "test_layer" in str(item.fspath) or "test_integration" in str(item.fspath):
            item.add_marker(pytest.mark.requires_deps)
