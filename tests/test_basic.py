"""Basic test to verify package installation and imports."""

import pytest


def test_package_imports():
    """Test that the package can be imported."""
    import pennylane_keras_layer
    assert pennylane_keras_layer.__version__ == "0.1.0"


def test_pennylane_available():
    """Test that PennyLane is available."""
    try:
        import pennylane as qml  # noqa: F401
    except ImportError:
        pytest.skip("PennyLane not installed")


def test_keras_available():
    """Test that Keras is available."""
    try:
        import keras  # noqa: F401
    except ImportError:
        pytest.skip("Keras not installed")
