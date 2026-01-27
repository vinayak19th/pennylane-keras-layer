"""Utility tests for the package."""

import pytest


def test_package_version():
    """Test that package version is accessible."""
    import pennylane_keras_layer
    assert hasattr(pennylane_keras_layer, '__version__')
    assert isinstance(pennylane_keras_layer.__version__, str)
    assert pennylane_keras_layer.__version__ == "0.1.0"


def test_package_author():
    """Test that package author is accessible."""
    import pennylane_keras_layer
    assert hasattr(pennylane_keras_layer, '__author__')
    assert isinstance(pennylane_keras_layer.__author__, str)


def test_package_exports():
    """Test that package exports expected symbols."""
    import pennylane_keras_layer
    
    # Check __all__ is defined
    assert hasattr(pennylane_keras_layer, '__all__')
    
    # Check KerasCircuitLayer is exported
    assert 'KerasCircuitLayer' in pennylane_keras_layer.__all__
    
    # Check KerasCircuitLayer is accessible
    assert hasattr(pennylane_keras_layer, 'KerasCircuitLayer')


def test_pennylanekeras_layer_module():
    """Test that the layer module exists and is importable."""
    from pennylane_keras_layer import layer
    assert layer is not None


def test_pennylanekeras_layer_class_exists():
    """Test that KerasCircuitLayer class exists in the layer module."""
    from pennylane_keras_layer.layer import KerasCircuitLayer
    assert KerasCircuitLayer is not None
    assert callable(KerasCircuitLayer)


def test_package_structure():
    """Test basic package structure."""
    import pennylane_keras_layer
    import os
    
    package_dir = os.path.dirname(pennylane_keras_layer.__file__)
    
    # Check that __init__.py exists
    init_file = os.path.join(package_dir, '__init__.py')
    assert os.path.exists(init_file)
    
    # Check that layer.py exists
    layer_file = os.path.join(package_dir, 'layer.py')
    assert os.path.exists(layer_file)


def test_import_from_package():
    """Test different ways to import KerasCircuitLayer."""
    # Method 1: Direct import
    from pennylane_keras_layer import KerasCircuitLayer
    assert KerasCircuitLayer is not None
    
    # Method 2: Module import
    import pennylane_keras_layer
    assert hasattr(pennylane_keras_layer, 'KerasCircuitLayer')
    
    # Method 3: From layer module
    from pennylane_keras_layer.layer import KerasCircuitLayer as QKL
    assert QKL is not None
    
    # All should reference the same class
    import pennylane_keras_layer
    from pennylane_keras_layer.layer import KerasCircuitLayer as QKL2
    assert pennylane_keras_layer.KerasCircuitLayer is QKL
    assert QKL is QKL2
