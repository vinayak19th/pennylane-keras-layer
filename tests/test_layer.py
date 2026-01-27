"""Tests for the QKerasLayer implementation."""

import pytest
import numpy as np


def test_qkeras_layer_import():
    """Test that QKerasLayer can be imported."""
    from pennylane_keras_layer import QKerasLayer
    assert QKerasLayer is not None


def test_qkeras_layer_initialization():
    """Test QKerasLayer initialization with default parameters."""
    try:
        from pennylane_keras_layer import QKerasLayer
        
        layer = QKerasLayer(layers=2, num_wires=1)
        assert layer.layers == 2
        assert layer.num_wires == 1
        assert layer.scaling == 1.0
        assert layer.circ_backend == "lightning.qubit"
        assert layer.circ_grad_method == "adjoint"
    except ImportError:
        pytest.skip("Required dependencies not installed")


def test_qkeras_layer_custom_parameters():
    """Test QKerasLayer with custom parameters."""
    try:
        from pennylane_keras_layer import QKerasLayer
        
        layer = QKerasLayer(
            layers=3,
            scaling=2.0,
            circ_backend="default.qubit",
            circ_grad_method="parameter-shift",
            num_wires=2
        )
        assert layer.layers == 3
        assert layer.scaling == 2.0
        assert layer.circ_backend == "default.qubit"
        assert layer.circ_grad_method == "parameter-shift"
        assert layer.num_wires == 2
    except ImportError:
        pytest.skip("Required dependencies not installed")


def test_qkeras_layer_build():
    """Test that QKerasLayer builds correctly."""
    try:
        import keras
        from pennylane_keras_layer import QKerasLayer
        
        layer = QKerasLayer(layers=2, num_wires=1)
        
        # Build the layer with input shape
        layer.build(input_shape=(None, 1))
        
        assert layer.is_built
        assert layer.layer_weights is not None
        assert layer.circuit is not None
        assert layer.layer_weights.shape == (3, 3)  # (layers+1, 3)
    except ImportError:
        pytest.skip("Required dependencies not installed")


def test_qkeras_layer_in_model():
    """Test QKerasLayer integration in a Keras model."""
    try:
        import keras
        from pennylane_keras_layer import QKerasLayer
        
        # Create a simple model
        inp = keras.layers.Input(shape=(1,))
        q_layer = QKerasLayer(layers=2, num_wires=1, circ_backend="default.qubit")
        out = q_layer(inp)
        model = keras.models.Model(inputs=inp, outputs=out)
        
        assert model is not None
        assert len(model.layers) == 2  # Input + QKerasLayer
        
        # Test model summary doesn't raise errors
        try:
            model.summary()
        except Exception as e:
            pytest.fail(f"Model summary failed: {e}")
    except ImportError:
        pytest.skip("Required dependencies not installed")


def test_qkeras_layer_forward_pass():
    """Test QKerasLayer forward pass with dummy data."""
    try:
        import keras
        import numpy as np
        from pennylane_keras_layer import QKerasLayer
        
        # Create model
        inp = keras.layers.Input(shape=(1,))
        q_layer = QKerasLayer(layers=1, num_wires=1, circ_backend="default.qubit")
        out = q_layer(inp)
        model = keras.models.Model(inputs=inp, outputs=out)
        
        # Create dummy input
        x = np.array([[0.5], [1.0], [1.5]])
        
        # Forward pass
        predictions = model(x)
        
        assert predictions is not None
        assert predictions.shape == (3, 1)  # (batch_size, num_wires)
        
        # Check that predictions are in valid range for expectation values
        assert np.all(predictions >= -1.0)
        assert np.all(predictions <= 1.0)
    except ImportError:
        pytest.skip("Required dependencies not installed")


def test_qkeras_layer_serialization():
    """Test QKerasLayer serialization and deserialization."""
    try:
        import keras
        from pennylane_keras_layer import QKerasLayer
        
        # Create layer
        layer = QKerasLayer(
            layers=2,
            scaling=1.5,
            circ_backend="default.qubit",
            num_wires=1,
            name="test_layer"
        )
        
        # Get config
        config = layer.get_config()
        
        assert config is not None
        assert "layers" in config
        assert "scaling" in config
        assert "circ_backend" in config
        assert "num_wires" in config
        
        # Recreate from config
        new_layer = QKerasLayer.from_config(config)
        
        assert new_layer.layers == layer.layers
        assert new_layer.scaling == layer.scaling
        assert new_layer.circ_backend == layer.circ_backend
        assert new_layer.num_wires == layer.num_wires
    except ImportError:
        pytest.skip("Required dependencies not installed")


def test_qkeras_layer_compute_output_shape():
    """Test QKerasLayer output shape computation."""
    try:
        from pennylane_keras_layer import QKerasLayer
        
        layer = QKerasLayer(layers=2, num_wires=3)
        
        # Test output shape computation
        input_shape = (32, 1)  # (batch_size, features)
        output_shape = layer.compute_output_shape(input_shape)
        
        assert output_shape == (32, 3)  # (batch_size, num_wires)
    except ImportError:
        pytest.skip("Required dependencies not installed")


def test_qkeras_layer_not_built_error():
    """Test that calling layer before building raises an error."""
    try:
        import keras
        import numpy as np
        from pennylane_keras_layer import QKerasLayer
        
        layer = QKerasLayer(layers=2, num_wires=1)
        
        # Try to call layer before building
        x = np.array([[0.5]])
        
        with pytest.raises(Exception, match="Layer not built"):
            layer.call(x)
    except ImportError:
        pytest.skip("Required dependencies not installed")


def test_qkeras_layer_draw_not_built_error():
    """Test that drawing circuit before building raises an error."""
    try:
        from pennylane_keras_layer import QKerasLayer
        
        layer = QKerasLayer(layers=2, num_wires=1)
        
        with pytest.raises(Exception, match="Layer not built"):
            layer.draw_qnode()
    except ImportError:
        pytest.skip("Required dependencies not installed")
