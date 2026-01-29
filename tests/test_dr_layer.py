"""Tests for the KerasCircuitLayer implementation."""

import pytest
import numpy as np
import keras


def test_pennylanekeras_layer_import():
    """Test that KerasCircuitLayer can be imported."""
    from pennylane_keras_layer import KerasDRCircuitLayer as KerasCircuitLayer
    assert KerasCircuitLayer is not None


def test_pennylanekeras_layer_initialization():
    """Test KerasCircuitLayer initialization with default parameters."""
    try:
        from pennylane_keras_layer import KerasDRCircuitLayer as KerasCircuitLayer
        
        layer = KerasCircuitLayer(layers=2, num_wires=1)
        assert layer.layers == 2
        assert layer.num_wires == 1
        assert layer.scaling == 1.0
        assert layer.circ_backend == "lightning.qubit"
        assert layer.circ_grad_method == "adjoint"
    except ImportError:
        pytest.skip("Required dependencies not installed")


def test_pennylanekeras_layer_custom_parameters():
    """Test KerasCircuitLayer with custom parameters."""
    try:
        from pennylane_keras_layer import KerasDRCircuitLayer as KerasCircuitLayer
        
        layer = KerasCircuitLayer(
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


def test_pennylanekeras_layer_build():
    """Test that KerasCircuitLayer builds correctly."""
    try:
        import keras
        from pennylane_keras_layer import KerasDRCircuitLayer as KerasCircuitLayer
        
        layer = KerasCircuitLayer(layers=2, num_wires=1)
        
        # Build the layer with input shape
        layer.build(input_shape=(None, 1))
        
        assert layer.built
        assert layer.layer_weights is not None
        assert layer.circuit is not None
        assert layer.layer_weights.shape == (3, 3)  # (layers+1, 3)
    except ImportError:
        pytest.skip("Required dependencies not installed")


def test_pennylanekeras_layer_in_model():
    """Test KerasCircuitLayer integration in a Keras model."""
    try:
        import keras
        from pennylane_keras_layer import KerasDRCircuitLayer as KerasCircuitLayer
        
        # Create a simple model
        inp = keras.layers.Input(shape=(1,))
        q_layer = KerasCircuitLayer(layers=2, num_wires=1, circ_backend="default.qubit")
        out = q_layer(inp)
        model = keras.models.Model(inputs=inp, outputs=out)
        
        assert model is not None
        assert len(model.layers) == 2  # Input + KerasCircuitLayer
        
        # Test model summary doesn't raise errors
        try:
            model.summary()
        except Exception as e:
            pytest.fail(f"Model summary failed: {e}")
    except ImportError:
        pytest.skip("Required dependencies not installed")


def test_pennylanekeras_layer_forward_pass():
    """Test KerasCircuitLayer forward pass with dummy data."""
    try:
        import keras
        import numpy as np
        from pennylane_keras_layer import KerasDRCircuitLayer as KerasCircuitLayer
        
        # Create model
        inp = keras.layers.Input(shape=(1,))
        q_layer = KerasCircuitLayer(layers=1, num_wires=1, circ_backend="default.qubit")
        out = q_layer(inp)
        model = keras.models.Model(inputs=inp, outputs=out)
        
        # Create dummy input
        x = np.array([[0.5], [1.0], [1.5]])
        
        # Forward pass
        predictions = model(x)
        predictions_np = keras.ops.convert_to_numpy(predictions)
        
        assert predictions is not None
        assert predictions.shape == (3, 1)  # (batch_size, num_wires)
        
        # Check that predictions are in valid range for expectation values
        assert np.all(predictions_np >= -1.0)
        assert np.all(predictions_np <= 1.0)
    except ImportError:
        pytest.skip("Required dependencies not installed")


def test_pennylanekeras_layer_serialization():
    """Test KerasCircuitLayer serialization and deserialization."""
    try:
        import keras
        from pennylane_keras_layer import KerasDRCircuitLayer as KerasCircuitLayer
        
        # Create layer
        layer = KerasCircuitLayer(
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
        new_layer = KerasCircuitLayer.from_config(config)
        
        assert new_layer.layers == layer.layers
        assert new_layer.scaling == layer.scaling
        assert new_layer.circ_backend == layer.circ_backend
        assert new_layer.num_wires == layer.num_wires
    except ImportError:
        pytest.skip("Required dependencies not installed")


def test_pennylanekeras_layer_compute_output_shape():
    """Test KerasCircuitLayer output shape computation."""
    try:
        from pennylane_keras_layer import KerasDRCircuitLayer as KerasCircuitLayer
        
        layer = KerasCircuitLayer(layers=2, num_wires=3)
        
        # Test output shape computation
        input_shape = (32, 1)  # (batch_size, features)
        output_shape = layer.compute_output_shape(input_shape)
        
        assert output_shape == (32, 3)  # (batch_size, num_wires)
    except ImportError:
        pytest.skip("Required dependencies not installed")


def test_pennylanekeras_layer_not_built_error():
    """Test that calling layer before building raises an error."""
    try:
        import keras
        import numpy as np
        from pennylane_keras_layer import KerasDRCircuitLayer as KerasCircuitLayer
        
        layer = KerasCircuitLayer(layers=2, num_wires=1)
        
        # Try to call layer before building
        x = np.array([[0.5]])
        
        with pytest.raises(RuntimeError, match="KerasDRCircuitLayer must be built before calling"):
            layer.call(x)
    except ImportError:
        pytest.skip("Required dependencies not installed")


def test_pennylanekeras_layer_draw_not_built_error():
    """Test that drawing circuit before building raises an error."""
    try:
        from pennylane_keras_layer import KerasDRCircuitLayer as KerasCircuitLayer
        
        layer = KerasCircuitLayer(layers=2, num_wires=1)
        
        with pytest.raises(RuntimeError, match="KerasDRCircuitLayer must be built before drawing"):
            layer.draw_qnode()
    except ImportError:
        pytest.skip("Required dependencies not installed")


def test_pennylanekeras_layer_invalid_layers():
    """Test that invalid layers parameter raises ValueError."""
    try:
        from pennylane_keras_layer import KerasDRCircuitLayer as KerasCircuitLayer
        
        # Test negative layers
        with pytest.raises(ValueError, match="layers must be a positive integer"):
            KerasCircuitLayer(layers=-1, num_wires=1)
        
        # Test zero layers
        with pytest.raises(ValueError, match="layers must be a positive integer"):
            KerasCircuitLayer(layers=0, num_wires=1)
        
        # Test non-integer layers
        with pytest.raises(ValueError, match="layers must be a positive integer"):
            KerasCircuitLayer(layers=2.5, num_wires=1)
    except ImportError:
        pytest.skip("Required dependencies not installed")


def test_pennylanekeras_layer_invalid_num_wires():
    """Test that invalid num_wires parameter raises ValueError."""
    try:
        from pennylane_keras_layer import KerasDRCircuitLayer as KerasCircuitLayer
        
        # Test negative wires
        with pytest.raises(ValueError, match="num_wires must be a positive integer"):
            KerasCircuitLayer(layers=2, num_wires=-1)
        
        # Test zero wires
        with pytest.raises(ValueError, match="num_wires must be a positive integer"):
            KerasCircuitLayer(layers=2, num_wires=0)
    except ImportError:
        pytest.skip("Required dependencies not installed")


def test_pennylanekeras_layer_invalid_scaling():
    """Test that invalid scaling parameter raises ValueError."""
    try:
        from pennylane_keras_layer import KerasDRCircuitLayer as KerasCircuitLayer
        
        # Test negative scaling
        with pytest.raises(ValueError, match="scaling must be a positive number"):
            KerasCircuitLayer(layers=2, num_wires=1, scaling=-1.0)
        
        # Test zero scaling
        with pytest.raises(ValueError, match="scaling must be a positive number"):
            KerasCircuitLayer(layers=2, num_wires=1, scaling=0)
    except ImportError:
        pytest.skip("Required dependencies not installed")
