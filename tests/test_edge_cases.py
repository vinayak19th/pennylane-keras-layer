"""Performance and edge case tests for KerasCircuitLayer."""

import pytest
import numpy as np
import keras


@pytest.mark.parametrize("layers", [1, 2, 3, 5])
def test_different_layer_counts(layers):
    """Test KerasCircuitLayer with different numbers of layers."""
    try:
        from pennylane_keras_layer import KerasDRCircuitLayer as KerasCircuitLayer
        
        layer = KerasCircuitLayer(layers=layers, num_wires=1)
        assert layer.layers == layers
        
        # Build and check weight shape
        layer.build(input_shape=(None, 1))
        assert layer.layer_weights.shape == (layers + 1, 3)
        
    except ImportError:
        pytest.skip("Required dependencies not installed")


@pytest.mark.parametrize("num_wires", [1, 2, 3])
def test_different_wire_counts(num_wires):
    """Test KerasCircuitLayer with different numbers of wires."""
    try:
        from pennylane_keras_layer import KerasDRCircuitLayer as KerasCircuitLayer
        
        layer = KerasCircuitLayer(layers=2, num_wires=num_wires)
        assert layer.num_wires == num_wires
        
        # Check output shape
        output_shape = layer.compute_output_shape((10, 1))
        assert output_shape == (10, num_wires)
        
    except ImportError:
        pytest.skip("Required dependencies not installed")


@pytest.mark.parametrize("scaling", [0.5, 1.0, 2.0, 5.0])
def test_different_scaling_factors(scaling):
    """Test KerasCircuitLayer with different scaling factors."""
    try:
        import keras
        from pennylane_keras_layer import KerasDRCircuitLayer as KerasCircuitLayer
        
        # Create model with specific scaling
        inp = keras.layers.Input(shape=(1,))
        q_layer = KerasCircuitLayer(
            layers=1, 
            scaling=scaling, 
            num_wires=1, 
            circ_backend="default.qubit"
        )
        out = q_layer(inp)
        model = keras.models.Model(inputs=inp, outputs=out)
        
        assert q_layer.scaling == scaling
        
        # Test prediction
        x = np.array([[1.0]])
        pred = model(x)
        assert pred is not None
        
    except ImportError:
        pytest.skip("Required dependencies not installed")


@pytest.mark.parametrize("backend", ["default.qubit", "lightning.qubit"])
def test_different_backends(backend):
    """Test KerasCircuitLayer with different quantum backends."""
    try:
        from pennylane_keras_layer import KerasDRCircuitLayer as KerasCircuitLayer
        
        layer = KerasCircuitLayer(layers=2, circ_backend=backend, num_wires=1)
        assert layer.circ_backend == backend
        
    except ImportError:
        pytest.skip("Required dependencies not installed")


def test_extreme_input_values():
    """Test KerasCircuitLayer with extreme input values."""
    try:
        import keras
        from pennylane_keras_layer import KerasDRCircuitLayer as KerasCircuitLayer
        
        # Create model
        inp = keras.layers.Input(shape=(1,))
        q_layer = KerasCircuitLayer(layers=1, num_wires=1, circ_backend="default.qubit")
        out = q_layer(inp)
        model = keras.models.Model(inputs=inp, outputs=out)
        
        # Test with extreme values
        extreme_values = np.array([
            [-1000.0],
            [-10.0],
            [-1e-10],
            [0.0],
            [1e-10],
            [10.0],
            [1000.0]
        ])
        
        predictions = model(extreme_values)
        predictions_np = keras.ops.convert_to_numpy(predictions)
        
        # Should still produce valid expectation values
        assert np.all(predictions_np >= -1.0)
        assert np.all(predictions_np <= 1.0)
        assert not np.any(np.isnan(predictions_np))
        
    except ImportError:
        pytest.skip("Required dependencies not installed")


def test_zero_input():
    """Test KerasCircuitLayer with zero input."""
    try:
        import keras
        from pennylane_keras_layer import KerasDRCircuitLayer as KerasCircuitLayer
        
        # Create model
        inp = keras.layers.Input(shape=(1,))
        q_layer = KerasCircuitLayer(layers=2, num_wires=1, circ_backend="default.qubit")
        out = q_layer(inp)
        model = keras.models.Model(inputs=inp, outputs=out)
        
        # Test with zero input
        x = np.zeros((5, 1))
        predictions = model(x)
        predictions_np = keras.ops.convert_to_numpy(predictions)
        
        assert predictions is not None
        assert predictions.shape == (5, 1)
        assert not np.any(np.isnan(predictions_np))
        
    except ImportError:
        pytest.skip("Required dependencies not installed")


def test_reproducibility_with_seed():
    """Test that results are reproducible with the same seed."""
    try:
        import keras
        from pennylane_keras_layer import KerasDRCircuitLayer as KerasCircuitLayer
        import numpy as np
        
        from keras import utils
        
        # Set seed
        np.random.seed(42)
        utils.set_random_seed(42)
        
        # Create first model
        inp1 = keras.layers.Input(shape=(1,))
        q_layer1 = KerasCircuitLayer(layers=2, num_wires=1, circ_backend="default.qubit")
        out1 = q_layer1(inp1)
        model1 = keras.models.Model(inputs=inp1, outputs=out1)
        
        # Get predictions
        x = np.array([[1.0], [2.0], [3.0]])
        pred1 = keras.ops.convert_to_numpy(model1(x))
        
        # Reset seed and create second model
        np.random.seed(42)
        utils.set_random_seed(42)
        
        inp2 = keras.layers.Input(shape=(1,))
        q_layer2 = KerasCircuitLayer(layers=2, num_wires=1, circ_backend="default.qubit")
        out2 = q_layer2(inp2)
        model2 = keras.models.Model(inputs=inp2, outputs=out2)
        
        # Get predictions
        pred2 = keras.ops.convert_to_numpy(model2(x))
        
        # Results should be identical
        np.testing.assert_array_almost_equal(pred1, pred2, decimal=10)
        
    except ImportError:
        pytest.skip("Required dependencies not installed")


def test_layer_naming():
    """Test that layer names work correctly."""
    try:
        from pennylane_keras_layer import KerasDRCircuitLayer as KerasCircuitLayer
        
        layer1 = KerasCircuitLayer(layers=2, num_wires=1, name="quantum_layer_1")
        assert layer1.name == "quantum_layer_1"
        
        layer2 = KerasCircuitLayer(layers=2, num_wires=1, name="quantum_layer_2")
        assert layer2.name == "quantum_layer_2"
        
    except ImportError:
        pytest.skip("Required dependencies not installed")


def test_multiple_calls_to_build():
    """Test that calling build multiple times doesn't break the layer."""
    try:
        from pennylane_keras_layer import KerasDRCircuitLayer as KerasCircuitLayer
        
        layer = KerasCircuitLayer(layers=2, num_wires=1)
        
        # Build once
        layer.build(input_shape=(None, 1))
        weights1 = keras.ops.convert_to_numpy(layer.layer_weights).copy()
        
        # Build again with same shape should not change weights
        layer.build(input_shape=(None, 1))
        weights2 = keras.ops.convert_to_numpy(layer.layer_weights).copy()
        
        # Weights should be the same (not reinitialized)
        np.testing.assert_array_equal(weights1, weights2)
        
    except ImportError:
        pytest.skip("Required dependencies not installed")
