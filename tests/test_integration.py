"""Integration tests for KerasCircuitLayer with different Keras backends."""

import pytest
import numpy as np
import os


@pytest.fixture
def sample_data():
    """Generate sample training data."""
    x = np.linspace(-2, 2, 20, dtype=np.float64)
    y = np.sin(x)
    return x.reshape(-1, 1), y.reshape(-1, 1)


@pytest.mark.integration
def test_model_training_basic(sample_data):
    """Test basic model training with KerasCircuitLayer."""
    try:
        import keras
        from pennylane_keras_layer import KerasDRCircuitLayer as KerasCircuitLayer
        
        x_train, y_train = sample_data
        
        # Build model
        inp = keras.layers.Input(shape=(1,))
        q_layer = KerasCircuitLayer(layers=1, num_wires=1, circ_backend="default.qubit")
        out = q_layer(inp)
        model = keras.models.Model(inputs=inp, outputs=out)
        
        # Compile
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.1),
            loss=keras.losses.mean_squared_error,
            run_eagerly=(keras.config.backend() == "tensorflow")
        )
        
        # Train for a few epochs
        history = model.fit(x_train, y_train, epochs=2, verbose=0)
        
        assert history is not None
        assert len(history.history['loss']) == 2
        
        # Check that loss is a reasonable number
        assert not np.isnan(history.history['loss'][-1])
        assert not np.isinf(history.history['loss'][-1])
        
    except ImportError:
        pytest.skip("Required dependencies not installed")


@pytest.mark.integration
def test_model_save_and_load(sample_data, tmp_path):
    """Test saving and loading a model with KerasCircuitLayer."""
    try:
        import keras
        from pennylane_keras_layer import KerasDRCircuitLayer as KerasCircuitLayer
        
        x_train, _ = sample_data
        
        # Build model
        inp = keras.layers.Input(shape=(1,))
        q_layer = KerasCircuitLayer(layers=2, num_wires=1, circ_backend="default.qubit")
        out = q_layer(inp)
        model = keras.models.Model(inputs=inp, outputs=out)
        
        # Get predictions before saving
        predictions_before = model(x_train)
        
        # Save model
        model_path = tmp_path / "test_model.keras"
        model.save(str(model_path))
        
        assert model_path.exists()
        
        # Load model
        loaded_model = keras.models.load_model(str(model_path))
        
        # Get predictions after loading
        predictions_after = loaded_model(x_train)
        
        # Compare predictions
        predictions_before_np = keras.ops.convert_to_numpy(predictions_before)
        predictions_after_np = keras.ops.convert_to_numpy(predictions_after)
        diff = np.abs(predictions_before_np - predictions_after_np).max()
        assert diff < 1e-6, f"Predictions differ by {diff}"
        
    except ImportError:
        pytest.skip("Required dependencies not installed")


@pytest.mark.integration
def test_model_with_multiple_layers(sample_data):
    """Test model with KerasCircuitLayer and other Keras layers."""
    try:
        import keras
        from pennylane_keras_layer import KerasDRCircuitLayer as KerasCircuitLayer
        
        x_train, y_train = sample_data
        
        # Build model with multiple layers
        inp = keras.layers.Input(shape=(1,))
        dense1 = keras.layers.Dense(4, activation='relu')(inp)
        q_layer = KerasCircuitLayer(layers=1, num_wires=1, circ_backend="default.qubit")(dense1[:, 0:1])
        dense2 = keras.layers.Dense(1)(q_layer)
        model = keras.models.Model(inputs=inp, outputs=dense2)
        
        # Compile and train
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.1),
            loss=keras.losses.mean_squared_error,
            run_eagerly=(keras.config.backend() == "tensorflow")
        )
        
        history = model.fit(x_train, y_train, epochs=2, verbose=0)
        
        assert history is not None
        assert not np.isnan(history.history['loss'][-1])
        
    except ImportError:
        pytest.skip("Required dependencies not installed")


@pytest.mark.integration
def test_model_prediction_range():
    """Test that model predictions are in valid expectation value range."""
    try:
        import keras
        from pennylane_keras_layer import KerasDRCircuitLayer as KerasCircuitLayer
        
        # Create model
        inp = keras.layers.Input(shape=(1,))
        q_layer = KerasCircuitLayer(layers=2, num_wires=1, circ_backend="default.qubit")
        out = q_layer(inp)
        model = keras.models.Model(inputs=inp, outputs=out)
        
        # Test with various inputs
        test_inputs = np.array([[-10.0], [-1.0], [0.0], [1.0], [10.0]])
        predictions = model(test_inputs)
        predictions_np = keras.ops.convert_to_numpy(predictions)
        
        # Expectation values should be in [-1, 1]
        assert np.all(predictions_np >= -1.0), f"Found predictions < -1: {predictions_np.min()}"
        assert np.all(predictions_np <= 1.0), f"Found predictions > 1: {predictions_np.max()}"
        
    except ImportError:
        pytest.skip("Required dependencies not installed")


@pytest.mark.integration
def test_batch_processing():
    """Test that layer handles different batch sizes correctly."""
    try:
        import keras
        from pennylane_keras_layer import KerasDRCircuitLayer as KerasCircuitLayer
        
        # Create model
        inp = keras.layers.Input(shape=(1,))
        q_layer = KerasCircuitLayer(layers=1, num_wires=1, circ_backend="default.qubit")
        out = q_layer(inp)
        model = keras.models.Model(inputs=inp, outputs=out)
        
        # Test different batch sizes
        for batch_size in [1, 5, 10, 32]:
            x = np.random.randn(batch_size, 1)
            predictions = model(x)
            assert predictions.shape == (batch_size, 1)
        
    except ImportError:
        pytest.skip("Required dependencies not installed")


@pytest.mark.integration
def test_gradient_flow():
    """Test that gradients flow through the quantum layer."""
    try:
        import keras
        from pennylane_keras_layer import KerasDRCircuitLayer as KerasCircuitLayer
        
        # Create model
        inp = keras.layers.Input(shape=(1,))
        q_layer = KerasCircuitLayer(layers=1, num_wires=1, circ_backend="default.qubit")
        out = q_layer(inp)
        model = keras.models.Model(inputs=inp, outputs=out)
        
        # Compile with optimizer
        model.compile(
            optimizer=keras.optimizers.SGD(learning_rate=0.01),
            loss=keras.losses.mean_squared_error,
            run_eagerly=(keras.config.backend() == "tensorflow")
        )
        
        # Get initial weights
        initial_weights = [keras.ops.convert_to_numpy(w).copy() for w in model.trainable_weights]
        
        # Train on a single sample
        x = np.array([[1.0]])
        y = np.array([[0.5]])
        model.fit(x, y, epochs=1, verbose=0)
        
        # Get updated weights
        updated_weights = [keras.ops.convert_to_numpy(w) for w in model.trainable_weights]
        
        # Check that at least one weight has changed
        weights_changed = any(
            not np.allclose(w1, w2) 
            for w1, w2 in zip(initial_weights, updated_weights)
        )
        
        assert weights_changed, "Weights did not update - gradients may not be flowing"
        
    except ImportError:
        pytest.skip("Required dependencies not installed")
