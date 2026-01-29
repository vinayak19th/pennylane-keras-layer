
import pytest
import pennylane as qml
import numpy as np
import keras
import os

from pennylane_keras_layer import KerasCircuitLayer, KerasDRCircuitLayer

def test_generic_layer_structure():
    """Test that KerasCircuitLayer creates weights correctly."""
    n_wires = 2
    dev = qml.device("default.qubit", wires=n_wires)
    
    @qml.qnode(dev)
    def circuit(w1, w2, x):
        qml.RX(x[0], wires=0)
        qml.Rot(*w1, wires=0)
        qml.Rot(*w2, wires=1)
        return qml.expval(qml.PauliZ(0))
        
    weight_shapes = {"w1": (3,), "w2": (3,)}
    
    KerasCircuitLayer.set_input_argument("x")
    KerasCircuitLayer.set_input_argument("x")
    layer = KerasCircuitLayer(circuit, weight_shapes, output_dim=1)
    
    # Build the layer
    layer.build((1, 1))
    
    assert len(layer.qnode_weights) == 2
    assert layer.qnode_weights["w1"].shape == (3,)
    assert layer.qnode_weights["w2"].shape == (3,)
    
def test_generic_layer_execution():
    """Test that KerasCircuitLayer executes correctly."""
    n_wires = 1
    dev = qml.device("default.qubit", wires=n_wires)
    
    @qml.qnode(dev)
    def circuit(w, x):
        qml.RX(x[0], wires=0)
        qml.Rot(*w, wires=0)
        return qml.expval(qml.PauliZ(0))
        
    weight_shapes = {"w": (3,)}
    
    # Mock inputs
    x = np.array([[np.pi]]) # RX(pi) -> state |1>, expect -1 if Rot is identity
    
    KerasCircuitLayer.set_input_argument("x")
    layer = KerasCircuitLayer(circuit, weight_shapes, output_dim=1)
    
    # Run call
    # We rely on Keras to initialize weights randomly, but we can inspect output shape
    out = layer(x)
    
    assert out.shape == (1,)

def test_dr_layer_preserved():
    """Test that KerasDRCircuitLayer is working (basic init)."""
    layer = KerasDRCircuitLayer(layers=2, num_wires=1)
    layer.build((1, 1))
    assert layer.layer_weights is not None
    assert layer.circuit is not None

def test_tf_graph_mode_error():
    """Test that KerasCircuitLayer raises error in TF graph mode."""
    # Only relevant if TF is installed and backend is TF
    if keras.config.backend() == "tensorflow":
        import tensorflow as tf
        
        n_wires = 1
        dev = qml.device("default.qubit", wires=n_wires)
        
        @qml.qnode(dev)
        def circuit(w, x):
            return qml.expval(qml.PauliZ(0))
            
        layer = KerasCircuitLayer(circuit, {"w": (3,)}, output_dim=1)
        
        @tf.function
        def graph_func(x):
            return layer(x)
            
        with pytest.raises(RuntimeError, match="does not support TensorFlow graph mode"):
            graph_func(tf.ones((1, 1)))

@pytest.mark.integration
def test_generic_layer_training():
    """Test that KerasCircuitLayer can be trained in a Keras model."""
    if keras.config.backend() == "tensorflow":
        import tensorflow as tf
        if not keras.config.run_eagerly() and not tf.executing_eagerly():
            pytest.skip("Test requires eager execution for now")

    n_wires = 1
    dev = qml.device("default.qubit", wires=n_wires)
    
    @qml.qnode(dev)
    def circuit(w, x):
        qml.RX(x[0], wires=0)
        qml.Rot(*w, wires=0)
        return qml.expval(qml.PauliZ(0))
        
    weight_shapes = {"w": (3,)}
    
    # Create model
    KerasCircuitLayer.set_input_argument("x")
    layer = KerasCircuitLayer(circuit, weight_shapes, output_dim=1)
    inp = keras.layers.Input(shape=(1,))
    out = layer(inp)
    model = keras.models.Model(inputs=inp, outputs=out)
    
    # Compile
    model.compile(optimizer=keras.optimizers.Adam(0.1), loss='mse')
    
    # Train on dummy data: input 0 -> output -1 (requires identity rotation if initialized near identity)
    # Actually, let's just check loss decreases or runs.
    x = np.zeros((10, 1))
    y = -np.ones((10, 1)) # Target -1 (state |1>)
    
    history = model.fit(x, y, epochs=2, verbose=0)
    assert len(history.history['loss']) == 2
    assert not np.isnan(history.history['loss'][0])

@pytest.mark.integration
def test_generic_layer_save_load(tmp_path):
    """Test saving and loading KerasCircuitLayer."""
    n_wires = 1
    dev = qml.device("default.qubit", wires=n_wires)
    
    @qml.qnode(dev)
    def circuit(w, x):
        qml.RX(x[0], wires=0)
        qml.Rot(*w, wires=0)
        return qml.expval(qml.PauliZ(0))
        
    weight_shapes = {"w": (3,)}
    
    # Create and build model
    KerasCircuitLayer.set_input_argument("x")
    layer = KerasCircuitLayer(circuit, weight_shapes, output_dim=1)
    inp = keras.layers.Input(shape=(1,))
    out = layer(inp)
    model = keras.models.Model(inputs=inp, outputs=out)
    
    # Run once to build
    x = np.zeros((1, 1))
    pred_before = keras.ops.convert_to_numpy(model(x))
    
    # Save
    model_path = tmp_path / "generic_model.keras"
    model.save(model_path)
    
    # Load
    loaded_model = keras.models.load_model(model_path)
    
    # Restore QNode
    # Note: Keras loads layers. We need to find our layer.
    # In this simple model, layer index 1 is likely the quantum layer (index 0 is InputLayer or Input)
    # Use name or type to find it
    found_layer = None
    for l in loaded_model.layers:
        if isinstance(l, KerasCircuitLayer):
            found_layer = l
            break
            
    assert found_layer is not None
    
    # QNode needs to be re-assigned
    found_layer.set_qnode(circuit)
    
    # Verify weights are preserved (approx) or at least output is same
    pred_after = keras.ops.convert_to_numpy(loaded_model(x))
    
    np.testing.assert_allclose(pred_before, pred_after, atol=1e-5)

if __name__ == "__main__":
    pytest.main([__file__])
