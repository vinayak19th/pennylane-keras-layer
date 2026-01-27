
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

if __name__ == "__main__":
    pytest.main([__file__])
