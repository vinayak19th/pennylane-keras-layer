# ==============================================================================
# Setup / Installation
# ==============================================================================
# To run this demo, you will need to install the following packages:
# pip install tensorflow  # Or jax, torch
# pip install pennylane
# pip install pennylane-keras-layer
#
# Note: You can select the backend by setting the KERAS_BACKEND environment variable.
# e.g., os.environ["KERAS_BACKEND"] = "jax"

import os

# Set backend to JAX (optional, but good for performance)
os.environ["KERAS_BACKEND"] = "jax"

import keras
import pennylane as qml
import numpy as np

# Import the layer
from pennylane_keras_layer import KerasCircuitLayer

# ==============================================================================
# 1. Create a Random QNode
# ==============================================================================
print("\n--- 1. Creating a Random QNode ---")
n_qubits = 2
n_layers = 2
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev)
def qnode(weights, inputs):
    """
    A simple QNode that embeds input data and then applies layers of weights.
    Structure:
    - AngleEmbedding for inputs
    - StronglyEntanglingLayers for weights
    """
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

# Define weight shapes for the Keras layer
# StronglyEntanglingLayers expects shape (n_layers, n_qubits, 3)
weight_shapes = {"weights": (n_layers, n_qubits, 3)}
print(f"QNode defined with {n_qubits} qubits and {n_layers} layers.")

# ==============================================================================
# 2. Wrap it into Keras
# ==============================================================================
print("\n--- 2. Wrapping QNode into Keras Layer ---")
# output_dim matches the number of measurements (2 expvals)
qlayer = KerasCircuitLayer(qnode, weight_shapes, output_dim=n_qubits)
print("KerasCircuitLayer created.")

# ==============================================================================
# 3. Create a Model and model.fit
# ==============================================================================
print("\n--- 3. Creating Model and Training ---")

# Simple model: Input -> Quantum Layer -> Output (Dense)
inputs = keras.Input(shape=(n_qubits,))
x = qlayer(inputs)
outputs = keras.layers.Dense(1)(x) # Regression output

model = keras.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer="adam", loss="mse")

# Generate some random dummy data
# batch_size=10, features=2
X = np.random.random((10, n_qubits))
y = np.random.random((10, 1))

print("Starting training...")
history = model.fit(X, y, epochs=2, batch_size=2, verbose=1)
print("Training complete.")

# ==============================================================================
# 4. model.save / model.load
# ==============================================================================
print("\n--- 4. Saving and Loading Model ---")
save_path = "simple_qnn.keras"
model.save(save_path)
print(f"Model saved to {save_path}")

# Load the model
# Custom Object Scope might be needed if not fully registered, 
# but KerasCircuitLayer uses @register_keras_serializable so it should be fine directly 
# or with custom_objects
try:
    loaded_model = keras.models.load_model(save_path)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# Verify QNode restoration
# IMPORTANT: QNodes are not serializable, so they must be re-set after loading!
print("Restoring QNode to loaded layer...")
# Assuming the quantum layer is the second layer (index 1) after InputLayer
# We iterate to find the KerasCircuitLayer just to be safe
q_layer_loaded = None
for layer in loaded_model.layers:
    if isinstance(layer, KerasCircuitLayer):
        q_layer_loaded = layer
        break

if q_layer_loaded:
    q_layer_loaded.set_qnode(qnode)
    print("QNode set on loaded layer.")
else:
    print("Warning: Could not find KerasCircuitLayer in loaded model.")

# Verify inference with loaded model
print("Verifying inference with loaded model...")
try:
    pred = loaded_model.predict(X[:2])
    print(f"Prediction shape: {pred.shape}")
    print("Inference successful!")
except Exception as e:
    print(f"Inference failed: {e}")

# Cleanup
if os.path.exists(save_path):
    os.remove(save_path)
    print(f"Cleaned up {save_path}")

print("\n--- Demo Complete ---")
