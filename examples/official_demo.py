import os

# ==============================================================================
# 0. Compatibility Note
# ==============================================================================
# This script is designed for general x86 systems. 
# On Apple Silicon systems using the JAX backend with GPU (Metal), 
# note that jax-metal currently has limited support for complex numbers 
# needed for quantum simulations.
# ==============================================================================

# Select your backend before importing Keras
# os.environ["KERAS_BACKEND"] = "jax"  # Options: "jax", "torch", "tensorflow"

import keras
import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

# Library-specific syntax from demo_simple.py
from pennylane_keras_layer import KerasCircuitLayer

# ==============================================================================
# 1. The Story: Unifying Quantum ML with Keras 3.0
# ==============================================================================
# For years, hybrid quantum-classical researchers were forced into "ecosystem
# silos." If you used Keras, you were locked into TensorFlow. If you used 
# PennyLane, you had to manage different backend-specific wrappers.
#
# Today, Keras 3.0 and PennyLane unite. With Keras's multi-backend engine 
# and this new unified layer, your hybrid models are now truly portable.
# ==============================================================================

# Step 1: Define a standard PennyLane QNode
n_qubits = 2
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev)
def qnode(weights, inputs):
    # Encoding classical data into quantum states
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    
    # Trainable quantum layers
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    
    # Measuring the result (Expval of PauliZ on each qubit)
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

# Specify the shapes for the trainable parameters
weight_shapes = {"weights": (2, n_qubits, 3)}

# ==============================================================================
# 2. Build and Train the Hybrid Model
# ==============================================================================
print(f"--- Running on Keras Backend: {keras.backend.backend()} ---")

# The 3-Step Keras Workflow:
# 1. Wrap the QNode in a Keras-native layer
qlayer = KerasCircuitLayer(qnode, weight_shapes, output_dim=n_qubits)

# 2. Assemble high-level components with Sequential or Functional APIs
model = keras.Sequential([
    keras.Input(shape=(n_qubits,)),
    qlayer,
    keras.layers.Dense(1, activation="sigmoid")
])

# 3. Use standard Keras tools like model.fit() 
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Generate a visualizable 2D dataset
X, y = make_moons(n_samples=200, noise=0.1, random_state=42)
X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0)) * 2 - 1 # Normalize to [-1, 1]
y = y.reshape(-1, 1).astype("float32")

# Training is now as simple as a classical neural network
print("Training hybrid model...")
# model.fit(X, y, epochs=10, batch_size=8, verbose=1)

# ==============================================================================
# 3. Beyond model.fit(): Multi-Backend Flexibility
# ==============================================================================
# The beauty of this integration is that it doesn't lock you into Keras APIs.
# You can use the same model in pure JAX or PyTorch optimization loops.

backend = keras.backend.backend()

if backend == "jax":
    import jax
    import jax.numpy as jnp
    
    print("Example: JAX functional optimization step")
    def jax_loss_fn(params, x, y):
        logits = model.stateless_call(params, x)
        return jnp.mean((logits - y)**2)
    
    # This is pure JAX. No Keras-specific training state required.
    # grads = jax.grad(jax_loss_fn)(model.trainable_variables, X[:1], y[:1])
    
elif backend == "torch":
    import torch
    
    print("Example: PyTorch dynamic optimization loop")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # model.train()
    # optimizer.zero_grad()
    # out = model(torch.tensor(X[:1]))
    # loss = torch.nn.BCELoss()(out, torch.tensor(y[:1]))
    # loss.backward()
    # optimizer.step()

# ==============================================================================
# 4. Visualization (The Decision Boundary)
# ==============================================================================
# A simple 2D classification task allows us to see exactly how the 
# quantum circuit is shaping the classical feature space.

print("Visualization snippet ready.")
# Use standard matplotlib/numpy for plotting results
# xx, yy = np.meshgrid(np.linspace(-1.1, 1.1, 20), np.linspace(-1.1, 1.1, 20))
# grid = np.c_[xx.ravel(), yy.ravel()]
# preds = model.predict(grid).reshape(xx.shape)
# plt.contourf(xx, yy, preds, alpha=0.8, cmap="RdBu")
# plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), edgecolors='k')
# plt.show()

print("\n--- Storytelling Demo Setup Complete ---")
