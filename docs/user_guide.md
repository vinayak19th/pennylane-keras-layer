# User Guide

This guide provides an in-depth understanding of PennyLane Keras Layer concepts, architecture, and best practices.

## Table of Contents

- [Core Concepts](#core-concepts)
- [Architecture](#architecture)
- [Data Re-Uploading Model](#data-re-uploading-model)
- [Multi-Backend Support](#multi-backend-support)
- [Model Building Patterns](#model-building-patterns)
- [Training Strategies](#training-strategies)
- [Saving and Loading](#saving-and-loading)
- [Performance Optimization](#performance-optimization)
- [Best Practices](#best-practices)

## Core Concepts

### Quantum Machine Learning

Quantum Machine Learning (QML) combines quantum computing with machine learning to:
- Leverage quantum superposition and entanglement
- Explore high-dimensional feature spaces efficiently
- Potentially achieve quantum advantage for certain tasks

### Hybrid Models

PennyLane Keras Layer enables hybrid quantum-classical models:
- Classical layers process input data
- Quantum layers perform quantum computations
- Classical layers process quantum outputs
- All parts trained end-to-end with gradient descent

### Variational Quantum Circuits

The quantum circuits in this library are variational:
- Contain trainable parameters (rotation angles)
- Parameters optimized via gradient-based methods
- Similar to weights in classical neural networks

## Architecture

The library provides two main layers for integrating quantum circuits:

1.  **`KerasDRCircuitLayer`**: A high-level, specialized layer for Data Re-Uploading models.
2.  **`KerasCircuitLayer`**: A generic layer for wrapping any PennyLane QNode.

### KerasDRCircuitLayer Structure

This layer implements specific Data Re-Uploading architecture:

```
Input Data (classical)
    ↓
Scaling (multiply by scaling factor)
    ↓
Quantum Circuit (PennyLane QNode)
    ├── Layer 1: Trainable Rotation → Data Encoding
    ├── ...
    └── Layer N: Trainable Rotation
    ↓
Measurement (PauliZ expectation)
    ↓
Output (classical)
```

### KerasCircuitLayer Structure

This layer is flexible:
-   Accepts a custom PennyLane QNode
-   Accepts a `weight_shapes` dictionary
-   Passes input as a named argument to the QNode
-   Handles weight management and backend compatibility

## Data Re-Uploading Model

### What is Data Re-Uploading?

Data Re-Uploading is a quantum machine learning technique where classical data is encoded multiple times into the circuit, interleaved with trainable gates.

### Mathematical Representation

For input `x` and weights `W = [w₀, w₁, ..., wₙ]`:

```
|ψ⟩ = U(wₙ) E(x) U(wₙ₋₁) E(x) ... U(w₁) E(x) U(w₀) |0⟩
```

Where `U(wᵢ)` are trainable rotations and `E(x)` encodes the data.

## Multi-Backend Support

### Supported Backends

| Backend    | Use Case                     | Performance | JIT Support |
|------------|------------------------------|-------------|-------------|
| JAX        | Research, High Performance   | Excellent   | Yes         |
| TensorFlow | Production, Deployment       | Good        | Yes         |
| PyTorch    | Research, Dynamic Graphs     | Good        | Limited     |

### Backend-Specific Considerations

#### JAX Backend

**Standard JAX Interface:**
```python
import os
os.environ["KERAS_BACKEND"] = "jax"
import keras
from pennylane_keras_layer import KerasDRCircuitLayer

q_layer = KerasDRCircuitLayer(layers=2, use_jax_python=False)
```

## Model Building Patterns

### Pattern 1: Data Re-Uploading (Simple)

```python
from pennylane_keras_layer import KerasDRCircuitLayer

model = keras.Sequential([
    keras.layers.Input(shape=(1,)),
    KerasDRCircuitLayer(layers=3, scaling=1.0),
    keras.layers.Dense(1)
])
```

### Pattern 2: Generic QNode (Flexible)

```python
import pennylane as qml
from pennylane_keras_layer import KerasCircuitLayer

# Define QNode
dev = qml.device("default.qubit", wires=2)
@qml.qnode(dev)
def circuit(inputs, w):
    qml.RX(inputs[0], wires=0)
    qml.Rot(*w, wires=0)
    return qml.expval(qml.Z(0))

# Create Layer
weight_shapes = {"w": (3,)}
KerasCircuitLayer.set_input_argument("inputs") # Important!
q_layer = KerasCircuitLayer(circuit, weight_shapes, output_dim=1)

model = keras.Sequential([
    keras.layers.Input(shape=(1,)),
    q_layer
])
```

## Training Strategies

### Optimizer Selection

**Adam (Recommended):**
```python
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01), loss="mse")
```

## Saving and Loading

### Saving Models

```python
model.save("my_quantum_model.keras")
```

### Loading KerasDRCircuitLayer

Automatically restored.

### Loading KerasCircuitLayer

QNodes are not serializable, so they must be re-attached:

```python
loaded_model = keras.models.load_model("my_quantum_model.keras")

# Find the layer and set the QNode
for layer in loaded_model.layers:
    if isinstance(layer, KerasCircuitLayer):
        layer.set_qnode(circuit)
```

## Best Practices

### 1. Model Saving
When using `KerasCircuitLayer`, always define your QNode in a way that it can be re-imported or re-defined for model loading.

### 2. Input Shapes
`KerasDRCircuitLayer` expects 1D inputs `(batch, 1)`. Use `KerasCircuitLayer` for multi-dimensional inputs.

### 3. Backend Selection
Use JAX for fastest training. Use TensorFlow for easiest graph export.
