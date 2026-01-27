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

### QKerasLayer Structure

```
Input Data (classical)
    ↓
Scaling (multiply by scaling factor)
    ↓
Quantum Circuit (PennyLane QNode)
    ├── Layer 1: Trainable Rotation → Data Encoding
    ├── Layer 2: Trainable Rotation → Data Encoding
    ├── ...
    └── Layer N: Trainable Rotation
    ↓
Measurement (PauliZ expectation)
    ↓
Output (classical)
```

### Weight Structure

Weights have shape `(layers + 1, 3)`:
- Each row: `[φ, θ, ω]` for `qml.Rot(φ, θ, ω)`
- `layers + 1` rows because the final layer has only rotation, no encoding
- Initialized randomly in `[0, 2π)`

### Circuit Components

#### 1. Data Encoding Block
```python
def data_encoding_block(self, x):
    qml.RX(x[:, 0], wires=0)
```
Encodes classical data into quantum state using RX rotation.

#### 2. Trainable Rotation Block
```python
def trainable_rotation_block(self, theta):
    qml.Rot(theta[0], theta[1], theta[2], wires=0)
```
Applies trainable three-parameter rotation.

#### 3. Serial Quantum Model
Alternates between trainable rotations and data encoding, implementing the Data Re-Uploading paradigm.

## Data Re-Uploading Model

### What is Data Re-Uploading?

Data Re-Uploading is a quantum machine learning technique where:
1. Classical data is encoded multiple times into the circuit
2. Between encodings, trainable quantum gates are applied
3. This creates a rich, parameterized transformation

### Mathematical Representation

For input `x` and weights `W = [w₀, w₁, ..., wₙ]`:

```
|ψ⟩ = U(wₙ) E(x) U(wₙ₋₁) E(x) ... U(w₁) E(x) U(w₀) |0⟩
```

Where:
- `U(wᵢ)`: Trainable rotation with parameters wᵢ
- `E(x)`: Data encoding operation
- `|0⟩`: Initial state

### Advantages

1. **Universal Function Approximation**: Can approximate any continuous function
2. **Rich Feature Space**: Multiple encodings create complex feature representations
3. **Efficient Parameterization**: Uses fewer qubits than other approaches

### Connection to Fourier Series

The Data Re-Uploading model can be viewed as a quantum Fourier series:
- Each layer contributes different frequency components
- Trainable parameters control amplitudes and phases
- Allows approximation of periodic and non-periodic functions

See the [Fourier series example](examples.md#fourier-series-approximation) for details.

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

q_layer = QKerasLayer(layers=2, use_jax_python=False)
```

- Uses `jax.jit` for compilation
- Best performance
- Recommended for most users

**JAX-Python Interface:**
```python
q_layer = QKerasLayer(layers=2, use_jax_python=True)
```

- Does NOT support JIT compilation
- Better compatibility with some JAX features
- Use if encountering JIT-related issues

#### TensorFlow Backend

```python
import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import keras

q_layer = QKerasLayer(layers=2)
```

- Stable and production-ready
- Good integration with TensorFlow ecosystem
- Graph mode optimization available

#### PyTorch Backend

```python
import os
os.environ["KERAS_BACKEND"] = "torch"
import keras

q_layer = QKerasLayer(layers=2)
```

- Dynamic computation graphs
- Good for research and experimentation
- Eager execution by default

### Switching Backends

**Important:** Backend must be set BEFORE importing Keras:

```python
import os
os.environ["KERAS_BACKEND"] = "jax"  # Must come first!
import keras  # Now uses JAX backend
```

## Model Building Patterns

### Pattern 1: Sequential API (Simplest)

```python
model = keras.Sequential([
    keras.layers.Input(shape=(1,)),
    QKerasLayer(layers=3, scaling=1.0),
    keras.layers.Dense(1)
])
```

**Use when:**
- Linear stack of layers
- Simple architectures

### Pattern 2: Functional API (Recommended)

```python
inputs = keras.layers.Input(shape=(10,))
x = keras.layers.Dense(5)(inputs)
x = keras.layers.Dense(1)(x)  # Reduce to quantum layer input size
x = QKerasLayer(layers=4, scaling=2.0, name="quantum")(x)
x = keras.layers.Dense(10)(x)
outputs = keras.layers.Dense(1)(x)
model = keras.models.Model(inputs=inputs, outputs=outputs)
```

**Use when:**
- Complex architectures
- Multiple inputs/outputs
- Need explicit layer naming

### Pattern 3: Subclassing API (Advanced)

```python
class HybridModel(keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = keras.layers.Dense(5)
        self.quantum = QKerasLayer(layers=3, scaling=1.0)
        self.dense2 = keras.layers.Dense(1)
    
    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.quantum(x)
        return self.dense2(x)

model = HybridModel()
```

**Use when:**
- Need custom training loops
- Complex forward pass logic
- Research implementations

### Multiple Quantum Layers

```python
model = keras.Sequential([
    keras.layers.Input(shape=(1,)),
    QKerasLayer(layers=2, scaling=1.0, name="quantum1"),
    QKerasLayer(layers=2, scaling=1.0, name="quantum2"),
    keras.layers.Dense(1)
])
```

**Note:** Each quantum layer has independent parameters.

## Training Strategies

### Optimizer Selection

**Adam (Recommended):**
```python
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01), loss="mse")
```
- Adaptive learning rates
- Good default choice
- Start with lr=0.01 to 0.03

**SGD with Momentum:**
```python
model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.1, momentum=0.9), loss="mse")
```
- More stable for some problems
- May require learning rate tuning

### Learning Rate Scheduling

```python
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.03,
    decay_steps=1000,
    decay_rate=0.96
)
optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
```

### Loss Functions

**Regression:**
```python
model.compile(optimizer="adam", loss="mse")  # Mean Squared Error
model.compile(optimizer="adam", loss="mae")  # Mean Absolute Error
```

**Classification:**
```python
model.compile(optimizer="adam", loss="binary_crossentropy")
model.compile(optimizer="adam", loss="categorical_crossentropy")
```

### Batch Size Considerations

Quantum circuits support batch processing:

```python
# Small batches for stability
model.fit(X, y, batch_size=32, epochs=100)

# Larger batches for speed (if memory allows)
model.fit(X, y, batch_size=128, epochs=100)
```

**Trade-offs:**
- Smaller batches: More stable gradients, slower training
- Larger batches: Faster training, may need learning rate adjustment

### Training Tips

1. **Start Simple**: Begin with few layers (2-3) and increase if needed
2. **Monitor Loss**: Use callbacks to track training progress
3. **Use Validation**: Split data to monitor overfitting
4. **Early Stopping**: Prevent overfitting

```python
callbacks = [
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=10),
    keras.callbacks.ModelCheckpoint('best_model.keras', save_best_only=True)
]

model.fit(X_train, y_train, 
          validation_data=(X_val, y_val),
          epochs=100, 
          callbacks=callbacks)
```

## Saving and Loading

### Save Complete Model

```python
# Save
model.save("my_quantum_model.keras")

# Load
loaded_model = keras.models.load_model("my_quantum_model.keras")
```

The saved model includes:
- Model architecture
- Layer configurations
- Trained weights
- Optimizer state

### Save Weights Only

```python
# Save weights
model.save_weights("weights.h5")

# Load weights
model.load_weights("weights.h5")
```

### Export for Production

```python
# SavedModel format (TensorFlow)
model.export("saved_model/")

# ONNX (if supported by backend)
# Requires additional tools
```

## Performance Optimization

### 1. Use Lightning Backend

```python
q_layer = QKerasLayer(layers=3, circ_backend="lightning.qubit")
```
Faster than `"default.qubit"` for CPU execution.

### 2. Enable JIT (JAX)

```python
# Automatically enabled for JAX backend
os.environ["KERAS_BACKEND"] = "jax"
q_layer = QKerasLayer(layers=3, use_jax_python=False)  # JIT enabled
```

### 3. Appropriate Gradient Method

```python
# Adjoint (fastest for simulators)
q_layer = QKerasLayer(layers=3, circ_grad_method="adjoint")

# Parameter-shift (for hardware/noisy simulators)
q_layer = QKerasLayer(layers=3, circ_grad_method="parameter-shift")
```

### 4. Batch Processing

Process data in batches instead of one-by-one:

```python
# Good: Batch processing
predictions = model.predict(X_test)  # X_test has batch dimension

# Less efficient: Single samples
predictions = [model.predict(x) for x in X_test]
```

### 5. Profile Your Code

```python
import time

start = time.time()
model.fit(X_train, y_train, epochs=10)
print(f"Training time: {time.time() - start:.2f}s")
```

## Best Practices

### 1. Input Scaling

Choose `scaling` parameter based on your data:

```python
# Data in [-1, 1]: Use scaling=1.0 to 2.0
q_layer = QKerasLayer(layers=3, scaling=1.0)

# Data in [-π, π]: Use scaling=1.0
q_layer = QKerasLayer(layers=3, scaling=1.0)

# Data in larger range: Normalize first or increase scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
q_layer = QKerasLayer(layers=3, scaling=1.0)
```

### 2. Number of Layers

Start small and increase:

```python
# Start with 2-3 layers
q_layer = QKerasLayer(layers=2)

# Increase if underfitting
q_layer = QKerasLayer(layers=5)

# Too many layers may overfit
```

### 3. Circuit Visualization

Understand your circuit:

```python
model = keras.Sequential([
    keras.layers.Input(shape=(1,)),
    QKerasLayer(layers=3, name="quantum")
])
model.build((None, 1))
model.get_layer("quantum").draw_qnode()
```

### 4. Reproducibility

Set random seeds:

```python
import numpy as np
import random
import os

# Python random
random.seed(42)

# NumPy
np.random.seed(42)

# Backend-specific (example for JAX)
if keras.backend.backend() == "jax":
    import jax
    jax.config.update("jax_random_seed", 42)
```

### 5. Error Handling

```python
try:
    q_layer = QKerasLayer(layers=3)
    model = keras.Sequential([keras.layers.Input(shape=(1,)), q_layer])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=10)
except RuntimeError as e:
    print(f"Runtime error: {e}")
except ValueError as e:
    print(f"Value error: {e}")
```

### 6. Testing

Always test your models:

```python
# Smoke test
model = create_my_model()
sample_input = np.random.randn(1, 10)
output = model(sample_input)
assert output.shape == (1, 1), "Output shape mismatch"

# Gradient test
with keras.backend.GradientTape() as tape:
    output = model(sample_input)
    loss = keras.losses.mse(output, target)
grads = tape.gradient(loss, model.trainable_variables)
assert all(g is not None for g in grads), "Some gradients are None"
```

## Common Patterns

### Pattern: Regression

```python
model = keras.Sequential([
    keras.layers.Input(shape=(1,)),
    QKerasLayer(layers=4, scaling=1.0),
    keras.layers.Dense(1, activation="linear")
])
model.compile(optimizer="adam", loss="mse", metrics=["mae"])
```

### Pattern: Binary Classification

```python
model = keras.Sequential([
    keras.layers.Input(shape=(features,)),
    keras.layers.Dense(1),
    QKerasLayer(layers=3, scaling=1.0),
    keras.layers.Dense(1, activation="sigmoid")
])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
```

### Pattern: Preprocessing Integration

```python
inputs = keras.layers.Input(shape=(10,))
x = keras.layers.Normalization()(inputs)
x = keras.layers.Dense(1)(x)
x = QKerasLayer(layers=3, scaling=1.0)(x)
outputs = keras.layers.Dense(1)(x)
model = keras.models.Model(inputs=inputs, outputs=outputs)
```

## Next Steps

- See [Tutorials](tutorials.md) for step-by-step examples
- Check [Examples](examples.md) for complete working code
- Review [API Reference](api_reference.md) for parameter details
- Visit [FAQ](faq.md) for common questions
