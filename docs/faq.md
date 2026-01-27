# Frequently Asked Questions (FAQ)

Common questions and answers about PennyLane Keras Layer.

## Table of Contents

- [General Questions](#general-questions)
- [Installation Issues](#installation-issues)
- [Usage Questions](#usage-questions)
- [Backend-Specific Questions](#backend-specific-questions)
- [Performance Questions](#performance-questions)
- [Error Messages](#error-messages)
- [Advanced Topics](#advanced-topics)

---

## General Questions

### What is PennyLane Keras Layer?

PennyLane Keras Layer is a library that integrates PennyLane's quantum computing framework with Keras 3, allowing you to use quantum circuits as layers in neural networks. It supports multiple backends (TensorFlow, JAX, PyTorch) and enables hybrid quantum-classical machine learning.

### Why would I want to use quantum layers in my model?

Quantum layers can:
- Explore high-dimensional feature spaces efficiently
- Potentially provide quantum advantage for certain tasks
- Enable novel approaches to machine learning problems
- Combine quantum and classical computing strengths

### Is this production-ready?

The library is currently in alpha stage (v0.1.0). It's suitable for:
- Research and experimentation
- Proof-of-concept projects
- Learning quantum machine learning

For production use, thoroughly test with your specific use case.

### Do I need quantum hardware?

No! The library uses quantum simulators by default. You can:
- Use PennyLane's CPU simulators (`default.qubit`, `lightning.qubit`)
- Use GPU-accelerated simulators (with appropriate backends)
- Potentially use real quantum hardware (via PennyLane plugins)

### What is Data Re-Uploading?

Data Re-Uploading is a quantum ML technique where classical data is encoded into the quantum circuit multiple times, interspersed with trainable quantum gates. This creates a rich, parameterized transformation that can approximate complex functions.

---

## Installation Issues

### "No module named 'pennylane_keras_layer'" error

**Solution:**
```bash
pip install pennylane-keras-layer
```

If installing from source:
```bash
git clone https://github.com/vinayak19th/pennylane-keras-layer.git
cd pennylane-keras-layer
pip install -e .
```

### "No module named 'keras'" error

**Solution:** Keras 3 requires a backend. Install one:
```bash
# For JAX (recommended)
pip install jax jaxlib keras

# For TensorFlow
pip install tensorflow keras

# For PyTorch
pip install torch keras
```

### Version conflicts with dependencies

**Solution:** Create a fresh virtual environment:
```bash
python -m venv qml_env
source qml_env/bin/activate  # On Windows: qml_env\Scripts\activate
pip install pennylane-keras-layer jax
```

### ImportError: "PennyLane version too old"

**Solution:** Update PennyLane:
```bash
pip install --upgrade pennylane>=0.30.0
```

---

## Usage Questions

### How do I set the Keras backend?

**Important:** Set backend BEFORE importing Keras:

```python
import os
os.environ["KERAS_BACKEND"] = "jax"  # or "tensorflow" or "torch"

import keras  # Import AFTER setting backend
from pennylane_keras_layer import QKerasLayer
```

### What input shape does QKerasLayer expect?

QKerasLayer expects 2D input: `(batch_size, features)`. Currently, the implementation uses only the first feature dimension.

```python
# Good: Input shape (batch, 1)
model = keras.Sequential([
    keras.layers.Input(shape=(1,)),
    QKerasLayer(layers=3)
])

# If you have more features, reduce first
model = keras.Sequential([
    keras.layers.Input(shape=(10,)),
    keras.layers.Dense(1),  # Reduce to 1D
    QKerasLayer(layers=3)
])
```

### How many layers should I use?

Start with 2-3 layers and increase if needed:

```python
# Start simple
QKerasLayer(layers=2)

# Increase if underfitting
QKerasLayer(layers=4)

# Too many layers may overfit
```

Rule of thumb: More layers = more expressivity but longer training time.

### What scaling factor should I use?

Depends on your data range:

```python
# Data in [-1, 1]: Use scaling=1.0 to 2.0
QKerasLayer(layers=3, scaling=1.0)

# Data in larger range: Normalize first
from sklearn.preprocessing import StandardScaler
X_scaled = StandardScaler().fit_transform(X)
QKerasLayer(layers=3, scaling=1.0)
```

### Can I use multiple quantum layers?

Yes! Each layer has independent parameters:

```python
model = keras.Sequential([
    keras.layers.Input(shape=(1,)),
    QKerasLayer(layers=2, name="quantum1"),
    QKerasLayer(layers=2, name="quantum2"),
    keras.layers.Dense(1)
])
```

### How do I visualize the quantum circuit?

```python
model = keras.Sequential([
    keras.layers.Input(shape=(1,)),
    QKerasLayer(layers=3, name="quantum")
])
model.build((None, 1))

# Draw circuit
import matplotlib.pyplot as plt
model.get_layer("quantum").draw_qnode()
plt.show()
```

### Can I save and load models?

Yes! Full serialization support:

```python
# Save
model.save("quantum_model.keras")

# Load
loaded_model = keras.models.load_model("quantum_model.keras")
```

---

## Backend-Specific Questions

### Which backend should I use?

| Backend    | Best For                          | Pros                      | Cons                    |
|------------|-----------------------------------|---------------------------|-------------------------|
| JAX        | Research, performance             | Fast, JIT compilation     | Newer, less ecosystem   |
| TensorFlow | Production, deployment            | Mature, stable            | Slightly slower         |
| PyTorch    | Research, dynamic graphs          | Flexible, popular         | Less Keras 3 support    |

**Recommendation:** Start with JAX for best performance.

### How do I switch backends?

Set before importing Keras:

```python
# Switch to JAX
import os
os.environ["KERAS_BACKEND"] = "jax"
import keras

# Switch to TensorFlow (in a new Python session)
import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import keras
```

### What is use_jax_python parameter?

For JAX backend only:

```python
# Standard JAX (default, recommended)
QKerasLayer(layers=2, use_jax_python=False)  # Uses jax.jit

# JAX-Python interface
QKerasLayer(layers=2, use_jax_python=True)  # No JIT, better compatibility
```

Use `use_jax_python=True` if encountering JIT-related errors.

### Can I use GPU acceleration?

Yes, depends on the backend:

**JAX:**
```bash
# Install JAX with CUDA support (check JAX docs for latest instructions)
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

**TensorFlow:**
```bash
# TensorFlow 2.15+ includes GPU support by default
pip install tensorflow>=2.15.0
```

**PyTorch:**
```bash
# Install PyTorch with CUDA
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

---

## Performance Questions

### Training is slow. How can I speed it up?

1. **Use lightning backend:**
   ```python
   QKerasLayer(layers=3, circ_backend="lightning.qubit")
   ```

2. **Use JAX with JIT:**
   ```python
   os.environ["KERAS_BACKEND"] = "jax"
   QKerasLayer(layers=3, use_jax_python=False)
   ```

3. **Use adjoint gradient method:**
   ```python
   QKerasLayer(layers=3, circ_grad_method="adjoint")
   ```

4. **Increase batch size:**
   ```python
   model.fit(X, y, batch_size=64)  # Instead of 32
   ```

5. **Reduce number of layers:**
   ```python
   QKerasLayer(layers=2)  # Instead of 5
   ```

### How does performance scale with layer count?

Approximately linear with number of layers:

| Layers | Relative Time |
|--------|---------------|
| 2      | 1x            |
| 4      | 2x            |
| 8      | 4x            |

More layers = longer training but potentially better accuracy.

### Can I train on GPU?

Yes, but quantum simulation doesn't always benefit from GPU. Best results:
- Use GPU for large batch sizes
- Use JAX or TensorFlow backends
- Some PennyLane devices support GPU (e.g., `lightning.gpu`)

### What's the largest model I can train?

Depends on:
- Number of qubits (currently limited to 1 in this implementation)
- Number of layers
- Batch size
- Available memory

Current implementation with 1 qubit can handle:
- Up to ~20 layers easily
- Batch sizes of 128+
- Thousands of training samples

---

## Error Messages

### "RuntimeError: QKerasLayer must be built before calling"

**Cause:** Calling layer before building.

**Solution:** Ensure layer is built:
```python
# Option 1: Use in a model
model = keras.Sequential([
    keras.layers.Input(shape=(1,)),
    QKerasLayer(layers=3)
])
# Layer is auto-built when model is used

# Option 2: Explicitly build
q_layer = QKerasLayer(layers=3)
q_layer.build((None, 1))
```

### "ValueError: layers must be a positive integer"

**Cause:** Invalid layer parameter.

**Solution:** Use positive integer:
```python
# Wrong
QKerasLayer(layers=0)
QKerasLayer(layers=-1)
QKerasLayer(layers=2.5)

# Correct
QKerasLayer(layers=3)
```

### "ValueError: Unsupported Keras backend"

**Cause:** Backend not set or invalid.

**Solution:**
```python
import os
os.environ["KERAS_BACKEND"] = "jax"  # Must be BEFORE import
import keras
```

### "ModuleNotFoundError: No module named 'jax'"

**Cause:** Backend not installed.

**Solution:**
```bash
# Install the backend you want to use
pip install jax jaxlib
# or
pip install tensorflow
# or
pip install torch
```

### JAX JIT compilation errors

**Cause:** Some operations incompatible with JIT.

**Solution:** Use JAX-Python interface:
```python
QKerasLayer(layers=3, use_jax_python=True)
```

---

## Advanced Topics

### Can I customize the quantum circuit?

Currently, the circuit architecture is fixed (Data Re-Uploading). For custom circuits, you would need to modify the `QKerasLayer` class or create your own layer.

Future versions may support custom circuit definitions.

### Can I use more than 1 qubit?

The `num_wires` parameter exists but the current implementation only uses wire 0. Multi-qubit circuits would require modifications to the `serial_quantum_model` method.

### How do I use real quantum hardware?

Install PennyLane hardware plugin and set backend:

```python
# Example: IBM Quantum
# pip install pennylane-qiskit

QKerasLayer(
    layers=2,
    circ_backend="qiskit.ibmq",
    circ_grad_method="parameter-shift"  # Adjoint not available on hardware
)
```

**Note:** Real hardware has limitations (noise, queue times, costs).

### Can I implement custom gradient methods?

Yes, via PennyLane's gradient methods:

```python
QKerasLayer(
    layers=3,
    circ_grad_method="parameter-shift"  # or "finite-diff", "adjoint", etc.
)
```

### How do I debug gradient issues?

```python
import keras
import numpy as np

# Check gradients (backend-specific)
# For TensorFlow:
import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import tensorflow as tf

model = create_model()
x = np.random.randn(1, 1)
y = np.random.randn(1, 1)

with tf.GradientTape() as tape:
    pred = model(x)
    loss = keras.losses.mse(y, pred)

grads = tape.gradient(loss, model.trainable_variables)

# Print gradients
for i, grad in enumerate(grads):
    if grad is None:
        print(f"Gradient {i}: None (WARNING!)")
    else:
        print(f"Gradient {i}: shape={grad.shape}, mean={np.mean(grad):.6f}")
```

### Can I integrate with other Keras features?

Yes! Compatible with:
- Callbacks (EarlyStopping, ModelCheckpoint, etc.)
- Learning rate schedules
- Custom training loops
- Mixed precision training (with care)
- Model subclassing

### What about quantum noise?

For noisy simulations or hardware:

```python
import pennylane as qml

# Create noisy device (example)
# This would require custom circuit creation
# Current QKerasLayer uses noise-free simulators by default
```

### How can I contribute improvements?

See [Contributing Guide](contributing.md) for details on:
- Reporting bugs
- Suggesting features
- Submitting pull requests
- Development setup

---

## Still Have Questions?

- Check the [User Guide](user_guide.md) for concepts
- See [Tutorials](tutorials.md) for examples
- Review [API Reference](api_reference.md) for details
- Open an issue on [GitHub](https://github.com/vinayak19th/pennylane-keras-layer/issues)
