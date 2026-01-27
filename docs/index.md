# PennyLane Keras Layer Documentation

Welcome to the PennyLane Keras Layer documentation! This library provides seamless integration between PennyLane quantum computing framework and Keras 3, enabling quantum machine learning with full multi-backend support.

## Quick Links

- [Getting Started](getting_started.md)
- [API Reference](api_reference.md)
- [User Guide](user_guide.md)
- [Tutorials](tutorials.md)
- [Examples](examples.md)
- [FAQ](faq.md)
- [Contributing](contributing.md)

## What is PennyLane Keras Layer?

PennyLane Keras Layer is a Python library that bridges quantum computing and classical deep learning by allowing you to:

- **Integrate quantum circuits** into Keras models as standard layers
- **Train quantum models** using familiar Keras APIs
- **Use multiple backends** - TensorFlow, JAX, or PyTorch
- **Leverage PennyLane's** powerful quantum computing features
- **Serialize and deserialize** quantum models with Keras' save/load mechanisms

## Key Features

### Multi-Backend Support
Train quantum models with any Keras 3 backend:
- **TensorFlow**: Industry-standard for production deployment
- **JAX**: High-performance with automatic differentiation and JIT compilation
- **PyTorch**: Popular framework with dynamic computation graphs

### Data Re-Uploading Architecture
Implements the Data Re-Uploading quantum machine learning paradigm, which allows:
- Efficient encoding of classical data into quantum states
- Multiple layers of quantum transformations
- Trainable quantum parameters optimized via gradient descent

### Seamless Keras Integration
- Drop-in replacement for standard Keras layers
- Compatible with Sequential, Functional, and Subclassing APIs
- Full support for model saving, loading, and checkpointing
- Works with all Keras optimizers, losses, and metrics

## Installation

```bash
pip install pennylane-keras-layer
```

For development:
```bash
git clone https://github.com/vinayak19th/pennylane-keras-layer.git
cd pennylane-keras-layer
pip install -e ".[dev]"
```

## Quick Example

```python
import os
os.environ["KERAS_BACKEND"] = "jax"  # or "tensorflow" or "torch"

import keras
from pennylane_keras_layer import QKerasLayer

# Create a quantum layer
q_layer = QKerasLayer(
    layers=2,
    scaling=1.0,
    num_wires=1,
    name="quantum_layer"
)

# Build a model
model = keras.Sequential([
    keras.layers.Input(shape=(1,)),
    q_layer,
    keras.layers.Dense(1)
])

# Train like any Keras model
model.compile(optimizer="adam", loss="mse")
model.fit(X_train, y_train, epochs=10)
```

## System Requirements

- Python >= 3.8
- PennyLane >= 0.30.0
- Keras >= 3.0.0
- NumPy >= 1.21.0

## Getting Help

- **Documentation**: You're reading it!
- **GitHub Issues**: [Report bugs or request features](https://github.com/vinayak19th/pennylane-keras-layer/issues)
- **Examples**: Check the `examples/` directory in the repository

## License

This project is licensed under the MIT License. See the [LICENSE](../LICENSE) file for details.

## Acknowledgments

This library builds upon:
- [PennyLane](https://pennylane.ai/) - Quantum machine learning framework
- [Keras](https://keras.io/) - Deep learning API
