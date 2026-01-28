# Release Notes - Version 1.1.0

**Release Date**: January 28, 2026

## Overview

We are excited to announce the release of **pennylane-keras3-layer v1.1.0**, bringing Keras 3 support to PennyLane with full multi-backend training capabilities! This release provides seamless integration between quantum computing and deep learning, enabling developers to build hybrid quantum-classical neural networks with ease.

## What's New

### 🎉 Major Features

#### 1. KerasCircuitLayer - Generic Quantum Layer
The `KerasCircuitLayer` class wraps PennyLane QNodes as Keras layers, providing:
- Full multi-backend support (TensorFlow, JAX, PyTorch)
- Automatic weight initialization and management
- Serialization support for model saving/loading
- Easy integration with existing Keras models

```python
import pennylane as qml
from pennylane_keras_layer import KerasCircuitLayer

# Define quantum device
dev = qml.device('default.qubit', wires=1)

# Define custom quantum circuit
@qml.qnode(dev)
def custom_circuit(inputs, weights):
    qml.RX(inputs[0], wires=0)
    qml.Rot(*weights[0], wires=0)
    return qml.expval(qml.PauliZ(0))

# Convert to Keras layer
weight_shapes = {"weights": (1, 3)}
q_layer = KerasCircuitLayer(custom_circuit, weight_shapes, output_dim=1)
```

#### 2. KerasDRCircuitLayer - Data Re-Uploading Circuit
Pre-built quantum layer implementing the powerful data re-uploading paradigm:
- Configurable number of layers
- Adjustable scaling parameters
- Ready-to-use architecture for quantum machine learning

```python
from pennylane_keras_layer import KerasDRCircuitLayer

# Create data re-uploading layer
q_layer = KerasDRCircuitLayer(layers=3, scaling=1.0, num_wires=1)
```

### 📚 Comprehensive Documentation

This release includes extensive documentation to help you get started:

- **[Getting Started](docs/getting_started.md)**: Quick installation and first steps
- **[User Guide](docs/user_guide.md)**: In-depth concepts and best practices
- **[API Reference](docs/api_reference.md)**: Complete API documentation
- **[Tutorials](docs/tutorials.md)**: Step-by-step guides for each backend
- **[Examples](docs/examples.md)**: Real-world usage examples
- **[FAQ](docs/faq.md)**: Common questions and troubleshooting
- **[Contributing](docs/contributing.md)**: How to contribute to the project

### 💡 Working Examples

The release includes several practical examples:

1. **Basic Example** (`examples/basic_example.py`)
   - Simple quantum layer integration
   - Demonstrates basic usage patterns

2. **Fourier Series Approximation** (`examples/fourier_series_example.py`)
   - Advanced quantum ML application
   - Shows quantum advantage in function approximation

3. **QNN Module Tutorial** (`examples/tutorial_qnn_module_keras.py`)
   - Comprehensive tutorial for building QNNs
   - Multi-backend compatibility demonstration

### 🧪 Testing Infrastructure

Robust test suite with:
- Unit tests for all major components
- Integration tests for backend compatibility
- Edge case handling
- pytest configuration with coverage reporting
- Support for testing across different backends

Run tests with:
```bash
pytest tests/ --cov=pennylane_keras_layer
```

### ⚙️ CI/CD Pipeline

Automated workflows for:
- Continuous integration with multiple Python versions (3.10, 3.11)
- Automated testing on pull requests
- PyPI publishing workflow
- Code quality checks

## Key Benefits

### Multi-Backend Flexibility
Choose your preferred deep learning framework without changing your quantum code:
- **JAX**: High-performance with JIT compilation
- **TensorFlow**: Industry-standard deep learning
- **PyTorch**: Research-friendly framework

### Easy Integration
Drop-in replacement for standard Keras layers:
```python
model = keras.Sequential([
    keras.layers.Input(shape=(1,)),
    KerasDRCircuitLayer(layers=3, num_wires=1),  # Quantum layer
    keras.layers.Dense(10),                       # Classical layer
])
```

### Production Ready
- Full serialization support for model persistence
- Comprehensive error handling
- Well-tested and documented
- MIT licensed for commercial use

## Technical Specifications

### Requirements
- Python >= 3.10
- PennyLane >= 0.30.0
- Keras >= 3.0.0
- NumPy >= 1.21.0

### Installation

From source:
```bash
git clone https://github.com/vinayak19th/pennylane-keras-layer.git
cd pennylane-keras-layer
pip install -e .
```

For development:
```bash
pip install -e ".[dev]"
```

## Performance Considerations

- **JAX Backend**: Recommended for best performance with JIT compilation
- **TensorFlow Backend**: Stable and widely-supported
- **PyTorch Backend**: Great for research and experimentation

## What's Next

This release establishes the foundation for quantum-classical hybrid models in Keras 3. Future releases will focus on:
- Additional pre-built quantum circuit architectures
- Performance optimizations
- Extended backend support
- More tutorials and examples

## Get Involved

We welcome contributions! Check out our [Contributing Guide](docs/contributing.md) to learn how you can help improve this project.

## Acknowledgments

Special thanks to the PennyLane and Keras teams for creating the excellent frameworks that make this integration possible.

---

**Full Changelog**: See [CHANGELOG.md](CHANGELOG.md) for detailed changes.

**Repository**: https://github.com/vinayak19th/pennylane-keras-layer

**Issues**: https://github.com/vinayak19th/pennylane-keras-layer/issues
