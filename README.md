# PennyLane Keras Layer

A library to introduce Keras 3 support to PennyLane with full support for multi-backend training.

## Overview

This package provides integration between PennyLane (a quantum computing framework) and Keras 3, enabling the use of quantum layers in neural networks with support for multiple backends.

### Key Features

- **🔄 Multi-Backend Support**: Works seamlessly with JAX, TensorFlow, and PyTorch
- **🎯 Easy Integration**: Drop-in replacement for standard Keras layers  
- **⚡ High Performance**: JIT compilation support with JAX backend
- **💾 Serialization**: Full support for model saving and loading
- **🔧 Flexible Configuration**: Customizable quantum circuit parameters
- **📊 Data Re-Uploading**: Implements powerful quantum ML paradigm

## Installation

### From source

```bash
git clone https://github.com/vinayak19th/pennylane-keras-layer.git
cd pennylane-keras-layer
pip install -e .
```

### Development installation

For development, install with additional dependencies:

```bash
pip install -e ".[dev]"
```

## Requirements

- Python >= 3.8
- PennyLane >= 0.30.0
- Keras >= 3.0.0
- NumPy >= 1.21.0

## Quick Start

```python
import os
os.environ["KERAS_BACKEND"] = "jax"  # Set backend before importing keras

import keras
from pennylane_keras_layer import QKerasLayer

# Create a quantum layer
q_layer = QKerasLayer(layers=3, scaling=1.0, num_wires=1)

# Build a model
model = keras.Sequential([
    keras.layers.Input(shape=(1,)),
    q_layer
])

# Compile and train
model.compile(optimizer="adam", loss="mse")
# model.fit(X_train, y_train, epochs=50)
```

## Documentation

Comprehensive documentation is available in the [`docs/`](docs/) directory:

- **[Getting Started](docs/getting_started.md)**: Installation and setup guide
- **[User Guide](docs/user_guide.md)**: In-depth concepts and best practices
- **[API Reference](docs/api_reference.md)**: Detailed API documentation
- **[Tutorials](docs/tutorials.md)**: Step-by-step tutorials for different backends
- **[Examples](docs/examples.md)**: Complete working examples
- **[FAQ](docs/faq.md)**: Frequently asked questions
- **[Contributing](docs/contributing.md)**: How to contribute to the project

## Examples

Check out the `examples/` directory for usage examples:

```bash
# Run basic example
python examples/basic_example.py

# Run Fourier series approximation demo
python examples/fourier_series_example.py
```

## Development

### Running tests

The package includes a comprehensive test suite in the `tests/` directory:

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=pennylane_keras_layer --cov-report=term-missing

# Run specific test file
pytest tests/test_layer.py -v

# Run only integration tests
pytest tests/ -m integration
```

See [tests/README.md](tests/README.md) for detailed information about the test suite.

### Code formatting

```bash
black pennylane_keras_layer/ tests/
```

### Linting

```bash
flake8 pennylane_keras_layer/ tests/
```

## Contributing

Contributions are welcome! Please see our [Contributing Guide](docs/contributing.md) for:

- How to report bugs
- How to suggest features
- Development setup
- Code standards
- Pull request process

Feel free to submit a Pull Request or open an issue!

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [PennyLane](https://pennylane.ai/) - Quantum machine learning framework
- [Keras](https://keras.io/) - Deep learning API
