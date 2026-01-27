# PennyLane Keras Layer

A library to introduce Keras 3 support to PennyLane with full support for multi-backend training.

## Overview

This package provides integration between PennyLane (a quantum computing framework) and Keras 3, enabling the use of quantum layers in neural networks with support for multiple backends.

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
import pennylane_keras_layer

# Core functionality will be added in future updates
```

## Examples

Check out the `examples/` directory for usage examples:

```bash
python examples/basic_example.py
```

## Development

### Running tests

```bash
pytest tests/
```

### Code formatting

```bash
black pennylane_keras_layer/ tests/
```

### Linting

```bash
flake8 pennylane_keras_layer/ tests/
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [PennyLane](https://pennylane.ai/) - Quantum machine learning framework
- [Keras](https://keras.io/) - Deep learning API
