# Documentation

This directory contains comprehensive documentation for the PennyLane Keras Layer library.

## Documentation Structure

### Core Documentation

- **[index.md](index.md)**: Main documentation landing page with overview and quick links
- **[getting_started.md](getting_started.md)**: Installation guide and first steps
- **[user_guide.md](user_guide.md)**: In-depth concepts, architecture, and best practices
- **[api_reference.md](api_reference.md)**: Detailed API documentation for QKerasLayer

### Learning Resources

- **[tutorials.md](tutorials.md)**: Step-by-step tutorials for different backends and use cases
- **[examples.md](examples.md)**: Complete working examples with explanations

### Additional Resources

- **[faq.md](faq.md)**: Frequently asked questions and troubleshooting
- **[contributing.md](contributing.md)**: Guidelines for contributing to the project

## Quick Start

If you're new to PennyLane Keras Layer:

1. Start with [Getting Started](getting_started.md) for installation
2. Read the [User Guide](user_guide.md) to understand core concepts
3. Try the [Tutorials](tutorials.md) for hands-on learning
4. Reference the [API Documentation](api_reference.md) as needed

## Documentation Topics

### By Audience

**Beginners:**
- [Getting Started](getting_started.md)
- [Tutorial 1: First Quantum Model](tutorials.md#tutorial-1-first-quantum-model-jax)
- [Basic Example](examples.md#basic-example)

**Intermediate Users:**
- [User Guide](user_guide.md)
- [All Tutorials](tutorials.md)
- [Complete Examples](examples.md)

**Advanced Users:**
- [API Reference](api_reference.md)
- [Performance Optimization](user_guide.md#performance-optimization)
- [Custom Training Loop](examples.md#custom-training-loop)

**Contributors:**
- [Contributing Guide](contributing.md)
- [Code Standards](contributing.md#code-standards)
- [Development Setup](contributing.md#development-setup)

### By Topic

**Installation & Setup:**
- [Installation](getting_started.md#installation)
- [Setting the Keras Backend](getting_started.md#setting-the-keras-backend)
- [Development Installation](getting_started.md#development-installation)

**Basic Usage:**
- [Quick Example](index.md#quick-example)
- [First Quantum Model Tutorial](tutorials.md#tutorial-1-first-quantum-model-jax)
- [Basic Example](examples.md#basic-example)

**Architecture & Concepts:**
- [Core Concepts](user_guide.md#core-concepts)
- [Data Re-Uploading Model](user_guide.md#data-re-uploading-model)
- [QKerasLayer Structure](user_guide.md#qkeraslayer-structure)

**Multi-Backend Support:**
- [Backend Overview](user_guide.md#multi-backend-support)
- [JAX Tutorial](tutorials.md#tutorial-1-first-quantum-model-jax)
- [TensorFlow Tutorial](tutorials.md#tutorial-2-using-tensorflow-backend)
- [PyTorch Tutorial](tutorials.md#tutorial-3-using-pytorch-backend)

**Model Building:**
- [Model Building Patterns](user_guide.md#model-building-patterns)
- [Sequential API](user_guide.md#pattern-1-sequential-api-simplest)
- [Functional API](user_guide.md#pattern-2-functional-api-recommended)
- [Subclassing API](user_guide.md#pattern-3-subclassing-api-advanced)

**Training:**
- [Training Strategies](user_guide.md#training-strategies)
- [Optimizer Selection](user_guide.md#optimizer-selection)
- [Custom Training Loop](examples.md#custom-training-loop)

**Advanced Topics:**
- [Fourier Series Approximation](tutorials.md#tutorial-4-fourier-series-approximation)
- [Hyperparameter Tuning](tutorials.md#tutorial-6-hyperparameter-tuning)
- [Performance Optimization](user_guide.md#performance-optimization)

**Troubleshooting:**
- [FAQ](faq.md)
- [Error Messages](faq.md#error-messages)
- [Performance Questions](faq.md#performance-questions)

## Building Documentation (Optional)

While these markdown files are readable as-is, you can optionally build them into a website using tools like:

### MkDocs

```bash
# Install MkDocs
pip install mkdocs mkdocs-material

# Create mkdocs.yml configuration
# Serve locally
mkdocs serve

# Build static site
mkdocs build
```

### Sphinx

```bash
# Install Sphinx with markdown support
pip install sphinx sphinx-rtd-theme myst-parser

# Configure Sphinx
# Build HTML
make html
```

## Contributing to Documentation

See the [Contributing Guide](contributing.md) for:
- Documentation standards
- Docstring format
- How to submit documentation improvements

## License

This documentation is part of the PennyLane Keras Layer project and is licensed under the MIT License.
