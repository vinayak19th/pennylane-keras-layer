# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2026-01-28

### Added

#### Core Features
- **KerasCircuitLayer**: Generic quantum layer wrapper for PennyLane QNodes
  - Full multi-backend support (TensorFlow, JAX, PyTorch)
  - Customizable quantum circuit parameters
  - Automatic weight initialization
  - Full serialization support for model saving/loading
  
- **KerasDRCircuitLayer**: Data Re-Uploading quantum circuit layer
  - Implements powerful quantum ML paradigm
  - Configurable number of layers and scaling
  - Built-in circuit architecture for easy deployment

#### Documentation
- Comprehensive documentation suite in `docs/` directory:
  - Getting Started guide
  - User Guide with in-depth concepts and best practices
  - API Reference with detailed documentation
  - Tutorials for different backends (JAX, TensorFlow, PyTorch)
  - Complete working examples
  - FAQ section
  - Contributing guidelines

#### Examples
- `basic_example.py`: Simple quantum layer integration
- `fourier_series_example.py`: Fourier series approximation demo
- `tutorial_qnn_module_keras.py`: QNN module tutorial
- Jupyter notebooks for interactive testing:
  - `RandomTrainingTest.ipynb`
  - `Testing.ipynb`

#### Testing Infrastructure
- Comprehensive test suite covering:
  - Basic functionality tests
  - Edge case handling
  - Generic layer tests
  - Integration tests
  - Utility function tests
- pytest configuration with coverage reporting
- Multi-backend testing support

#### CI/CD
- GitHub Actions workflows for:
  - Python package testing
  - Automated publishing to PyPI
  - Multi-Python version support (3.10, 3.11)

#### Project Infrastructure
- MIT License
- Complete project metadata in `pyproject.toml`
- Package configuration for setuptools
- `.gitignore` for Python projects
- Distribution packages (wheels and source distributions)

### Features Highlights

- 🔄 **Multi-Backend Support**: Seamless integration with JAX, TensorFlow, and PyTorch
- 🎯 **Easy Integration**: Drop-in replacement for standard Keras layers
- ⚡ **High Performance**: JIT compilation support with JAX backend
- 💾 **Serialization**: Full support for model saving and loading
- 🔧 **Flexible Configuration**: Customizable quantum circuit parameters
- 📊 **Data Re-Uploading**: Implements powerful quantum ML paradigm

### Technical Details

- Python compatibility: >= 3.10
- Core dependencies:
  - PennyLane >= 0.30.0
  - Keras >= 3.0.0
  - NumPy >= 1.21.0
- Development dependencies:
  - pytest >= 7.0.0
  - pytest-cov >= 4.0.0
  - black >= 23.0.0
  - flake8 >= 6.0.0
  - mypy >= 1.0.0

### Files Changed
- 41 files changed
- 7,805 insertions(+)
- Initial release with complete codebase

---

[1.1.0]: https://github.com/vinayak19th/pennylane-keras-layer/releases/tag/v1.1.0
