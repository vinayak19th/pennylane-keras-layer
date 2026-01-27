# Contributing to PennyLane Keras Layer

Thank you for your interest in contributing to PennyLane Keras Layer! This document provides guidelines for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Development Workflow](#development-workflow)
- [Code Standards](#code-standards)
- [Testing](#testing)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)
- [Release Process](#release-process)

---

## Code of Conduct

### Our Standards

- Be respectful and inclusive
- Welcome newcomers and help them learn
- Focus on constructive feedback
- Assume good intentions

### Unacceptable Behavior

- Harassment or discrimination
- Trolling or insulting comments
- Publishing private information
- Unprofessional conduct

---

## Getting Started

### Ways to Contribute

You can contribute in many ways:

1. **Report Bugs**: Open an issue describing the bug
2. **Suggest Features**: Propose new features or improvements
3. **Fix Issues**: Submit pull requests for open issues
4. **Improve Documentation**: Fix typos, add examples, clarify concepts
5. **Write Tests**: Increase test coverage
6. **Review Code**: Review and comment on pull requests

### First Time Contributors

Look for issues labeled:
- `good first issue`: Good for newcomers
- `help wanted`: Extra attention needed
- `documentation`: Documentation improvements

---

## Development Setup

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/pennylane-keras-layer.git
cd pennylane-keras-layer
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate it
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Development Dependencies

```bash
# Install package in editable mode with dev dependencies
pip install -e ".[dev]"

# Install a backend (choose one or more)
pip install jax jaxlib      # For JAX
pip install tensorflow      # For TensorFlow
pip install torch           # For PyTorch
```

### 4. Verify Installation

```bash
# Run tests to verify setup
pytest tests/ -v
```

---

## How to Contribute

### Reporting Bugs

When reporting bugs, include:

1. **Description**: Clear description of the bug
2. **Reproduction Steps**: Minimal code to reproduce
3. **Expected Behavior**: What you expected to happen
4. **Actual Behavior**: What actually happened
5. **Environment**:
   - Python version
   - Package versions (`pip list`)
   - Operating system
   - Keras backend

**Template:**

```markdown
## Bug Description
Brief description of the bug.

## Reproduction
```python
import keras
from pennylane_keras_layer import QKerasLayer

# Minimal code to reproduce
```

## Expected Behavior
What should happen.

## Actual Behavior
What actually happens.

## Environment
- Python: 3.9.7
- pennylane-keras-layer: 0.1.0
- PennyLane: 0.31.0
- Keras: 3.0.2
- Backend: JAX 0.4.13
- OS: Ubuntu 22.04
```

### Suggesting Features

When suggesting features:

1. **Use Case**: Describe the use case
2. **Proposed Solution**: How you envision it working
3. **Alternatives**: Other approaches you considered
4. **Implementation**: Ideas for implementation (optional)

---

## Development Workflow

### 1. Create a Branch

```bash
# Update main branch
git checkout main
git pull upstream main

# Create feature branch
git checkout -b feature/your-feature-name
# or
git checkout -b fix/bug-description
```

### Branch Naming

- `feature/`: New features
- `fix/`: Bug fixes
- `docs/`: Documentation changes
- `test/`: Test additions/changes
- `refactor/`: Code refactoring

### 2. Make Changes

- Write clear, focused commits
- Follow code standards (see below)
- Add tests for new features
- Update documentation as needed

### 3. Commit Changes

```bash
# Stage changes
git add .

# Commit with descriptive message
git commit -m "Add support for multi-qubit circuits

- Implement multi-qubit encoding
- Add tests for 2+ qubits
- Update documentation"
```

**Commit Message Guidelines:**

- Use present tense ("Add feature" not "Added feature")
- First line: brief summary (50 chars or less)
- Blank line after first line
- Detailed description if needed
- Reference issues: "Fixes #123" or "Relates to #456"

### 4. Push Changes

```bash
git push origin feature/your-feature-name
```

### 5. Open Pull Request

1. Go to GitHub repository
2. Click "New Pull Request"
3. Select your branch
4. Fill in PR template
5. Submit for review

---

## Code Standards

### Python Style

Follow PEP 8 with these tools:

**Black (Code Formatting):**
```bash
# Format code
black pennylane_keras_layer/ tests/

# Check formatting
black --check pennylane_keras_layer/ tests/
```

**Flake8 (Linting):**
```bash
# Lint code
flake8 pennylane_keras_layer/ tests/

# Configuration in setup.cfg or .flake8
```

### Code Quality

- **Line Length**: Max 100 characters (configured in Black)
- **Imports**: Group stdlib, third-party, local; sorted alphabetically
- **Docstrings**: Use Google-style docstrings
- **Type Hints**: Add where it improves clarity (optional)

### Example Code Style

```python
"""
Module description.

This module provides...
"""

import os
from typing import Optional

import keras
import numpy as np
import pennylane as qml


class MyLayer(keras.layers.Layer):
    """Brief description.
    
    Longer description explaining the layer's purpose and behavior.
    
    Args:
        param1 (int): Description of param1.
        param2 (float, optional): Description of param2. Defaults to 1.0.
    
    Example:
        >>> layer = MyLayer(param1=5)
        >>> output = layer(input_tensor)
    """
    
    def __init__(self, param1: int, param2: float = 1.0, **kwargs):
        """Initialize the layer."""
        super().__init__(**kwargs)
        self.param1 = param1
        self.param2 = param2
    
    def call(self, inputs):
        """Forward pass.
        
        Args:
            inputs: Input tensor.
            
        Returns:
            Output tensor.
        """
        # Implementation
        return output
```

---

## Testing

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_layer.py -v

# Run with coverage
pytest tests/ --cov=pennylane_keras_layer --cov-report=term-missing

# Run specific test
pytest tests/test_layer.py::test_layer_creation -v
```

### Writing Tests

Use pytest framework:

```python
"""
Test module for MyFeature.
"""
import pytest
import numpy as np
import keras
from pennylane_keras_layer import QKerasLayer


def test_feature_basic():
    """Test basic functionality."""
    layer = QKerasLayer(layers=2)
    assert layer.layers == 2


def test_feature_with_data():
    """Test with actual data."""
    X = np.random.randn(10, 1)
    
    model = keras.Sequential([
        keras.layers.Input(shape=(1,)),
        QKerasLayer(layers=2)
    ])
    
    output = model(X)
    
    assert output.shape == (10, 1)


@pytest.mark.parametrize("layers,scaling", [
    (2, 1.0),
    (3, 2.0),
    (4, 0.5),
])
def test_feature_parametrized(layers, scaling):
    """Test with multiple parameter combinations."""
    layer = QKerasLayer(layers=layers, scaling=scaling)
    assert layer.layers == layers
    assert layer.scaling == scaling


def test_feature_error_handling():
    """Test error conditions."""
    with pytest.raises(ValueError):
        QKerasLayer(layers=0)  # Should raise ValueError
```

### Test Guidelines

- **Coverage**: Aim for >80% code coverage
- **Isolation**: Each test should be independent
- **Clear Names**: Test names should describe what they test
- **Assertions**: Use specific assertions with messages
- **Edge Cases**: Test boundary conditions and edge cases

---

## Documentation

### Docstring Format

Use Google-style docstrings:

```python
def my_function(arg1, arg2, kwarg1=None):
    """Brief description of function.
    
    More detailed description explaining what the function does,
    how it works, and any important details.
    
    Args:
        arg1 (type): Description of arg1.
        arg2 (type): Description of arg2.
        kwarg1 (type, optional): Description of kwarg1. Defaults to None.
    
    Returns:
        type: Description of return value.
    
    Raises:
        ValueError: When arg1 is invalid.
        TypeError: When arg2 has wrong type.
    
    Example:
        >>> result = my_function(1, 2, kwarg1=3)
        >>> print(result)
        6
    """
    # Implementation
```

### Documentation Files

Update relevant docs in `docs/`:

- `api_reference.md`: API changes
- `user_guide.md`: Usage patterns, concepts
- `tutorials.md`: Step-by-step examples
- `examples.md`: Complete examples
- `faq.md`: Common questions
- `CHANGELOG.md`: Version history

### Documentation Standards

- **Clarity**: Write for users, not just developers
- **Examples**: Include working code examples
- **Completeness**: Document all public APIs
- **Updates**: Keep docs in sync with code

---

## Pull Request Process

### Before Submitting

- [ ] Code follows style guidelines
- [ ] Tests pass locally
- [ ] New tests added for new features
- [ ] Documentation updated
- [ ] Commit messages are clear
- [ ] Branch is up to date with main

### PR Checklist

```markdown
## Description
Brief description of changes.

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing
Describe testing done:
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Manual testing performed

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex code
- [ ] Documentation updated
- [ ] No new warnings generated
- [ ] Tests pass locally
```

### Review Process

1. **Automated Checks**: CI runs tests, linting
2. **Code Review**: Maintainers review code
3. **Feedback**: Address review comments
4. **Approval**: Get approval from maintainer
5. **Merge**: Maintainer merges PR

### After Merge

- Delete your feature branch
- Update your local main branch
- Close related issues (if applicable)

---

## Release Process

For maintainers:

### 1. Version Bump

Update version in:
- `pyproject.toml`
- `pennylane_keras_layer/__init__.py`

### 2. Update Changelog

Update `CHANGELOG.md`:

```markdown
## [0.2.0] - 2024-01-15

### Added
- New feature description

### Changed
- Changed behavior description

### Fixed
- Bug fix description

### Deprecated
- Deprecated feature warning
```

### 3. Create Release

```bash
# Tag release
git tag -a v0.2.0 -m "Release version 0.2.0"
git push origin v0.2.0

# Create GitHub release with notes
```

### 4. Publish to PyPI

```bash
# Build package
python -m build

# Upload to PyPI
python -m twine upload dist/*
```

---

## Questions?

- **General**: Open a [GitHub Discussion](https://github.com/vinayak19th/pennylane-keras-layer/discussions)
- **Bugs**: Open an [Issue](https://github.com/vinayak19th/pennylane-keras-layer/issues)
- **Security**: Email maintainers privately

Thank you for contributing! 🎉
