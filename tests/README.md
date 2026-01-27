# Test Suite for PennyLane-Keras-Layer

This directory contains comprehensive unit tests for the pennylane-keras-layer package.

## Test Structure

```
tests/
├── __init__.py              # Test package initialization
├── conftest.py              # Pytest configuration and fixtures
├── test_basic.py            # Basic import and dependency tests
├── test_layer.py            # Unit tests for QKerasLayer
├── test_integration.py      # Integration tests (model training, saving, etc.)
├── test_edge_cases.py       # Edge cases and parametric tests
└── test_utils.py            # Utility and package structure tests
```

## Test Categories

### Basic Tests (`test_basic.py`)
- Package import verification
- Dependency availability checks (PennyLane, Keras)

### Layer Tests (`test_layer.py`)
- QKerasLayer initialization
- Layer building and weight creation
- Forward pass functionality
- Serialization/deserialization
- Error handling

### Integration Tests (`test_integration.py`)
- Model training end-to-end
- Model saving and loading
- Integration with other Keras layers
- Batch processing
- Gradient flow verification

### Edge Case Tests (`test_edge_cases.py`)
- Different layer counts (1, 2, 3, 5 layers)
- Different wire counts (1, 2, 3 wires)
- Different scaling factors
- Different quantum backends
- Extreme input values
- Reproducibility tests

### Utility Tests (`test_utils.py`)
- Package version and metadata
- Package structure verification
- Import path testing

## Running Tests

### Run all tests
```bash
pytest tests/
```

### Run with verbose output
```bash
pytest tests/ -v
```

### Run specific test file
```bash
pytest tests/test_layer.py
```

### Run specific test function
```bash
pytest tests/test_layer.py::test_qkeras_layer_initialization
```

### Run tests with coverage
```bash
pytest tests/ --cov=pennylane_keras_layer --cov-report=html
```

### Run only integration tests
```bash
pytest tests/ -m integration
```

### Skip slow tests
```bash
pytest tests/ -m "not slow"
```

## Test Markers

- `@pytest.mark.integration` - Integration tests (may be slower)
- `@pytest.mark.slow` - Slow running tests
- `@pytest.mark.backend(name)` - Tests for specific backends

## Dependencies

Most tests will skip gracefully if required dependencies are not installed:
- `pennylane` - Required for quantum circuit functionality
- `keras` - Required for neural network integration
- `numpy` - Required for numerical operations
- `matplotlib` - Optional, for visualization tests

## Writing New Tests

When adding new tests:

1. **Use descriptive names**: `test_<what_is_being_tested>`
2. **Add docstrings**: Explain what the test validates
3. **Handle imports gracefully**: Use try/except and `pytest.skip()` for missing dependencies
4. **Use fixtures**: Define reusable test data in `conftest.py`
5. **Add markers**: Mark integration tests, slow tests, etc.
6. **Test edge cases**: Consider boundary conditions and error cases

### Example Test Template

```python
def test_feature_name():
    """Test that feature works correctly."""
    try:
        from pennylane_keras_layer import QKerasLayer
        
        # Setup
        layer = QKerasLayer(layers=2, num_wires=1)
        
        # Exercise
        result = layer.some_method()
        
        # Verify
        assert result is not None
        assert some_condition
        
    except ImportError:
        pytest.skip("Required dependencies not installed")
```

## Continuous Integration

These tests are designed to run in CI/CD pipelines. They handle missing dependencies gracefully and provide clear error messages when tests fail.

## Test Coverage Goals

- **Unit Tests**: >80% code coverage
- **Integration Tests**: All major user workflows
- **Edge Cases**: Common boundary conditions and error scenarios

## Troubleshooting

### Tests are skipped
- Check that all dependencies are installed: `pip install -e ".[dev]"`

### Import errors
- Ensure package is installed in development mode: `pip install -e .`

### Backend-specific failures
- Some tests may only work with specific Keras backends (TensorFlow, JAX, PyTorch)
- Set backend with: `export KERAS_BACKEND=jax`

## Contributing

When contributing new features:
1. Add corresponding unit tests
2. Ensure existing tests still pass
3. Update this README if adding new test categories
