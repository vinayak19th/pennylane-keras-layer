# Getting Started

This guide will help you get up and running with PennyLane Keras Layer.

## Installation

### Prerequisites

Before installing PennyLane Keras Layer, ensure you have:
- Python 3.8 or higher
- pip package manager

### Basic Installation

Install the package using pip:

```bash
pip install pennylane-keras-layer
```

This will install the core dependencies:
- PennyLane (>= 0.30.0)
- Keras (>= 3.0.0)
- NumPy (>= 1.21.0)

### Choosing a Backend

Keras 3 requires a backend. Install one of the following:

**TensorFlow (recommended for production):**
```bash
pip install tensorflow>=2.15.0
```

**JAX (recommended for performance):**
```bash
pip install jax jaxlib
```

**PyTorch:**
```bash
pip install torch>=2.0.0
```

### Development Installation

For development work, clone the repository and install in editable mode:

```bash
git clone https://github.com/vinayak19th/pennylane-keras-layer.git
cd pennylane-keras-layer
pip install -e ".[dev]"
```

This includes additional development dependencies:
- pytest (for testing)
- pytest-cov (for coverage reports)
- black (for code formatting)
- flake8 (for linting)
- mypy (for type checking)

## Verify Installation

Create a simple script to verify your installation:

```python
import pennylane as qml
import keras
from pennylane_keras_layer import QKerasLayer

print(f"✓ PennyLane version: {qml.__version__}")
print(f"✓ Keras version: {keras.__version__}")
print(f"✓ Keras backend: {keras.backend.backend()}")
print(f"✓ QKerasLayer imported successfully")
```

## Setting the Keras Backend

**Important:** The Keras backend must be set BEFORE importing Keras.

### Via Environment Variable (Recommended)

```python
import os
os.environ["KERAS_BACKEND"] = "jax"  # or "tensorflow" or "torch"

import keras  # Import AFTER setting the backend
```

### Via Configuration File

Create or edit `~/.keras/keras.json`:

```json
{
    "backend": "jax",
    "floatx": "float32"
}
```

## First Quantum Model

Here's a complete example to create and train your first quantum model:

```python
import os
os.environ["KERAS_BACKEND"] = "jax"

import numpy as np
import keras
from pennylane_keras_layer import QKerasLayer

# Generate sample data
X = np.linspace(-np.pi, np.pi, 100)
y = np.sin(X)

# Create a quantum layer
q_layer = QKerasLayer(
    layers=3,           # Number of quantum layers
    scaling=1.0,        # Input scaling factor
    num_wires=1,        # Number of qubits
    name="quantum_layer"
)

# Build model
model = keras.Sequential([
    keras.layers.Input(shape=(1,)),
    q_layer
])

# Compile model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.01),
    loss="mse"
)

# Train model
history = model.fit(X, y, epochs=50, verbose=1)

# Make predictions
predictions = model.predict(X)

print("Training complete!")
```

## Next Steps

Now that you have PennyLane Keras Layer installed:

1. **Read the [User Guide](user_guide.md)** to understand core concepts
2. **Check [API Reference](api_reference.md)** for detailed parameter descriptions
3. **Explore [Tutorials](tutorials.md)** for backend-specific examples
4. **Run the [Examples](examples.md)** in the repository

## Troubleshooting

### Import Errors

**Problem:** `ImportError: No module named 'pennylane_keras_layer'`

**Solution:** Ensure the package is installed:
```bash
pip install pennylane-keras-layer
```

### Backend Issues

**Problem:** `ValueError: Unsupported Keras backend`

**Solution:** Set the backend before importing Keras:
```python
import os
os.environ["KERAS_BACKEND"] = "jax"  # Must be BEFORE import keras
import keras
```

### Version Conflicts

**Problem:** Dependency version conflicts

**Solution:** Create a fresh virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install pennylane-keras-layer
```

For more issues, see the [FAQ](faq.md) or [open an issue on GitHub](https://github.com/vinayak19th/pennylane-keras-layer/issues).
