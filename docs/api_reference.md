# API Reference

This page provides detailed documentation for the PennyLane Keras Layer API.

## QKerasLayer

```python
pennylane_keras_layer.QKerasLayer(
    layers,
    scaling=1.0,
    circ_backend="lightning.qubit",
    circ_grad_method="adjoint",
    num_wires=1,
    use_jax_python=False,
    **kwargs
)
```

A Keras Layer that wraps a PennyLane quantum circuit (QNode).

This layer implements a Data Re-Uploading quantum machine learning model that can be integrated into Keras models with full multi-backend support (TensorFlow, JAX, PyTorch).

### Parameters

#### layers : int
Number of layers in the Data Re-Uploading model. Must be a positive integer.

Each layer consists of:
1. A trainable rotation block (`qml.Rot`)
2. A data encoding block (`qml.RX`)

The final layer only includes the trainable rotation block.

**Example:**
```python
q_layer = QKerasLayer(layers=3)  # 3 layers
```

#### scaling : float, optional (default=1.0)
Scaling factor applied to input data before encoding into the quantum circuit.

This parameter allows you to control the range of input values fed into the quantum gates. Higher values increase the sensitivity to input variations.

**Example:**
```python
q_layer = QKerasLayer(layers=2, scaling=2.0)
```

#### circ_backend : str, optional (default="lightning.qubit")
PennyLane device/backend to use for quantum circuit execution.

Common options:
- `"default.qubit"`: Default simulator (slower but more general)
- `"lightning.qubit"`: Fast C++ simulator (recommended for CPU)
- `"default.qubit.jax"`: JAX-based simulator
- Any other PennyLane device

**Example:**
```python
q_layer = QKerasLayer(layers=2, circ_backend="default.qubit")
```

#### circ_grad_method : str, optional (default="adjoint")
Gradient computation method for the quantum circuit.

Options:
- `"adjoint"`: Efficient adjoint method (fast, works for simulators)
- `"parameter-shift"`: Parameter-shift rule (works on hardware)
- `"backprop"`: Backpropagation (only for simulators)
- `"finite-diff"`: Finite differences

**Example:**
```python
q_layer = QKerasLayer(layers=2, circ_grad_method="parameter-shift")
```

#### num_wires : int, optional (default=1)
Number of qubits/wires to initialize the quantum device with.

**Note:** Current implementation uses only wire 0 for computations. Multi-wire support may be added in future versions.

**Example:**
```python
# Currently, use num_wires=1
q_layer = QKerasLayer(layers=2, num_wires=1)
```

#### use_jax_python : bool, optional (default=False)
Flag to use the JAX-Python interface instead of the standard JAX interface.

When `True`, uses PennyLane's "jax-python" interface which:
- Does NOT support `jax.jit` compilation
- Provides better compatibility with some JAX features
- May have different performance characteristics

Only relevant when Keras backend is set to JAX.

**Example:**
```python
q_layer = QKerasLayer(layers=2, use_jax_python=True)
```

#### **kwargs
Additional keyword arguments passed to the parent `keras.layers.Layer` class.

Common options:
- `name`: String name for the layer
- `trainable`: Boolean, whether layer weights are trainable
- `dtype`: Data type for the layer

**Example:**
```python
q_layer = QKerasLayer(layers=2, name="my_quantum_layer", trainable=True)
```

### Attributes

#### layer_weights
Trainable weight tensor of shape `(layers + 1, 3)`.

Each row contains three rotation angles for the `qml.Rot` gate in each layer.
Initialized randomly in the range [0, 2π).

Accessible after the layer is built.

#### circuit
The PennyLane QNode representing the quantum circuit.

For JAX backend (non-python), this is a JIT-compiled version of the circuit.

#### is_built : bool
Flag indicating whether the layer has been built.

#### interface : str
PennyLane interface being used ("tf", "torch", "jax", or "jax-python").

Automatically determined from the Keras backend.

### Methods

#### build(input_shape)
Initialize the layer weights based on input shape.

This method is called automatically by Keras when the layer is first used.

**Parameters:**
- `input_shape` (tuple): Shape of the input tensor

**Example:**
```python
q_layer = QKerasLayer(layers=2)
q_layer.build((None, 1))  # Batch size None, input dim 1
```

#### call(inputs)
Forward pass of the layer.

**Parameters:**
- `inputs`: Input tensor

**Returns:**
- Output tensor with shape `(batch_size, num_wires)`

**Raises:**
- `RuntimeError`: If called before the layer is built

#### compute_output_shape(input_shape)
Compute the output shape of the layer.

**Parameters:**
- `input_shape` (tuple): Input tensor shape

**Returns:**
- `tuple`: Output shape `(batch_size, num_wires)`

#### draw_qnode()
Visualize the quantum circuit.

Creates a matplotlib figure showing the circuit structure with random input.

**Raises:**
- `RuntimeError`: If called before the layer is built

**Example:**
```python
import matplotlib.pyplot as plt

q_layer = QKerasLayer(layers=2)
model = keras.Sequential([keras.layers.Input(shape=(1,)), q_layer])
model.build((None, 1))

q_layer.draw_qnode()
plt.show()
```

#### get_config()
Return the layer configuration as a dictionary.

Used for serialization when saving models.

**Returns:**
- `dict`: Configuration dictionary

#### from_config(config)
Create a layer instance from a configuration dictionary.

Class method used for deserialization when loading models.

**Parameters:**
- `config` (dict): Configuration dictionary

**Returns:**
- `QKerasLayer`: New layer instance

## Usage Examples

### Basic Usage

```python
import keras
from pennylane_keras_layer import QKerasLayer

# Create layer
q_layer = QKerasLayer(layers=3, scaling=1.0)

# Use in a model
model = keras.Sequential([
    keras.layers.Input(shape=(1,)),
    q_layer,
    keras.layers.Dense(1)
])
```

### With Custom Parameters

```python
q_layer = QKerasLayer(
    layers=5,
    scaling=2.5,
    circ_backend="default.qubit",
    circ_grad_method="parameter-shift",
    num_wires=1,
    name="custom_quantum_layer"
)
```

### Functional API

```python
inputs = keras.layers.Input(shape=(10,))
x = keras.layers.Dense(1)(inputs)  # Reduce to 1D for quantum layer
x = QKerasLayer(layers=4, scaling=1.0)(x)
outputs = keras.layers.Dense(1)(x)
model = keras.models.Model(inputs=inputs, outputs=outputs)
```

### Model Serialization

```python
# Save model
model.save("my_quantum_model.keras")

# Load model
loaded_model = keras.models.load_model("my_quantum_model.keras")
```

## Module-Level Exports

### \_\_version\_\_
String containing the package version.

```python
from pennylane_keras_layer import __version__
print(__version__)  # '0.1.0'
```

### \_\_author\_\_
String containing the package author.

```python
from pennylane_keras_layer import __author__
```

## Constants

The package uses the following internal constants:

- Default circuit backend: `"lightning.qubit"`
- Default gradient method: `"adjoint"`
- Default scaling factor: `1.0`
- Weight initialization range: `[0, 2π)`

## Type Hints

While the library does not enforce strict typing, the expected types are:

```python
layers: int
scaling: float
circ_backend: str
circ_grad_method: str
num_wires: int
use_jax_python: bool
```

## See Also

- [User Guide](user_guide.md): Conceptual overview and best practices
- [Tutorials](tutorials.md): Step-by-step examples
- [Examples](examples.md): Complete working examples
