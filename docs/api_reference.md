# API Reference

This page provides detailed documentation for the PennyLane Keras Layer API.

## KerasDRCircuitLayer

```python
pennylane_keras_layer.KerasDRCircuitLayer(
    layers,
    scaling=1.0,
    circ_backend="lightning.qubit",
    circ_grad_method="adjoint",
    num_wires=1,
    use_jax_python=False,
    **kwargs
)
```

A Keras Layer wrapping a PennyLane QNode for Data Re-Uploading.

This layer implements a Data Re-Uploading quantum machine learning model that can be integrated into Keras models with full multi-backend support (TensorFlow, JAX, PyTorch).

### Parameters

#### layers : int
Number of layers in the Data Re-Uploading model. Must be a positive integer.

#### scaling : float, optional (default=1.0)
Scaling factor applied to input data before encoding into the quantum circuit.

#### circ_backend : str, optional (default="lightning.qubit")
PennyLane device/backend to use for quantum circuit execution.

#### circ_grad_method : str, optional (default="adjoint")
Gradient computation method for the quantum circuit.

#### num_wires : int, optional (default=1)
Number of qubits/wires to initialize the quantum device with.

#### use_jax_python : bool, optional (default=False)
Flag to use the JAX-Python interface instead of the standard JAX interface.

#### **kwargs
Additional keyword arguments passed to the parent `keras.layers.Layer` class.

### Example

```python
import os
os.environ["KERAS_BACKEND"] = "jax"
import keras
from pennylane_keras_layer import KerasDRCircuitLayer

# Create a quantum layer
q_layer = KerasDRCircuitLayer(layers=2, scaling=1.0, num_wires=1)

# Build a model
inp = keras.layers.Input(shape=(1,))
out = q_layer(inp)
model = keras.models.Model(inputs=inp, outputs=out)
```

## KerasCircuitLayer

```python
pennylane_keras_layer.KerasCircuitLayer(
    qnode,
    weight_shapes,
    output_dim=1,
    use_jax_python=False,
    weight_specs=None,
    **kwargs
)
```

A generic Keras Layer wrapping a PennyLane QNode.

This layer enables the integration of arbitrary PennyLane quantum circuits into Keras models.

### Parameters

#### qnode : qml.QNode
The PennyLane QNode to be converted into a Keras layer. The QNode must accept an input argument (default name "inputs") and trainable weights.

#### weight_shapes : dict[str, tuple]
A dictionary mapping from all weights used in the QNode to their corresponding shapes.

#### output_dim : int or tuple, optional (default=1)
The output dimension of the QNode.

#### use_jax_python : bool, optional (default=False)
Flag to use the vectorized jax backend.

#### weight_specs : dict[str, dict], optional
An optional dictionary for users to provide additional specifications for weights, such as initialization methods.

#### **kwargs
Additional keyword arguments for the Keras Layer class.

### Methods

#### set_input_argument(input_name="inputs")
Static method to set the name of the input argument in the QNode signature.

```python
KerasCircuitLayer.set_input_argument("x")
```

#### set_qnode(qnode)
Method to restore the QNode after loading a saved model (since QNodes are not serializable).

### Example

```python
import pennylane as qml
from pennylane_keras_layer import KerasCircuitLayer

# 1. Define QNode
dev = qml.device("default.qubit", wires=2)

@qml.qnode(dev)
def circuit(inputs, w):
    qml.RX(inputs[0], wires=0)
    qml.Rot(*w, wires=0)
    return qml.expval(qml.Z(0))

# 2. Define shapes
weight_shapes = {"w": (3,)}

# 3. Create Layer
qlayer = KerasCircuitLayer(circuit, weight_shapes, output_dim=1)
```

## Module-Level Exports

### \_\_version\_\_
String containing the package version.

### \_\_author\_\_
String containing the package author.
