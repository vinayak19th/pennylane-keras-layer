"""
Keras Layer implementation for PennyLane quantum circuits.

This module provides the KerasCircuitLayer class that wraps PennyLane QNodes
as Keras layers, enabling quantum circuits in neural networks with full
multi-backend support (TensorFlow, JAX, PyTorch).
"""

import keras
from keras.utils import register_keras_serializable
from keras.saving import serialize_keras_object, deserialize_keras_object
from keras import ops
import pennylane as qml
import numpy as np
import inspect
from collections.abc import Iterable
from typing import Optional, Text

@register_keras_serializable(package="PennylaneKeras", name="KerasCircuitLayer")
class KerasCircuitLayer(keras.layers.Layer):
    """A Keras Layer wrapping a PennyLane QNode.
    
    This layer enables the integration of PennyLane quantum circuits into 
    Keras models with full multi-backend support.
    
    Args:
        qnode (qml.QNode): The PennyLane QNode to be executed.
        weight_shapes (dict): A dictionary mapping weight argument names to their shapes.
        output_dim (int): The output dimension of the QNode. Optional.
        use_jax_python (bool): Flag to use the vectorized jax backend. 
        weight_specs (dict[str, dict]): An optional dictionary for users to provide additional
            specifications for weights used in the QNode, such as the method of parameter
            initialization. This specification is provided as a dictionary with keys given by the
            arguments of the `add_weight()
            <https://keras.io/api/layers/base_layer/#addweight-method>`__
            method and values being the corresponding specification.
        **kwargs: Additional keyword arguments for the Keras Layer class.
    """
    
    def __init__(
        self,
        qnode:qml.QNode,
        weight_shapes: dict,
        output_dim: int = None,
        use_jax_python: bool = False,
        weight_specs = None,
        **kwargs
    ):
        """Initialize the KerasCircuitLayer."""
        super().__init__(**kwargs)
        
        
        self.qnode = qnode
        self.weight_shapes = weight_shapes
        self.output_dim = output_dim
        self.qnode_weights = {}
        self.use_jax_python = use_jax_python
        self.weight_specs = weight_specs if weight_specs is not None else {}

        if qnode==None:
            print("Warning: QNode loaded as None. This is normally the case after loading the model from a file. Please use 'set_qnode' method to restore the qnode")
        else:
            self._signature_validation(qnode, weight_shapes)

        # Allows output_dim to be specified as an int or as a tuple, e.g, 5, (5,), (5, 2), [5, 2]
        # Note: Single digit values will be considered an int and multiple as a tuple, e.g [5,] or (5,)
        # are passed as integer 5 and [5, 2] will be passes as tuple (5, 2)
        if isinstance(output_dim, Iterable) and len(output_dim) > 1:
            self.output_dim = tuple(output_dim)
        else:
            self.output_dim = output_dim[0] if isinstance(output_dim, Iterable) else output_dim

        # Define Keras Layer flags
        self.is_built: bool = False
        
        # Selecting the Pennylane interface based on keras backend
        backend = keras.config.backend()
        if backend == "torch":
            self.interface = "torch"
        elif backend == "tensorflow":
            self.interface = "tf"
        elif backend == "jax":
            if use_jax_python:
                self.interface = "jax-python"
            else:
                self.interface = "jax"
        else:
            raise ValueError(
                f"Unsupported Keras backend: {backend}. "
                f"Supported backends are: 'torch', 'tensorflow', 'jax'"
            )
        
        self.build(None)

    def _signature_validation(self, qnode, weight_shapes):
        sig = inspect.signature(qnode.func).parameters

        if self.input_arg not in sig:
            raise TypeError(
                f"QNode must include an argument with name {self.input_arg} for inputting data"
            )

        if self.input_arg in set(weight_shapes.keys()):
            raise ValueError(
                f"{self.input_arg} argument should not have its dimension specified in "
                f"weight_shapes"
            )

        param_kinds = [p.kind for p in sig.values()]

        if inspect.Parameter.VAR_POSITIONAL in param_kinds:
            raise TypeError("Cannot have a variable number of positional arguments")

        if inspect.Parameter.VAR_KEYWORD not in param_kinds:
            if set(weight_shapes.keys()) | {self.input_arg} != set(sig.keys()):
                raise ValueError("Must specify a shape for every non-input parameter in the QNode")
    @property
    def input_arg(self):
        """Name of the argument to be used as the input to the Keras
        `Layer <https://keras.io/api/layers/base_layer/#layer-class>`__. Set to
        ``"inputs"``."""
        return self._input_arg
    
    @staticmethod
    def set_input_argument(input_name: Text = "inputs") -> None:
        """
        Set the name of the input argument.

        Args:
            input_name (str): Name of the input argument
        """
        KerasCircuitLayer._input_arg = input_name
             
    def build(self, input_shape):
        """Initialize the layer weights."""
        for weight, size in self.weight_shapes.items():
            spec = self.weight_specs.get(weight, {})
            if 'initializer' not in spec:
                self.qnode_weights[weight] = self.add_weight(name=weight, 
                    shape=size, 
                    initializer=keras.initializers.random_uniform(minval=0, maxval=2 * np.pi),
                    **spec)
            else:
                self.qnode_weights[weight] = self.add_weight(name=weight, 
                    shape=size, **spec)

        self.built = True
        # Create Quantum Circuit
        self.circuit = self.create_circuit()
        self.is_built = True
    
    def create_circuit(self):
        """Create the PennyLane device and QNode."""
        # Create device once
        if self.qnode == None:
            print("Delaying circuit creation till QNode is set using the 'set_qnode' method")
            return None
        else:
            circuit_node = self.qnode
            circuit_node.interface = self.interface
            if self.interface == "jax":
                import jax
                return jax.jit(circuit_node)
            else:
                return circuit_node
    
    def draw_qnode(self,input, **kwargs):
        """Draw the quantum circuit.

        Args:
            input (tensor-like): Input data to the circuit.
            **kwargs: Additional keyword arguments to be passed to `qml.draw_mpl`.

        Raises:
            RuntimeError: If the layer has not been built.
        """
        if not self.is_built:
            raise RuntimeError(
                "KerasDRCircuitLayer must be built before drawing."
            )

        weight_values = [self.qnode_weights[k] for k in self.weight_shapes.keys()]
        if keras.config.backend() == "jax":
            # Use .value to get the underlying value for JIT compatibility
            weight_values = [w.value for w in self.qnode_weights.values()]
            qml.draw_mpl(self.circuit.func,**kwargs)(weight_values, input)
        else:
            qml.draw_mpl(self.circuit,**kwargs)(weight_values, input)
        
    def call(self, inputs):
        """Execute the QNode.
        
        The inputs are passed as the last positional argument to the QNode,
        after the weights (which are unpacked from the dictionary in order).
        """
        # Check for TF graph mode
        if keras.config.backend() == "tensorflow" and not keras.config.run_eagerly():
             import tensorflow as tf
             if not tf.executing_eagerly():
                 raise RuntimeError(
                     "KerasCircuitLayer does not support TensorFlow graph mode (e.g. inside @tf.function) "
                     "directly. Please use Eager execution or the 'tf' interface in PennyLane."
                 )

        # Prepare arguments
        # We pass weights as positional arguments in the order of keys in weight_shapes
        # User requested "inputs last", so we append inputs at the end.
        
        weight_values = [self.qnode_weights[k] for k in self.weight_shapes.keys()]
        
        if keras.config.backend() == "jax":
            # Use .value to get the underlying value for JIT compatibility
            weight_values = [w.value for w in weight_values]
            
        res = self.circuit(*weight_values, inputs)
        
        # If the QNode returns a list of results (multiple measurements), stack them
        if isinstance(res, (list, tuple)):
            return ops.stack(res, axis=-1)
            
        return res

    # def __getattr__(self, item):
    #     """If the given attribute does not exist in the class, look for it in the wrapped QNode."""
    #     if self._initialized and hasattr(self.qnode, item):
    #         return getattr(self.qnode, item)

    #     return super().__getattr__(item)

    # def __setattr__(self, item, val):
    #     """If the given attribute does not exist in the class, try to set it in the wrapped QNode."""
    #     if self._initialized and hasattr(self.qnode, item):
    #         setattr(self.qnode, item, val)
    #     else:
    #         super().__setattr__(item, val)
    
    def compute_output_shape(self, input_shape:tuple):
        """Computes the output shape after passing data of shape ``input_shape`` through the
        QNode.

        Args:
            input_shape (tuple): shape of input data

        Returns:
            tuple: shape of output data
        """
        if self.output_dim:
             if self.output_dim == 1:
                 return (input_shape[0],)
             return (input_shape[0], self.output_dim)
        return input_shape 

    def get_config(self):
        config = super().get_config()
        config.update({
            "weight_shapes": serialize_keras_object(self.weight_shapes),
            "output_dim": serialize_keras_object(self.output_dim),
            "use_jax_python": serialize_keras_object(self.use_jax_python),
        })
        return config
    
    def __str__(self):
        return f"<Quantum Keras ({self.interface} backend) Layer: func={self.qnode.func.__name__}>"

    def set_qnode(self, qnode:qml.QNode):
        """_summary_

        Args:
            qnode (qml.QNode): _description_
        """
        self.qnode = qnode
        print("Verifying QNode compatibility")
        self._signature_validation(qnode, self.weight_shapes)
        print("Setting QNode")
        self.circuit = self.create_circuit()

    @classmethod
    def from_config(cls, config):
        weight_shapes = deserialize_keras_object(config.pop("weight_shapes", {}))
        output_dim = deserialize_keras_object(config.pop("output_dim", None))
        use_jax_python = deserialize_keras_object(config.pop("use_jax_python", False))
        
        return cls(
            qnode=None, # Cannot restore QNode
            weight_shapes=weight_shapes,
            output_dim=output_dim,
            use_jax_python = use_jax_python,
            **config
        )

KerasCircuitLayer.set_input_argument()

@register_keras_serializable(package="PennylaneKeras", name="KerasDRCircuitLayer")
class KerasDRCircuitLayer(keras.layers.Layer):
    """A Keras Layer wrapping a PennyLane Q-Node.
    
    This layer implements a Data Re-Uploading quantum machine learning model
    that can be integrated into Keras models with full multi-backend support.
    
    Args:
        layers (int): Number of layers in the DR Model.
        scaling (float): Scaling factor for the input data. Defaults to 1.0
        circ_backend (str): Backend for the quantum circuit. Defaults to 'lightning.qubit'
        circ_grad_method (str): Gradient method for the quantum circuit. Defaults to 'adjoint'
        num_wires (int): Number of wires to initialize the qml.device. Defaults to 1.
        use_jax_python (bool): Flag to use the vectorized jax backend. 
            NOTE: This does not support jax.jit compilation. 
            See: https://docs.pennylane.ai/en/stable/introduction/interfaces/jax.html for details
        **kwargs: Additional keyword arguments for the keras Layer class such as 'name'.
    
    Example:
        >>> import keras
        >>> from pennylane_keras_layer import KerasDRCircuitLayer
        >>> 
        >>> # Create a quantum layer
        >>> q_layer = KerasDRCircuitLayer(layers=2, scaling=1.0, num_wires=1)
        >>> 
        >>> # Build a model
        >>> inp = keras.layers.Input(shape=(1,))
        >>> out = q_layer(inp)
        >>> model = keras.models.Model(inputs=inp, outputs=out)
    """
    
    def __init__(
        self,
        layers: int,
        scaling: float = 1.0,
        circ_backend: str = "lightning.qubit",
        circ_grad_method: str = "adjoint",
        num_wires: int = 1,
        use_jax_python: bool = False,
        **kwargs
    ):
        """Initialize the KerasDRCircuitLayer."""
        super().__init__(**kwargs)
        
        # Input validation
        if not isinstance(layers, int) or layers <= 0:
            raise ValueError(f"layers must be a positive integer, got {layers}")
        if not isinstance(num_wires, int) or num_wires <= 0:
            raise ValueError(f"num_wires must be a positive integer, got {num_wires}")
        if not isinstance(scaling, (int, float)) or scaling <= 0:
            raise ValueError(f"scaling must be a positive number, got {scaling}")
        
        # Defining the circuit parameters
        self.layers = layers
        self.scaling = scaling
        self.circ_backend = circ_backend
        self.circ_grad_method = circ_grad_method
        self.num_wires = num_wires
        self.use_jax_python = use_jax_python
        
        # Define Keras Layer flags
        self.is_built: bool = False
        
        # Selecting the Pennylane interface based on keras backend
        backend = keras.config.backend()
        if backend == "torch":
            import torch
            self.interface = "torch"
        elif backend == "tensorflow":
            import tensorflow as tf
            self.interface = "tf"
        elif backend == "jax":
            import jax
            if use_jax_python:
                self.interface = "jax-python"
            else:
                self.interface = "jax"
        else:
            raise ValueError(
                f"Unsupported Keras backend: {backend}. "
                f"Supported backends are: 'torch', 'tensorflow', 'jax'"
            )
    
    def build(self, input_shape):
        """Initialize the layer weights based on input_shape."""
        self._circuit_input_shape = input_shape[1:]
        
        # Initialize weights
        self.layer_weights = self.add_weight(
            shape=(self.layers + 1, 3),
            initializer=keras.initializers.random_uniform(minval=0, maxval=2 * np.pi),
            trainable=True
        )
        
        # Create Quantum Circuit
        self.circuit = self.create_circuit()
        self.is_built = True
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.num_wires)

    def data_encoding_block(self, x):
        """Data-encoding circuit block.
        Note: Current implementation operates on wire 0 only.
        """
        qml.RX(x[:, 0], wires=0)
    
    def trainable_rotation_block(self, theta):
        """Trainable circuit block."""
        qml.Rot(theta[0], theta[1], theta[2], wires=0)

    def serial_quantum_model(self, weights, x):
        """Data Re-Uploading QML model."""
        for theta in weights[:-1]:
            self.trainable_rotation_block(theta)
            self.data_encoding_block(x)
        
        # (L+1)'th unitary
        self.trainable_rotation_block(weights[-1])
        return qml.expval(qml.PauliZ(wires=0))
    
    def create_circuit(self):
        """ Creates the PennyLane device and QNode"""
        if self.interface == "jax":
            @jax.jit
            def create_circuit_jax_jit(layer_weights, x):
                dev = qml.device(self.circ_backend, wires = self.num_wires)
                circuit_node = qml.QNode(self.serial_quantum_model, dev, diff_method=self.circ_grad_method, interface=self.interface)
                return circuit_node(layer_weights, x)
            return create_circuit_jax_jit
        else:
            dev = qml.device(self.circ_backend, wires = self.num_wires)
            return qml.QNode(self.serial_quantum_model, dev, diff_method=self.circ_grad_method, interface=self.interface)
    
    def call(self, inputs):
        """Define the forward pass of the layer."""
        if not self.is_built:
            raise RuntimeError(
                "KerasDRCircuitLayer must be built before calling."
            )
        
        x = ops.multiply(self.scaling, inputs)
        
        if self.interface == "jax":
            out = self.circuit(self.layer_weights.value, x)
        else:
            out = self.circuit(self.layer_weights, x)
        return out

    def draw_qnode(self, **kwargs):
        """Draw the layer circuit."""
        if not self.is_built:
            raise RuntimeError(
                "KerasDRCircuitLayer must be built before drawing."
            )
        
        x = ops.expand_dims(keras.random.uniform(shape=self._circuit_input_shape), 0)
        qml.draw_mpl(self.circuit,**kwargs)(self.layer_weights.numpy(), x)

    def get_config(self):
        """Create layer config for layer saving."""
        base_config = super(KerasDRCircuitLayer, self).get_config()
        
        config = {
            "layers": serialize_keras_object(self.layers),
            "scaling": serialize_keras_object(self.scaling),
            "circ_backend": serialize_keras_object(self.circ_backend),
            "circ_grad_method": serialize_keras_object(self.circ_grad_method),
            "num_wires": serialize_keras_object(self.num_wires),
            "use_jax_python": serialize_keras_object(self.use_jax_python),
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        """Create an instance of layer from config."""
        layers = deserialize_keras_object(config.pop("layers"))
        scaling = deserialize_keras_object(config.pop("scaling"))
        circ_backend = deserialize_keras_object(config.pop("circ_backend"))
        circ_grad_method = deserialize_keras_object(config.pop("circ_grad_method"))
        num_wires = deserialize_keras_object(config.pop("num_wires"))
        use_jax_python = deserialize_keras_object(config.pop("use_jax_python", False))
        
        return cls(
            layers=layers,
            scaling=scaling,
            circ_backend=circ_backend,
            circ_grad_method=circ_grad_method,
            num_wires=num_wires,
            use_jax_python=use_jax_python,
            **config
        )