"""
Keras Layer implementation for PennyLane quantum circuits.

This module provides the QKerasLayer class that wraps PennyLane QNodes
as Keras layers, enabling quantum circuits in neural networks with full
multi-backend support (TensorFlow, JAX, PyTorch).
"""

import keras
from keras.utils import register_keras_serializable
from keras.saving import serialize_keras_object, deserialize_keras_object
from keras import ops
import pennylane as qml
import numpy as np


@register_keras_serializable(package="QKeras", name="QKerasLayer")
class QKerasLayer(keras.layers.Layer):
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
        >>> from pennylane_keras_layer import QKerasLayer
        >>> 
        >>> # Create a quantum layer
        >>> q_layer = QKerasLayer(layers=2, scaling=1.0, num_wires=1)
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
        """Initialize the QKerasLayer."""
        super().__init__(**kwargs)  # Passing the keyword arguments to the parent class
        
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
    
    def build(self, input_shape):
        """Initialize the layer weights based on input_shape.
        
        Args:
            input_shape (tuple): The shape of the input
        """
        # Save input_shape without batch to be used later for the draw_circuit function
        # Using a different name to avoid shadowing parent class attribute
        self._circuit_input_shape = input_shape[1:]
        
        # Initialize weights in the same way as the numpy array in the previous section.
        # Randomly initialize weights to uniform distribution in the a range of [0,2pi)
        self.layer_weights = self.add_weight(
            shape=(self.layers + 1, 3),
            initializer=keras.initializers.random_uniform(minval=0, maxval=2 * np.pi),
            trainable=True
        )
        
        # Create Quantum Circuit
        self.circuit = self.create_circuit()
        
        # Set the layer as built
        self.is_built = True
    
    def compute_output_shape(self, input_shape):
        """Return output shape as a function of the input shape.
        
        For this model we return an expectation value per qubit. The '0' index 
        of the input_shape is always the batch, so we return an output shape 
        of (batch, num_wires).
        
        Args:
            input_shape (tuple): Shape of the input tensor
            
        Returns:
            tuple: Output shape (batch, num_wires)
        """
        return (input_shape[0], self.num_wires)

    def data_encoding_block(self, x):
        """Data-encoding circuit block.
        
        Note: Current implementation operates on wire 0 only.
        
        Args:
            x: Input data (expects batch dimension)
        """
        # Use the [:,0] syntax for batch support
        qml.RX(x[:, 0], wires=0)
    
    def trainable_rotation_block(self, theta):
        """Trainable circuit block with parametrized rotations.
        
        Args:
            theta: Weight parameters for the rotation gate
        """
        qml.Rot(theta[0], theta[1], theta[2], wires=0)

    def serial_quantum_model(self, weights, x):
        """Data Re-Uploading QML model.
        
        This method defines the quantum circuit architecture without 
        the qml.qnode decorator.
        
        Note: Current implementation operates on wire 0 only, regardless
        of the num_wires parameter.
        
        Args:
            weights: Trainable weights for the quantum circuit
            x: Input data
            
        Returns:
            Expectation value of PauliZ measurement
        """
        for theta in weights[:-1]:
            self.trainable_rotation_block(theta)
            self.data_encoding_block(x)
        
        # (L+1)'th unitary
        self.trainable_rotation_block(weights[-1])
        return qml.expval(qml.PauliZ(wires=0))
    
    def create_circuit(self):
        """Create the PennyLane device and QNode.
        
        For JAX backend (non-python), creates the device and QNode once,
        then wraps only the execution in jax.jit for performance.
        
        Returns:
            Callable: PennyLane QNode or JIT-compiled wrapper
        """
        # Create device once (outside JIT boundary for efficiency)
        dev = qml.device(self.circ_backend, wires=self.num_wires)
        circuit_node = qml.QNode(
            self.serial_quantum_model,
            dev,
            diff_method=self.circ_grad_method,
            interface=self.interface
        )
        
        if self.interface == "jax":
            import jax
            # JIT compile only the circuit execution, not device creation
            return jax.jit(circuit_node)
        else:
            return circuit_node
    
    def call(self, inputs):
        """Define the forward pass of the layer.
        
        Args:
            inputs: Input tensor
            
        Returns:
            Output tensor with quantum circuit predictions
            
        Raises:
            Exception: If layer is called before being built
        """
        # We need to prevent the layer from being called before the weights and circuit are built
        if not self.is_built:
            raise RuntimeError(
                "QKerasLayer must be built before calling. "
                "Ensure the layer is connected in a model or call layer.build(input_shape) explicitly."
            )
        
        # We multiply the input with the scaling factor outside the circuit for optimized vector execution.
        x = ops.multiply(self.scaling, inputs)
        
        # We call the circuit with the weight variables.
        if self.interface == "jax":
            out = self.circuit(self.layer_weights.value, x)
        else:
            out = self.circuit(self.layer_weights, x)
        return out

    def draw_qnode(self):
        """Draw the layer circuit.
        
        Creates a visualization of the quantum circuit with random input.
        
        Raises:
            RuntimeError: If layer is called before being built
        """
        # We want to raise an exception if this function is called before our QNode is created
        if not self.is_built:
            raise RuntimeError(
                "QKerasLayer must be built before drawing. "
                "Ensure the layer is connected in a model or call layer.build(input_shape) explicitly."
            )
        
        # Create a random input using the input_shape defined earlier with a single batch dim
        x = ops.expand_dims(keras.random.uniform(shape=self._circuit_input_shape), 0)
        qml.draw_mpl(self.circuit)(self.layer_weights.numpy(), x)

    def get_config(self):
        """Create layer config for layer saving.
        
        Returns:
            dict: Configuration dictionary
        """
        # Load the basic config parameters of the keras.layer parent class
        base_config = super(QKerasLayer, self).get_config()
        
        # Create a custom configuration for the instance variables unique to the QNode
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
        """Create an instance of layer from config.
        
        Args:
            config (dict): Configuration dictionary
            
        Returns:
            QKerasLayer: New instance of the layer
        """
        # The cls argument is the specific layer config and the config object contains general keras.layer arguments
        layers = deserialize_keras_object(config.pop("layers"))
        scaling = deserialize_keras_object(config.pop("scaling"))
        circ_backend = deserialize_keras_object(config.pop("circ_backend"))
        circ_grad_method = deserialize_keras_object(config.pop("circ_grad_method"))
        num_wires = deserialize_keras_object(config.pop("num_wires"))
        use_jax_python = deserialize_keras_object(config.pop("use_jax_python", False))
        
        # Call the init function of the layer from the config
        return cls(
            layers=layers,
            scaling=scaling,
            circ_backend=circ_backend,
            circ_grad_method=circ_grad_method,
            num_wires=num_wires,
            use_jax_python=use_jax_python,
            **config
        )
