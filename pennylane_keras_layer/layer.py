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
        
        # Defining the circuit parameters
        self.layers = layers
        self.scaling = scaling
        self.circ_backend = circ_backend
        self.circ_grad_method = circ_grad_method
        self.num_wires = num_wires
        
        # Define Keras Layer flags
        self.is_built: bool = False
        
        # Selecting the Pennylane interface based on keras backend
        if keras.config.backend() == "torch":
            self.interface = "torch"
        elif keras.config.backend() == "tensorflow":
            self.interface = "tf"
        elif keras.config.backend() == "jax":
            if use_jax_python:
                self.interface = "jax-python"
            else:
                self.interface = "jax"
    
    def build(self, input_shape):
        """Initialize the layer weights based on input_shape.
        
        Args:
            input_shape (tuple): The shape of the input
        """
        # Save input_shape without batch to be used later for the draw_circuit function
        self.input_shape = input_shape[1:]
        
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

    def S(self, x):
        """Data-encoding circuit block.
        
        Args:
            x: Input data (expects batch dimension)
        """
        # Use the [:,0] syntax for batch support
        qml.RX(x[:, 0], wires=0)
    
    def W(self, theta):
        """Trainable circuit block.
        
        Args:
            theta: Weight parameters for the rotation gate
        """
        qml.Rot(theta[0], theta[1], theta[2], wires=0)

    def serial_quantum_model(self, weights, x):
        """Data Re-Uploading QML model.
        
        This method defines the quantum circuit architecture without 
        the qml.qnode decorator.
        
        Args:
            weights: Trainable weights for the quantum circuit
            x: Input data
            
        Returns:
            Expectation value of PauliZ measurement
        """
        for theta in weights[:-1]:
            self.W(theta)
            self.S(x)
        
        # (L+1)'th unitary
        self.W(weights[-1])
        return qml.expval(qml.PauliZ(wires=0))
    
    def create_circuit(self):
        """Create the PennyLane device and QNode.
        
        For JAX backend, the circuit is wrapped in jax.jit for performance.
        
        Returns:
            Callable: PennyLane QNode or JIT-compiled function
        """
        if self.interface == "jax":
            import jax
            
            @jax.jit
            def create_circuit_jax_jit(layer_weights, x):
                dev = qml.device(self.circ_backend, wires=self.num_wires)
                circuit_node = qml.QNode(
                    self.serial_quantum_model,
                    dev,
                    diff_method=self.circ_grad_method,
                    interface=self.interface
                )
                return circuit_node(layer_weights, x)
            
            return create_circuit_jax_jit
        else:
            dev = qml.device(self.circ_backend, wires=self.num_wires)
            return qml.QNode(
                self.serial_quantum_model,
                dev,
                diff_method=self.circ_grad_method,
                interface=self.interface
            )
    
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
            raise Exception("Layer not built") from None
        
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
            Exception: If layer is called before being built
        """
        # We want to raise an exception if this function is called before our QNode is created
        if not self.is_built:
            raise Exception("Layer not built") from None
        
        # Create a random input using the input_shape defined earlier with a single batch dim
        x = ops.expand_dims(keras.random.uniform(shape=self.input_shape), 0)
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
        
        # Call the init function of the layer from the config
        return cls(
            layers=layers,
            scaling=scaling,
            circ_backend=circ_backend,
            circ_grad_method=circ_grad_method,
            num_wires=num_wires,
            **config
        )
