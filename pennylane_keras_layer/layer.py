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
    
    This layer enables the integration of PennyLane quantum circuits into Keras models with full multi-backend support.
    The backend can be selected by setting the `KERAS_BACKEND` environment variable to "tensorflow", "jax", or "torch" for example:
    
    >>> os.environ["KERAS_BACKEND"] = "jax"
    >>> import keras
    >>> import pennylane as qml
    >>> from pennylane_keras_layer import KerasCircuitLayer

    The signature of the QNode must contain an inputs named argument for input data, with all other arguments to be treated as internal weights. We can then convert to a Keras layer with:

    >>> weight_shapes = {"weights_0": 3, "weight_1": 1}
    >>> qlayer = KerasCircuitLayer(qnode, weight_shapes, output_dim=n_qubits)

    Args:
        qnode (qml.QNode): The PennyLane QNode to be converted into a Keras layer.
        weight_shapes (dict[str, tuple]): A dictionary mapping from all weights used in the QNode to their corresponding shapes.
        output_dim (int|tuple): The output dimension of the QNode passed as a single output dimension or tuple of output dimensions. Optional. Default is 1.
        use_jax_python (bool): Flag to use the vectorized jax backend. Default is False.
        weight_specs (dict[str, dict]): An optional dictionary for users to provide additional
            specifications for weights used in the QNode, such as the method of parameter
            initialization. This specification is provided as a dictionary with keys given by the
            arguments of the `add_weight() <https://keras.io/api/layers/base_layer/#addweight-method>`__
            method and values being the corresponding specification. Optional. Default is None.
        **kwargs: Additional keyword arguments for the Keras Layer class.

    **Example**

    First let's define the QNode that we want to convert into a Keras layer:

    .. code-block:: python

        import pennylane as qml
        from pennylane_keras_layer import KerasCircuitLayer

        n_qubits = 2
        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev)
        def qnode(inputs, weights_0, weight_1):
            qml.RX(inputs[0], wires=0)
            qml.RX(inputs[1], wires=1)
            qml.Rot(*weights_0, wires=0)
            qml.RY(weight_1, wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.Z(0)), qml.expval(qml.Z(1))

    The signature of the QNode **must** contain an ``inputs`` named argument for input data, with all other arguments to be treated as internal weights. We can then convert to a Keras layer with:

    >>> weight_shapes = {"weights_0": 3, "weight_1": 1}
    >>> qlayer = KerasCircuitLayer(qnode, weight_shapes, output_dim=n_qubits)

    The internal weights of the QNode are automatically initialized within the :class:`~.KerasCircuitLayer` and must have their shapes specified in a ``weight_shapes`` dictionary. It is then easy to combine with other neural network layers from the `keras.layers` module and create a hybrid:

    >>> import keras
    >>> clayer = keras.layers.Dense(2)
    >>> model = keras.models.Sequential([qlayer, clayer])

    Building on Keras 3, the KerasCircuitLayer can be used used purely keras models or those of the underlying backend. For example, when using the `torch` backend, the model can be used with `torch.nn.Modules` and `torch.nn.Sequential` models as follows:

    1. First initialize the backend to torch and import the necessary packages:
    >>> os.environ["KERAS_BACKEND"] = "torch"
    >>> import torch
    >>> import keras
    >>> import pennylane as qml
    >>> from pennylane_keras_layer import KerasCircuitLayer
    
    2. Define the QNode and convert to a Keras layer:
    >>> weight_shapes = {"weights_0": 3, "weight_1": 1}
    >>> qlayer = KerasCircuitLayer(qnode, weight_shapes, output_dim=n_qubits)
    
    3. Define the model:
    >>> model = torch.nn.Sequential(
    ...     qlayer,
    ...     torch.nn.Linear(n_qubits, 2),
    ... )
    
    This can then be used as a normal torch model:
    >>> model(torch.tensor([0.1, 0.2]))
    
    .. details::
        :title: Usage Details

        **QNode signature**

        The QNode must have a signature that satisfies the following conditions:

        - Contain an ``inputs`` named argument for input data.
        - All other arguments must accept an array or tensor and are treated as internal weights of the QNode.
        - All other arguments must have no default value.
        - The ``inputs`` argument is permitted to have a default value provided the gradient with respect to ``inputs`` is not required.
        - There cannot be a variable number of positional or keyword arguments, e.g., no ``*args`` or ``**kwargs`` present in the signature.

        **Output shape**

        The ``output_dim`` argument determines the output shape of the layer.
        If ``output_dim`` is an integer, the output shape is ``(batch_dim, output_dim)``.
        If ``output_dim`` is a tuple, the output shape is ``(batch_dim, *output_dim)``.

        **Initializing weights**

        If ``weight_specs`` is not specified, weights are randomly initialized from the uniform distribution on the interval :math:`[0, 2 pi]`.

        The optional ``weight_specs`` argument allows for the initialization method of the QNode weights to be specified. The dictionary passed to the argument must be a dictionary where keys are weight names and values are dictionaries of arguments passed to `add_weight`.

        For example, weights can be randomly initialized from the normal distribution by passing:

        .. code-block:: python

            weight_specs = {
                "weights_0": {"initializer": "random_normal"},
                "weight_1": {"initializer": "ones"}
            }

        **Model saving**

        Instances of ``KerasCircuitLayer`` can be saved using the usual ``model.save()`` utility.
        However, since PennyLane QNodes are not natively serializable, loading the model requires a specific step to restore the QNode.

        .. code-block:: python

            model.save("model.keras")

        To load the layer again:

        .. code-block:: python

            model = keras.models.load_model("model.keras")
            # The QNode is not restored automatically.
            # We must set it manually for each KerasCircuitLayer in the model
            # For a sequential model with the quantum layer at index 2:
            model.layers[2].set_qnode(qnode)

        **Full code example**

        The code block below shows how a circuit composed of templates from the :doc:`/introduction/templates` module can be combined with classical `Dense` layers to learn the two-dimensional `moons <https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html>`__ dataset.

        .. code-block:: python

            import pennylane as qml
            import keras
            import sklearn.datasets
            from pennylane_keras_layer import KerasCircuitLayer

            n_qubits = 2
            dev = qml.device("default.qubit", wires=n_qubits)

            @qml.qnode(dev)
            def qnode(weights, inputs):
                qml.AngleEmbedding(inputs, wires=range(n_qubits))
                qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
                return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]

            n_layers = 6
            weight_shapes = {"weights": (n_layers, n_qubits)}

            qlayer = KerasCircuitLayer(qnode, weight_shapes, output_dim=n_qubits)

            clayer1 = keras.layers.Dense(2)
            clayer2 = keras.layers.Dense(2, activation="softmax")
            model = keras.models.Sequential([keras.Input(shape=(2,)), clayer1, qlayer, clayer2])

            opt = keras.optimizers.SGD(learning_rate=0.2)
            model.compile(opt, loss="mae", metrics=["accuracy"])

            data = sklearn.datasets.make_moons(n_samples=200, noise=0.1)
            X = data[0]
            y_hot = keras.ops.one_hot(data[1], 2)

            model.fit(X, y_hot, epochs=6, batch_size=5, validation_split=0.25)
    """
    
    def __init__(
        self,
        qnode:qml.QNode,
        weight_shapes: dict,
        output_dim: int | tuple = 1,
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
            self._signature_validation(self.qnode, weight_shapes)

        # Allows output_dim to be specified as an int or as a tuple, e.g, 5, (5,), (5, 2), [5, 2]
        # Note: Single digit values will be considered an int and multiple as a tuple, e.g [5,] or (5,)
        # are passed as integer 5 and [5, 2] will be passes as tuple (5, 2)
        if isinstance(output_dim, Iterable) and len(output_dim) > 1:
            self.output_dim = tuple(output_dim)
        else:
            self.output_dim = output_dim[0] if isinstance(output_dim, Iterable) else output_dim

        # Define Keras Layer flags
        self.built: bool = False
        
        # Selecting the Pennylane interface based on keras backend and set them to double precisions
        backend = keras.config.backend()
        if backend == "torch":
            import torch
            torch.set_default_dtype(torch.float64)
            self.interface = "torch"
        elif backend == "tensorflow":
            import tensorflow as tf
            self.interface = "tf"
        elif backend == "jax":
            import jax
            jax.config.update("jax_enable_x64", True)
            if use_jax_python:
                self.interface = "jax-python"
            else:
                self.interface = "jax"
        else:
            raise ValueError(
                f"Unsupported Keras backend: {backend}. "
                f"Supported backends are: 'torch', 'tensorflow', 'jax'"
            )
        # Build the layer as output shape and weight shapes are input_shape independent
        self.build(None)
        # Update qnode interface and compile
        self.update_qnode()

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
        if self.built:
            return
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
    
    def update_qnode(self) -> None:
        """Update the QNode with the correct interface"""
        # Create device once
        if self.qnode == None:
            print("Delaying circuit creation till QNode is set using the 'set_qnode' method")
            return None
        else:
            self.qnode.interface = self.interface
            if self.interface == "jax":
                import jax
                self.qnode = jax.jit(self.qnode)
    
    def draw_qnode(self,input, **kwargs):
        """Draw the quantum circuit.

        Args:
            input (tensor-like): Input data to the circuit.
            **kwargs: Additional keyword arguments to be passed to `qml.draw_mpl`.

        Raises:
            RuntimeError: If the layer has not been built.
        """
        if not self.built:
            raise RuntimeError(
                "KerasDRCircuitLayer must be built before drawing."
            )

        weight_values = [self.qnode_weights[k] for k in self.weight_shapes.keys()]
        if self.interface == "jax":
            # Use .value to get the underlying value for JIT compatibility
            weight_values = [w.value for w in self.qnode_weights.values()]
            qml.draw_mpl(self.qnode.func,**kwargs)(weight_values, input)
        else:
            qml.draw_mpl(self.qnode,**kwargs)(weight_values, input)
        
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
            
        res = self.qnode(*weight_values, inputs)
        
        # If the QNode returns a list of results (multiple measurements), stack them
        if isinstance(res, (list, tuple)):
            res = ops.stack(res, axis=-1)
            
        return ops.cast(res, self.compute_dtype)

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
        print("Updating QNode")
        self.update_qnode()

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
        self.built: bool = False
        
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
        if self.built:
             return
        self._circuit_input_shape = input_shape[1:]
        
        # Initialize weights
        self.layer_weights = self.add_weight(
            shape=(self.layers + 1, 3),
            initializer=keras.initializers.random_uniform(minval=0, maxval=2 * np.pi),
            trainable=True
        )
        
        # Create Quantum Circuit
        self.circuit = self.create_circuit()
        self.built = True
    
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
            import jax
            @jax.jit # noqa
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
        if not self.built:
            raise RuntimeError(
                "KerasDRCircuitLayer must be built before calling."
            )
        
        x = ops.multiply(self.scaling, inputs)
        
        if self.interface == "jax":
            import jax.numpy as jnp
            # Cast inputs to float64 to ensure consistent JVP
            x = ops.cast(x, "float64")
            w = ops.cast(self.layer_weights.value, "float64")
            out = self.circuit(w, x)
        else:
            out = self.circuit(self.layer_weights, x)
            
        if len(out.shape) == 1:
            out = ops.reshape(out, (-1, 1))
            
        return ops.cast(out, self.compute_dtype)

    def draw_qnode(self, **kwargs):
        """Draw the layer circuit."""
        if not self.built:
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