"""
Turning quantum nodes into Keras Layers
=======================================

This example is a Keras 3 adaptation of the PennyLane tutorial 
"Turning quantum nodes into Torch Layers".
"""

import os

# Set backend to JAX, TensorFlow, or Torch (defaulting to jax for this example if available)
# os.environ["KERAS_BACKEND"] = "jax" 

import keras
from keras import ops
import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

# Set random seeds
np.random.seed(42)
keras.utils.set_random_seed(42)

# Import KerasCircuitLayer from the package
# Assuming we are running from the root of the repo or package is installed
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pennylane_keras_layer import KerasCircuitLayer

##############################################################################
# 1. Fixing the dataset and problem
# ---------------------------------

X, y = make_moons(n_samples=200, noise=0.1)
y_hot = keras.utils.to_categorical(y, num_classes=2)

print(f"X shape: {X.shape}")
print(f"y_hot shape: {y_hot.shape}")

# Visualize (optional)
# c = ["#1f77b4" if y_ == 0 else "#ff7f0e" for y_ in y]
# plt.axis("off")
# plt.scatter(X[:, 0], X[:, 1], c=c)
# plt.show()

##############################################################################
# 2. Defining a QNode
# -------------------

n_qubits = 2
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev)
def qnode(weights, inputs):
    """
    QNode ready for KerasCircuitLayer.
    Inputs must be the last argument.
    """
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]

##############################################################################
# 3. Interfacing with Keras
# -------------------------

n_layers = 6
weight_shapes = {"weights": (n_layers, n_qubits)}

# Create the layer
qlayer = KerasCircuitLayer(qnode, weight_shapes, output_dim=n_qubits)

##############################################################################
# 4. Creating a Hybrid Model (Sequential)
# ---------------------------------------

model = keras.Sequential([
    keras.layers.Dense(2, input_shape=(2,)),
    qlayer,
    keras.layers.Dense(2),
    keras.layers.Softmax()
])

model.summary()

##############################################################################
# 5. Training the Model
# ---------------------

model.compile(
    optimizer=keras.optimizers.SGD(learning_rate=0.2),
    loss="mae", # Using MAE to match the demo's L1Loss
    metrics=["accuracy"]
)

history = model.fit(
    X, y_hot,
    epochs=6,
    batch_size=5,
    validation_split=0.25, # Validating on a subset implicitly mimicking manual split
    shuffle=True
)

##############################################################################
# 6. Creating Non-Sequential Models
# ---------------------------------
# We used keras.ops to handle tensor operations in a backend-agnostic way.

class HybridModel(keras.Model):
    def __init__(self):
        super().__init__()
        self.clayer_1 = keras.layers.Dense(4)
        # Note: We need separate QLayers if we want separate weights or states, 
        # reusing one might be intended if weights are shared. 
        # The demo creates two TorchLayers with the same qnode, which implies separate weights.
        self.qlayer_1 = KerasCircuitLayer(qnode, weight_shapes, output_dim=n_qubits)
        self.qlayer_2 = KerasCircuitLayer(qnode, weight_shapes, output_dim=n_qubits)
        self.clayer_2 = keras.layers.Dense(2)
        self.softmax = keras.layers.Softmax()

    def call(self, inputs):
        x = self.clayer_1(inputs)
        
        # Use keras.ops.split to split the 4-dim output into two 2-dim tensors
        # keras.ops.split(x, indices_or_sections, axis=...)
        # Splitting into 2 equal sections along axis 1
        x_1, x_2 = ops.split(x, 2, axis=1)
        
        x_1 = self.qlayer_1(x_1)
        x_2 = self.qlayer_2(x_2)
        
        # Use keras.ops.concatenate
        x = ops.concatenate([x_1, x_2], axis=1)
        
        x = self.clayer_2(x)
        return self.softmax(x)

# Instantiate and compile
hybrid_model = HybridModel()
# We need to build the model or run one forward pass to initialize weights if not using Functional API explicitly with Input
# passing some dummy data to build
hybrid_model(ops.convert_to_tensor(X[:5]))

hybrid_model.compile(
    optimizer=keras.optimizers.SGD(learning_rate=0.2),
    loss="mae",
    metrics=["accuracy"]
)

print("\nTraining Non-Sequential Hybrid Model:")
history_hybrid = hybrid_model.fit(
    X, y_hot,
    epochs=6,
    batch_size=5,
    shuffle=True
)
