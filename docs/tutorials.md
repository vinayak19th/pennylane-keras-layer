# Tutorials

Step-by-step tutorials for using PennyLane Keras Layer with different backends and use cases.

## Table of Contents

- [Tutorial 1: First Quantum Model (JAX)](#tutorial-1-first-quantum-model-jax)
- [Tutorial 2: Using TensorFlow Backend](#tutorial-2-using-tensorflow-backend)
- [Tutorial 3: Using PyTorch Backend](#tutorial-3-using-pytorch-backend)
- [Tutorial 4: Fourier Series Approximation](#tutorial-4-fourier-series-approximation)
- [Tutorial 5: Model Saving and Loading](#tutorial-5-model-saving-and-loading)
- [Tutorial 6: Hyperparameter Tuning](#tutorial-6-hyperparameter-tuning)

---

## Tutorial 1: First Quantum Model (JAX)

This tutorial walks you through creating your first quantum machine learning model using the JAX backend.

### Step 1: Setup Environment

```python
import os
# Set JAX as backend BEFORE importing Keras
os.environ["KERAS_BACKEND"] = "jax"

# Configure JAX for 64-bit precision BEFORE importing keras
import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import keras
from pennylane_keras_layer import QKerasLayer

# Set Keras float precision
keras.backend.set_floatx('float64')

print(f"Backend: {keras.backend.backend()}")
```

### Step 2: Generate Training Data

```python
# Create a simple sine wave dataset
X_train = np.linspace(-np.pi, np.pi, 100).reshape(-1, 1)
y_train = np.sin(X_train)

# Create test data
X_test = np.linspace(-np.pi, np.pi, 50).reshape(-1, 1)
y_test = np.sin(X_test)

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
```

### Step 3: Create Quantum Layer

```python
# Create quantum layer with 3 layers
q_layer = QKerasLayer(
    layers=3,
    scaling=1.0,
    circ_backend="lightning.qubit",
    num_wires=1,
    name="quantum_layer"
)

print("Quantum layer created successfully!")
```

### Step 4: Build Model

```python
# Build model using Sequential API
model = keras.Sequential([
    keras.layers.Input(shape=(1,)),
    q_layer
], name="QuantumModel")

# Print model summary
model.summary()
```

### Step 5: Compile Model

```python
# Compile with Adam optimizer
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.01),
    loss=keras.losses.MeanSquaredError(),
    metrics=[keras.metrics.MeanAbsoluteError()]
)

print("Model compiled successfully!")
```

### Step 6: Train Model

```python
# Train the model
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1
)

print("Training complete!")
```

### Step 7: Evaluate and Predict

```python
# Evaluate on test set
test_loss, test_mae = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test MAE: {test_mae:.4f}")

# Make predictions
predictions = model.predict(X_test)

# Calculate accuracy
accuracy = np.mean(np.abs(predictions - y_test) < 0.1)
print(f"Accuracy (within 0.1): {accuracy*100:.2f}%")
```

### Step 8: Visualize Results (Optional)

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 4))

# Training history
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training History')

# Predictions vs Actual
plt.subplot(1, 2, 2)
plt.plot(X_test, y_test, 'b-', label='Actual', alpha=0.7)
plt.plot(X_test, predictions, 'r--', label='Predicted', alpha=0.7)
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Predictions vs Actual')

plt.tight_layout()
plt.savefig('tutorial1_results.png')
plt.show()
```

---

## Tutorial 2: Using TensorFlow Backend

Learn how to use PennyLane Keras Layer with TensorFlow backend.

### Setup

```python
import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
import keras
from pennylane_keras_layer import QKerasLayer

print(f"Using backend: {keras.backend.backend()}")
```

### Create and Train Model

```python
# Generate data
X = np.random.randn(200, 1)
y = np.cos(X * 2).reshape(-1)

# Split data
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Create model
model = keras.Sequential([
    keras.layers.Input(shape=(1,)),
    QKerasLayer(
        layers=4,
        scaling=2.0,
        circ_backend="default.qubit",
        name="quantum"
    )
])

# Compile
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.02),
    loss="mse"
)

# Train
history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=16,
    validation_split=0.2,
    verbose=1
)

# Evaluate
test_loss = model.evaluate(X_test, y_test)
print(f"Test MSE: {test_loss:.4f}")
```

### TensorFlow-Specific Features

```python
# Use TensorFlow data pipeline
import tensorflow as tf

dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
dataset = dataset.shuffle(buffer_size=100).batch(32)

model.fit(dataset, epochs=10)
```

---

## Tutorial 3: Using PyTorch Backend

Learn how to use PennyLane Keras Layer with PyTorch backend.

### Setup

```python
import os
os.environ["KERAS_BACKEND"] = "torch"

import numpy as np
import torch
import keras
from pennylane_keras_layer import QKerasLayer

print(f"Using backend: {keras.backend.backend()}")
print(f"PyTorch version: {torch.__version__}")
```

### Create Model

```python
# Generate data
X = np.linspace(-1, 1, 150).reshape(-1, 1).astype(np.float32)
y = (X ** 2).ravel()

# Create model
model = keras.Sequential([
    keras.layers.Input(shape=(1,)),
    QKerasLayer(
        layers=3,
        scaling=1.5,
        num_wires=1,
        name="quantum"
    ),
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.01),
    loss="mse"
)

# Train
model.fit(X, y, epochs=40, batch_size=32, verbose=1)

# Predict
predictions = model.predict(X)
print(f"Mean Absolute Error: {np.mean(np.abs(predictions - y)):.4f}")
```

### PyTorch-Specific Features

```python
# Access PyTorch tensors
for layer in model.layers:
    if isinstance(layer, QKerasLayer):
        # Weights are PyTorch tensors
        weights = layer.layer_weights
        print(f"Weight type: {type(weights)}")
        print(f"Weight shape: {weights.shape}")
```

---

## Tutorial 4: Fourier Series Approximation

Approximate a truncated Fourier series using quantum circuits.

### Theory

The Data Re-Uploading model can represent Fourier series:
```
f(x) = c₀ + Σ(cₙ e^(inx) + c̄ₙ e^(-inx))
```

### Implementation

```python
import os
os.environ["KERAS_BACKEND"] = "jax"

import numpy as np
import keras
from pennylane_keras_layer import QKerasLayer

# Target function: truncated Fourier series
def fourier_target(x, degree=1, scaling=1.0):
    coeffs = [0.15 + 0.15j] * degree
    coeff0 = 0.1
    res = coeff0
    for idx, coeff in enumerate(coeffs):
        exponent = np.complex128(scaling * (idx + 1) * x * 1j)
        conj_coeff = np.conjugate(coeff)
        res += coeff * np.exp(exponent) + conj_coeff * np.exp(-exponent)
    return np.real(res)

# Generate data
degree = 1
scaling = 1.0
x_data = np.linspace(-6, 6, 100)
y_data = np.array([fourier_target(x, degree, scaling) for x in x_data])

# Create quantum model with same degree
model = keras.Sequential([
    keras.layers.Input(shape=(1,)),
    QKerasLayer(
        layers=degree + 1,  # Match Fourier degree
        scaling=scaling,
        num_wires=1
    )
])

# Train
model.compile(optimizer=keras.optimizers.Adam(0.03), loss="mse")
history = model.fit(x_data, y_data, epochs=50, verbose=1)

# Results
predictions = model.predict(x_data)
final_mse = np.mean((predictions - y_data) ** 2)
print(f"Final MSE: {final_mse:.6f}")
```

### Visualization

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'])
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training Progress')
plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(x_data, y_data, 'k-', label='Target', linewidth=2)
plt.plot(x_data, predictions, 'r--', label='Quantum Model', linewidth=2)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Fourier Series Approximation')
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('fourier_approximation.png')
plt.show()
```

---

## Tutorial 5: Model Saving and Loading

Learn how to save and load quantum models.

### Saving Models

```python
import os
os.environ["KERAS_BACKEND"] = "jax"

import keras
import numpy as np
from pennylane_keras_layer import QKerasLayer

# Create and train a model
model = keras.Sequential([
    keras.layers.Input(shape=(1,)),
    QKerasLayer(layers=2, scaling=1.0, name="quantum")
])

X = np.linspace(-1, 1, 50).reshape(-1, 1)
y = X ** 2

model.compile(optimizer="adam", loss="mse")
model.fit(X, y, epochs=20, verbose=0)

# Save complete model
model.save("quantum_model.keras")
print("✓ Model saved to quantum_model.keras")

# Save only weights
model.save_weights("quantum_weights.h5")
print("✓ Weights saved to quantum_weights.h5")
```

### Loading Models

```python
# Load complete model
loaded_model = keras.models.load_model("quantum_model.keras")
print("✓ Model loaded successfully")

# Verify loaded model
original_pred = model.predict(X)
loaded_pred = loaded_model.predict(X)

difference = np.abs(original_pred - loaded_pred).max()
print(f"Max prediction difference: {difference:.10f}")

if difference < 1e-6:
    print("✓ Model loaded correctly!")
```

### Loading Weights Only

```python
# Create new model with same architecture
new_model = keras.Sequential([
    keras.layers.Input(shape=(1,)),
    QKerasLayer(layers=2, scaling=1.0, name="quantum")
])

# Build model first
new_model.build((None, 1))

# Load weights
new_model.load_weights("quantum_weights.h5")
print("✓ Weights loaded successfully")

# Verify
new_pred = new_model.predict(X)
difference = np.abs(original_pred - new_pred).max()
print(f"Max prediction difference: {difference:.10f}")
```

---

## Tutorial 6: Hyperparameter Tuning

Find optimal hyperparameters for your quantum model.

### Grid Search

```python
import os
os.environ["KERAS_BACKEND"] = "jax"

import numpy as np
import keras
from pennylane_keras_layer import QKerasLayer
from itertools import product

# Generate data
X = np.linspace(-2, 2, 100).reshape(-1, 1)
y = np.sin(2 * X).ravel()

# Define hyperparameter grid
param_grid = {
    'layers': [2, 3, 4],
    'scaling': [0.5, 1.0, 2.0],
    'learning_rate': [0.01, 0.03, 0.05]
}

# Grid search
results = []

for layers, scaling, lr in product(
    param_grid['layers'],
    param_grid['scaling'],
    param_grid['learning_rate']
):
    # Create model
    model = keras.Sequential([
        keras.layers.Input(shape=(1,)),
        QKerasLayer(layers=layers, scaling=scaling)
    ])
    
    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="mse"
    )
    
    # Train
    history = model.fit(X, y, epochs=30, verbose=0)
    
    # Record results
    final_loss = history.history['loss'][-1]
    results.append({
        'layers': layers,
        'scaling': scaling,
        'learning_rate': lr,
        'final_loss': final_loss
    })
    
    print(f"Layers={layers}, Scaling={scaling}, LR={lr}: Loss={final_loss:.4f}")

# Find best parameters
best = min(results, key=lambda x: x['final_loss'])
print(f"\nBest parameters:")
print(f"  Layers: {best['layers']}")
print(f"  Scaling: {best['scaling']}")
print(f"  Learning Rate: {best['learning_rate']}")
print(f"  Final Loss: {best['final_loss']:.4f}")
```

### Random Search

```python
import random

# Random search
n_iterations = 20
results = []

for i in range(n_iterations):
    # Random hyperparameters
    layers = random.choice([2, 3, 4, 5])
    scaling = random.uniform(0.5, 3.0)
    lr = 10 ** random.uniform(-3, -1)  # Log-uniform between 0.001 and 0.1
    
    # Create and train model
    model = keras.Sequential([
        keras.layers.Input(shape=(1,)),
        QKerasLayer(layers=layers, scaling=scaling)
    ])
    
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss="mse")
    history = model.fit(X, y, epochs=30, verbose=0)
    
    final_loss = history.history['loss'][-1]
    results.append({
        'layers': layers,
        'scaling': scaling,
        'learning_rate': lr,
        'final_loss': final_loss
    })

# Best result
best = min(results, key=lambda x: x['final_loss'])
print(f"Best random search result:")
print(f"  Layers: {best['layers']}")
print(f"  Scaling: {best['scaling']:.3f}")
print(f"  Learning Rate: {best['learning_rate']:.5f}")
print(f"  Final Loss: {best['final_loss']:.4f}")
```

### Cross-Validation

```python
from sklearn.model_selection import KFold

# K-Fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

for train_idx, val_idx in kf.split(X):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    # Create model
    model = keras.Sequential([
        keras.layers.Input(shape=(1,)),
        QKerasLayer(layers=3, scaling=1.0)
    ])
    
    model.compile(optimizer=keras.optimizers.Adam(0.02), loss="mse")
    model.fit(X_train, y_train, epochs=30, verbose=0)
    
    # Evaluate
    val_loss = model.evaluate(X_val, y_val, verbose=0)
    cv_scores.append(val_loss)

print(f"Cross-validation scores: {cv_scores}")
print(f"Mean CV score: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
```

---

## Next Steps

- Explore [Examples](examples.md) for complete working code
- Read [User Guide](user_guide.md) for in-depth concepts
- Check [API Reference](api_reference.md) for detailed parameters
- Visit [FAQ](faq.md) for common questions
