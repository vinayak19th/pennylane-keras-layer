# Examples

Complete working examples demonstrating different use cases of PennyLane Keras Layer.

## Table of Contents

- [Basic Example](#basic-example)
- [Fourier Series Approximation](#fourier-series-approximation)
- [Regression Example](#regression-example)
- [Binary Classification](#binary-classification)
- [Multi-Layer Quantum Network](#multi-layer-quantum-network)
- [Custom Training Loop](#custom-training-loop)
- [Hyperparameter Comparison](#hyperparameter-comparison)

---

## Basic Example

Simple example demonstrating the basic usage of QKerasLayer.

```python
"""
Basic example demonstrating pennylane-keras-layer usage.
"""
import os
os.environ["KERAS_BACKEND"] = "jax"

import numpy as np
import keras
from pennylane_keras_layer import QKerasLayer

# Generate simple data
X_train = np.linspace(-np.pi, np.pi, 100).reshape(-1, 1)
y_train = np.sin(X_train)

# Create quantum model
model = keras.Sequential([
    keras.layers.Input(shape=(1,)),
    QKerasLayer(layers=3, scaling=1.0, name="quantum")
])

# Compile and train
model.compile(optimizer=keras.optimizers.Adam(0.01), loss="mse")
model.fit(X_train, y_train, epochs=50, verbose=1)

# Predict
predictions = model.predict(X_train)
print(f"MSE: {np.mean((predictions - y_train)**2):.4f}")
```

**Output:**
```
Epoch 1/50
4/4 ━━━━━━━━━━━━━━━━━━━━ 2s 50ms/step - loss: 0.5234
...
Epoch 50/50
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 12ms/step - loss: 0.0123
MSE: 0.0115
```

---

## Fourier Series Approximation

Demonstrates quantum circuit's ability to approximate Fourier series.

**See: [`examples/fourier_series_example.py`](../examples/fourier_series_example.py)**

This example shows:
- Creating truncated Fourier series as target function
- Building quantum model to approximate it
- Training and evaluation
- Model saving and loading
- Visualization of results

**Key Code Snippet:**

```python
def create_target_function(degree=1, scaling=1, coeffs=None, coeff0=0.1):
    """Create a truncated Fourier series function."""
    if coeffs is None:
        coeffs = [0.15 + 0.15j] * degree
    
    def target_function(x):
        res = coeff0
        for idx, coeff in enumerate(coeffs):
            exponent = np.complex128(scaling * (idx + 1) * x * 1j)
            conj_coeff = np.conjugate(coeff)
            res += coeff * np.exp(exponent) + conj_coeff * np.exp(-exponent)
        return np.real(res)
    
    return target_function

# Create model matching Fourier degree
target_fn = create_target_function(degree=1, scaling=1.0)
model = keras.Sequential([
    keras.layers.Input(shape=(1,)),
    QKerasLayer(layers=2, scaling=1.0)  # layers = degree + 1
])
```

**Run Example:**
```bash
cd examples
python fourier_series_example.py
```

---

## Regression Example

Complete regression example with train/test split and evaluation.

```python
"""
Regression example: Fit a quadratic function.
"""
import os
os.environ["KERAS_BACKEND"] = "jax"

import numpy as np
import keras
from pennylane_keras_layer import QKerasLayer
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# Generate quadratic data with noise
np.random.seed(42)
X = np.linspace(-2, 2, 200).reshape(-1, 1)
y = 0.5 * X**2 + 0.3 * X + np.random.normal(0, 0.1, X.shape)
y = y.ravel()

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# Create model
model = keras.Sequential([
    keras.layers.Input(shape=(1,)),
    QKerasLayer(
        layers=4,
        scaling=1.5,
        circ_backend="lightning.qubit",
        name="quantum"
    )
], name="QuantumRegressor")

# Compile
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.02),
    loss="mse",
    metrics=["mae"]
)

# Train with validation
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    batch_size=32,
    verbose=1
)

# Evaluate
test_loss, test_mae = model.evaluate(X_test, y_test)
predictions = model.predict(X_test)

# Additional metrics
r2 = r2_score(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)

print("\n" + "="*50)
print("RESULTS")
print("="*50)
print(f"Test Loss (MSE): {test_loss:.4f}")
print(f"Test MAE: {mae:.4f}")
print(f"R² Score: {r2:.4f}")

# Visualize
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.title('Training History')
plt.grid(alpha=0.3)

plt.subplot(1, 3, 2)
plt.scatter(X_test, y_test, alpha=0.5, label='Actual')
plt.scatter(X_test, predictions, alpha=0.5, label='Predicted')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Predictions vs Actual')
plt.grid(alpha=0.3)

plt.subplot(1, 3, 3)
residuals = y_test - predictions.ravel()
plt.scatter(predictions, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('regression_results.png', dpi=100)
print("\n✓ Results saved to regression_results.png")
plt.show()
```

---

## Binary Classification

Binary classification example using quantum circuits.

```python
"""
Binary classification with quantum circuits.
"""
import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
import keras
from pennylane_keras_layer import QKerasLayer
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Generate synthetic dataset
X, y = make_moons(n_samples=300, noise=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"Class distribution: {np.bincount(y_train)}")

# Create hybrid model
inputs = keras.layers.Input(shape=(2,))
x = keras.layers.Dense(1)(inputs)  # Reduce to 1D for quantum layer
x = QKerasLayer(
    layers=3,
    scaling=2.0,
    num_wires=1,
    name="quantum"
)(x)
x = keras.layers.Dense(8, activation='relu')(x)
outputs = keras.layers.Dense(1, activation='sigmoid')(x)

model = keras.models.Model(inputs=inputs, outputs=outputs)

# Compile
model.compile(
    optimizer=keras.optimizers.Adam(0.01),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("\nModel Architecture:")
model.summary()

# Train
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=16,
    verbose=1
)

# Evaluate
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"\nTest Accuracy: {test_acc*100:.2f}%")

# Confusion matrix
from sklearn.metrics import confusion_matrix, classification_report

y_pred = (model.predict(X_test) > 0.5).astype(int).ravel()
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Visualize decision boundary
import matplotlib.pyplot as plt

def plot_decision_boundary(model, X, labels):
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu')
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='RdYlBu', edgecolors='k')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training History')
plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
plot_decision_boundary(model, X_test, y_test)
plt.title(f'Decision Boundary (Acc: {test_acc*100:.1f}%)')

plt.tight_layout()
plt.savefig('classification_results.png', dpi=100)
print("\n✓ Results saved to classification_results.png")
plt.show()
```

---

## Multi-Layer Quantum Network

Example with multiple quantum layers.

```python
"""
Multi-layer quantum network.
"""
import os
os.environ["KERAS_BACKEND"] = "jax"

import numpy as np
import keras
from pennylane_keras_layer import QKerasLayer

# Generate complex data
X = np.linspace(-3, 3, 150).reshape(-1, 1)
y = np.sin(X) + 0.5 * np.cos(2*X)
y = y.ravel()

# Multi-layer quantum model
model = keras.Sequential([
    keras.layers.Input(shape=(1,)),
    
    # First quantum layer
    QKerasLayer(
        layers=2,
        scaling=1.0,
        name="quantum_layer_1"
    ),
    
    # Second quantum layer
    QKerasLayer(
        layers=2,
        scaling=1.0,
        name="quantum_layer_2"
    ),
    
    # Optional: Add classical layer
    keras.layers.Dense(1)
], name="MultiQuantumModel")

print("Model Architecture:")
model.summary()

# Compile and train
model.compile(
    optimizer=keras.optimizers.Adam(0.02),
    loss="mse"
)

history = model.fit(X, y, epochs=60, batch_size=32, verbose=1)

# Predictions
predictions = model.predict(X)
mse = np.mean((predictions - y.reshape(-1, 1))**2)
print(f"\nFinal MSE: {mse:.4f}")

# Visualize
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'])
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training Progress')
plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(X, y, 'b-', label='Target', linewidth=2)
plt.plot(X, predictions, 'r--', label='Multi-Quantum Model', linewidth=2)
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Multi-Layer Quantum Network')
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('multi_layer_results.png')
plt.show()
```

---

## Custom Training Loop

Advanced example with custom training loop for fine-grained control.

```python
"""
Custom training loop example.
"""
import os
os.environ["KERAS_BACKEND"] = "jax"

import numpy as np
import keras
from pennylane_keras_layer import QKerasLayer

# Data
X_train = np.linspace(-1, 1, 80).reshape(-1, 1)
y_train = np.tanh(2 * X_train)

# Model
model = keras.Sequential([
    keras.layers.Input(shape=(1,)),
    QKerasLayer(layers=3, scaling=1.5)
])

# Optimizer and loss
optimizer = keras.optimizers.Adam(learning_rate=0.02)
loss_fn = keras.losses.MeanSquaredError()

# Training loop
epochs = 50
batch_size = 16
num_batches = len(X_train) // batch_size

train_losses = []

# For JAX backend, use JAX-specific gradient computation
import jax

for epoch in range(epochs):
    epoch_loss = 0
    
    # Shuffle data
    indices = np.random.permutation(len(X_train))
    X_shuffled = X_train[indices]
    y_shuffled = y_train[indices]
    
    for batch in range(num_batches):
        start = batch * batch_size
        end = start + batch_size
        
        X_batch = X_shuffled[start:end]
        y_batch = y_shuffled[start:end]
        
        # Forward pass with gradient computation
        def compute_loss(trainable_vars):
            # Set model weights
            for var, val in zip(model.trainable_variables, trainable_vars):
                var.assign(val)
            predictions = model(X_batch, training=True)
            return loss_fn(y_batch, predictions)
        
        # Compute gradients
        loss_value, gradients = jax.value_and_grad(
            lambda vars: compute_loss(vars).numpy()
        )(model.trainable_variables)
        
        # Update weights
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        epoch_loss += loss.numpy()
    
    # Average loss for epoch
    avg_loss = epoch_loss / num_batches
    train_losses.append(avg_loss)
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

# Final evaluation
predictions = model(X_train)
final_mse = np.mean((predictions.numpy() - y_train)**2)
print(f"\nFinal MSE: {final_mse:.4f}")

# Plot
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(train_losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Custom Training Loop')
plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(X_train, y_train, 'b-', label='Target')
plt.plot(X_train, predictions.numpy(), 'r--', label='Predicted')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Results')
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('custom_training_results.png')
plt.show()
```

---

## Hyperparameter Comparison

Compare different hyperparameter configurations.

```python
"""
Compare different layer counts and scaling factors.
"""
import os
os.environ["KERAS_BACKEND"] = "jax"

import numpy as np
import keras
from pennylane_keras_layer import QKerasLayer
import matplotlib.pyplot as plt

# Data
X = np.linspace(-2, 2, 100).reshape(-1, 1)
y = np.sin(np.pi * X).ravel()

# Test configurations
configs = [
    {'layers': 2, 'scaling': 1.0, 'name': '2 layers, scale=1.0'},
    {'layers': 3, 'scaling': 1.0, 'name': '3 layers, scale=1.0'},
    {'layers': 2, 'scaling': 2.0, 'name': '2 layers, scale=2.0'},
    {'layers': 4, 'scaling': 1.5, 'name': '4 layers, scale=1.5'},
]

results = []

for config in configs:
    print(f"\nTraining: {config['name']}")
    
    # Create model
    model = keras.Sequential([
        keras.layers.Input(shape=(1,)),
        QKerasLayer(layers=config['layers'], scaling=config['scaling'])
    ])
    
    # Train
    model.compile(optimizer=keras.optimizers.Adam(0.02), loss="mse")
    history = model.fit(X, y, epochs=40, verbose=0)
    
    # Evaluate
    predictions = model.predict(X, verbose=0)
    final_loss = history.history['loss'][-1]
    
    results.append({
        'config': config,
        'history': history,
        'predictions': predictions,
        'final_loss': final_loss
    })
    
    print(f"  Final loss: {final_loss:.4f}")

# Visualize comparison
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

for idx, result in enumerate(results):
    row = idx // 2
    col = idx % 2
    ax = axes[row, col]
    
    config = result['config']
    predictions = result['predictions']
    
    ax.plot(X, y, 'k-', label='Target', linewidth=2, alpha=0.7)
    ax.plot(X, predictions, 'r--', label='Predicted', linewidth=2)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f"{config['name']}\nMSE: {result['final_loss']:.4f}")
    ax.legend()
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('hyperparameter_comparison.png', dpi=100)
print("\n✓ Comparison saved to hyperparameter_comparison.png")
plt.show()

# Print summary
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
best = min(results, key=lambda x: x['final_loss'])
print(f"Best configuration: {best['config']['name']}")
print(f"Best MSE: {best['final_loss']:.4f}")
```

---

## Running Examples

All examples can be run from the command line:

```bash
# Navigate to examples directory
cd /path/to/pennylane-keras-layer/examples

# Run basic example
python basic_example.py

# Run Fourier series example
python fourier_series_example.py
```

Or copy-paste the code directly into a Python script or Jupyter notebook.

## Next Steps

- Read [Tutorials](tutorials.md) for step-by-step guides
- Check [User Guide](user_guide.md) for concepts and best practices
- See [API Reference](api_reference.md) for detailed documentation
