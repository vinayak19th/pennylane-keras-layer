"""
Fourier Series Approximation with Quantum Circuits
===================================================

This example demonstrates how to use QKerasLayer to approximate a truncated 
Fourier series using a Data Re-Uploading quantum model. This is based on the
'Quantum models as Fourier series' demo.

The example shows:
1. Creating a target function (truncated Fourier series)
2. Building a quantum model with QKerasLayer
3. Training the model to fit the target function
4. Saving and loading the trained model

Requirements:
- pennylane-keras-layer
- keras (with your choice of backend: TensorFlow, JAX, or PyTorch)
- numpy
- matplotlib (for visualization)
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# Set Keras backend (can be 'jax', 'tensorflow', or 'torch')
os.environ["KERAS_BACKEND"] = "jax"  # Change to your preferred backend

import keras
from keras import ops

# Configure float precision
keras.backend.set_floatx('float64')
if keras.backend.backend() == "jax":
    print("Setting jax to use float64")
    import jax
    jax.config.update("jax_enable_x64", True)

print(f"Using Keras backend: {keras.backend.backend()}")

# Import PennyLane and our custom layer
import pennylane as qml
from pennylane_keras_layer import QKerasLayer


# Define the target function
def create_target_function(degree=1, scaling=1, coeffs=None, coeff0=0.1):
    """Create a truncated Fourier series function.
    
    Args:
        degree (int): Degree of the target function
        scaling (float): Scaling of the data
        coeffs (list): Coefficients of non-zero frequencies
        coeff0 (float): Coefficient of zero frequency
        
    Returns:
        callable: Target function
    """
    if coeffs is None:
        coeffs = [0.15 + 0.15j] * degree
    
    def target_function(x):
        """Generate a truncated Fourier series, where the data gets re-scaled."""
        res = coeff0
        for idx, coeff in enumerate(coeffs):
            exponent = np.complex128(scaling * (idx + 1) * x * 1j)
            conj_coeff = np.conjugate(coeff)
            res += coeff * np.exp(exponent) + conj_coeff * np.exp(-exponent)
        return np.real(res)
    
    return target_function


def main():
    """Main function to run the Fourier series approximation demo."""
    
    # Create target function
    print("\n" + "="*60)
    print("FOURIER SERIES APPROXIMATION WITH QUANTUM CIRCUITS")
    print("="*60)
    
    degree = 1
    scaling = 1.0
    target_fn = create_target_function(degree=degree, scaling=scaling)
    
    # Generate data
    x = np.linspace(-6, 6, 70)
    target_y = np.array([target_fn(x_) for x_ in x])
    
    print(f"\nTarget function: Degree {degree} Fourier series")
    print(f"Data points: {len(x)}")
    
    # Plot target function
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(x, target_y, c="black", label="Target")
    plt.scatter(x, target_y, facecolor="white", edgecolor="black")
    plt.ylim(-1, 1)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Target Function (Fourier Series)")
    plt.legend()
    
    # Create quantum model
    print("\nBuilding quantum model...")
    layers = 2  # Number of quantum layers
    
    q_layer = QKerasLayer(
        layers=layers,
        scaling=scaling,
        circ_backend="default.qubit",
        num_wires=1,
        name="QuantumLayer"
    )
    
    # Build Keras model
    inp = keras.layers.Input(shape=(1,))
    out = q_layer(inp)
    model = keras.models.Model(inputs=inp, outputs=out, name="QuantumModel")
    
    print("\nModel architecture:")
    model.summary()
    
    # Make predictions with random weights
    print("\nMaking predictions with random weights...")
    random_predictions = model(x)
    
    # Plot untrained predictions
    plt.subplot(1, 2, 2)
    plt.plot(x, target_y, c="black", label="Target", alpha=0.3)
    plt.plot(x, random_predictions, c="blue", label="Untrained Model")
    plt.ylim(-1, 1)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Untrained Model Predictions")
    plt.legend()
    plt.tight_layout()
    plt.savefig("/tmp/quantum_model_before_training.png", dpi=100)
    print("Saved plot: /tmp/quantum_model_before_training.png")
    plt.close()
    
    # Train the model
    print("\nTraining model...")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.03),
        loss=keras.losses.mean_squared_error,
        run_eagerly=True
    )
    
    history = model.fit(x=x, y=target_y, epochs=30, verbose=1)
    
    # Make predictions with trained model
    print("\nMaking predictions with trained model...")
    predictions = model(x)
    
    # Calculate final loss
    final_loss = keras.losses.mean_squared_error(target_y, predictions).numpy().mean()
    print(f"\nFinal MSE: {final_loss:.6f}")
    
    # Plot trained predictions
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(x, target_y, c="black", label="Target")
    plt.scatter(x, target_y, facecolor="white", edgecolor="black")
    plt.plot(x, predictions, c="blue", label="Trained Model", linewidth=2)
    plt.ylim(-1, 1)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Trained Model (MSE: {final_loss:.6f})")
    plt.legend()
    plt.tight_layout()
    plt.savefig("/tmp/quantum_model_after_training.png", dpi=100)
    print("Saved plot: /tmp/quantum_model_after_training.png")
    plt.close()
    
    # Save the model
    model_path = "/tmp/quantum_model.keras"
    print(f"\nSaving model to {model_path}...")
    model.save(model_path)
    
    # Load and test the model
    print(f"Loading model from {model_path}...")
    loaded_model = keras.models.load_model(model_path)
    
    # Make predictions with loaded model
    loaded_predictions = loaded_model(x)
    
    # Verify loaded model
    diff = np.abs(predictions - loaded_predictions).max()
    print(f"Max difference between original and loaded model: {diff:.10f}")
    
    if diff < 1e-6:
        print("✓ Model saved and loaded successfully!")
    else:
        print("⚠ Warning: Loaded model predictions differ from original")
    
    print("\n" + "="*60)
    print("Demo completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()
