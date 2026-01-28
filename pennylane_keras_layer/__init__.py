"""
PennyLane Keras Layer
=====================

A library to introduce Keras 3 support to PennyLane with full support 
for multi-backend training.

This package provides integration between PennyLane quantum computing 
framework and Keras 3, enabling quantum layers in neural networks.
"""

__version__ = "0.1.0"
__author__ = "Vinayak"

# Import main components
from pennylane_keras_layer.layer import KerasCircuitLayer, KerasDRCircuitLayer

__all__ = ["KerasCircuitLayer", "KerasDRCircuitLayer"]
