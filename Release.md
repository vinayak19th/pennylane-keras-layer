# Release 0.1.0

## What's Changed

### 🚀 New Features & Enhancements
- **Keras 3 & Multi-Backend Support**: Full compatibility with TensorFlow, JAX, and PyTorch (set via `KERAS_BACKEND` env var).
- **Generic `KerasCircuitLayer`**: New flexible layer to wrap any PennyLane QNode.
- **Improved `KerasDRCircuitLayer`**: Specialized layer for Data Re-Uploading models with easier configuration.

### 🐛 Bug Fixes
- Fixed casting issues and prevented layer rebuilding if already built (d309472)
- Corrected `draw_qnode` function to use `self.interface` instead of `keras.backend` (2ecabae)
- Fixed pi symbol rendering in docstrings (d968a95)
- Updated unit tests to fix failures (692fc85)

### 📚 Documentation
- Added detailed docstrings for `KerasDRCircuitLayer` (e65a3b6)
- Consolidated examples into the correct folder (1c0aa25)
- Updated `Getting Started`, `User Guide`, and `API Reference` to reflect new architecture.

### ⚙️ CI/CD & Tests
- Created multi-environment testing grid for robust backend testing (2158872)

### 🧹 Chores
- Updated examples for clarity (e3cf8ca)
- Unified `self.circuit` vs `self.qnode` arguments (0cdf2b0)
- Refactored examples:
    - Renamed `basic_example.py` to `import_tests.py` (27fa10b)
    - Renamed `RandomTrainingTest.ipynb` to `KerasCircuitLayer_example.ipynb` (d5b64b0)
    - Removed redundant `tutorial_qnn_module_keras.py` (17537cb)

## Breaking Changes
- `QKerasLayer` has been removed/renamed to `KerasCircuitLayer` and `KerasDRCircuitLayer`. Update your imports and initialization accordingly.

**Full Changelog**: https://github.com/vinayak19th/pennylane-keras-layer/compare/v0.0.1...v0.1.0
