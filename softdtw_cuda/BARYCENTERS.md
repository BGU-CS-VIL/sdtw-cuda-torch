# SoftDTW Barycenters

Time series averaging using soft Dynamic Time Warping geometry.

## Overview

A **barycenter** is the "average" of a set of sequences in DTW space. Unlike Euclidean averaging (point-wise mean), DTW barycenters align sequences first and then average, capturing the underlying pattern shape even when sequences have different timing.

**Reference:**
- Cuturi, M., & Blondel, M. (2017). Soft-DTW: a Differentiable Loss Function for Time-Series. In ICML.
- tslearn implementation: https://github.com/tslearn-team/tslearn/blob/main/tslearn/barycenters/softdtw.py

## API

### `softdtw_barycenter()`

Compute a SoftDTW barycenter through gradient-based optimization.

```python
from softdtw_cuda import softdtw_barycenter
import torch

# Input: batch of time series (B, N, D)
X = torch.randn(16, 100, 3, device="cuda", requires_grad=False)

# Compute barycenter
barycenter = softdtw_barycenter(
    X,
    gamma=1.0,           # Regularization parameter
    weights=None,        # Optional per-sequence weights
    max_iter=100,        # Optimization iterations
    lr=0.01,            # Learning rate
    init=None,          # Initial barycenter (None = Euclidean mean)
    device="cuda",      # Compute device
)

# Output: (N, D)
print(barycenter.shape)  # torch.Size([100, 3])
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `X` | Tensor | required | Input sequences (B, N, D) |
| `gamma` | float | 1.0 | SoftDTW regularization (higher = smoother) |
| `weights` | Tensor | None | Per-sequence weights (B,); if None, uniform |
| `max_iter` | int | 100 | Maximum optimization iterations |
| `lr` | float | 0.1 | Adam learning rate (decayed via cosine annealing) |
| `init` | Tensor | None | Initial barycenter (N, D); if None, weighted mean |
| `device` | str/device | None | Compute device; if None, uses X's device |
| `fused` | bool/None | None | Fused mode: `None` (auto), `True` (require), `False` (never) |
| `verbose` | bool | False | Print iteration progress with convergence info |
| `early_stopping` | bool | True | Stop early if loss plateaus (usually saves 30-50% iterations) |
| `patience` | int | 10 | Iterations without improvement before stopping |
| `tol` | float | 1e-5 | Absolute improvement threshold (handles negative SoftDTW values) |

### Returns

- **barycenter** (Tensor): Computed barycenter of shape (N, D)

## Algorithm

The barycenter is found by solving:

```
minimize   Σ_i w_i * SoftDTW(barycenter, X_i)
```

where:
- `w_i` are the (optional) weights
- `SoftDTW(·, ·)` is the soft DTW distance
- Optimization uses Adam with automatic gradients

**Steps:**

1. **Initialize**: Weighted mean of input sequences (better than unweighted mean)
2. For each iteration:
   - Compute SoftDTW distances from barycenter to all sequences
   - Compute weighted loss (can be negative due to soft-min aggregation)
   - Backpropagate gradients with clipping for stability
   - Update barycenter via Adam optimizer with cosine annealing LR decay
3. **Early Stopping** (optional): Stop if loss plateaus for `patience` iterations

**Optimizations:**
- ✓ **Weighted initialization**: Better starting point than Euclidean mean
- ✓ **Cosine annealing**: Learning rate decays from `lr` to `lr * 0.01` over iterations
- ✓ **Gradient clipping**: Prevents unstable updates (max norm = 1.0)
- ✓ **Early stopping**: Detects convergence, typically saves 30-50% of iterations
- ✓ **Device handling**: Automatic data movement to target device
- ✓ **CUDA sync**: Accurate timing via `torch.cuda.synchronize()`

## Example

### Basic Usage

```python
import torch
from softdtw_cuda import softdtw_barycenter

# Generate example sequences
sequences = torch.randn(10, 50, 3)

# Compute barycenter
barycenter = softdtw_barycenter(
    sequences,
    gamma=1.0,
    max_iter=50,
)

print(barycenter.shape)  # (50, 3)
```

### Weighted Averaging

```python
# Weight sequences by importance
weights = torch.tensor([2.0, 1.0, 1.0, 1.0, 1.0])  # Emphasize first sequence
barycenter = softdtw_barycenter(
    sequences,
    gamma=1.0,
    weights=weights,
)
```

### With Custom Initialization

```python
# Start from a specific barycenter
init = sequences.mean(0)  # Euclidean mean
barycenter = softdtw_barycenter(
    sequences,
    gamma=1.0,
    init=init,
    max_iter=100,
)
```

### Explicit Fused Mode Selection

```python
# Auto-select (default) - uses fused if CUDA available
barycenter = softdtw_barycenter(sequences, fused=None)

# Force fused mode (saves 98% memory, slower)
barycenter = softdtw_barycenter(sequences, fused=True)

# Force unfused mode (uses distance matrix, faster)
barycenter = softdtw_barycenter(sequences, fused=False)
```

**When to use each:**

- **`fused=None` (auto)**: Recommended for most use cases; balances convenience and efficiency
- **`fused=True`**: When memory is the bottleneck; accept slower runtime
- **`fused=False`**: When speed is critical; accept higher memory usage

### Early Stopping (Convergence Detection)

```python
# Enable early stopping (default) - stops when loss plateaus
barycenter = softdtw_barycenter(
    sequences,
    max_iter=100,           # Will stop early if converged
    early_stopping=True,    # Enable convergence detection
    patience=10,            # Stop after 10 iterations without improvement
    tol=1e-5,              # Improvement threshold (absolute)
    verbose=True,           # Show convergence progress
)

# Disable early stopping (always run full max_iter)
barycenter = softdtw_barycenter(
    sequences,
    max_iter=100,
    early_stopping=False,
)
```

**Typical speedup:** With `early_stopping=True`, optimization usually completes in 50-70% of `max_iter` iterations.

## Visualization Example

See [examples/barycenter_example.py](../examples/barycenter_example.py) for a complete working example that:

1. Generates example time series
2. Computes the SoftDTW barycenter
3. Plots original sequences and barycenter side-by-side

Run with:
```bash
python examples/barycenter_example.py
```

## Best Practices

**For faster convergence:**
```python
barycenter = softdtw_barycenter(
    X,
    gamma=1.0,
    max_iter=100,
    lr=0.05,              # Higher LR (still safe with clipping)
    early_stopping=True,  # Enable convergence detection
    patience=10,
    verbose=True,         # Monitor progress
)
```

**For stable convergence (many iterations):**
```python
barycenter = softdtw_barycenter(
    X,
    gamma=1.0,
    max_iter=200,
    lr=0.01,              # Conservative LR
    early_stopping=False, # Run full iterations
)
```

**For memory-constrained settings:**
```python
barycenter = softdtw_barycenter(
    X,
    gamma=1.0,
    max_iter=100,
    lr=0.01,
    fused=True,           # Save 98% memory
    early_stopping=True,  # May help more with fused mode
)
```


## Limitations

- Currently supports only squared Euclidean distance (`dist='sqeuclidean'`)
- All input sequences must have the same length
- Optimization is local, not global (depends on initialization)
- Requires gradient computation (some overhead vs. non-differentiable methods)
- SoftDTW distance can be negative (soft-min aggregation), not a true metric

## References

1. Petitjean, F., Ketterlin, A., & Gançarski, P. (2011). A global averaging method for dynamic time warping, with applications to clustering.
2. Cuturi, M., & Blondel, M. (2017). Soft-DTW: a Differentiable Loss Function for Time-Series. ICML.
3. Tavenard, R., et al. (2017). tslearn: A machine learning toolkit dedicated to time-series data. JMLR.


## Citation

If you use this implementation in research, please cite:

```bibtex
@inproceedings{cuturi2017soft,
  title={Soft-DTW: a Differentiable Loss Function for Time-Series},
  author={Cuturi, Marco and Blondel, Mathieu},
  booktitle={ICML},
  year={2017}
}
```
