# SoftDTW-CUDA (PyTorch + Numba)

A **GPU-accelerated, memory-efficient, and numerically stable** implementation of
**Soft Dynamic Time Warping (SoftDTW)** for PyTorch.

This package is designed primarily as a **loss function for training neural networks**, with additional support for **time series averaging** (barycenters). Strong emphasis on:

* 🔥 **GPU memory efficiency**
* 📏 **Long sequence support** (lengths > 1024)
* 🧮 **Numerical stability** (log-space backward)
* ⚡ **Optional fused distance computation** (no `(B,N,M)` tensor)
* 📊 **Time series averaging** (SoftDTW barycenters)

---

## Why This Implementation?

Compared to the popular CUDA implementation by [Maghoumi et al.](https://github.com/mblondel/soft-dtw), this repo fixes critical limitations for real training workloads:

### Feature Comparison

| Feature | Maghoumi CUDA | This Repo |
|---|---|---|
| CUDA forward | ✅ | ✅ |
| CUDA backward | ⚠️ numerically unstable | ✅ log-space stable |
| Max sequence length | ❌ ≤ 1024 | ✅ unbounded (tiled) |
| Memory-efficient fused mode | ❌ | ✅ |

### Key Benchmark (B=32, N=512, D=64)

| | Maghoumi | Ours (Unfused) | Ours (Fused) |
|---|---|---|---|
| **Peak Memory** | 8,256 MB | 257 MB | 161 MB |
| **Runtime** | 2,791 ms | **42 ms** | 430 ms |
| **vs. Maghoumi memory** | — | 96.9% less | 98.0% less |
| **vs. Maghoumi speed** | — | **67× faster** | 6.5× faster |

### When to Use Each Mode

| Scenario | Mode | Reason |
|---|---|---|
| Large D, big batches | Fused | ~98% memory savings |
| Speed-critical / inference | Unfused | 10–67× faster than Fused |
| N > 1024 | Both modes | Both use tiled anti-diagonal execution; fused saves more memory |
| Small D (D=1–4) | Unfused | Fused savings are small (~30%) |

### Limitations

* Fused mode requires **CUDA** and **squared Euclidean distance only**
* Fused is 10–25× slower in runtime than unfused (memory/compute trade-off)
* CPU implementation is for testing only, not performance

> Full benchmark tables and analysis: [bench/README.md](bench/README.md)

---

## Installation

### Requirements

* Python ≥ 3.10
* NVIDIA GPU with CUDA toolkit **≤ 12.6**
* PyTorch with CUDA support (see below)
* Numba ≥ 0.60

> ⚠️ Tested with CUDA ≤ 12.6. Compatibility with newer CUDA versions is not guaranteed.

### Step 1 — Install PyTorch with CUDA

PyTorch must be installed **before** this package, with the correct CUDA variant for your system. See [pytorch.org/get-started](https://pytorch.org/get-started/locally/) for the right command. Example for CUDA 12.4:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
```


### Step 2 — Install this package

```bash
git clone https://github.com/BGU-CS-VIL/sdtw-cuda-torch
pip install -e sdtw-cuda-torch
```

---

## Usage

### Basic (Unfused)

```python
from softdtw_cuda import SoftDTW

loss_fn = SoftDTW(gamma=1.0)

x = torch.randn(B, N, D, device="cuda", requires_grad=True)
y = torch.randn(B, M, D, device="cuda", requires_grad=True)

loss = loss_fn(x, y).mean()
loss.backward()
```

* Explicit distance computation
* More flexible
* Higher memory usage

---

### Fused Mode (Recommended for Training)

```python
loss_fn = SoftDTW(
    gamma=1.0,
    dist="sqeuclidean",
    fused=True
)

loss = loss_fn(x, y).mean()
loss.backward()
```

**Fused mode**

* No distance tensor
* Much lower GPU memory
* Best choice for large `N`, `D`

---
# Applications
## Forecasting

![Forecasting](https://github.com/BGU-CS-VIL/sdtw-cuda-torch/blob/main/examples/forecasting_results.png)
Train a simple forecaster using SoftDTW as the loss function:

```python
import torch
from softdtw_cuda import SoftDTW

model = MyForecaster().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = SoftDTW(gamma=1.0, fused=True)

for x_batch, y_batch in dataloader:
    y_pred = model(x_batch.cuda())           # (B, N, D)
    loss = loss_fn(y_pred, y_batch.cuda()).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

See [examples/forecasting_example.py](examples/forecasting_example.py) for a complete working example with sine wave data.



## Time Series Barycenters (Averaging)
![SoftDTW Barycenter](https://github.com/BGU-CS-VIL/sdtw-cuda-torch/blob/main/examples/softdtw_barycenter_example.png)

Compute a DTW-space average (barycenter) for a batch of sequences:

```python
from softdtw_cuda import softdtw_barycenter

sequences = torch.randn(10, 100, 3, device="cuda")  # 10 sequences

barycenter = softdtw_barycenter(
    sequences,
    gamma=1.0,
    max_iter=100,
    lr=0.1,
)

print(barycenter.shape)  # (100, 3)
```

**Key options:**

* `gamma`: Regularization strength (higher = smoother)
* `max_iter`: Optimization iterations
* `lr`: Adam learning rate (0.1 default)
* `fused`: Auto-select fused mode (memory vs speed trade-off)
* `early_stopping=True`: Detects convergence, saves ~30-50% iterations

See [BARYCENTERS.md](softdtw_cuda/BARYCENTERS.md) for detailed docs and [examples/barycenter_example.py](examples/barycenter_example.py) for visualization.


---

## Normalization

Supports the common normalized variant:

$$\mathrm{SoftDTW\_norm}(x,y) = \mathrm{SoftDTW}(x,y) - \tfrac{1}{2}\bigl(\mathrm{SoftDTW}(x,x) + \mathrm{SoftDTW}(y,y)\bigr)$$

Enable with:

```python
SoftDTW(normalize=True)
```

⚠️ **Current constraint:** normalization requires equal sequence lengths
`x.shape == y.shape == (B, N, D)`

---

## Notes

* SoftDTW **may return negative values** (expected)
* Squared Euclidean distances are always ≥ 0
* Negativity arises from the soft-min aggregation

---

## Tests

```bash
pytest -v
```

| Test file | What it covers |
|---|---|
| `test_softdtw_small.py` | CPU and CUDA forward/backward, gradient correctness |
| `test_softdtw_long.py` | Sequences longer than 1024 (tiled kernel) |
| `test_softdtw_log_backward.py` | Log-space backward numerical stability |
| `test_fused_sqeuclid.py` | Fused vs unfused equivalence for squared Euclidean |
| `test_sqeuclidean.py` | Distance computation correctness |
| `test_validation.py` | Input validation: gamma, device, empty sequences, shape mismatches |

---

## Benchmarking

Full benchmark suite available in `bench/` directory. Key results:

**SoftDTW Loss Function:**
* Memory efficiency: 92-98% reduction vs. Maghoumi et al.
* Supports arbitrary sequence lengths (no 1024 limit)
* Numerically stable via log-space backward pass

**Barycenter Optimization:**
* Early stopping typically saves 30-50% of iterations
* Cosine annealing + gradient clipping ensures stability
* Supports both fused and unfused modes

Run benchmarks with:
```bash
python bench/bench_memory.py
python examples/barycenter_example.py --compare
```

---

## Acknowledgments

**SoftDTW Loss:**
> Cuturi & Blondel,
> *Soft-DTW: a Differentiable Loss Function for Time-Series*, ICML 2017

**Barycenter Implementation:**
> Based on [tslearn](https://github.com/tslearn-team/tslearn) implementation, originally from Cuturi & Blondel (ICML 2017)

**Prior PyTorch/CUDA implementations this work builds on:**
* [Sleepwalking/pytorch-softdtw](https://github.com/Sleepwalking/pytorch-softdtw) — PyTorch GPU implementation
* [Maghoumi/pytorch-softdtw-cuda](https://github.com/Maghoumi/pytorch-softdtw-cuda) — CUDA implementation (motivation for memory and stability improvements)
* [keonlee9420/Soft-DTW-Loss](https://github.com/keonlee9420/Soft-DTW-Loss) — additional PyTorch reference implementation

---

## License

MIT

