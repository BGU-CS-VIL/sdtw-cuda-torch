from __future__ import annotations

import math
import pytest
import torch

from softdtw_cuda import SoftDTW, softdtw


# ---------------------------------------------------------------------------
# gamma validation
# ---------------------------------------------------------------------------

def test_gamma_zero_raises():
    with pytest.raises(ValueError, match="gamma must be > 0"):
        SoftDTW(gamma=0.0)


def test_gamma_negative_raises():
    with pytest.raises(ValueError, match="gamma must be > 0"):
        SoftDTW(gamma=-1.0)


def test_gamma_inf_raises():
    with pytest.raises(ValueError, match="gamma must be finite"):
        SoftDTW(gamma=math.inf)


def test_gamma_nan_raises():
    with pytest.raises(ValueError, match="gamma must be finite"):
        SoftDTW(gamma=math.nan)


# ---------------------------------------------------------------------------
# fused=True on CPU raises
# ---------------------------------------------------------------------------

def test_fused_true_on_cpu_raises():
    sdtw = SoftDTW(gamma=1.0, fused=True)
    x = torch.randn(2, 5, 3)
    y = torch.randn(2, 5, 3)
    with pytest.raises(ValueError, match="fused=True requires CUDA"):
        sdtw(x, y)


# ---------------------------------------------------------------------------
# normalize=True with unequal sequence lengths raises
# ---------------------------------------------------------------------------

def test_normalize_unequal_lengths_raises():
    sdtw = SoftDTW(gamma=1.0, normalize=True)
    x = torch.randn(2, 5, 3)
    y = torch.randn(2, 7, 3)
    with pytest.raises(ValueError, match="normalize=True"):
        sdtw(x, y)


# ---------------------------------------------------------------------------
# Empty sequences raise
# ---------------------------------------------------------------------------

def test_empty_sequence_x_raises():
    sdtw = SoftDTW(gamma=1.0)
    x = torch.randn(2, 0, 3)
    y = torch.randn(2, 5, 3)
    with pytest.raises(ValueError, match="Sequence lengths must be > 0"):
        sdtw(x, y)


def test_empty_sequence_y_raises():
    sdtw = SoftDTW(gamma=1.0)
    x = torch.randn(2, 5, 3)
    y = torch.randn(2, 0, 3)
    with pytest.raises(ValueError, match="Sequence lengths must be > 0"):
        sdtw(x, y)


# ---------------------------------------------------------------------------
# Batch size mismatch raises
# ---------------------------------------------------------------------------

def test_batch_size_mismatch_raises():
    sdtw = SoftDTW(gamma=1.0)
    x = torch.randn(2, 5, 3)
    y = torch.randn(3, 5, 3)
    with pytest.raises(ValueError, match="Batch sizes must match"):
        sdtw(x, y)


# ---------------------------------------------------------------------------
# Feature dim mismatch raises
# ---------------------------------------------------------------------------

def test_feature_dim_mismatch_raises():
    sdtw = SoftDTW(gamma=1.0)
    x = torch.randn(2, 5, 3)
    y = torch.randn(2, 5, 4)
    with pytest.raises(ValueError, match="Feature dims must match"):
        sdtw(x, y)


# ---------------------------------------------------------------------------
# functional API exposes fused parameter
# ---------------------------------------------------------------------------

def test_functional_fused_none_runs():
    """softdtw() with fused=None (default) should work on CPU."""
    torch.manual_seed(0)
    x = torch.randn(2, 5, 3)
    y = torch.randn(2, 5, 3)
    out = softdtw(x, y, fused=None)
    assert out.shape == (2,)
    assert torch.isfinite(out).all()


def test_functional_fused_true_on_cuda():
    """softdtw() with fused=True should work when CUDA is available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    torch.manual_seed(0)
    x = torch.randn(2, 8, 3, device="cuda", requires_grad=True)
    y = torch.randn(2, 8, 3, device="cuda")
    out = softdtw(x, y, fused=True)
    assert out.shape == (2,)
    assert torch.isfinite(out).all()
    out.sum().backward()
    assert torch.isfinite(x.grad).all()
