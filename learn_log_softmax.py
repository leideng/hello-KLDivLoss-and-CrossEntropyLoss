"""Learn torch.nn.LogSoftmax: log(softmax(x)) applied along a dimension.

Run: uv run python learn_log_softmax.py
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _header(title: str) -> None:
    print(f"\n=== {title} ===")


def test_1d_dim0() -> None:
    """1D tensor, softmax along dim=0 (the only axis)."""
    _header("1D, dim=0 (default)")
    x = torch.tensor([1.0, 2.0, 3.0])
    y = F.log_softmax(x, dim=0)
    manual = x - torch.logsumexp(x, dim=0)
    print("input: ", x)       # tensor([1., 2., 3.])
    print("output:", y)       # tensor([-2.4076, -1.4076, -0.4076])
    print("manual log_softmax:", manual)  # tensor([-2.4076, -1.4076, -0.4076])
    print("exp(output) sums to:", y.exp().sum().item())  # 1.0


def test_2d_dim1() -> None:
    """2D tensor, softmax along dim=1 (each row independently)."""
    _header("2D, dim=1 (per row)")
    x = torch.tensor([[1.0, 2.0, 3.0], [3.0, 1.0, 2.0]])
    y = F.log_softmax(x, dim=1)
    print("input:\n", x)
    # tensor([[1., 2., 3.],
    #         [3., 1., 2.]])
    print("output:\n", y)
    # tensor([[-2.4076, -1.4076, -0.4076],
    #         [-0.4076, -2.4076, -1.4076]])
    print("row sums after exp:", y.exp().sum(dim=1))  # tensor([1.0000, 1.0000])


def test_2d_dim0() -> None:
    """2D tensor, softmax along dim=0 (each column independently)."""
    _header("2D, dim=0 (per column)")
    x = torch.tensor([[1.0, 2.0, 3.0], [3.0, 1.0, 2.0]])
    y = F.log_softmax(x, dim=0)
    print("output:\n", y)
    # tensor([[-2.1269, -0.3133, -0.3133],
    #         [-0.1269, -1.3133, -1.3133]])


def test_3d_last_dim() -> None:
    """3D tensor, softmax along last axis (dim=-1)."""
    _header("3D, dim=-1 (last axis)")
    torch.manual_seed(0)
    x = torch.randn(2, 2, 3)
    y = F.log_softmax(x, dim=-1)
    print("input shape:", x.shape)    # torch.Size([2, 2, 3])
    print("output shape:", y.shape)   # torch.Size([2, 2, 3])
    print("slice [0,0]:", y[0, 0])   # tensor([-0.1689, -2.0033, -3.8886])


def test_3d_middle_dim() -> None:
    """3D tensor, softmax along middle axis (dim=1)."""
    _header("3D, dim=1 (middle axis)")
    x = torch.tensor(
        [
            [[1.0, 2.0, 3.0], [3.0, 1.0, 2.0]],
            [[2.0, 3.0, 1.0], [1.0, 2.0, 3.0]],
        ]
    )
    y = F.log_softmax(x, dim=1)
    print("slice [0,:,0]:", y[0, :, 0])  # tensor([-2.1269, -0.1269])


def test_dtypes() -> None:
    """Output dtype matches input dtype."""
    _header("dtypes: float32 vs float64")
    x32 = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    x64 = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    y32 = F.log_softmax(x32, dim=0)
    y64 = F.log_softmax(x64, dim=0)
    print("float32 output dtype:", y32.dtype)  # torch.float32
    print("float64 output dtype:", y64.dtype)  # torch.float64


def test_large_logits() -> None:
    """Large logits: numerically stable (no inf/nan)."""
    _header("large logits (numerical stability)")
    x = torch.tensor([1000.0, 1001.0, 1002.0])
    y = F.log_softmax(x, dim=0)
    print("input: ", x)    # tensor([1000., 1001., 1002.])
    print("output:", y)    # tensor([-2.4076, -1.4076, -0.4076])  same as [1,2,3]
    print("(no inf/nan; same centered log-probs as small logits)")


def test_negative_logits() -> None:
    """Negative logits work the same way."""
    _header("negative logits")
    x = torch.tensor([-1.0, 0.0, 1.0])
    y = F.log_softmax(x, dim=0)
    print("input: ", x)    # tensor([-1.,  0.,  1.])
    print("output:", y)    # tensor([-2.4076, -1.4076, -0.4076])


def test_uniform() -> None:
    """Equal logits => uniform distribution => log(1/C) for each class."""
    _header("all-equal logits => uniform log-probs")
    x = torch.tensor([5.0, 5.0, 5.0])
    y = F.log_softmax(x, dim=0)
    expected = math.log(1.0 / 3.0)
    print("input: ", x)    # tensor([5., 5., 5.])
    print("output:", y)    # tensor([-1.0986, -1.0986, -1.0986])
    print(f"log(1/3) ≈ {expected:.4f}")  # log(1/3) ≈ -1.0986


def test_vs_log_softmax() -> None:
    """F.log_softmax is equivalent to log(F.softmax) (tiny float error)."""
    _header("LogSoftmax vs log(softmax): equivalent")
    x = torch.tensor([1.0, 2.0, 3.0])
    a = F.log_softmax(x, dim=0)
    b = torch.log(F.softmax(x, dim=0))
    print("max |diff|:", (a - b).abs().max().item())  # 2.9802322387695312e-08


def test_module() -> None:
    """nn.LogSoftmax module wraps the same functional."""
    _header("nn.LogSoftmax module (same as functional)")
    x = torch.tensor([1.0, 2.0, 3.0])
    m = nn.LogSoftmax(dim=0)
    print("module output:", m(x))  # tensor([-2.4076, -1.4076, -0.4076])


def main() -> None:
    test_1d_dim0()
    test_2d_dim1()
    test_2d_dim0()
    test_3d_last_dim()
    test_3d_middle_dim()
    test_dtypes()
    test_large_logits()
    test_negative_logits()
    test_uniform()
    test_vs_log_softmax()
    test_module()


if __name__ == "__main__":
    main()
