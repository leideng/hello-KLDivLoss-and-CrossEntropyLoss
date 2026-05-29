"""Learn torch.nn.CrossEntropyLoss: NLLLoss(LogSoftmax(logits), target).

Run: uv run python learn_cross_entropy_loss.py
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _header(title: str) -> None:
    print(f"\n=== {title} ===")


def _logits_batch() -> tuple[torch.Tensor, torch.Tensor]:
    logits = torch.tensor([[2.0, 1.0, 0.0], [0.0, 1.0, 3.0]])
    target = torch.tensor([0, 2])
    return logits, target


def test_2d_class_indices() -> tuple[torch.Tensor, torch.Tensor]:
    """Most common: logits (N, C), target = class index per sample."""
    _header("class indices, 2D logits (N, C) — most common")
    logits, target = _logits_batch()
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(logits, target)
    print("logits:\n", logits)
    # tensor([[2., 1., 0.],
    #         [0., 1., 3.]])
    print("target (class idx):", target)  # tensor([0, 2])
    print("loss (default mean):", f"{loss.item():.4f}")  # loss (default mean): 0.2887
    return logits, target


def test_reduction_modes(logits: torch.Tensor, target: torch.Tensor) -> None:
    """reduction: 'none' | 'sum' | 'mean'."""
    _header("reduction modes")
    none = nn.CrossEntropyLoss(reduction="none")(logits, target)
    sum_ = nn.CrossEntropyLoss(reduction="sum")(logits, target)
    mean = nn.CrossEntropyLoss(reduction="mean")(logits, target)
    print("none: ", none)               # tensor([0.4076, 0.1698])
    print("sum:  ", f"{sum_.item():.4f}")   # sum: 0.5775
    print("mean: ", f"{mean.item():.4f}")  # mean: 0.2887


def test_per_sample(logits: torch.Tensor, target: torch.Tensor) -> None:
    """reduction='none' gives one loss per sample."""
    _header("per-sample loss (reduction='none')")
    per = nn.CrossEntropyLoss(reduction="none")(logits, target)
    for i, (t, l) in enumerate(zip(target.tolist(), per.tolist())):
        print(f"sample {i} (true class {t}): {l:.4f}")
        # sample 0 (true class 0): 0.4076
        # sample 1 (true class 2): 0.1698


def test_1d_logits() -> None:
    """Single sample: logits (C,), target is a scalar index."""
    _header("1D logits (single sample, C classes)")
    logits = torch.tensor([2.0, 1.0, 0.0])
    target = torch.tensor(0)
    loss = nn.CrossEntropyLoss()(logits, target)
    print("logits:", logits)   # tensor([2., 1., 0.])
    print("target:", target)   # tensor(0)
    print("loss:", f"{loss.item():.4f}")  # loss: 0.4076


def test_ignore_index() -> None:
    """Samples with target == ignore_index are excluded from the loss."""
    _header("ignore_index: skip label 1")
    logits = torch.tensor([[2.0, 1.0, 0.0], [0.0, 1.0, 3.0], [1.0, 2.0, 0.0]])
    target = torch.tensor([0, 1, 2])
    loss = nn.CrossEntropyLoss(ignore_index=1)(logits, target)
    print("target:", target)  # tensor([0, 1, 2])
    print("loss (mean over non-ignored):", f"{loss.item():.4f}")  # loss: 1.4076


def test_class_weights() -> None:
    """weight[c] scales the loss for class c."""
    _header("class weights: penalize rare classes more")
    logits, target = _logits_batch()
    weight = torch.tensor([1.0, 2.0, 0.5])
    loss = nn.CrossEntropyLoss(weight=weight)(logits, target)
    print("weight:", weight)  # tensor([1.0000, 2.0000, 0.5000])
    print("loss:", f"{loss.item():.4f}")  # loss: 0.3284


def test_label_smoothing(logits: torch.Tensor, target: torch.Tensor) -> None:
    """label_smoothing mixes hard labels with a uniform distribution."""
    _header("label_smoothing: soften hard targets")
    l0 = nn.CrossEntropyLoss(label_smoothing=0.1)(logits, target)
    l1 = nn.CrossEntropyLoss(label_smoothing=0.0)(logits, target)
    print("smoothing=0.1, loss:", f"{l0.item():.4f}")  # smoothing=0.1, loss: 0.4221
    print("smoothing=0.0, loss:", f"{l1.item():.4f}")  # smoothing=0.0, loss: 0.2887


def test_soft_targets() -> None:
    """Target can be a probability vector (N, C) instead of class indices."""
    _header("soft targets (probability vector per sample)")
    logits = torch.tensor([[2.0, 1.0, 0.0], [0.0, 1.0, 3.0]])
    target = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    loss = nn.CrossEntropyLoss()(logits, target)
    print("prob target shape:", target.shape)  # torch.Size([2, 3])
    print("loss:", f"{loss.item():.4f}")       # loss: 0.2887


def test_3d_sequence() -> None:
    """Sequence/segmentation: logits (N, C, L), target (N, L)."""
    _header("3D logits (N, C, L) for sequence / segmentation")
    N, C, L = 2, 3, 4
    torch.manual_seed(0)
    logits = torch.randn(N, C, L)
    target = torch.randint(0, C, (N, L))
    loss = nn.CrossEntropyLoss()(logits, target)
    print("logits shape:", logits.shape)   # torch.Size([2, 3, 4])
    print("target shape:", target.shape)   # torch.Size([2, 4])
    print("loss:", f"{loss.item():.4f}")   # loss: 1.5128


def test_manual(logits: torch.Tensor, target: torch.Tensor) -> None:
    """Cross-entropy per sample = -log_softmax[i, target[i]]."""
    _header("manual: -log_softmax[range(N), target]")
    log_probs = F.log_softmax(logits, dim=1)
    manual = -log_probs[torch.arange(logits.size(0)), target]
    print("manual mean:", f"{manual.mean().item():.4f}")  # manual mean: 0.2887
    print("nn loss:    ", f"{nn.CrossEntropyLoss()(logits, target).item():.4f}")  # nn loss: 0.2887


def test_equivalence_nll(logits: torch.Tensor, target: torch.Tensor) -> None:
    """CrossEntropyLoss = NLLLoss(LogSoftmax(logits))."""
    _header("equivalence: CrossEntropy = NLL(LogSoftmax(logits))")
    ce = nn.CrossEntropyLoss()(logits, target)
    nll = nn.NLLLoss()(F.log_softmax(logits, dim=1), target)
    print("CE loss: ", f"{ce.item():.4f}")   # CE loss: 0.2887
    print("NLL+LSM:", f"{nll.item():.4f}")  # NLL+LSM: 0.2887


def test_target_dtype() -> None:
    """Class-index targets must be torch.long (int64)."""
    _header("wrong target dtype raises (need Long for class indices)")
    target = torch.tensor([0, 2], dtype=torch.long)
    print("target dtype for indices:", target.dtype)  # torch.int64


def test_confident_prediction() -> None:
    """Correct class with large logit => very small loss."""
    _header("high confidence correct prediction => low loss")
    logits = torch.tensor([[0.0, 0.0, 10.0]])
    target = torch.tensor([2])
    loss = nn.CrossEntropyLoss()(logits, target)
    print("logits favor class 2 strongly, target=2, loss:", f"{loss.item():.4f}")  # loss: 0.0001


def main() -> None:
    logits, target = test_2d_class_indices()
    test_reduction_modes(logits, target)
    test_per_sample(logits, target)
    test_1d_logits()
    test_ignore_index()
    test_class_weights()
    test_label_smoothing(logits, target)
    test_soft_targets()
    test_3d_sequence()
    test_manual(logits, target)
    test_equivalence_nll(logits, target)
    test_target_dtype()
    test_confident_prediction()


if __name__ == "__main__":
    main()
