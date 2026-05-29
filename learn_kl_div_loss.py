"""Learn torch.nn.KLDivLoss: KL(target || input) with input = log-probabilities.

Run: uv run python learn_kl_div_loss.py
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _header(title: str) -> None:
    print(f"\n=== {title} ===")


def _example_pair() -> tuple[torch.Tensor, torch.Tensor]:
    """Model log-probs and target probability distribution over 3 classes."""
    target = torch.tensor([0.7, 0.2, 0.1])
    logits = torch.tensor([0.0, 1.0, 2.0])
    log_input = F.log_softmax(logits, dim=0)
    return log_input, target


def test_setup() -> tuple[torch.Tensor, torch.Tensor]:
    """input = log-probabilities; target = probabilities (sum to 1)."""
    _header("setup: log-probs (input) vs probs (target)")
    log_input, target = _example_pair()
    print("log_input (model):", log_input)  # tensor([-2.4076, -1.4076, -0.4076])
    print("target  (true):  ", target)      # tensor([0.7000, 0.2000, 0.1000])
    return log_input, target


def test_reduction_batchmean(log_input: torch.Tensor, target: torch.Tensor) -> None:
    """Default: sum over classes, mean over batch (matches KL definition)."""
    _header("reduction='batchmean' (default)")
    loss_fn = nn.KLDivLoss(reduction="batchmean")
    loss = loss_fn(log_input.unsqueeze(0), target.unsqueeze(0))
    print("loss:", f"{loss.item():.4f}")  # loss: 1.2058
    print("formula per element: target * (log(target) - input); mean over batch")


def test_reduction_mean(log_input: torch.Tensor, target: torch.Tensor) -> None:
    """Mean over all N*C elements (differs from batchmean)."""
    _header("reduction='mean'")
    loss_fn = nn.KLDivLoss(reduction="mean")
    loss = loss_fn(log_input.unsqueeze(0), target.unsqueeze(0))
    print("loss:", f"{loss.item():.4f}")  # loss: 0.4019
    print("(mean over all N*C elements; differs from batchmean!)")


def test_reduction_sum(log_input: torch.Tensor, target: torch.Tensor) -> None:
    """Sum over all elements."""
    _header("reduction='sum'")
    loss_fn = nn.KLDivLoss(reduction="sum")
    loss = loss_fn(log_input.unsqueeze(0), target.unsqueeze(0))
    print("loss:", f"{loss.item():.4f}")  # loss: 1.2058


def test_reduction_none(log_input: torch.Tensor, target: torch.Tensor) -> None:
    """Per-element loss, shape preserved."""
    _header("reduction='none'")
    loss_fn = nn.KLDivLoss(reduction="none")
    out = loss_fn(log_input.unsqueeze(0), target.unsqueeze(0))
    print(out)  # tensor([[ 1.4357, -0.0404, -0.1895]])
    print("shape:", out.shape)  # torch.Size([1, 3])


def test_log_target(log_input: torch.Tensor, target: torch.Tensor) -> None:
    """Pass log(target) instead of target when log_target=True."""
    _header("log_target=True (both sides in log space)")
    log_target = torch.log(target)
    loss_fn = nn.KLDivLoss(reduction="batchmean", log_target=True)
    loss = loss_fn(log_input.unsqueeze(0), log_target.unsqueeze(0))
    print("loss (batchmean):", f"{loss.item():.4f}")  # loss (batchmean): 1.2058


def test_2d_batch() -> None:
    """Batched input: shape (N, C)."""
    _header("2D batch: N=2, C=3")
    target = torch.tensor([[0.7, 0.2, 0.1], [0.1, 0.7, 0.2]])
    logits = torch.tensor([[0.0, 1.0, 2.0], [2.0, 0.0, 1.0]])
    log_input = F.log_softmax(logits, dim=1)
    print("log_input:\n", log_input)
    # tensor([[-2.4076, -1.4076, -0.4076],
    #         [-0.4076, -2.4076, -1.4076]])
    print("target:\n", target)
    # tensor([[0.7000, 0.2000, 0.1000],
    #         [0.1000, 0.7000, 0.2000]])
    batchmean = nn.KLDivLoss(reduction="batchmean")(log_input, target)
    none = nn.KLDivLoss(reduction="none")(log_input, target)
    print("batchmean:", f"{batchmean.item():.4f}")  # batchmean: 1.2058
    print("per-sample (none):\n", none)
    # tensor([[ 1.4357, -0.0404, -0.1895],
    #         [-0.1895,  1.4357, -0.0404]])


def test_manual_kl(log_input: torch.Tensor, target: torch.Tensor) -> None:
    """KL(target || input) = sum target * (log(target) - log_input)."""
    _header("manual KL vs KLDivLoss")
    manual = (target * (torch.log(target) - log_input)).sum()
    batchmean = manual / 1  # batch size 1
    loss = nn.KLDivLoss(reduction="batchmean")(
        log_input.unsqueeze(0), target.unsqueeze(0)
    )
    print("manual batchmean:", f"{batchmean.item():.4f}")  # manual batchmean: 1.2058
    print("nn loss:          ", f"{loss.item():.4f}")       # nn loss: 1.2058


def test_target_is_probs() -> None:
    """With log_target=False, target must be a valid distribution."""
    _header("target must be probabilities (unless log_target)")
    target = torch.tensor([0.7, 0.2, 0.1])
    print("with log_target=False, target sums per row should be ~1")
    print("row sum:", target.sum().item())  # row sum: 1.0


def test_element_contrib(log_input: torch.Tensor, target: torch.Tensor) -> None:
    """Each element contributes target * (log(target) - log_input)."""
    _header("element-wise contribution (reduction='none')")
    contrib = target * (torch.log(target) - log_input)
    print("contrib = target * (log(target) - log_input)")
    print(contrib.unsqueeze(0))  # tensor([[ 1.4357, -0.0404, -0.1895]])


def test_with_log_softmax(log_input: torch.Tensor, target: torch.Tensor) -> None:
    """Typical pipeline: logits -> LogSoftmax -> KLDivLoss."""
    _header("pairing with LogSoftmax (typical training)")
    logits = torch.tensor([[0.0, 1.0, 2.0]])
    log_probs = F.log_softmax(logits, dim=1)
    loss = nn.KLDivLoss(reduction="batchmean")(log_probs, target.unsqueeze(0))
    print("logits -> LogSoftmax -> KLDivLoss vs softmax target")
    print("loss:", f"{loss.item():.4f}")  # loss: 1.2058


def main() -> None:
    log_input, target = test_setup()
    test_reduction_batchmean(log_input, target)
    test_reduction_mean(log_input, target)
    test_reduction_sum(log_input, target)
    test_reduction_none(log_input, target)
    test_log_target(log_input, target)
    test_2d_batch()
    test_manual_kl(log_input, target)
    test_target_is_probs()
    test_element_contrib(log_input, target)
    test_with_log_softmax(log_input, target)


if __name__ == "__main__":
    main()
