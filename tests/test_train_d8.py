import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from graph_retriever.dataset import Sample
from graph_retriever.train import dispersion_penalty, future_targets, multi_target_loss


def _prep(sessions):
    train = [Sample(prefix=s[:i], target=s[i]) for s in sessions for i in range(1, len(s))]
    return SimpleNamespace(train_sessions=sessions, train=train)


def test_future_targets_aligned_with_samples():
    prep = _prep([[0, 1, 2, 3], [4, 5]])
    assert future_targets(prep, 2) == [[2, 3], [3], [], []]


def test_future_targets_detects_misalignment():
    prep = _prep([[0, 1, 2]])
    prep.train = list(reversed(prep.train))
    with pytest.raises(RuntimeError):
        future_targets(prep, 2)


def test_multi_target_loss_reduces_to_ce_without_futures():
    logits = torch.randn(3, 5)
    target = torch.tensor([0, 2, 4])
    got = multi_target_loss(logits, target, [[], [], []], 0.5)
    assert got.item() == pytest.approx(F.cross_entropy(logits, target).item(), rel=1e-6)


def test_multi_target_loss_weights():
    logits = torch.zeros(1, 4)  # uniform: every NLL = log 4
    logits[0, 1] = 1.0
    logp = F.log_softmax(logits, dim=1)[0]
    got = multi_target_loss(logits, torch.tensor([1]), [[2, 3]], 0.5)
    want = -(logp[1] + 0.5 * logp[2] + 0.25 * logp[3]) / 1.75
    assert got.item() == pytest.approx(want.item(), rel=1e-6)


def test_dispersion_penalty_bounds():
    assert dispersion_penalty(torch.zeros(4, 10)).item() == pytest.approx(1.0)
    peaked = torch.full((4, 10), -1e4)
    peaked[:, 3] = 0.0
    assert dispersion_penalty(peaked).item() == pytest.approx(10.0)
    assert not math.isnan(dispersion_penalty(torch.randn(8, 10)).item())
