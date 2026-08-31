import math

import pytest
import torch

from distributions.gumbel import gumbel_softmax_sample
from evaluate.metrics import TopKAccumulator
from modules.model import _strip_dedup_col
from modules.scheduler.inv_sqrt import InverseSquareRootScheduler


def test_gumbel_softmax_samples_are_probabilities():
    torch.manual_seed(11)
    sample = gumbel_softmax_sample(
        logits=torch.zeros(4, 6),
        temperature=0.7,
        device=torch.device("cpu"),
    )

    assert sample.shape == (4, 6)
    assert torch.isfinite(sample).all()
    assert torch.all(sample >= 0)
    assert torch.allclose(sample.sum(dim=-1), torch.ones(4))


def test_top_k_accumulator_computes_hit_rate_and_ndcg():
    accumulator = TopKAccumulator(ks=[1, 2])
    actual = torch.tensor([[1, 2, 3], [7, 8, 9]])
    predictions = torch.tensor(
        [
            [[1, 2, 3], [4, 5, 6]],
            [[0, 0, 0], [7, 8, 9]],
        ]
    )

    accumulator.accumulate(actual=actual, top_k=predictions)
    metrics = accumulator.reduce()

    assert metrics["h@1"] == pytest.approx(0.5)
    assert metrics["h@2"] == pytest.approx(1.0)
    assert metrics["ndcg"] == pytest.approx((1 + 1 / math.log2(3)) / 2)


def test_strip_dedup_column_preserves_hierarchical_tokens():
    ids = torch.tensor([[10, 11, 12, 99, 20, 21, 22, 98]])

    stripped = _strip_dedup_col(ids, sem_ids_dim=4, n_layers=3)

    assert torch.equal(stripped, torch.tensor([[10, 11, 12, 20, 21, 22]]))


def test_inverse_square_root_scheduler_warms_then_decays():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=1.0)
    scheduler = InverseSquareRootScheduler(optimizer, warmup_steps=2)

    initial = scheduler.get_last_lr()[0]
    optimizer.step()
    scheduler.step()
    warmup = scheduler.get_last_lr()[0]
    optimizer.step()
    scheduler.step()
    decayed = scheduler.get_last_lr()[0]

    assert initial == pytest.approx(1.0)
    assert warmup == pytest.approx(1.0)
    assert decayed < warmup
