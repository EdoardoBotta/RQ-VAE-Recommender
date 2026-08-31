import random

import pytest
import torch

from data.processed import SeqData


@pytest.mark.parametrize("history_as_tensor", [False, True])
def test_subsampled_sequence_normalizes_storage_and_ignores_padding(
    history_as_tensor,
):
    dataset = SeqData.__new__(SeqData)
    history = [1, 2, 3, 4, -1]
    dataset.sequence_data = {
        "userId": torch.tensor([[7]]),
        "itemId": torch.tensor([history]) if history_as_tensor else [history],
        "itemId_fut": torch.tensor([[5]]),
    }
    dataset.subsample = True
    dataset._max_seq_len = 5
    dataset.item_data = torch.arange(6 * 768, dtype=torch.float32).reshape(6, 768)
    random.seed(0)

    batch = dataset[0]

    assert batch.ids.shape == (5,)
    assert batch.ids_fut.shape == (1,)
    assert batch.ids_fut.item() >= 0
    assert batch.seq_mask.sum() >= 2
    assert torch.all(batch.ids[batch.seq_mask] >= 0)
    assert torch.all(batch.x[~batch.seq_mask] == -1)


def test_subsampled_sequence_requires_enough_valid_items():
    dataset = SeqData.__new__(SeqData)
    dataset.sequence_data = {
        "userId": torch.tensor([[7]]),
        "itemId": [[1, -1]],
        "itemId_fut": torch.tensor([[2]]),
    }
    dataset.subsample = True
    dataset._max_seq_len = 5
    dataset.item_data = torch.zeros(3, 768)

    with pytest.raises(ValueError, match="at least three valid items"):
        dataset[0]
