from typing import NamedTuple
from torch import Tensor


class SeqBatch(NamedTuple):
    user_ids: Tensor
    ids: Tensor
    x: Tensor
    seq_mask: Tensor
