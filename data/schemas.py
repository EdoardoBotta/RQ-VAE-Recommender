from typing import NamedTuple
from torch import Tensor

FUT_SUFFIX = "_fut"


class SeqBatch(NamedTuple):
    user_ids: Tensor
    ids: Tensor
    ids_fut: Tensor
    x: Tensor
    x_fut: Tensor
    seq_mask: Tensor
