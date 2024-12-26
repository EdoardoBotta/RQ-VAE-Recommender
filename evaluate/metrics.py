from collections import defaultdict
from einops import rearrange
from torch import Tensor


class TopKAccumulator:
    def __init__(self, ks=[1, 5, 10]):
        self.ks = ks
        self.reset()

    def reset(self):
        self.total = 0
        self.metrics = defaultdict(int)

    def accumulate(self, actual: Tensor, top_k: Tensor) -> None:
        match_found, rank = (rearrange(actual, "b d -> b 1 d") == top_k).all(axis=-1).max(axis=-1)
        matched_rank = rank[match_found]
        self.total += len(rank)
        for k in self.ks:
            self.metrics[f"h@{k}"] += len(matched_rank[matched_rank < k])
        
    def reduce(self) -> dict:
        return {k: v/self.total for k, v in self.metrics.items()}
