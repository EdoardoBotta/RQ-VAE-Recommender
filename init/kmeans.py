import numpy as np
import torch

from typing import NamedTuple


def kmeans_init_(tensor: torch.Tensor, x: torch.Tensor):
    assert tensor.dim() == 2
    assert x.dim() == 2

    with torch.no_grad():
        k, _ = tensor.shape
        # TODO: Check out scipy kmeans2
        kmeans_out = Kmeans(k=k).run(x)
        tensor[:, :] = kmeans_out.centroids[:, :]


class KmeansOutput(NamedTuple):
    centroids: torch.Tensor
    assignment: torch.Tensor


class Kmeans:
    def __init__(self,
                 k: int,
                 max_iters: int = None,
                 stop_threshold: float = 1e-2) -> None:
        self.k = k
        self.iters = max_iters
        self.stop_threshold = stop_threshold
        self.centroids = None
        self.assignment = None

    def _init_centroids(self, x: torch.Tensor) -> None:
        B, D = x.shape
        init_idx = np.random.choice(B, self.k, replace=False)
        self.centroids = x[init_idx, :]
        self.assignment = None

    def _update_centroids(self, x) -> torch.Tensor:
        squared_pw_dist = (x.unsqueeze(1) - self.centroids.unsqueeze(0))**2
        centroid_idx = (squared_pw_dist.sum(axis=2)).min(axis=1).indices
        assigned = (
            torch.arange(self.k, device=x.device).unsqueeze(1) == centroid_idx
        )
        for cluster in range(self.k):
            is_assigned_to_c = assigned[cluster]
            if not is_assigned_to_c.any():
                continue
            self.centroids[cluster, :] = x[is_assigned_to_c, :].mean(axis=0)
        self.assignment = centroid_idx

    def run(self, x):
        self._init_centroids(x)

        i = 0
        while self.iters is None or i < self.iters:
            old_c = self.centroids.clone()
            self._update_centroids(x)
            if torch.norm(self.centroids - old_c) < self.stop_threshold:
                break
            i += 1

        return KmeansOutput(
            centroids=self.centroids,
            assignment=self.assignment
        )
