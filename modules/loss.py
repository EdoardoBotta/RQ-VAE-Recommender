from torch import nn


class ReconstructionLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x_hat, x):
        return ((x_hat - x)**2).sum(axis=-1)


class RqVaeLoss(nn.Module):
    def __init__(self, commitment_weight: float = 1.0) -> None:
        super().__init__()
        self.commitment_weight = commitment_weight

    def forward(self, query, value):
        emb_loss = ((query.detach() - value)**2).sum(axis=[-1, -2])
        query_loss = ((query - value.detach())**2).sum(axis=[-1, -2])
        return emb_loss + self.commitment_weight * query_loss

