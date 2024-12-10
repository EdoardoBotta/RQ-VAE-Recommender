from torch import nn
from torch import Tensor


class ReconstructionLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x_hat, x) -> Tensor:
        return ((x_hat - x)**2).sum(axis=-1)


class CategoricalReconstuctionLoss(nn.Module):
    def __init__(self, n_cat_feats) -> None:
        super().__init__()
        self.reconstruction_loss = ReconstructionLoss()
        self.n_cat_feats = n_cat_feats
    
    def forward(self, x_hat, x) -> Tensor:
        cont_reconstr = self.reconstruction_loss(
            x_hat[:, :-self.n_cat_feats],
            x[:, :-self.n_cat_feats]
        )
        cat_reconstr = nn.functional.binary_cross_entropy_with_logits(
            x_hat[:, -self.n_cat_feats:],
            x[:, -self.n_cat_feats:],
            reduction='none'
        ).sum(axis=-1)
        return cont_reconstr + cat_reconstr


class RqVaeLoss(nn.Module):
    def __init__(self, commitment_weight: float = 1.0) -> None:
        super().__init__()
        self.commitment_weight = commitment_weight

    def forward(self, query, value) -> Tensor:
        emb_loss = ((query.detach() - value)**2).sum(axis=[-1, -2])
        query_loss = ((query - value.detach())**2).sum(axis=[-1, -2])
        return emb_loss + self.commitment_weight * query_loss

