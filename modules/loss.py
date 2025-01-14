from einops import rearrange
from torch import nn
from torch import Tensor
from torch.nn import functional as F


def efficient_rotation_trick_transform(u, q, e):
    """
    4.2 in https://arxiv.org/abs/2410.06424
    """
    e = rearrange(e, 'b d -> b 1 d')
    w = F.normalize(u + q, p=2, dim=1, eps=1e-6).detach()

    return (
        e -
        2 * (e @ rearrange(w, 'b d -> b d 1') @ rearrange(w, 'b d -> b 1 d')) +
        2 * (e @ rearrange(u, 'b d -> b d 1').detach() @ rearrange(q, 'b d -> b 1 d').detach())
    ).squeeze()


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
        reconstr = self.reconstruction_loss(
            x_hat[:, :-self.n_cat_feats],
            x[:, :-self.n_cat_feats]
        )
        if self.n_cat_feats > 0:
            cat_reconstr = nn.functional.binary_cross_entropy_with_logits(
                x_hat[:, -self.n_cat_feats:],
                x[:, -self.n_cat_feats:],
                reduction='none'
            ).sum(axis=-1)
            reconstr += cat_reconstr
        return reconstr


class RqVaeLoss(nn.Module):
    def __init__(self, commitment_weight: float = 1.0) -> None:
        super().__init__()
        self.commitment_weight = commitment_weight

    def forward(self, query, value) -> Tensor:
        emb_loss = ((query.detach() - value)**2).sum(axis=[-1, -2])
        value = query + (value - query).detach()
        query_loss = ((query - value.detach())**2).sum(axis=[-1, -2])
        return emb_loss + self.commitment_weight * query_loss
