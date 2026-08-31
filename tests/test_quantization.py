import numpy as np
import pytest
import torch

from init.kmeans import Kmeans
from modules.quantize import Quantize
from modules.quantize import QuantizeForwardMode
from modules.rqvae import RqVae


@pytest.mark.parametrize(
    "mode",
    [
        QuantizeForwardMode.GUMBEL_SOFTMAX,
        QuantizeForwardMode.STE,
        QuantizeForwardMode.ROTATION_TRICK,
    ],
)
def test_quantize_modes_produce_valid_outputs_and_gradients(mode):
    torch.manual_seed(7)
    layer = Quantize(
        embed_dim=4,
        n_embed=8,
        do_kmeans_init=False,
        forward_mode=mode,
    )
    x = torch.randn(6, 4, requires_grad=True)

    output = layer(x, temperature=0.5)

    assert output.embeddings.shape == x.shape
    assert output.ids.shape == (6,)
    assert output.loss.shape == (6,)
    assert torch.all((0 <= output.ids) & (output.ids < 8))
    assert torch.isfinite(output.embeddings).all()
    assert torch.isfinite(output.loss).all()

    (output.embeddings.square().mean() + output.loss.mean()).backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert layer.embedding.weight.grad is not None
    assert torch.isfinite(layer.embedding.weight.grad).all()


def test_kmeans_separates_two_compact_clusters():
    np.random.seed(2)
    x = torch.tensor(
        [
            [-5.1, -5.0],
            [-4.9, -5.0],
            [-5.0, -4.9],
            [5.1, 5.0],
            [4.9, 5.0],
            [5.0, 4.9],
        ]
    )

    output = Kmeans(k=2, max_iters=50).run(x)
    centers = output.centroids[output.centroids[:, 0].argsort()]

    assert output.assignment.shape == (6,)
    assert torch.allclose(centers[0], torch.tensor([-5.0, -4.9667]), atol=0.05)
    assert torch.allclose(centers[1], torch.tensor([5.0, 4.9667]), atol=0.05)


def test_rqvae_emits_one_semantic_id_per_residual_level():
    torch.manual_seed(3)
    model = RqVae(
        input_dim=6,
        embed_dim=4,
        hidden_dims=[8],
        codebook_size=7,
        codebook_kmeans_init=False,
        codebook_mode=QuantizeForwardMode.STE,
        n_layers=3,
        n_cat_features=0,
    )
    model.eval()

    output = model.get_semantic_ids(torch.randn(5, 6))

    assert output.embeddings.shape == (5, 4, 3)
    assert output.residuals.shape == (5, 4, 3)
    assert output.sem_ids.shape == (5, 3)
    assert output.quantize_loss.shape == (5,)
    assert torch.all((0 <= output.sem_ids) & (output.sem_ids < 7))


def test_later_ste_commitment_losses_reach_the_encoder():
    torch.manual_seed(5)
    model = RqVae(
        input_dim=4,
        embed_dim=3,
        hidden_dims=[6],
        codebook_size=5,
        codebook_kmeans_init=False,
        codebook_mode=QuantizeForwardMode.STE,
        n_layers=2,
        n_cat_features=0,
    )
    model.train()
    # Isolate the second level's encoder-side commitment gradient.
    model.layers[0].quantize_loss.commitment_weight = 0.0
    items = torch.randn(7, 4, requires_grad=True)

    model.get_semantic_ids(items).quantize_loss.mean().backward()

    assert items.grad is not None
    assert torch.isfinite(items.grad).all()
    assert items.grad.norm() > 0
