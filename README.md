# RQ-VAE Recommender

[![Tests](https://github.com/EdoardoBotta/RQ-VAE-Recommender/actions/workflows/tests.yml/badge.svg)](https://github.com/EdoardoBotta/RQ-VAE-Recommender/actions/workflows/tests.yml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22076162.svg)](https://doi.org/10.5281/zenodo.22076162)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

RQ-VAE Recommender is a PyTorch implementation of semantic-ID tokenization and
generative retrieval for recommender-system research. It implements the
two-stage workflow introduced by *Recommender Systems with Generative
Retrieval*:

1. A residual-quantized variational autoencoder (RQ-VAE) maps each catalog item
   to a short tuple of discrete semantic IDs.
2. A Transformer consumes a user's semantic-ID history and generates the IDs of
   likely next items.

![RQ-VAE semantic-ID tokenization followed by generative retrieval](https://github.com/EdoardoBotta/RQ-VAE/assets/64335373/199b38ac-a282-4ba1-bd89-3291617e6aa5)

## Statement of need

Generative recommendation represents catalog items as tokens and retrieves an
item by generating its identifier. Research on this approach requires more than
an RQ-VAE layer: researchers must prepare sequential datasets, learn stable
hierarchical identifiers, handle identifier collisions, train a constrained
sequence model, and evaluate ranked outputs. RQ-VAE Recommender provides these
parts in one inspectable PyTorch workflow. It is intended for recommender-system
researchers and practitioners studying semantic tokenization, vector
quantization, cold-item representations, and generative retrieval.

The project prioritizes a compact implementation of the complete experimental
pipeline. General-purpose vector-quantization libraries provide reusable neural
layers, while broader recommendation frameworks cover many unrelated model
families. This repository focuses on the interface between residual semantic-ID
learning and next-item generation, with automatic preprocessing for established
recommendation datasets and gin-based experiment configurations.

## Capabilities

- Residual quantization with K-means initialization and Euclidean or cosine
  assignment.
- Gumbel-Softmax, straight-through, and rotation-trick gradient estimators.
- Semantic-ID collision disambiguation and corpus-prefix-constrained generation.
- A T5 encoder-decoder retrieval model with hit-rate and NDCG evaluation.
- Automatic preprocessing for Amazon Reviews and MovieLens datasets.
- Optional multi-device/mixed-precision training through Accelerate and optional
  Weights & Biases logging.
- A published Amazon Beauty RQ-VAE checkpoint on
  [Hugging Face](https://huggingface.co/edobotta/rqvae-amazon-beauty).

Dataset support differs by stage:

| Dataset | RQ-VAE tokenization | Retrieval-model training |
| --- | :---: | :---: |
| Amazon Reviews: Beauty, Sports, Toys | Yes | Yes |
| MovieLens 1M | Yes | Not currently exposed |
| MovieLens 32M | Yes | Not currently exposed |

## Installation

Python 3.10 or newer is required. A CUDA or MPS accelerator is recommended for
full training runs but is not needed for the unit tests.

```bash
git clone https://github.com/EdoardoBotta/RQ-VAE-Recommender.git
cd RQ-VAE-Recommender
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

The `requirements.txt` file remains available for environments that use a
requirements-based workflow. To install development and test tools, use:

```bash
python -m pip install -e ".[test]"
pytest
```

The unit tests run on CPU without downloading datasets or model weights. The
training data are downloaded automatically on first use and cached under the
configured dataset directory.

## Quick API example

The following example verifies semantic-ID generation with a small untrained
model:

```python
import torch

from modules.quantize import QuantizeForwardMode
from modules.rqvae import RqVae

model = RqVae(
    input_dim=8,
    embed_dim=4,
    hidden_dims=[16, 8],
    codebook_size=16,
    codebook_kmeans_init=False,
    codebook_mode=QuantizeForwardMode.STE,
    n_layers=3,
    n_cat_features=0,
)
model.eval()

items = torch.randn(5, 8)
semantic_ids = model.get_semantic_ids(items).sem_ids
assert semantic_ids.shape == (5, 3)
```

See the [API overview](docs/api.md) for the model, tokenizer, data, and tensor
interfaces.

## Training

Training arguments are managed with
[gin-config](https://github.com/google/gin-config). The `train` functions in
`train_rqvae.py` and `train_decoder.py` are configurable, so model dimensions,
datasets, optimization settings, evaluation intervals, logging, and output paths
can be recorded in a `.gin` file.

### Amazon Reviews

Train the tokenizer, then train the retrieval model from the resulting
checkpoint:

```bash
rqvae-train configs/rqvae_amazon.gin
rqvae-train-decoder configs/decoder_amazon.gin
```

The equivalent source-tree commands are:

```bash
python train_rqvae.py configs/rqvae_amazon.gin
python train_decoder.py configs/decoder_amazon.gin
```

`configs/decoder_amazon.gin` currently points to the example Beauty checkpoint
in `trained_models/rqvae_amazon_beauty/`. Change `train.dataset_split` and
`train.pretrained_rqvae_path` together when using another Amazon split.

### MovieLens 32M tokenizer

The checked-in MovieLens configuration trains the semantic-ID tokenizer only:

```bash
rqvae-train configs/rqvae_ml32m.gin
```

Full configurations are research-scale workloads. Runtime and memory needs vary
with the dataset, batch size, text-embedding model, and accelerator. See
[Reproducibility and verification](docs/reproducibility.md) for what to record
and how reviewers can perform a fast objective check.

## Documentation and community

- [API overview](docs/api.md)
- [Reproducibility and verification](docs/reproducibility.md)
- [Contributing guide](CONTRIBUTING.md)
- [Support](SUPPORT.md)
- [Security policy](SECURITY.md)
- [Changelog](CHANGELOG.md)

Bug reports and feature proposals belong in the
[issue tracker](https://github.com/EdoardoBotta/RQ-VAE-Recommender/issues).

## Research use

The repository has been used as the open-source RQ-VAE recommender framework for
the public-dataset experiments in *Stop Treating Collisions Equally:
Qualification-Aware Semantic ID Learning for Recommendation at Industrial
Scale*. The source release is preserved on
[Zenodo](https://doi.org/10.5281/zenodo.22076163).

## Citation

If this software supports your research, cite the archived release:

```bibtex
@software{botta2026rqvae,
  author  = {Edoardo Botta},
  title   = {RQ-VAE Recommender},
  version = {1.0.1},
  year    = {2026},
  doi     = {10.5281/zenodo.22076163},
  url     = {https://doi.org/10.5281/zenodo.22076163}
}
```

Machine-readable citation metadata are available in [CITATION.cff](CITATION.cff).

## References

- Shashank Rajput et al., [*Recommender Systems with Generative
  Retrieval*](https://proceedings.neurips.cc/paper_files/paper/2023/hash/20dcab0f14046a5c6b02b61da9f13229-Abstract-Conference.html),
  Advances in Neural Information Processing Systems 36, 2023.
- Eric Jang, Shixiang Gu, and Ben Poole,
  [*Categorical Reparameterization with Gumbel-Softmax*](https://openreview.net/forum?id=rkE3y85ee),
  International Conference on Learning Representations, 2017.
- Christopher Fifty et al., [*Restructuring Vector Quantization with the
  Rotation Trick*](https://arxiv.org/abs/2410.06424), 2024.
- [vector-quantize-pytorch](https://github.com/lucidrains/vector-quantize-pytorch)
  and [deep-vector-quantization](https://github.com/karpathy/deep-vector-quantization)
  for related low-level implementations.

## License

RQ-VAE Recommender is distributed under the [MIT License](LICENSE).
