# API overview

RQ-VAE Recommender exposes its modeling components as ordinary PyTorch modules.
Training scripts use the same classes documented here. Tensor dimensions use
`B` for batch size, `N` for sequence length, `D` for feature dimension, `H` for
the number of residual codebooks, and `K` for codebook size.

## `modules.rqvae.RqVae`

`RqVae` maps continuous item features to hierarchical semantic IDs and
reconstructs the original feature vectors.

```python
import torch

from modules.quantize import QuantizeForwardMode
from modules.rqvae import RqVae

model = RqVae(
    input_dim=768,
    embed_dim=32,
    hidden_dims=[512, 256, 128],
    codebook_size=256,
    codebook_kmeans_init=False,
    codebook_mode=QuantizeForwardMode.STE,
    n_layers=3,
    n_cat_features=0,
)
model.eval()

item_features = torch.randn(16, 768)
output = model.get_semantic_ids(item_features)
assert output.sem_ids.shape == (16, 3)
```

Constructor parameters of particular interest are:

- `input_dim`: item-feature width.
- `embed_dim`: latent and codeword width.
- `hidden_dims`: encoder hidden widths; the decoder mirrors them.
- `codebook_size`: number of entries in each residual codebook.
- `n_layers`: number of residual quantization levels and semantic-ID tokens.
- `codebook_mode`: gradient estimator used while training.
- `codebook_kmeans_init`: whether the first training batch initializes each
  codebook with K-means.
- `n_cat_features`: number of categorical features at the end of the input
  vector; use `0` for continuous-only features.

`get_semantic_ids(x, gumbel_t=0.001)` returns `RqVaeOutput` with:

| Field | Shape | Meaning |
| --- | --- | --- |
| `embeddings` | `[B, embed_dim, H]` | Selected codeword at each level |
| `residuals` | `[B, embed_dim, H]` | Residual presented to each level |
| `sem_ids` | `[B, H]` | Discrete codeword indices |
| `quantize_loss` | `[B]` | Per-item residual quantization loss |

`forward(batch, gumbel_t)` accepts a `data.schemas.SeqBatch` and returns the
combined training loss, reconstruction loss, quantization loss, embedding
norms, and semantic-ID diversity statistic. `load_pretrained(path)` loads the
model state from a checkpoint written by `train_rqvae.py`.

## `modules.quantize.Quantize`

`Quantize` implements one codebook level. Its `forward(x, temperature)` method
accepts `[B, embed_dim]` inputs and returns selected embeddings, integer IDs,
and the quantization loss. `QuantizeForwardMode` provides three training
estimators:

- `GUMBEL_SOFTMAX`: differentiable relaxed categorical sampling.
- `STE`: nearest-codeword selection with a straight-through estimator.
- `ROTATION_TRICK`: nearest-codeword selection with the rotation-based gradient
  transformation.

`QuantizeDistance` selects squared Euclidean or cosine assignment distance.

## `modules.tokenizer.semids.SemanticIdTokenizer`

`SemanticIdTokenizer` combines an `RqVae` with the sequence-data interface.
Call `precompute_corpus_ids(item_dataset)` before tokenizing interaction
sequences. It computes `[num_items, H + 1]` cached IDs; the extra column is a
deterministic within-corpus disambiguation index for items whose first `H`
semantic tokens collide.

Calling the tokenizer with a `SeqBatch` returns `TokenizedSeqBatch`, containing
flattened semantic-ID histories, next-item IDs, masks, and token-type IDs. The
tokenizer temporarily switches its RQ-VAE to evaluation mode and does not track
gradients.

## `modules.model.EncoderDecoderRetrievalModel`

This model predicts a next item's semantic-ID tuple from a user's semantic-ID
history. Its constructor accepts the precomputed corpus codebook and T5 model
dimensions. `forward(tokenized_batch)` returns the summed cross-entropy loss
across semantic-ID levels. `generate_next_sem_id(tokenized_batch)` returns
`GenerationOutput`:

| Field | Shape | Meaning |
| --- | --- | --- |
| `sem_ids` | `[B, top_k, H]` | Ranked candidate semantic-ID tuples |
| `log_probas` | `[B, top_k]` | Cumulative candidate log probabilities |

Generation rejects partial ID prefixes that do not occur in the precomputed
corpus codebook. The model optionally inserts a learned separator between item
IDs and can prepend a hashed user embedding.

## Data interfaces

- `data.processed.ItemData` returns item-feature batches for RQ-VAE training.
- `data.processed.SeqData` returns user histories and next-item targets.
- `data.processed.RecDataset` identifies the Amazon Reviews, MovieLens 1M, and
  MovieLens 32M preprocessing backends.
- `data.schemas.SeqBatch` and `TokenizedSeqBatch` define the fields passed
  between the dataset, tokenizer, and models.

The data backends download their public source data on first use and cache
processed tensors below the configured dataset root.

## Training entry points

The installed commands are equivalent to the repository scripts:

```text
rqvae-train CONFIG_PATH
rqvae-train-decoder CONFIG_PATH
```

Both parse a gin configuration file before calling their `train` function.
Every training argument can therefore be set in a checked-in configuration
without modifying Python code.
