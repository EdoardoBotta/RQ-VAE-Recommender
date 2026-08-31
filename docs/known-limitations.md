# Known limitations

The supported research scope is intentionally narrower than a general-purpose
recommendation framework.

- The RQ-VAE tokenizer supports the documented Amazon Reviews and MovieLens
  datasets. The checked-in retrieval-training command currently supports Amazon
  Reviews only and rejects other datasets explicitly.
- Full preprocessing and training are research-scale jobs. Initial preparation
  downloads data and computes text embeddings with Sentence-T5 XXL, so the
  complete examples are not suitable as CPU smoke tests.
- Semantic-ID assignment is a nearest-codeword operation. An item close to an
  assignment boundary can change ID across checkpoints, numerical precision,
  hardware, or PyTorch kernels. Use evaluation mode and archive the checkpoint,
  environment, and generated corpus-ID table when exact IDs matter.
- Codebook tuples are not guaranteed to be unique. The tokenizer appends a
  deterministic occurrence column when it builds the complete corpus cache;
  downstream retrieval code relies on that column to disambiguate collisions.
- Gumbel-Softmax and rotation-trick training are experimental gradient
  estimators. Their optimization behavior can differ materially from the
  straight-through mode and should be reported with the complete gin
  configuration.
- Custom datasets require adapting the preprocessing interface to emit the
  documented `SeqBatch` tensors. The project does not currently expose a
  schema-driven data-ingestion command for arbitrary production catalogs.

Reports that fall within the supported scope should be filed in the public
[issue tracker](https://github.com/EdoardoBotta/RQ-VAE-Recommender/issues) with
the smallest reproducible configuration and environment details.
