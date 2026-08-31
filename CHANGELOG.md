# Changelog

Notable user-visible changes are documented here. This project follows
[Semantic Versioning](https://semver.org/).

## Unreleased

- Add standards-based Python packaging and command-line entry points.
- Add automated tests, continuous integration, and JOSS paper validation.
- Expand installation, usage, API, contribution, support, and citation guidance.
- Preserve encoder gradients through every residual-quantization level and add
  a regression test for the multi-level STE path.
- Normalize list/tensor sequence storage during training subsampling and reject
  invalid corpus-cache IDs with actionable errors.
- Document supported scope, collision behavior, and numerical limitations.

## 1.0.1 - 2026-08-24

- Archive the `v1_0_1` source release on Zenodo.
- Include RQ-VAE checkpoints for the documented Amazon and MovieLens workflows.
- Add NDCG evaluation and codebook-collapse diagnostics.

## 1.0.0 - 2026-06-30

- First tagged release of the two-stage semantic-ID tokenization and generative
  retrieval workflow.
