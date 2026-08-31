# Reproducibility and verification

## Fast verification

The automated test suite checks residual K-means, all three quantization
estimators, gradients, semantic-ID shapes and ranges, Gumbel-Softmax sampling,
ranking metrics, duplicate-column handling, and learning-rate scheduling. It
does not download datasets or pretrained models.

```bash
python -m pip install -e ".[test]"
pytest
```

GitHub Actions runs the same suite on Python 3.10 and 3.12. This is the fastest
way for reviewers to verify core numerical behavior on CPU.

## Reproducing a training workflow

The example gin files in `configs/` record model dimensions, optimizer values,
dataset selection, evaluation intervals, and output paths. Training remains
stochastic; for a research run, record at least:

- the repository version or commit;
- the complete gin configuration and any command-line environment settings;
- Python, PyTorch, accelerator, and package versions;
- hardware type and number of devices;
- downloaded dataset version and preprocessing date;
- random seeds used by Python, NumPy, PyTorch, and data-loader workers;
- generated checkpoints and evaluation logs.

The RQ-VAE and retrieval model are deliberately trained in separate stages.
Retain the exact RQ-VAE checkpoint used to construct semantic IDs when reporting
retrieval results, because a different tokenizer changes the target vocabulary.
Generate corpus IDs with the model in evaluation mode. Items close to a
codebook decision boundary can still receive different IDs when hardware,
PyTorch kernels, precision, or the checkpoint changes; archive the generated ID
table when exact downstream reproduction matters.

Full example configurations target research-scale datasets and are not CI smoke
tests. Their runtime and memory needs depend strongly on the dataset, embedding
model, batch size, and accelerator. Weights & Biases logging is optional and can
be disabled with `train.wandb_logging=False` in a gin configuration.

## Public artifacts

- The source release is archived at
  [Zenodo](https://doi.org/10.5281/zenodo.22076163).
- A trained Amazon Beauty RQ-VAE checkpoint is available from
  [Hugging Face](https://huggingface.co/edobotta/rqvae-amazon-beauty).
- Additional repository checkpoints are examples, not substitutes for reporting
  the configuration and software version used in a new experiment.
