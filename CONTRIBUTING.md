# Contributing

Contributions that improve correctness, documentation, portability, dataset
support, or reproducibility are welcome. By participating, you agree to follow
the [Code of Conduct](CODE_OF_CONDUCT.md).

## Before opening a change

- Use the [issue tracker](https://github.com/EdoardoBotta/RQ-VAE-Recommender/issues)
  to report bugs, request features, or propose a substantial design change.
- Search existing issues and pull requests first.
- Do not post private data, access tokens, or proprietary datasets.
- For questions about using the software, follow [SUPPORT.md](SUPPORT.md).

Small fixes can go directly to a pull request. For changes to model behavior,
data preparation, or public interfaces, opening an issue first makes it easier
to agree on expected behavior and a verification plan.

## Development setup

RQ-VAE Recommender supports Python 3.10 and newer. From a clone of the
repository:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
pytest
```

The unit tests do not download datasets or pretrained weights. Full training
runs require substantially more compute and download the selected public
dataset on first use.

## Pull requests

Please:

1. Keep the change focused and explain its motivation.
2. Add or update tests for behavior that can be checked automatically.
3. Update the README or API documentation when an interface or workflow changes.
4. Run `pytest` locally and report any platform-specific checks you performed.
5. Avoid committing datasets, experiment logs, credentials, or generated model
   checkpoints.

Pull requests are reviewed for correctness, maintainability, documentation, and
compatibility with the existing training configurations. Research results and
performance claims should include enough configuration and evaluation detail to
be checked independently.

## Reporting security concerns

Do not open a public issue for a suspected vulnerability that could expose
credentials, private data, or systems. Follow the private reporting instructions
in [SECURITY.md](SECURITY.md).
