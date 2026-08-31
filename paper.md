---
title: "RQ-VAE Recommender: Semantic-ID tokenization and generative retrieval in PyTorch"
tags:
  - Python
  - recommender systems
  - generative retrieval
  - semantic IDs
  - vector quantization
authors:
  - name: Edoardo Botta
    affiliation: 1
    orcid: 0009-0003-6246-8477
affiliations:
  - name: Independent Researcher
    index: 1
date: 30 August 2026
bibliography: paper.bib
---

# Summary

Recommender systems help people navigate large catalogs of products, media, and
other items. Conventional retrieval models score catalog items, whereas a
generative retriever predicts an item's identifier as a sequence of tokens.
`RQ-VAE Recommender` is a PyTorch research implementation of this approach. It
first learns short, hierarchical *semantic IDs* from item descriptions with a
residual-quantized variational autoencoder (RQ-VAE). A Transformer then learns
to generate the semantic ID of the next item from a user's interaction history.
The repository connects data preparation, semantic tokenization, collision
handling, constrained generation, and ranking evaluation in a reproducible
two-stage workflow.

# Statement of need

Semantic IDs replace arbitrary catalog keys with compact codes learned from
item features. They can reduce output-vocabulary growth and let related or
previously unseen items share statistical structure. The TIGER framework showed
that combining residual semantic IDs with sequence-to-sequence prediction can
improve retrieval and cold-item generalization [@rajput2023tiger]. Reproducing
that research idea, however, requires coordinated implementations of dataset
preprocessing, residual quantization, collision disambiguation, sequential
training, corpus-constrained decoding, and top-$K$ evaluation.

`RQ-VAE Recommender` targets researchers and practitioners studying generative
recommendation, learned discrete item representations, vector-quantization
optimization, and semantic-ID collisions. Its purpose is reproducible training,
not inference from pretrained checkpoints. Executable gin configurations and
end-to-end PyTorch loops let users locally reproduce data preparation,
semantic-ID and retriever training, and ranking evaluation for the supported
Amazon and MovieLens workflows; optional experiment tracking and CPU tests
support this process.

# State of the field

Residual quantization was introduced for hierarchical discrete image
representations by Lee et al. [@lee2022rqvae], while TIGER applied RQ-VAE IDs to
generative recommendation [@rajput2023tiger]. General-purpose libraries such as
`vector-quantize-pytorch` [@lucidrains2020vectorquantize] offer a broad catalog
of quantization layers, but do not define the recommendation data, tokenizer,
collision, and constrained next-item retrieval workflow implemented here.

Broader generative-recommendation systems have since appeared. GRID combines
multiple semantic-ID learners with TIGER-style generation [@ju2025grid], and
GenRec is a model zoo spanning conventional and generative recommenders
[@lu2025genrec]. Those projects serve comparative and large-framework use
cases. `RQ-VAE Recommender` predates them and occupies a narrower niche: a
compact end-to-end PyTorch implementation centered on the RQ-VAE-to-retriever
interface, with three interchangeable quantization gradient estimators and
minimal framework abstraction. Building a standalone implementation made the
full pipeline available when the project began and remains useful for focused
method development, teaching, and baselines; GenRec explicitly identifies this
repository as related prior software.

# Software design

The architecture separates semantic-ID learning from next-item generation. In
the first stage, an MLP encoder maps item features to a latent vector. A stack of
codebooks successively quantizes its residual; codeword indices form a
coarse-to-fine ID and the summed codewords are decoded back to the input.
Researchers can choose K-means initialization, Euclidean or cosine assignment,
and Gumbel-Softmax [@jang2017gumbel], straight-through, or rotation-trick
[@fifty2024rotation] gradients. This separation makes codebook behavior and ID
diversity directly inspectable, and lets one tokenizer checkpoint be reused
across retrieval experiments, at the cost of not jointly optimizing both
stages.

A tokenizer caches IDs for the complete corpus. Because a finite product of
codebooks can assign the same tuple to multiple items, it appends a deterministic
disambiguation value. The retrieval model embeds the semantic tokens in a shared
table and uses a T5-style encoder and autoregressive decoder
[@raffel2020t5]. During generation, it masks partial ID sequences absent from
the corpus and keeps the highest-scoring candidates. This enforces catalog-valid
outputs while preserving the compact hierarchical vocabulary. Hit rate and
normalized discounted cumulative gain provide end-to-end ranking checks.

PyTorch [@paszke2019pytorch] supplies the numerical and accelerator interface,
while gin keeps experiment choices outside the training code. The intentionally
minimal design exposes ordinary PyTorch modules and configurations for ease of
use; separate components let researchers replace one stage without rewriting
the pipeline.

# Research impact statement

The project has been publicly developed since June 2024 and, as of August 2026,
has received more than 840 GitHub stars and 120 forks, with external issues and
pull requests documenting use and extension by the community. A pretrained
Amazon Beauty tokenizer is distributed through Hugging Face, and version 1.0.1
is preserved in a citable Zenodo archive [@botta2026rqvae].

More directly, Hu et al. based all public-dataset experiments for their QuaSID
research on the open-source `RQ-VAE Recommender` framework and linked this
repository in their implementation details [@hu2026quasid]. Their study used
the framework for Amazon Beauty and Toys baselines and for testing new
collision-aware semantic-ID objectives. This independent reuse demonstrates
that the software functions as an extensible research baseline rather than only
as code for a single analysis.

# AI usage disclosure

OpenAI Codex using GPT-5 (accessed in August 2026) was used to audit JOSS
requirements and assist with initial drafts of packaging metadata, automated
tests, repository documentation, and this manuscript. It did not make the core
research or software-design decisions. The author reviewed and validated the
AI-assisted code changes. The documentation and manuscript were checked against
the implementation, cited sources, and automated validation results.

# Acknowledgements

This work received no specific external funding.

# References
