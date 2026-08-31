# JOSS submission checklist

This checklist tracks the current Journal of Open Source Software requirements
and the project-specific actions remaining before submission.

## Eligibility and pre-review gates

- [x] Research software with an obvious research application.
- [x] OSI-approved license in a plain-text `LICENSE` file (MIT).
- [x] Public, cloneable source repository with more than six months of iterative
  history (public since June 2024).
- [x] Demonstrated research use: Hu et al. (2026) used this repository as the
  basis of all public-dataset experiments in their QuaSID paper.
- [x] Community evidence: external issues and pull requests, more than 840 stars,
  and more than 120 forks as of August 2026.
- [ ] Repository owner: remove GitHub's current issue-creation restriction so
  any GitHub user can file an issue, as JOSS requires.
- [x] Correctness audit: fix the multi-level STE gradient path and sequence
  subsampling defects; add cache-boundary validation and regression tests.
- [ ] Repository owner: confirm that the software is feature-complete for the
  deliberately limited scope claimed in the paper.

## Software and open-source practice

- [x] Standards-based `pyproject.toml` and `pip install` workflow.
- [x] Dependency list includes all direct imports.
- [x] Automated CPU tests for core numerical behavior.
- [x] GitHub Actions test matrix for supported Python versions.
- [x] Installation, example usage, API, and reproducibility documentation.
- [x] Contribution, issue reporting, support, security, and conduct guidance.
- [x] Changelog and tagged source archive with DOI.
- [x] Open public pull request
  [#68](https://github.com/EdoardoBotta/RQ-VAE-Recommender/pull/68).
- [x] Confirm the pull-request test matrix passes on Python 3.10 and 3.12 and
  the official Open Journals draft action builds the paper.
- [x] Confirm installation and tests succeed in fresh GitHub-hosted Python 3.10
  and 3.12 environments; also build and import the wheel locally.
- [ ] Review and merge pull request #68 through the normal public workflow.
- [ ] Optional: ask a colleague to follow the documented fast verification
  steps independently and record any resulting fixes publicly.

## JOSS paper

- [x] `paper.md` and `paper.bib` use JOSS Markdown/YAML and citation syntax.
- [x] Draft is within the required 750--1750 word range.
- [x] Required sections: Summary, Statement of need, State of the field,
  Software design, Research impact statement, AI usage disclosure,
  Acknowledgements, and References.
- [x] State-of-field comparison and build-versus-contribute rationale.
- [x] Software archive and independent research reuse are cited.
- [x] Official Open Journals draft-PDF workflow.
- [x] Author: confirm that Edoardo Botta is the sole author.
- [x] Author: add ORCID identifier `0009-0003-6246-8477`.
- [x] Author: use the supplied spelling, Edoardo Botta, and list the affiliation
  as Independent Researcher.
- [x] Author: remove the AI-review placeholder after confirming review and
  validation of the AI-assisted code changes; document the checks applied to
  the documentation and manuscript.
- [x] Author: confirm that the work received no specific external funding.
- [ ] Author: confirm related papers or submissions and disclose them on the
  JOSS form; the JOSS paper must describe software rather than new results.
- [x] Author: confirm there are no financial, personal, or professional
  conflicts of interest to disclose in the submission form.
- [x] Download and visually inspect the three-page `paper.pdf`; citations,
  links, and page layout render cleanly. Before the final disclosure rebuild,
  the body is 808 words.
- [ ] Reinspect author metadata, AI disclosure, and acknowledgements after the
  author-only placeholders are finalized.

## Submission and review

- [x] Submitting author is the primary software contributor and has the GitHub
  account `EdoardoBotta`.
- [x] Sole-author submission; no co-author consent is required.
- [ ] Submit the repository, release/archive version, and paper through the JOSS
  new-paper form.
- [ ] Do not use generative AI for conversations with JOSS editors or reviewers
  except for translation; JOSS's current author policy prohibits it.
- [ ] Respond to reviewer questions within roughly two weeks and changes within
  four to six weeks, or communicate delays in the public review thread.
- [ ] At acceptance, create the requested GitHub Release, archive that exact
  tagged version, and give the version and DOI to the editor.
