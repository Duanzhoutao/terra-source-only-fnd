# TERRA: Source-Only Unknown-Domain Fake News Detection

This repository contains the minimal reproducibility package for TERRA
(Text-form-aware Expert Reliability Regulation), a source-only unknown-domain
fake news detection method evaluated on the FND5-LOCKED-V1 leave-one-domain-out
protocol.

The repository is a private pre-release artifact until the paper is published.
It is designed to make the core conclusion reproducible without distributing
raw news text, model checkpoints, LLM raw outputs, or internal exploration
materials.

## What Is Included

- Clean implementation of the TERRA scoring rule:
  - text-form-aware expert activation
  - bounded auxiliary residual regulation
  - source-validation threshold selection
  - Macro-F1, fake-class AUC, fake-class AUPRC, fake recall, and ECE-10
- De-identified expert-score inputs for the five target settings and three
  seeds. These files contain labels, split membership, text form, text length,
  event-quality mask, and expert probabilities.
- Representative baseline summary table used to rebuild the main comparison.
- Verification script that checks the released package against expected values.

## What Is Not Included

- Raw article, title, or statement text.
- LLM prompts or raw LLM JSON outputs.
- Model weights, checkpoints, Hugging Face caches, or training logs.
- Internal project notes, paper drafts, temporary files, or exploratory route
  search code.

## Reproduce The Main Result

```bash
python -m pip install -e .
python scripts/verify_package.py
```

Expected main TERRA result:

| Setting | Mean Macro-F1 | Mean AUC | Worst-target F1 | Fake AUPRC | ECE-10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed-method peer-seed | 0.609986 +/- 0.001920 | 0.642599 +/- 0.000235 | 0.535794 +/- 0.000987 | 0.477204 +/- 0.000195 | 0.246752 +/- 0.000095 |
| fixed peer ensemble | 0.608843 | 0.642394 | 0.537112 | 0.477089 | 0.245580 |

To run only one setting:

```bash
python scripts/reproduce_terra.py --mode fixed_peer_seed
python scripts/reproduce_terra.py --mode fixed_peer_ensemble
```

The scripts write outputs under `outputs/`.

## Protocol Summary

- Domains: PolitiFact, GossipCop, CoAID, PMD/MD, LIAR.
- Split: leave one domain out as unseen target-test; the remaining four domains
  form source train/source validation.
- Selection: method configuration and threshold are selected only from
  source-validation scores.
- Test use: target-test labels are used only for final evaluation.
- Positive class: `fake=1`.

The packaged score inputs are derived from the locked FND5-LOCKED-V1 artifacts.
They are sufficient to reproduce the paper's core TERRA fusion and evaluation
numbers, but they are not a replacement for the original public datasets.
Dataset provenance is documented in `docs/DATA.md`.

## Repository Layout

```text
data/
  terra_score_inputs/      # de-identified expert-score tables
  summary/                 # released manifests and representative baselines
docs/
  DATA.md                  # data provenance and release boundaries
  REPRODUCIBILITY.md       # commands and expected outputs
scripts/
  reproduce_terra.py       # recompute TERRA from score inputs
  make_main_table.py       # rebuild representative comparison table
  verify_package.py        # end-to-end verification
src/terra/
  metrics.py
  model.py
```

## Citation

Citation metadata will be updated after publication.

