# GitHub Upload Plan

## Goal

Publish a compact, private reproducibility repository for the TERRA paper before
publication. After the paper is accepted or published, the repository can be
made public with citation metadata and any final camera-ready updates.

## Scope

The repository is intentionally limited to the materials needed to reproduce
the paper's core empirical conclusion:

- TERRA text-form-aware expert activation.
- Bounded auxiliary residual regulation.
- Source-validation threshold selection.
- Main metrics and representative comparison table.
- De-identified expert-score inputs for the locked five-domain protocol.

## Included Data

The uploaded data consists of compact CSV score tables:

- one file per seed and held-out target domain,
- source-validation and target-test rows only,
- binary labels,
- text-form metadata and text length,
- event-quality mask,
- expert probability scores needed by TERRA.

This is enough to rerun the released TERRA scoring and evaluation pipeline.

## Excluded Materials

The upload excludes raw dataset text, model weights, checkpoints, cache
directories, raw LLM outputs, training logs, paper drafts, temporary files, and
large intermediate artifacts.

## Repository Structure

```text
README.md
requirements.txt
pyproject.toml
data/
  terra_score_inputs/
  summary/
docs/
  DATA.md
  REPRODUCIBILITY.md
  UPLOAD_PLAN.md
scripts/
  reproduce_terra.py
  make_main_table.py
  verify_package.py
src/terra/
  metrics.py
  model.py
```

## Verification Command

```bash
python -m pip install -e .
python scripts/verify_package.py
```

The verification checks both the main fixed-method peer-seed result and the
fixed peer ensemble supplement against expected values.

