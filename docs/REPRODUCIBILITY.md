# Reproducibility Guide

## Environment

Use Python 3.9 or newer.

```bash
python -m pip install -e .
```

The package depends only on NumPy, pandas, and scikit-learn for the released
reproduction path.

## Main Verification

```bash
python scripts/verify_package.py
```

This command:

1. recomputes TERRA fixed-method peer-seed results,
2. recomputes TERRA fixed peer ensemble results,
3. rebuilds the representative main-result table, and
4. checks numerical outputs against the expected values.

## Individual Commands

Main paper table setting:

```bash
python scripts/reproduce_terra.py --mode fixed_peer_seed
```

Deterministic fixed peer ensemble supplement:

```bash
python scripts/reproduce_terra.py --mode fixed_peer_ensemble
```

Representative comparison table:

```bash
python scripts/make_main_table.py
```

Outputs are written to:

```text
outputs/terra_reproduction/
outputs/main_table.md
outputs/main_table.csv
```

## Expected Values

| Mode | Mean Macro-F1 | Mean AUC | Fake AUPRC | ECE-10 | Worst-target F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `fixed_peer_seed` | 0.609986 | 0.642599 | 0.477204 | 0.246752 | 0.535794 |
| `fixed_peer_ensemble` | 0.608843 | 0.642394 | 0.477089 | 0.245580 | 0.537112 |

Small floating-point differences below `5e-7` are treated as equivalent.

## Evaluation Notes

- The threshold is chosen from source-validation labels using balanced
  accuracy as the primary criterion and Macro-F1 as the tie-breaker.
- AUC and AUPRC are computed directly from fake-class scores.
- ECE-10 uses 10 equal-width bins over fake probability and reports the
  sample-weighted absolute gap between mean predicted fake probability and
  empirical fake rate.
- The fixed-method peer-seed result first computes metrics for each seed across
  five target domains, then aggregates over seeds.

