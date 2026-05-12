# Data Card

## Dataset Sources

FND5-LOCKED-V1 combines five public fake news or fact-checking data domains:

| Domain | Source | Label mapping |
| --- | --- | --- |
| PolitiFact | FakeNewsNet PolitiFact | fake=1, real=0 |
| GossipCop | FakeNewsNet GossipCop | fake=1, real=0 |
| CoAID | CoAID COVID-19 misinformation dataset | fake=1, real=0 |
| PMD/MD | Fake and Real News / fakenews2020-style MD data | -1 -> fake=1, 1 -> real=0 |
| LIAR | LIAR political statement dataset | false and pants-fire -> fake=1; other labels -> real=0 |

The paper uses a MERMAID-aligned five-domain extension, not an official
MERMAID reproduction. PolitiFact, GossipCop, and CoAID use the project's
audited full-table versions rather than the exact MERMAID subset.

## Released Inputs

The released files under `data/terra_score_inputs/` are de-identified
expert-score tables. Each file corresponds to one seed and one held-out target
setting and contains only source-validation and target-test rows.

Columns:

| Column | Meaning |
| --- | --- |
| `sample_id` | Stable sample identifier used for alignment. |
| `target_setting` | Held-out target setting. |
| `seed` | Reproduction seed. |
| `split` | `val` or `test`. |
| `label` | Binary label, with `fake=1`. |
| `domain` | Source or target domain name. |
| `text_source` | Text form, e.g. `title_only`, `article_text`, `statement_only`. |
| `text_length_chars` | Character length of the model text. Raw text is not included. |
| `event_clean_mask` | Event-evidence quality mask. |
| expert score columns | Source-only expert probabilities used by TERRA. |

The package does not include raw text, model checkpoints, LLM raw outputs, or
training logs. This keeps the artifact compact and avoids redistributing
dataset content whose licensing and provenance should be handled through the
original dataset sources.

## Locked Protocol

- Leave-one-domain-out target settings:
  - `target_politifact`
  - `target_gossipcop`
  - `target_coaid`
  - `target_pmd`
  - `target_liar`
- Source-validation scores are used to choose the decision threshold.
- Target-test labels are used only for final evaluation.
- Main metric is mean Macro-F1 over the five target domains.
- Additional metrics are fake-class AUC, fake-class AUPRC, fake recall,
  worst-target Macro-F1, and ECE-10.

## Public Data Reconstruction

To rebuild the full training artifacts from scratch, obtain the original
datasets from their maintainers and reproduce the same FND5-LOCKED-V1
preprocessing, split, LLM event extraction, and upstream expert training. This
minimal repository intentionally starts from the fixed expert-score tables so
that the paper's core fusion and evaluation claims can be verified without
shipping large or sensitive intermediates.

