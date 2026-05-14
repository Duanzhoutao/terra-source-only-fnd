# Model Details

This document describes the released TERRA model configuration used by the
minimal reproducibility package. The implementation lives in
`src/terra/model.py`; the metric and threshold utilities live in
`src/terra/metrics.py`; the runnable reproduction entry point is
`scripts/reproduce_terra.py`.

## Released Inputs

TERRA is recomputed from de-identified expert-score tables under
`data/terra_score_inputs/`. Each table contains source-validation and
target-test rows for one seed and one held-out target setting.

The released columns used by the scoring rule are:

| Column | Role |
| --- | --- |
| `text_source` | Text form indicator, such as `title_only`, `article_text`, or `statement_only`. |
| `text_length_chars` | Character length used by the text-form activation rule. |
| `event_clean_mask` | Indicator that structured event evidence is available for clean event use. |
| `event_reliability_lr_bal_c0p03` | Event reliability base expert score. |
| `source_sparse_top2` | Source sparse lexical expert score. |
| `plm_deberta_rank` | Pretrained language model semantic expert score. |
| `event_semantic_rank` | Structured event semantic expert score. |
| `peer_mermaid_moe` | Peer source-expert MoE score used in the fixed-method peer-seed setting. |
| `peer_mermaid_moe_seedlogitmean` | Seed-averaged peer source-expert MoE score used in the deterministic supplement. |

Raw article text, LLM raw outputs, training logs, checkpoints, and Hugging Face
caches are not included in this release.

## Scoring Rule

The released TERRA score is computed in three stages.

1. `text_form_activation` starts from the event reliability base expert. For
   short article text, it blends the base score with the sparse source expert.
   For statement-only samples, it blends the base score with the PLM semantic
   expert. For very short title-only samples, it applies a lighter PLM blend.
   The released configuration uses `short_cut=575`, `title_cut=40`,
   `source_weight=1.0`, `statement_plm_weight=1.0`, and
   `title_plm_weight=0.2`.
2. `confidence_blend` builds a source-only support signal from
   `source_sparse_top2` and `event_reliability_lr_bal_c0p03` when no explicit
   support column is provided.
3. `bounded_residual` adds the peer MoE score as a bounded residual rather than
   replacing the activated base score. The released configuration uses
   `weight=0.15`, `gate_k=8.0`, and a confidence gate. The residual scale is
   `0.5` for `fixed_peer_seed` and `0.35` for `fixed_peer_ensemble`.

All operations are performed in probability/logit space with clipping at
`1e-6` and `1 - 1e-6`.

## Selection And Evaluation

For each held-out target setting, the score is computed for source-validation
and target-test rows. The decision threshold is selected only on
source-validation rows using the balanced-accuracy rule implemented in
`choose_threshold`. Target-test labels are used only for final evaluation.

The released scripts report Macro-F1, AUC, fake-class AUPRC, fake recall,
ECE-10, and worst-target Macro-F1. The main paper table focuses on Macro-F1,
AUC, and worst-target Macro-F1.

## Reproduction Modes

`scripts/reproduce_terra.py --mode fixed_peer_seed` recomputes the main
three-seed setting from `peer_mermaid_moe`.

`scripts/reproduce_terra.py --mode fixed_peer_ensemble` recomputes the
deterministic supplement from `peer_mermaid_moe_seedlogitmean`.

Run `python scripts/verify_package.py` to verify the released package against
the expected aggregate values.
