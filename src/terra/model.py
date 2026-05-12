from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import clipped_logit, logit_blend, sigmoid


def text_form_activation(
    frame: pd.DataFrame,
    *,
    base_col: str = "event_reliability_lr_bal_c0p03",
    source_col: str = "source_sparse_top2",
    plm_col: str = "plm_deberta_rank",
    event_col: str = "event_semantic_rank",
    short_cut: float = 575.0,
    title_cut: float = 40.0,
    source_weight: float = 1.0,
    statement_plm_weight: float = 1.0,
    title_plm_weight: float = 0.2,
    long_event_weight: float = 0.0,
) -> np.ndarray:
    """Apply the paper's text-form-aware expert activation rule."""
    base = frame[base_col].to_numpy(dtype=np.float64)
    out = base.copy()
    text_source = frame["text_source"].fillna("unknown").astype(str).to_numpy()
    length = frame["text_length_chars"].to_numpy(dtype=np.float64)
    event_mask = frame.get("event_clean_mask", pd.Series(1.0, index=frame.index)).to_numpy(dtype=np.float64)

    source_score = frame[source_col].to_numpy(dtype=np.float64)
    plm_score = frame[plm_col].to_numpy(dtype=np.float64)
    event_score = frame[event_col].to_numpy(dtype=np.float64)

    article_short = (text_source == "article_text") & (length < short_cut)
    out[article_short] = logit_blend(base[article_short], source_score[article_short], source_weight)

    statement = text_source == "statement_only"
    out[statement] = logit_blend(base[statement], plm_score[statement], statement_plm_weight)

    title = (text_source == "title_only") & (length <= title_cut)
    out[title] = logit_blend(base[title], plm_score[title], title_plm_weight)

    if long_event_weight > 0:
        article_long = (text_source == "article_text") & (length >= short_cut) & (event_mask > 0.5)
        out[article_long] = logit_blend(out[article_long], event_score[article_long], long_event_weight)
    return out.clip(1e-6, 1.0 - 1e-6)


def confidence_blend(frame: pd.DataFrame, members: list[str], power: float = 1.0) -> np.ndarray:
    """Confidence-weighted logit blend used as a source-only support signal."""
    probs = [frame[col].to_numpy(dtype=np.float64).clip(1e-6, 1.0 - 1e-6) for col in members]
    conf = np.column_stack([(np.abs(p - 0.5) * 2.0) ** power for p in probs])
    logits = np.column_stack([clipped_logit(p) for p in probs])
    return sigmoid((logits * (conf + 1e-3)).sum(axis=1) / np.maximum((conf + 1e-3).sum(axis=1), 1e-12)).clip(1e-6, 1.0 - 1e-6)


def bounded_residual(
    base: np.ndarray,
    aux: np.ndarray,
    support: np.ndarray,
    *,
    weight: float = 0.15,
    scale: float = 0.5,
    gate_k: float = 8.0,
    confidence_gate: bool = True,
) -> np.ndarray:
    """Apply the bounded auxiliary residual regulation rule."""
    base_p = np.asarray(base, dtype=np.float64).clip(1e-6, 1.0 - 1e-6)
    aux_p = np.asarray(aux, dtype=np.float64).clip(1e-6, 1.0 - 1e-6)
    support_p = np.asarray(support, dtype=np.float64).clip(1e-6, 1.0 - 1e-6)
    base_z = clipped_logit(base_p)
    raw_delta = clipped_logit(aux_p) - base_z
    smooth_delta = np.tanh(raw_delta / max(float(scale), 1e-6)) * float(scale)
    align_gate = (1.0 + np.tanh(float(gate_k) * (aux_p - 0.5) * (support_p - 0.5))) * 0.5
    if confidence_gate:
        conf_base = np.abs(base_p - 0.5) * 2.0
        conf_aux = np.abs(aux_p - 0.5) * 2.0
        conf_gate = sigmoid(float(gate_k) * (conf_aux - conf_base))
    else:
        conf_gate = 1.0
    return sigmoid(base_z + float(weight) * align_gate * conf_gate * smooth_delta).clip(1e-6, 1.0 - 1e-6)


def terra_score(
    frame: pd.DataFrame,
    peer_col: str,
    *,
    support_col: str | None = None,
    residual_scale: float = 0.5,
) -> np.ndarray:
    """Compute TERRA scores from the released expert-score table."""
    base = text_form_activation(frame)
    if support_col is None:
        support = confidence_blend(frame, ["source_sparse_top2", "event_reliability_lr_bal_c0p03"], power=1.0)
    else:
        support = frame[support_col].to_numpy(dtype=np.float64)
    return bounded_residual(
        base,
        frame[peer_col].to_numpy(dtype=np.float64),
        support,
        weight=0.15,
        scale=residual_scale,
        gate_k=8.0,
        confidence_gate=True,
    )
