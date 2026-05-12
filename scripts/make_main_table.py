from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REPRESENTATIVE_ROWS = [
    ("TERRA (ours)", "terra_fixed_method_peer_seed", "TERRA fixed-method peer-seed reproduction"),
    ("GroupDRO + DeBERTa-v3-base", "groupdro_deberta_v3_base", "full source-only run, seed 20260424"),
    ("DeBERTa-v3-base", "deberta_v3_base", "three-seed source-only peer reproduction"),
    ("MERMAID-style MoE, trainable encoder", "mermaid_style_trainable_encoder", "single-seed diagnostic run"),
    ("TF-IDF + LinearSVM", "tfidf_linear_svm", "three-seed classic baseline"),
    ("IRM + DeBERTa-v3-base", "irm_deberta_v3_base", "full source-only run, seed 20260424"),
    ("MDFEND", "mdfend", "three-seed neural baseline"),
    ("MERMAID-style source-expert MoE", "mermaid_style_source_expert_moe", "three-seed peer reproduction"),
    ("M3FEND", "m3fend", "three-seed peer reproduction"),
]


def format_mean_std(mean: float, std: float | None) -> str:
    if std is None or abs(float(std)) < 1e-12:
        return f"{float(mean):.6f}"
    return f"{float(mean):.6f} +/- {float(std):.6f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the representative main-result table.")
    parser.add_argument("--metrics", type=Path, default=Path("data/summary/representative_baselines.csv"))
    parser.add_argument("--out", type=Path, default=Path("outputs/main_table.md"))
    args = parser.parse_args()
    metrics = pd.read_csv(args.metrics)
    rows = []
    for display, model_id, note in REPRESENTATIVE_ROWS:
        row = metrics.loc[metrics["model_id"].eq(model_id)]
        if row.empty:
            raise ValueError(f"Missing model_id={model_id}")
        item = row.iloc[0]
        rows.append(
            {
                "Model": display,
                "Mean Macro-F1": format_mean_std(item["mean_macro_f1"], item.get("std_macro_f1")),
                "Mean AUC": format_mean_std(item["mean_auc"], item.get("std_auc")),
                "Fake AUPRC": format_mean_std(item["mean_fake_auprc"], item.get("std_fake_auprc")),
                "ECE-10": format_mean_std(item["mean_ece_10bin_fake_prob"], item.get("std_ece_10bin_fake_prob")),
                "Worst-target F1": format_mean_std(item["mean_worst_target_f1"], item.get("std_worst_target_f1")),
                "Fake recall": format_mean_std(item["mean_fake_recall"], item.get("std_fake_recall")),
                "Note": note,
            }
        )
    out = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    text = [
        "# Representative FND5-LOCKED-V1 Results",
        "",
        "Positive class is `fake=1`. AUPRC is Average Precision over fake probability. ECE-10 uses 10 equal-width fake-probability bins.",
        "",
        out.to_markdown(index=False),
        "",
    ]
    args.out.write_text("\n".join(text), encoding="utf-8")
    out.to_csv(args.out.with_suffix(".csv"), index=False)
    print(args.out)


if __name__ == "__main__":
    main()

