from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from terra.metrics import binary_metrics, choose_threshold
from terra.model import terra_score


TARGETS = ["target_politifact", "target_gossipcop", "target_coaid", "target_pmd", "target_liar"]
SEEDS = [20260424, 20260425, 20260426]


def score_one(
    data_dir: Path,
    seed: int,
    target: str,
    peer_col: str,
    support_col: str | None,
    residual_scale: float,
) -> dict[str, float | int | str]:
    path = data_dir / f"seed{seed}" / f"{target}.csv"
    frame = pd.read_csv(path)
    frame["terra_score"] = terra_score(frame, peer_col=peer_col, support_col=support_col, residual_scale=residual_scale)

    val = frame.loc[frame["split"].eq("val")].copy()
    test = frame.loc[frame["split"].eq("test")].copy()
    threshold, val_macro_f1 = choose_threshold(val["label"].to_numpy(), val["terra_score"].to_numpy(), "balanced_acc")
    metrics = binary_metrics(test["label"].to_numpy(), test["terra_score"].to_numpy(), threshold)
    return {
        "target_setting": target,
        "seed": seed,
        "threshold_rule": "balanced_acc",
        "threshold_used": threshold,
        "val_macro_f1_at_threshold": val_macro_f1,
        "n_test": int(len(test)),
        "n_test_fake": int(test["label"].sum()),
        **metrics,
    }


def aggregate_by_seed(per_target: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for seed, group in per_target.groupby("seed"):
        rows.append(
            {
                "seed": int(seed),
                "n_targets": int(group["target_setting"].nunique()),
                "mean_macro_f1": float(group["macro_f1"].mean()),
                "mean_auc": float(group["auc"].mean()),
                "mean_fake_recall": float(group["fake_recall"].mean()),
                "mean_fake_auprc": float(group["fake_auprc"].mean()),
                "mean_ece_10bin_fake_prob": float(group["ece_10bin_fake_prob"].mean()),
                "worst_target_f1": float(group["macro_f1"].min()),
                "mean_threshold_used": float(group["threshold_used"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("seed").reset_index(drop=True)


def aggregate_over_seeds(by_seed: pd.DataFrame, model_name: str) -> pd.DataFrame:
    metric_cols = [
        "mean_macro_f1",
        "mean_auc",
        "mean_fake_recall",
        "mean_fake_auprc",
        "mean_ece_10bin_fake_prob",
        "worst_target_f1",
    ]
    row: dict[str, float | int | str] = {
        "model": model_name,
        "n_seeds": int(by_seed["seed"].nunique()),
        "n_target_runs": int(by_seed["n_targets"].sum()),
    }
    for col in metric_cols:
        row[col] = float(by_seed[col].mean())
        row[f"std_{col}"] = float(by_seed[col].std(ddof=1)) if len(by_seed) > 1 else 0.0
    row["mean_threshold_used"] = float(by_seed["mean_threshold_used"].mean())
    return pd.DataFrame([row])


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce TERRA results from released expert-score tables.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/terra_score_inputs"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/terra_reproduction"))
    parser.add_argument(
        "--mode",
        choices=["fixed_peer_seed", "fixed_peer_ensemble"],
        default="fixed_peer_seed",
        help="fixed_peer_seed is the main paper table; fixed_peer_ensemble is the deterministic supplement.",
    )
    args = parser.parse_args()

    config = {
        "fixed_peer_seed": {
            "peer_col": "peer_mermaid_moe",
            "support_col": None,
            "residual_scale": 0.5,
        },
        "fixed_peer_ensemble": {
            "peer_col": "peer_mermaid_moe_seedlogitmean",
            "support_col": "event_reliability_lr_bal_c0p03",
            "residual_scale": 0.35,
        },
    }[args.mode]
    peer_col = str(config["peer_col"])
    support_col = config["support_col"]
    residual_scale = float(config["residual_scale"])
    model_name = {
        "fixed_peer_seed": "TERRA_fixed_method_peer_seed",
        "fixed_peer_ensemble": "TERRA_fixed_peer_ensemble",
    }[args.mode]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        score_one(
            args.data_dir,
            seed,
            target,
            peer_col=peer_col,
            support_col=str(support_col) if support_col is not None else None,
            residual_scale=residual_scale,
        )
        for seed in SEEDS
        for target in TARGETS
    ]
    per_target = pd.DataFrame(rows).sort_values(["seed", "target_setting"]).reset_index(drop=True)
    by_seed = aggregate_by_seed(per_target)
    aggregate = aggregate_over_seeds(by_seed, model_name=model_name)

    per_target.to_csv(args.out_dir / f"{args.mode}_per_target.csv", index=False)
    by_seed.to_csv(args.out_dir / f"{args.mode}_by_seed.csv", index=False)
    aggregate.to_csv(args.out_dir / f"{args.mode}_aggregate.csv", index=False)

    summary = {
        "mode": args.mode,
        "config": config,
        "aggregate": aggregate.iloc[0].to_dict(),
    }
    (args.out_dir / f"{args.mode}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
