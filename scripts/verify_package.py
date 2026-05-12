from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


EXPECTED = {
    "fixed_peer_seed": {
        "mean_macro_f1": 0.6099860784182729,
        "std_mean_macro_f1": 0.0019203253141012856,
        "mean_auc": 0.6425992043286674,
        "std_mean_auc": 0.00023537707777027445,
        "mean_fake_auprc": 0.47720353749608946,
        "mean_ece_10bin_fake_prob": 0.24675170872293103,
        "worst_target_f1": 0.535794000569473,
        "std_worst_target_f1": 0.000986743548720164,
    },
    "fixed_peer_ensemble": {
        "mean_macro_f1": 0.6088434348790626,
        "mean_auc": 0.6423943375394311,
        "mean_fake_auprc": 0.4770889121104614,
        "mean_ece_10bin_fake_prob": 0.24558011888740236,
        "worst_target_f1": 0.5371121891328502,
    },
}


def run(cmd: list[str], cwd: Path) -> None:
    subprocess.run(cmd, cwd=cwd, check=True)


def check_close(actual: float, expected: float, name: str, tol: float = 5e-7) -> None:
    if abs(float(actual) - float(expected)) > tol:
        raise AssertionError(f"{name}: actual={actual} expected={expected}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run and verify the released TERRA reproduction package.")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.repo_root
    for mode in EXPECTED:
        run([sys.executable, "scripts/reproduce_terra.py", "--mode", mode], cwd=root)
        agg_path = root / "outputs/terra_reproduction" / f"{mode}_aggregate.csv"
        row = pd.read_csv(agg_path).iloc[0]
        for key, expected in EXPECTED[mode].items():
            check_close(row[key], expected, f"{mode}.{key}")
    run([sys.executable, "scripts/make_main_table.py"], cwd=root)
    print("verification passed")


if __name__ == "__main__":
    main()

