from __future__ import annotations

import argparse
from pathlib import Path
from typing import Final

import pandas as pd
from scipy import stats

METHODS: Final[list[str]] = ["simsiam", "vicreg", "rvicreg"]
THRESHOLDS: Final[list[float]] = [0.0, 0.005, 0.01]

PAPER_PPLUS: Final[dict[str, tuple[float, float, float]]] = {
    "simsiam": (48.9, 46.7, 40.0),
    "vicreg": (44.4, 40.0, 37.8),
    "rvicreg": (64.4, 62.2, 51.1),
}

PAPER_RESCUE: Final[dict[float, tuple[int, int]]] = {
    0.0: (13, 4),
    0.005: (12, 2),
    0.01: (7, 1),
}

PAPER_GEOMETRY: Final[dict[str, float]] = {
    "kappa_mean": -0.934,
    "kappa_min": -0.992,
    "kappa_max": -0.820,
    "boundary_mean": 0.987,
    "boundary_q95": 0.996,
}

PAPER_PER_DATASET: Final[dict[str, tuple[float, float, float]]] = {
    "csgo": (0.40, 0.00, 0.80),
    "pokemon_sec": (0.20, 0.00, 0.60),
    "petfinder": (0.40, 0.20, 0.60),
    "pokemon_pri": (0.20, 0.20, 0.40),
    "lol": (0.40, 0.60, 0.60),
    "hs_race": (0.20, 0.20, 0.20),
    "hs_cardclass": (1.00, 1.00, 1.00),
    "hs_set": (1.00, 1.00, 1.00),
    "hs_spell": (0.60, 0.80, 0.60),
}


def verdict(ok: bool) -> str:
    return "MATCH" if ok else "DIFFERS"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv", type=Path, default=Path("results/modality_ablation_full_analysis.csv")
    )
    args = parser.parse_args()

    frame = pd.read_csv(args.csv)
    frame["TG"] = frame["S_Tri"] - frame["B_star"]
    failures = 0

    print(f"paper analysis table: {args.csv} rows={len(frame)}")
    print(f"datasets={frame['dataset'].nunique()} seeds={frame['seed'].nunique()}")
    print()

    print("Table 1: positive-gain rate p+(tau)")
    for method in METHODS:
        series = frame.loc[frame["method"] == method, "TG"]
        got = tuple(round(100.0 * float((series > t).mean()), 1) for t in THRESHOLDS)
        ok = got == PAPER_PPLUS[method]
        failures += not ok
        print(f"  {method:9s} computed={got} paper={PAPER_PPLUS[method]} {verdict(ok)}")

    print()
    print("Selective rescue vs VICReg")
    pivot = frame.pivot_table(index=["dataset", "seed"], columns="method", values="TG")
    for threshold, expected in PAPER_RESCUE.items():
        rescue = int(((pivot["vicreg"] <= threshold) & (pivot["rvicreg"] > threshold)).sum())
        harm = int(((pivot["vicreg"] > threshold) & (pivot["rvicreg"] <= threshold)).sum())
        p_value = float(stats.binomtest(rescue, rescue + harm, 0.5).pvalue)
        ok = (rescue, harm) == expected
        failures += not ok
        print(
            f"  tau={threshold:<5} computed R={rescue} H={harm} p={p_value:.3f} "
            f"paper R={expected[0]} H={expected[1]} {verdict(ok)}"
        )

    print()
    print("Per-dataset p+(0)")
    for dataset, expected in PAPER_PER_DATASET.items():
        got = tuple(
            round(
                float(
                    (
                        frame.loc[
                            (frame["method"] == m) & (frame["dataset"] == dataset), "TG"
                        ]
                        > 0
                    ).mean()
                ),
                2,
            )
            for m in METHODS
        )
        ok = got == expected
        failures += not ok
        print(f"  {dataset:13s} computed={got} paper={expected} {verdict(ok)}")

    print()
    print("Geometry appendix")
    curved = frame[frame["method"] == "rvicreg"]
    got_geom = {
        "kappa_mean": round(float(curved["kappa"].mean()), 3),
        "kappa_min": round(float(curved["kappa"].min()), 3),
        "kappa_max": round(float(curved["kappa"].max()), 3),
        "boundary_mean": round(float(curved["boundary_max"].mean()), 3),
        "boundary_q95": round(float(curved["boundary_max"].quantile(0.95)), 3),
    }
    for key, expected_value in PAPER_GEOMETRY.items():
        ok = got_geom[key] == expected_value
        failures += not ok
        print(
            f"  {key:14s} computed={got_geom[key]:+.3f} paper={expected_value:+.3f} {verdict(ok)}"
        )

    print()
    print(f"total mismatches: {failures}")


if __name__ == "__main__":
    main()
