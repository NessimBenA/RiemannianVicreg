from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, TypedDict

import pandas as pd

Method = Literal["simsiam", "vicreg", "rvicreg"]
Subset = Literal["I", "T", "Tab", "I_T", "I_Tab", "T_Tab", "I_T_Tab"]

DATASET_KEYS: Final[dict[str, str]] = {
    "csgo_skin_quality": "csgo",
    "leagueoflegends_skin_category": "lol",
    "pokemon_primary_type": "pokemon_pri",
    "pokemon_secondary_type": "pokemon_sec",
    "hearthstone_minion_race": "hs_race",
    "hearthstone_all_cardclass": "hs_cardclass",
    "hearthstone_all_set": "hs_set",
    "hearthstone_spell_spellschool": "hs_spell",
    "petfinder": "petfinder",
}
METHODS: Final[tuple[Method, ...]] = ("simsiam", "vicreg", "rvicreg")
SUBSETS: Final[dict[tuple[bool, bool, bool], Subset]] = {
    (True, False, False): "I",
    (False, True, False): "T",
    (False, False, True): "Tab",
    (True, True, False): "I_T",
    (True, False, True): "I_Tab",
    (False, True, True): "T_Tab",
    (True, True, True): "I_T_Tab",
}
UNIMODAL: Final[tuple[Subset, ...]] = ("I", "T", "Tab")
BIMODAL: Final[tuple[Subset, ...]] = ("I_T", "I_Tab", "T_Tab")
PAIRS_WITH: Final[dict[Subset, tuple[Subset, Subset]]] = {
    "I": ("I_T", "I_Tab"),
    "T": ("I_T", "T_Tab"),
    "Tab": ("I_Tab", "T_Tab"),
}
COLUMNS: Final[tuple[str, ...]] = (
    "dataset", "seed", "method",
    "S_I", "S_T", "S_Tab", "S_IT", "S_ITab", "S_TTab", "S_Tri",
    "U_star_mod", "U_star", "B_star_pair", "B_star", "B_star_Ustar",
    "step1", "step2", "step2_prime", "tri_minus_Ustar",
    "kappa", "boundary_max",
)


class MetricsFile(TypedDict):
    dataset_type: str
    ssl_objective: str
    seed: int
    image: bool
    text: bool
    tab: bool
    summary: dict[str, float]


@dataclass(frozen=True)
class RunRecord:
    dataset: str
    method: Method
    seed: int
    subset: Subset
    test_accuracy: float
    kappa: float | None
    boundary_max: float | None


def parse_method(value: str) -> Method:
    for method in METHODS:
        if value == method:
            return method
    raise ValueError(f"Unknown ssl_objective: {value}")


def to_record(data: MetricsFile) -> RunRecord:
    summary = data["summary"]
    if "test/accuracy" not in summary:
        raise ValueError(f"Run without test accuracy: {data}")
    return RunRecord(
        dataset=DATASET_KEYS[data["dataset_type"]],
        method=parse_method(data["ssl_objective"]),
        seed=int(data["seed"]),
        subset=SUBSETS[(bool(data["image"]), bool(data["text"]), bool(data["tab"]))],
        test_accuracy=float(summary["test/accuracy"]),
        kappa=summary.get("pretrain/kappa"),
        boundary_max=summary.get("pretrain/boundary_proximity_max"),
    )


def load_records(runs_dir: Path) -> list[RunRecord]:
    records: list[RunRecord] = []
    for path in sorted(runs_dir.rglob("metrics.json")):
        data: MetricsFile = json.loads(path.read_text())
        records.append(to_record(data))
    return records


def build_table(records: list[RunRecord]) -> pd.DataFrame:
    index: dict[tuple[str, int, Method, Subset], RunRecord] = {}
    for record in records:
        key = (record.dataset, record.seed, record.method, record.subset)
        if key in index:
            raise ValueError(f"Duplicate run: {key}")
        index[key] = record

    groups = sorted({(r.dataset, r.seed, r.method) for r in records})
    rows: list[dict[str, str | int | float | None]] = []
    for dataset, seed, method in groups:
        missing = [s for s in SUBSETS.values() if (dataset, seed, method, s) not in index]
        if missing:
            raise ValueError(f"Missing subsets {missing} for {dataset} seed {seed} {method}")
        acc: dict[Subset, float] = {
            s: index[(dataset, seed, method, s)].test_accuracy for s in SUBSETS.values()
        }
        u_mod = max(UNIMODAL, key=lambda s: (acc[s], -UNIMODAL.index(s)))
        b_pair = max(BIMODAL, key=lambda s: (acc[s], -BIMODAL.index(s)))
        u_star = acc[u_mod]
        b_star = acc[b_pair]
        b_star_u = max(acc[p] for p in PAIRS_WITH[u_mod])
        tri = acc["I_T_Tab"]
        tri_run = index[(dataset, seed, method, "I_T_Tab")]
        rows.append({
            "dataset": dataset,
            "seed": seed,
            "method": method,
            "S_I": acc["I"],
            "S_T": acc["T"],
            "S_Tab": acc["Tab"],
            "S_IT": acc["I_T"],
            "S_ITab": acc["I_Tab"],
            "S_TTab": acc["T_Tab"],
            "S_Tri": tri,
            "U_star_mod": u_mod,
            "U_star": u_star,
            "B_star_pair": b_pair,
            "B_star": b_star,
            "B_star_Ustar": b_star_u,
            "step1": b_star_u - u_star,
            "step2": tri - b_star_u,
            "step2_prime": tri - b_star,
            "tri_minus_Ustar": tri - u_star,
            "kappa": tri_run.kappa,
            "boundary_max": tri_run.boundary_max,
        })
    return pd.DataFrame(rows, columns=list(COLUMNS))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, default=Path("output/modality_ablation"))
    parser.add_argument("--out", type=Path, default=Path("results/modality_ablation_rerun.csv"))
    args = parser.parse_args()
    runs_dir: Path = args.runs
    out_path: Path = args.out

    records = load_records(runs_dir)
    table = build_table(records)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_path, index=False)
    print(f"{len(records)} runs -> {len(table)} rows -> {out_path}")


if __name__ == "__main__":
    main()
