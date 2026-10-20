# On the Geometry of Multimodal Saturation: Riemannian VICReg

Code for the NeurReps 2026 paper (Proceedings of Machine Learning Research) by Nessim Ben Abbes, Duc Han Le, Sabri Mtibaa, and Van-Tam Nguyen.

Riemannian VICReg (R-VICReg) aligns views by squared geodesic distance on a product of learnable negative-curvature factors and recovers VICReg as curvature vanishes. Across nine image-text-tabular datasets it raises the probability that a third modality helps from 44.4% (VICReg) to 64.4%.

This repository contains the R-VICReg implementation, the 945-run modality-ablation sweep behind the paper's results, the per-run results table, and a script that checks every reported number against it.

## Layout

```
train_lanistr.py                  entry point: SSL pretraining followed by supervised finetuning
lanistr_impl/                     model, losses (VICReg, SimSiam, R-VICReg), kappa-stereographic geometry, trainer
data_loaders/                     MuG-Bench and PetFinder.my loaders
configs/                          configs used by the sweep
run_modality_ablation.sh          the full sweep: 9 datasets x 7 modality subsets x 3 methods x 5 seeds
run_geometry_showcase.sh          the two CSGO seed-456 runs shown in Figure 2
analysis/aggregate_runs.py        builds the results table from a sweep's metrics.json files
analysis/verify_paper_numbers.py  checks the paper's numbers against a results table
analysis/geometry_showcase.py     Figure 2 and its statistics
results/                          results table and Figure 2 data used in the paper
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The pinned versions are those of the environment the experiments were run from. Image and text encoders (`WinKawaks/vit-tiny-patch16-224`, `prajjwal1/bert-mini`) are downloaded from the Hugging Face Hub on first use.

## Checking the paper's numbers (CPU, seconds)

```bash
python analysis/verify_paper_numbers.py
python analysis/geometry_showcase.py
```

The first script recomputes Table 1 (positive-gain rates), the selective-rescue counts, the per-dataset rates, and the learned-geometry statistics from `results/modality_ablation_full_analysis.csv` and compares each to the value reported in the paper. The second recomputes the Figure 2 statistics (most-utilized factor and its curvature, tangent PCA variance, fraction of codes beyond geodesic radius 2, shell area ratio) from `results/geometry_showcase_data.npz` and draws the figure.

## Data

The datasets are public and not redistributed here.

- MuG-Bench: https://github.com/lupantech/MuG-Bench, placed under `data/MuG/` with one folder per task (`CSGO-Skin-quality`, `LeagueOfLegends-Skin-category`, `Pokemon-primary_type`, `Pokemon-secondary_type`, `Hearthstone-Minion-race`, `Hearthstone-All-cardClass`, `Hearthstone-All-set`, `Hearthstone-Spell-spellSchool`), each with `train.csv`, `dev.csv`, `test.csv`.
- PetFinder.my Adoption Prediction: https://www.kaggle.com/c/petfinder-adoption-prediction, placed under `data/PetfindAdoptionPrediction/` (`train.csv`, `train_images/`, `train_sentiment/`).

## Reproducing the sweep (GPU)

```bash
./run_modality_ablation.sh full 2
python analysis/aggregate_runs.py --runs output/modality_ablation --out results/modality_ablation_rerun.csv
python analysis/verify_paper_numbers.py --csv results/modality_ablation_rerun.csv
```

The sweep runs 945 jobs spread over the given number of GPUs (about 200 GPU-hours in total for the paper). Each run writes a `metrics.json` with its test accuracy and final pretraining geometry statistics; finished runs are skipped when the script is restarted. A single run is a plain call to `train_lanistr.py`; see `run_job` in `run_modality_ablation.sh` for the exact arguments.

Reruns on other hardware or library versions will not be bitwise identical to the paper's runs; `verify_paper_numbers.py` then reports which numbers differ.

Figure 2 from fresh runs:

```bash
./run_geometry_showcase.sh 0
python analysis/geometry_showcase.py --runs output/geometry_showcase
```

## Configuration notes

All reported R-VICReg runs use 64 factors of dimension 4, per-factor learnable curvature initialized at kappa = -1 and bounded by kappa_max = 1.5, encoders finetuned during pretraining, and variance/covariance computed on tangent codes at the origin of each factor (`riemannian.use_frechet_mean=false`, the default). The Fréchet-mean-centered variant is available with `riemannian.use_frechet_mean=true`.

## Citation

```bibtex
@InProceedings{benabbes26,
  title    = {{O}n the {G}eometry of {M}ultimodal {S}aturation: {R}iemannian {VICReg}},
  author   = {Ben Abbes, Nessim and Le, Duc Han and Mtibaa, Sabri and Nguyen, Van-Tam},
  software = {https://github.com/NessimBenA/RiemannianVicreg}
}
```

## License

Code is released under the MIT License. The datasets remain under their original licenses.
