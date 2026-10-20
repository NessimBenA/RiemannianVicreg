#!/bin/bash
set -u

cd "$(dirname "$0")"
source .venv/bin/activate

ROOT="${OUTPUT_BASE:-output/geometry_showcase}"
GPU="${1:-0}"
mkdir -p "$ROOT"

run_job() {
    local name="$1"
    local objective="$2"
    local output="$ROOT/$name"

    local riemannian_args=(riemannian.enabled=false)
    if [ "$objective" = "rvicreg" ]; then
        riemannian_args=(
            riemannian.enabled=true
            riemannian.num_factors=64
            riemannian.factor_dim=4
            riemannian.kappa=-1
            riemannian.learnable_kappa=true
            riemannian.per_factor_kappa=true
            riemannian.factor_separated_projector=false
            riemannian.use_isotropy=false
            riemannian.use_frechet_mean=false
            riemannian.frechet_normalize_tangents=false
        )
    fi

    echo "START $name"
    CUDA_VISIBLE_DEVICES=$GPU TOKENIZERS_PARALLELISM=false python train_lanistr.py \
        --config configs/lanistr_mug_simsiam_finetune.yaml \
        seed=456 \
        dataset.name=mug \
        dataset.dataset_type=csgo_skin_quality \
        dataset.data_dir=data/MuG \
        dataset.image=true dataset.text=true dataset.tab=true \
        training.mode=simsiam_finetune \
        training.task=pretrain \
        training.ssl_objective="$objective" \
        training.num_epochs=50 \
        training.batch_size=64 \
        training.learning_rate=1e-4 \
        training.freeze_encoders=false \
        training.output_dir="$output" \
        training.keep_all_checkpoints=false \
        training.export_mechanism_representations=true \
        encoders.image.pretrained=true \
        encoders.text.pretrained=true \
        projection.output_dim=256 \
        "${riemannian_args[@]}" \
        > "$output.log" 2>&1
    local status=$?
    echo "END $name (exit $status)"
    return $status
}

run_job baseline_csgo_I_T_Tab_vicreg_seed456 vicreg || exit $?
run_job baseline_csgo_I_T_Tab_rvicreg_learned_seed456 rvicreg || exit $?

echo "Figure 2 statistics: python analysis/geometry_showcase.py --runs $ROOT"
