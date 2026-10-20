#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

source .venv/bin/activate

EXPERIMENT=${1:-full}
NUM_GPUS=${2:-2}

OUTPUT_BASE="${OUTPUT_BASE:-output/modality_ablation}"
CONFIG_FILE="configs/lanistr_mug_simsiam_finetune.yaml"

SEEDS=(42 123 456 789 1337)

declare -A METHODS
METHODS["simsiam"]="simsiam|false"
METHODS["vicreg"]="vicreg|false"
METHODS["rvicreg"]="rvicreg|true"

declare -A MODALITY_COMBOS
MODALITY_COMBOS["I"]="true|false|false"
MODALITY_COMBOS["T"]="false|true|false"
MODALITY_COMBOS["Tab"]="false|false|true"
MODALITY_COMBOS["I_T"]="true|true|false"
MODALITY_COMBOS["I_Tab"]="true|false|true"
MODALITY_COMBOS["T_Tab"]="false|true|true"
MODALITY_COMBOS["I_T_Tab"]="true|true|true"

declare -A DATASETS
DATASETS["csgo"]="mug|csgo_skin_quality|data/MuG"
DATASETS["hs_race"]="mug|hearthstone_minion_race|data/MuG"
DATASETS["pokemon_pri"]="mug|pokemon_primary_type|data/MuG"
DATASETS["hs_cardclass"]="mug|hearthstone_all_cardclass|data/MuG"
DATASETS["hs_set"]="mug|hearthstone_all_set|data/MuG"
DATASETS["hs_spell"]="mug|hearthstone_spell_spellschool|data/MuG"
DATASETS["pokemon_sec"]="mug|pokemon_secondary_type|data/MuG"
DATASETS["lol"]="mug|leagueoflegends_skin_category|data/MuG"
DATASETS["petfinder"]="petfinder|petfinder|data/PetfindAdoptionPrediction"

ALL_DATASETS=("csgo" "hs_race" "pokemon_pri" "hs_cardclass" "hs_set" "hs_spell" "pokemon_sec" "lol" "petfinder")
MODALITY_NAMES=("I" "T" "Tab" "I_T" "I_Tab" "T_Tab" "I_T_Tab")
METHOD_NAMES=("simsiam" "vicreg" "rvicreg")

print_usage() {
    echo "Usage: $0 [full|list] [num_gpus]"
    echo ""
    echo "  full  - 3 methods x 7 modality subsets x 9 datasets x 5 seeds = 945 runs"
    echo "  list  - print the run grid without running"
    echo ""
    echo "Runs are written to \$OUTPUT_BASE (default: output/modality_ablation)."
    echo "A run is skipped when its metrics.json already exists, so the sweep can be resumed."
}

is_completed() {
    compgen -G "${OUTPUT_BASE}/$1/*/metrics.json" > /dev/null
}

run_job() {
    local gpu=$1
    local run_name=$2
    local ssl_obj=$3
    local riem_enabled=$4
    local seed=$5
    local ds_name=$6
    local ds_type=$7
    local ds_dir=$8
    local use_image=$9
    local use_text=${10}
    local use_tab=${11}
    local output_dir="${OUTPUT_BASE}/${run_name}"

    echo "[GPU ${gpu}] Starting: ${run_name}"

    local riem_args="riemannian.enabled=false"
    if [ "$riem_enabled" = "true" ]; then
        riem_args="riemannian.enabled=true"
        riem_args="${riem_args} riemannian.num_factors=64"
        riem_args="${riem_args} riemannian.factor_dim=4"
        riem_args="${riem_args} riemannian.learnable_kappa=true"
        riem_args="${riem_args} riemannian.factor_separated_projector=false"
        riem_args="${riem_args} riemannian.per_factor_kappa=true"
        riem_args="${riem_args} riemannian.use_isotropy=false"
        riem_args="${riem_args} riemannian.use_frechet_mean=false"
        riem_args="${riem_args} riemannian.frechet_normalize_tangents=false"
    fi

    CUDA_VISIBLE_DEVICES=${gpu} python train_lanistr.py \
        --config ${CONFIG_FILE} \
        seed=${seed} \
        dataset.name=${ds_name} \
        dataset.dataset_type=${ds_type} \
        dataset.data_dir=${ds_dir} \
        dataset.image=${use_image} dataset.text=${use_text} dataset.tab=${use_tab} \
        training.mode=simsiam_finetune \
        training.task=pretrain \
        training.ssl_objective=${ssl_obj} \
        training.num_epochs=50 \
        training.batch_size=64 \
        training.learning_rate=1e-4 \
        training.freeze_encoders=false \
        training.output_dir="${output_dir}" \
        encoders.image.pretrained=true \
        encoders.text.pretrained=true \
        projection.output_dim=256 \
        ${riem_args} \
        > "${output_dir}.log" 2>&1

    local status=$?
    if [ $status -eq 0 ]; then
        echo "[GPU ${gpu}] Finished: ${run_name}"
    else
        echo "[GPU ${gpu}] FAILED: ${run_name} (exit code: $status)"
    fi
    return $status
}

build_jobs() {
    local -n jobs_ref=$1
    for dataset_name in "${ALL_DATASETS[@]}"; do
        IFS='|' read -r ds_name ds_type ds_dir <<< "${DATASETS[$dataset_name]}"
        for modality_name in "${MODALITY_NAMES[@]}"; do
            IFS='|' read -r use_image use_text use_tab <<< "${MODALITY_COMBOS[$modality_name]}"
            for method_name in "${METHOD_NAMES[@]}"; do
                IFS='|' read -r ssl_obj riem_enabled <<< "${METHODS[$method_name]}"
                for seed in "${SEEDS[@]}"; do
                    run_name="${dataset_name}_${modality_name}_${method_name}_seed${seed}"
                    jobs_ref+=("${run_name}|${ssl_obj}|${riem_enabled}|${seed}|${ds_name}|${ds_type}|${ds_dir}|${use_image}|${use_text}|${use_tab}")
                done
            done
        done
    done
}

run_experiment() {
    JOBS=()
    build_jobs JOBS
    local total_jobs=${#JOBS[@]}

    echo "Modality ablation sweep: ${total_jobs} runs on ${NUM_GPUS} GPUs -> ${OUTPUT_BASE}"
    mkdir -p "${OUTPUT_BASE}"

    declare -a GPU_PIDS
    for ((i=0; i<NUM_GPUS; i++)); do
        GPU_PIDS[$i]=0
    done

    local completed=0
    local failed=0
    local skipped=0

    for ((job_idx=0; job_idx<total_jobs; job_idx++)); do
        gpu=$((job_idx % NUM_GPUS))

        IFS='|' read -r run_name ssl_obj riem_enabled seed ds_name ds_type ds_dir use_image use_text use_tab <<< "${JOBS[$job_idx]}"

        if is_completed "$run_name"; then
            ((skipped++))
            continue
        fi

        if [ ${GPU_PIDS[$gpu]} -ne 0 ]; then
            wait ${GPU_PIDS[$gpu]} 2>/dev/null
            if [ $? -eq 0 ]; then
                ((completed++))
            else
                ((failed++))
            fi
            echo "[Progress] ${completed} completed, ${failed} failed, ${skipped} skipped / ${total_jobs} total"
        fi

        run_job $gpu "$run_name" "$ssl_obj" "$riem_enabled" "$seed" "$ds_name" "$ds_type" "$ds_dir" "$use_image" "$use_text" "$use_tab" &
        GPU_PIDS[$gpu]=$!
    done

    for ((i=0; i<NUM_GPUS; i++)); do
        if [ ${GPU_PIDS[$i]} -ne 0 ]; then
            wait ${GPU_PIDS[$i]} 2>/dev/null
            if [ $? -eq 0 ]; then
                ((completed++))
            else
                ((failed++))
            fi
        fi
    done

    echo "Sweep finished: ${completed} completed, ${failed} failed, ${skipped} skipped"
    echo "Aggregate with: python analysis/aggregate_runs.py --runs ${OUTPUT_BASE}"
}

list_jobs() {
    JOBS=()
    build_jobs JOBS
    printf '%s\n' "${JOBS[@]}" | cut -d'|' -f1
    echo "${#JOBS[@]} runs"
}

case $EXPERIMENT in
    full) run_experiment ;;
    list) list_jobs ;;
    -h|--help|help) print_usage ;;
    *)
        echo "Unknown experiment: $EXPERIMENT"
        print_usage
        exit 1
        ;;
esac
