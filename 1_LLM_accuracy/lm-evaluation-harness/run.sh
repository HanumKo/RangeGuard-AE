#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ENV_FILE="${REPO_ROOT}/env.sh"
NO_ERROR_FILE="${REPO_ROOT}/1_LLM_accuracy/no_error"

if [[ ! -f "${ENV_FILE}" ]]; then
    echo "Missing env.sh: ${ENV_FILE}" >&2
    exit 1
fi

# shellcheck source=../../env.sh
source "${ENV_FILE}"

##################### Fixed parameters #####################
TASKS="arc_easy"
BATCH_SIZE="auto"
DEVICE="cuda:0"

PYTHON_SEED=0
NUMPY_SEED=1234
FEWSHOT_SEED=1234

######################## Sweep setup ########################
NUM_SWEEPS=100
mapfile -t TORCH_SEEDS < <(shuf -i 0-999999 -n "${NUM_SWEEPS}")

####################### ECC schemes ########################
MODEL_NAME_ARRAY=(
    "Llama-3.2-1B"
    "Llama-3.1-8B"
)

ECC_SCHEME_ARRAY=(
    "NONE"
    "NULLING_BF16"
    "VAPI"
    "RANGEGUARD_BF16_SSC_FAST"
    "RANGEGUARD_BF16_DSC_FAST"
)

PROB_EXP_ARRAY=(10 9 8 7 6 5 4)

for MODEL_NAME in "${MODEL_NAME_ARRAY[@]}"; do
    MODEL_PATH="${MODEL_ROOT}/${MODEL_NAME}"

    if [[ ! -d "${MODEL_PATH}" ]]; then
        echo "Model directory not found: ${MODEL_PATH}" >&2
        exit 1
    fi

    RUN_ROOT_BASE="${OUTPUT_ROOT}/${MODEL_NAME}/${TASKS}"

    echo "========================================================"
    echo ">>> Starting Experiments for Model: ${MODEL_NAME}"
    echo "========================================================"

    for SCHEME in "${ECC_SCHEME_ARRAY[@]}"; do
        echo "========================================================"
        echo ">>> Starting Experiments for ECC Scheme: ${SCHEME}"
        echo "========================================================"

        for EXP in "${PROB_EXP_ARRAY[@]}"; do
            RUN_ROOT="${RUN_ROOT_BASE}/${SCHEME}/exp_${EXP}"
            mkdir -p "${RUN_ROOT}"

            echo ""
            echo "   [Condition] Model=${MODEL_NAME}, Scheme=${SCHEME}, BER=10^(-${EXP})"

            for (( i=0; i<NUM_SWEEPS; i++ )); do
                TORCH_SEED="${TORCH_SEEDS[$i]}"
                OUT_DIR="${RUN_ROOT}"
                mkdir -p "${OUT_DIR}"

                MODEL_ARGS="pretrained=${MODEL_PATH},error_injection_enable=True,error_prob=${EXP},ecc_enable=True,ecc_scheme=${SCHEME}"

                echo "      ---- Running Sweep $((i+1))/${NUM_SWEEPS} (Seed=${TORCH_SEED}) ----"
                echo "           Output: ${OUT_DIR}"

                lm_eval \
                    --model hf \
                    --model_args "${MODEL_ARGS}" \
                    --tasks "${TASKS}" \
                    --batch_size "${BATCH_SIZE}" \
                    --output_path "${OUT_DIR}" \
                    --device "${DEVICE}" \
                    --confirm_run_unsafe_code \
                    --seed "${PYTHON_SEED},${NUMPY_SEED},${TORCH_SEED},${FEWSHOT_SEED}"

                sleep 5
            done
        done
    done
done

echo "All experiments completed!"
touch "${NO_ERROR_FILE}"
echo "Success marker created: ${NO_ERROR_FILE}"
