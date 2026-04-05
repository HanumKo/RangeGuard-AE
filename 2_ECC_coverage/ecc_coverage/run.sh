#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="${ROOT_DIR}/build/ecc_cov"
BUILD_SCRIPT="${ROOT_DIR}/build_ecc_cov.sh"
OUT_DIR="${ROOT_DIR}/output"
TRIALS="${TRIALS:-1000000000}"
SEED="${SEED:-123}"

if [[ ! -x "${BIN}" ]]; then
  echo "Missing executable: ${BIN}" >&2
  echo "Attempting local rebuild via ${BUILD_SCRIPT}" >&2
  bash "${BUILD_SCRIPT}"
fi

mkdir -p "${OUT_DIR}"

MASTER_CSV="${OUT_DIR}/coverage_1b_all_cases.csv"
MASTER_LOG="${OUT_DIR}/run_1b_all_cases.log"

: > "${MASTER_LOG}"
echo "Running all cases with trials=${TRIALS}" | tee -a "${MASTER_LOG}"
echo "Seed: ${SEED}" | tee -a "${MASTER_LOG}"
echo "Output directory: ${OUT_DIR}" | tee -a "${MASTER_LOG}"
echo "CSV: ${MASTER_CSV}" | tee -a "${MASTER_LOG}"
echo "Start: $(date -u '+%Y-%m-%d %H:%M:%S UTC')" | tee -a "${MASTER_LOG}"

"${BIN}" \
  --trials "${TRIALS}" \
  --seed "${SEED}" \
  --patterns all \
  --csv "${MASTER_CSV}" | tee -a "${MASTER_LOG}"

echo "End: $(date -u '+%Y-%m-%d %H:%M:%S UTC')" | tee -a "${MASTER_LOG}"
