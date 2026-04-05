#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${ROOT_DIR}/build"
BIN="${BUILD_DIR}/ecc_cov"
: "${CXX:=g++}"
: "${CXXFLAGS:=-std=c++17 -O3 -DNDEBUG}"

mkdir -p "${BUILD_DIR}"

"${CXX}" ${CXXFLAGS} \
  -I"${ROOT_DIR}/include" \
  -I"${ROOT_DIR}/src" \
  "${ROOT_DIR}/src/main.cpp" \
  "${ROOT_DIR}/src/hbm3_crc_ssc.cpp" \
  "${ROOT_DIR}/src/secded.cpp" \
  "${ROOT_DIR}/src/weight_nulling.cpp" \
  "${ROOT_DIR}/src/vapi_dec78.cpp" \
  "${ROOT_DIR}/src/rs_10_8.cpp" \
  "${ROOT_DIR}/src/rs_12_8.cpp" \
  -o "${BIN}"

echo "Built ${BIN}"
