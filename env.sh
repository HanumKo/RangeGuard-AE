#!/bin/bash

# Resolve paths relative to the repository root so the artifact remains
# portable after `git clone` into any directory.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export LLM_ACCURACY_ROOT="${LLM_ACCURACY_ROOT:-${REPO_ROOT}/1_LLM_accuracy}"
export MODEL_ROOT="${MODEL_ROOT:-${LLM_ACCURACY_ROOT}/pretrained_models}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${LLM_ACCURACY_ROOT}/output}"
