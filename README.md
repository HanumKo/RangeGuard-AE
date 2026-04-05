# RangeGuard Artifact

This repository contains the artifact used to reproduce the currently maintained experimental results of RangeGuard.

The repository is organized into two main parts:

- `1_LLM_accuracy/`: LLM-side accuracy evaluation. This directory reproduces Figure 8(b) and Figure 8(c).
- `2_ECC_coverage/`: ECC coverage analysis using a standalone C++ implementation.

## Repository Structure
- `1_LLM_accuracy/`
  Contains a modified version of `lm-evaluation-harness` with RangeGuard ECC simulation support.
- `2_ECC_coverage/`
  Contains the ECC coverage analysis code and run scripts.
- `env.sh`
  Contains repository-relative path settings used by the experiment scripts.

## Setup
Directory-related paths are configured in:

- `env.sh`

Before running the experiments, review `env.sh` and prepare the required models and datasets in the expected directories.

Detailed setup instructions are provided in:

- `1_LLM_accuracy/README.md`
- `2_ECC_coverage/README.md`

## Recommended Order
The recommended order for reviewing or running the artifact is:

1. `1_LLM_accuracy`
2. `2_ECC_coverage`

## Entry Points
Main run scripts:

- `1_LLM_accuracy/lm-evaluation-harness/run.sh`
- `2_ECC_coverage/ecc_coverage/run.sh`

## Outputs
Outputs are generated inside each top-level experiment directory:

- `1_LLM_accuracy/output/`
- `2_ECC_coverage/ecc_coverage/output/`

For details on output formats and expected files, see the README in each subdirectory.
