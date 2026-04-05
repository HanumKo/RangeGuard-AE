# 2_ECC_coverage

This directory contains the code used to reproduce the ECC coverage analysis for RangeGuard.

The code in `ecc_coverage/` builds a standalone C++ executable that evaluates coverage across error patterns for multiple ECC schemes.

The results produced by this directory correspond to Table III in the paper.

## Setup
From this directory, build the coverage analysis binary:

```bash
cd ecc_coverage
bash build_ecc_cov.sh
```

This produces:

- `ecc_coverage/build/ecc_cov`

## Running
The main entry points are:

- `ecc_coverage/run.sh`

Run the experiment with:

```bash
cd ecc_coverage
bash run.sh
```

By default, `run.sh` uses `TRIALS=1000000000`, so a full run can take a long time.
For a quick smoke test, you can use:

```bash
cd ecc_coverage
TRIALS=1000 SEED=1 bash run.sh
```

## Output
Outputs are written under:

```txt
ecc_coverage/output/
```
