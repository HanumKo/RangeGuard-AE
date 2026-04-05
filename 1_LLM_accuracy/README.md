# 1_LLM_accuracy

This directory contains the code used to reproduce the LLM-side accuracy evaluation for RangeGuard.

The code in `lm-evaluation-harness/` is a modified version of EleutherAI's `lm-evaluation-harness`. The original project is available at:

- https://github.com/EleutherAI/lm-evaluation-harness

RangeGuard-related ECC simulation logic has been added under `lm-evaluation-harness/lm_eval/models/eccsim/`.

The results produced by this directory correspond to Figure 8(b) and Figure 8(c) in the paper.

## Setup
From this directory, install the local modified `lm-evaluation-harness` package:

```bash
cd lm-evaluation-harness
pip install -e .
```

## Model Preparation
Before running the experiment, the following models should be prepared under:

```txt
pretrained_models/
```

Expected model directories:

- `pretrained_models/Llama-3.2-1B`
- `pretrained_models/Llama-3.1-8B`

Reference model pages:

- https://huggingface.co/meta-llama/Llama-3.2-1B
- https://huggingface.co/meta-llama/Llama-3.1-8B

You can either:

- download these models in advance into `pretrained_models/`, or
- modify `MODEL_PATH` usage in the script so that `lm-evaluation-harness` loads the model remotely from Hugging Face

Because `lm-evaluation-harness` supports remote model loading, it is also possible to use a Hugging Face model identifier directly through `--model_args` instead of a local path.

## Running
The main entry point is:

- `lm-evaluation-harness/run.sh`

This script evaluates two models:

- `Llama-3.2-1B`
- `Llama-3.1-8B`

It runs the `arc_easy` task and sweeps:

- multiple ECC schemes
- BER from `10^-10` to `10^-4`
- 100 repeated runs per condition

Run it with:

```bash
cd lm-evaluation-harness
bash run.sh
```

## Output
Experiment outputs are written under:

```txt
output/
```

The output structure is organized by model, task, ECC scheme, and BER.

## Figure Generation
After the evaluation finishes, you can convert the result JSON files into CSV files and generate the LLM accuracy figures from `figure/`.

First, aggregate the JSON outputs into per-model CSV files:

```bash
cd figure
python json2csv.py
```

By default, this script scans:

```txt
output/
```

and writes CSV files under:

```txt
figure/source_csv/
```

Then generate the plots:

```bash
cd figure
python make_figure.py
```

By default, the generated figures are written under:

```txt
figure/output/
```

The plotting script expects the CSV files for:

- `Llama-3.2-1B`
- `Llama-3.1-8B`

to be present in `figure/source_csv/`.

## Path Configuration
Directory-related paths are configured in:

- `../env.sh`

By default, `env.sh` sets:

- `MODEL_ROOT=1_LLM_accuracy/pretrained_models`
- `OUTPUT_ROOT=1_LLM_accuracy/output`

If the repository is cloned to another location, update `env.sh` or override the exported path variables before running the script.
