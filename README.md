# Adversarial image detection from vision-language response profiles

This repository contains the complete reproduction pipeline for a detector that
uses a frozen vision-language model's image-text response profile. The
vision-language backbone and prompt embeddings remain frozen; only the
lightweight detector is trained.

## One-command reproduction

Python 3.11 or newer and a CUDA GPU are recommended. Create and activate a
virtual environment, install the dependencies, and run the suite without
arguments:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -u reproduce.py
```

On Windows PowerShell, activate the environment with:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -u reproduce.py
```

The runner downloads missing datasets and pretrained weights automatically. It
uses the complete public datasets, creates deterministic 70/15/15 partitions,
and executes the three seeds in reverse order: **44, 43, 42**.

## Complete experiment matrix

Every reported detector uses the same frozen image encoder, deterministic split,
attack configuration, and seed assignment.

- **Datasets:** CIFAR-10, STL-10, Tiny-ImageNet.
- **Backbones:** OpenCLIP ViT-B/16, ViT-B/32, and ConvNeXt-Base-W.
- **Attacks:** PGD, MI-FGSM, Flow, AutoAttack, and Auto-PGD-CE.
- **Detectors:** response profile, maximum softmax probability (MSP), KNN,
  Mahalanobis distance, KDE, and LID.
- **Seeds:** 44, 43, and 42.
- **Ablations:** seven response-profile feature variants on CIFAR-10 with
  ViT-B/16 and PGD, repeated for all three seeds.

For each dataset/backbone/seed combination, the response-profile detector is
trained once with PGD and evaluated unchanged on all five attacks. The four
embedding-geometry baselines select their hyperparameters and thresholds with
clean and PGD validation embeddings, then reuse them unchanged for the four
other attacks. The Auto-PGD evaluation therefore reuses the already fitted PGD
detectors; it does not create an Auto-PGD-specific detector.

The no-argument plan contains 197 resumable jobs. These jobs produce 135
attack evaluations for the response-profile detector, 135 for MSP, 540 for the
four geometry baselines, 21 ablation measurements, the illustrative figures,
and conservative cleanup records.

## Full datasets and deterministic splits

No fractional dataset mode exists in the reproduction code. Official labeled
training and test/validation pools are combined before a seeded random
permutation creates the following disjoint partitions:

| Dataset | Training | Validation | Test |
|---|---:|---:|---:|
| CIFAR-10 | 42,000 | 9,000 | 9,000 |
| STL-10 | 9,100 | 1,950 | 1,950 |
| Tiny-ImageNet | 77,000 | 16,500 | 16,500 |

CIFAR-10 and STL-10 are downloaded through torchvision. Tiny-ImageNet is
downloaded over HTTPS, validated, extracted safely, and reused on later runs.
Set `REPRODUCTION_DATA_DIR` only when a different persistent dataset directory
is needed.

## Results, caches, and resuming

All metrics are stored in one atomic JSON file:

```text
outputs/reproduction/results.json
```

The previous valid document is retained as `results.backup.json`. Each completed
job is keyed by its full configuration. Stop with `Ctrl+C` and rerun the same
command; completed jobs are skipped, and incomplete adversarial-generation
shards are resumed when possible.

Downloaded datasets remain under `data/`. Pretrained weights use the Hugging
Face and Torch caches. Detector bundles, fitted baseline state, score arrays,
figures, and resumable attack shards are kept under the output tree until their
dependent results are safely recorded. Cleanup never removes datasets,
`results.json`, detector bundles, score arrays, figures, logs, or configuration
files.

Useful path overrides are:

```bash
export REPRODUCTION_OUTPUT_DIR=/persistent/path/outputs/reproduction
export REPRODUCTION_DATA_DIR=/persistent/path/data
export HF_HOME=/persistent/path/cache/huggingface
export TORCH_HOME=/persistent/path/cache/torch
export REPRODUCTION_MODEL_CACHE_DIR=/persistent/path/cache/validated_models
export CUDA_DEVICE_INDEX=0
```

`REPRODUCTION_SEEDS` may assign a reverse-ordered subset of the canonical seeds
to one machine, for example `44` or `43,42`. Running `reproduce.py` without that
environment variable always executes all three seeds.

## Linux server

The supplied launcher selects CUDA device 0, retains persistent data/model
caches, and runs the complete matrix:

```bash
source .venv/bin/activate
tmux new -s response-profile
bash run_server.sh
```

Detach with `Ctrl+B`, then `D`. Reattach with:

```bash
tmux attach -t response-profile
```

Progress is also available in
`outputs/reproduction/reproduction.log`. Closing the SSH terminal or local
computer does not stop a detached tmux session.

## Method and attack settings

The response profile contains 43 features: category statistics, prompt-graph
measurements, clean-reference residuals, graph-diffusion features,
transformation stability, and the raw image-embedding norm. The MLP has hidden
widths 256/128/64. Validation ROC-AUC selects its checkpoint, and validation F1
selects the decision threshold.

PGD and MI-FGSM use random starts. CIFAR-10 and Tiny-ImageNet use an
`L_inf` budget of 0.04, step size 0.01, and 20 iterations; STL-10 uses 0.03,
0.0075, and 10 iterations. MI-FGSM uses momentum 1. Flow uses the corresponding
dataset budget/step/iteration settings and smoothness weight 0.05. AutoAttack
uses batch size 32. Auto-PGD-CE uses 100 steps, one restart, and the same
dataset-specific `L_inf` budget.

## Verification

Run all fast offline tests with:

```bash
python -m unittest discover -s tests -v
```

The tests check the reported feature dimension, clean-reference covariance,
attack budget, full-dataset split behavior, complete experiment matrix,
resumable attack shards, and atomic result recovery.
