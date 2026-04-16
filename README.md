# MovingMNIST-OcclusionBench

Benchmarking CNN robustness to structured occlusions on Moving MNIST / FashionMNIST.

## Overview

This repo evaluates how different occlusion patterns affect classification performance and confidence. Masks are applied within digit motion regions and calibrated to a target coverage.

Includes:

* Moving digit generation (`RollingMovingMNIST`)
* Multiple occlusion types (Bernoulli, Perlin, CGOL, triangular, branches)
* Batched evaluation + summary metrics (accuracy, NLL, Brier, confidence)
* Optional semantic feature comparison

## Setup

```bash
conda create -n occ python=3.12
conda activate occ
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## Usage

### Train

```bash
python scripts/train.py --dataset <mnist|fashion>
```

### Evaluate

```bash
python scripts/evaluate.py <checkpoint.pt> --dataset <mnist|fashion> --subset-size <int>
```

Outputs:

* CSVs (per-video + summary)
* Plots in `outputs/plots/`

## Occlusions

* **Bernoulli** — random pixels
* **Perlin** — smooth noise
* **CGOL** — cellular automata
* **Triangular** — geometric shapes
* **Branches** — connected line structures

All matched to target coverage (±2%).