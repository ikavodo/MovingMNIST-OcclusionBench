# MovingMNIST-OcclusionBench

Evaluating classifier robustness to various occlusions masks applied over Moving MNIST.

## Overview

This repository provides a framework to test how different structured occlusion patterns (Bernoulli, Perlin, CGOL,
triangular, branches) affect the performance of a simple CNN classifier on the Moving MNIST dataset. Occlusions are
applied only within the motion bounds of the digit, and density is controlled to match a target coverage.

Key features:

- Deterministic video generation with bouncing digits (`RollingMovingMNIST`)
- Five occlusion families with precise density control
- Batched evaluation for speed
- Semantic feature loss computation via network hooks
- Comprehensive plotting and metrics (ECE, reliability diagrams, etc.)

## Project Structure

```
.
├── data/                # MNIST data, MovingMNIST datasets
├── loss/                # Metrics and semantic loss
├── models/              # SmallCNN definition
├── training/            # Training loop, evaluation, configs
├── scripts/             # Evaluation runner (no training)
├── occluders.py         # All occlusion mask generators
├── utils.py             # Plotting helpers, seeding, paths
├── train.py             # Main training script
├── tests/               # Unit tests
├── outputs/             # Generated plots and CSVs
└── training_runs/       # Checkpoints and logs
```

## Installation

Create a conda environment (Python 3.12) and install dependencies:

```bash
# Create and activate conda environment
conda create -n occ python=3.12
conda activate occ

# Install dependencies from requirements.txt
pip install -r requirements.txt

# For CUDA support (if needed), first install PyTorch with CUDA:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# Then install the rest:
pip install -r requirements.txt
```

Or using pip-only (without conda):

```bash
# Create virtual environment
python -m venv occ_env
source occ_env/bin/activate  # On Windows: occ_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

Note: If you need CUDA support, install PyTorch with CUDA first (see comment above), then install the remaining
dependencies.

## Usage

### Train a model

```bash
python train.py --train
```

This trains a SmallCNN on moving MNIST frames and saves the best checkpoint to
`training_runs/moving_mnist/checkpoints/`.

### Evaluate a trained model

```bash
python scripts/evaluate.py path/to/checkpoint.pt
```

Runs occlusion sweep and generates plots in `outputs/plots/`.

## Testing

Run unit tests:

```bash
pytest tests/ -v
```

## Occlusion Families

- **BERNOULLI**: Independent pixel masking.
- **CGOL**: Cellular automaton (Game of Life) patterns.
- **TRIANGULAR**: Overlapping random triangles.
- **PERLIN**: Perlin noise thresholded.
- **BRANCHES**: Random tree‑like line segments.

All masks are tuned to achieve the target coverage within a 2% tolerance.
See [notebook](https://github.com/ikavodo/MovingMNIST-OcclusionBench/tree/master/notebooks/occ_moving_mnist.ipynb) 
for examples of masks and occluded videos. 