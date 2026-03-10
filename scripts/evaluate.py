#!/usr/bin/env python
"""
Load a trained model checkpoint and run the occlusion evaluation + plotting.
No training is performed.
"""
import argparse
from pathlib import Path

import pandas as pd
import torch

from models.small_cnn import SmallCNN
from training.config import EvalConfig
from training.evaluation import evaluate_occlusion_sweep, load_best_model
from utils import seed_everything, get_project_root, plot_occlusion_results
from train import build_loaders  # reuse data loader builder

OUT_DIR = get_project_root() / "outputs"

# Internal defaults chosen for a 16 GB GPU / reasonable CPU usage
DEFAULT_LOADER_BATCH_SIZE = 64
DEFAULT_EVAL_BATCH_SIZE = 16
DEFAULT_NUM_WORKERS = 4


def main(
        checkpoint_path,
        subset_size=128,
):
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    repo_root = get_project_root()
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_absolute():
        checkpoint_path = repo_root / checkpoint_path

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Using device: {device}")
    print(f"Resolved checkpoint: {checkpoint_path}")
    print(f"Evaluation subset size: {subset_size}")
    print(f"Internal loader batch size: {DEFAULT_LOADER_BATCH_SIZE}")
    print(f"Internal eval batch size: {DEFAULT_EVAL_BATCH_SIZE}")
    print(f"Num workers: {DEFAULT_NUM_WORKERS}")

    # Build test dataset only
    _, _, test_moving = build_loaders(
        seed=42,
        batch_size=DEFAULT_LOADER_BATCH_SIZE,
        num_workers=DEFAULT_NUM_WORKERS,
        test_subset_size=subset_size,
        data_dir=repo_root / "data",
    )

    model = load_best_model(SmallCNN, checkpoint_path, device=device)

    p_values = torch.linspace(0.1, 0.9, 9)
    df, summary = evaluate_occlusion_sweep(
        model,
        test_moving,
        p_values=p_values,
        out_csv=OUT_DIR / "occlusion_eval.csv",
        eval_cfg=EvalConfig(k_frames=5, n_mask_seeds=3, static_masks=True),
        device=device,
        batch_size=DEFAULT_EVAL_BATCH_SIZE,
        num_workers=DEFAULT_NUM_WORKERS,
    )

    plot_occlusion_results(
        df,
        out_dir=OUT_DIR / "plots",
    )


def plot_from_csv(in_csv):
    df = pd.read_csv(in_csv)
    plot_occlusion_results(
        df,
        out_dir=OUT_DIR / "plots",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run occlusion evaluation on a trained model."
    )
    parser.add_argument(
        "checkpoint",
        type=str,
        help="Path to model checkpoint (.pt file), absolute or relative to repo root.",
    )
    parser.add_argument(
        "--subset-size",
        type=int,
        default=64,
        help="Number of test videos to evaluate.",
    )
    args = parser.parse_args()

    main(
        checkpoint_path=args.checkpoint,
        subset_size=args.subset_size,
    )
