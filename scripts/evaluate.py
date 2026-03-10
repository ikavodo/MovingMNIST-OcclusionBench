#!/usr/bin/env python
"""
Load a trained model checkpoint and run the occlusion evaluation + plotting.
No training is performed.
"""
import argparse

import pandas as pd
import torch

from models.small_cnn import SmallCNN
from training.config import EvalConfig
from training.evaluation import evaluate_occlusion_sweep, load_best_model
from utils import seed_everything, get_project_root, plot_occlusion_results
from train import build_loaders  # reuse data loader builder

OUT_DIR = get_project_root() / "outputs"


def main(
        checkpoint_path,
        loader_batch_size=64,
        eval_batch_size=16,
        num_workers=4,
):
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build test dataset only
    _, _, test_moving = build_loaders(
        seed=42,
        batch_size=loader_batch_size,
        num_workers=num_workers,
        test_subset_size=64,
        data_dir=get_project_root() / "data",
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
        batch_size=eval_batch_size,
        num_workers=num_workers,
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
    parser = argparse.ArgumentParser(description="Run occlusion evaluation on a trained model.")
    parser.add_argument(
        "checkpoint",
        type=str,
        help="Path to model checkpoint (.pt file)",
    )
    parser.add_argument(
        "--loader-batch-size",
        type=int,
        default=128,
        help="Batch size used when building loaders/subsets.",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=16,
        help="Batch size used inside evaluate_occlusion_sweep. Lower this first if you hit CUDA OOM.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of DataLoader workers.",
    )
    args = parser.parse_args()

    main(
        checkpoint_path=args.checkpoint,
        loader_batch_size=args.loader_batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
    )
