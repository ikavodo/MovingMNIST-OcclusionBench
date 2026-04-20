#!/usr/bin/env python
import argparse
import os
from pathlib import Path

import torch

from data.mnist_base import DatasetType
from models.small_cnn import SmallCNN
from training.config import EvalConfig
from training.evaluation import load_best_model, evaluate_occlusion_sweep
from utils import seed_everything, get_project_root, plot_results

from data.loaders import (
    OUT_DIR,
    DEFAULT_LOADER_BATCH_SIZE,
    DEFAULT_EVAL_BATCH_SIZE,
    DEFAULT_NUM_WORKERS,
    parse_dataset,
    build_test_dataset,
)


def run_occlusion_eval(
        model,
        test_moving,
        out_dir=OUT_DIR,
        device=None,
        eval_batch_size=DEFAULT_EVAL_BATCH_SIZE,
        num_workers=DEFAULT_NUM_WORKERS,
        progress_bar = True
):
    p_values = torch.linspace(0.1, 0.9, 9)

    df, summary = evaluate_occlusion_sweep(
        model,
        test_moving,
        p_values=p_values,
        out_csv=out_dir / "occlusion_eval.csv",
        eval_cfg=EvalConfig(k_frames=5, n_mask_seeds=3, static_masks=True),
        device=device,
        batch_size=eval_batch_size,
        num_workers=num_workers,
        progress_bar = progress_bar
    )

    plot_results(
        df,
        out_dir=out_dir / "plots",
    )

    return df, summary


def main(checkpoint_path, subset_size=128, dataset=DatasetType.MNIST, progress_bar=True):
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
    print(f"Dataset type: {dataset.value}")

    test_moving = build_test_dataset(
        seed=42,
        subset_size=subset_size,
        num_workers=DEFAULT_NUM_WORKERS,
        data_dir=repo_root / "data",
        dataset=dataset,
    )

    model = load_best_model(SmallCNN, checkpoint_path, device=device)
    out_dir_dataset = OUT_DIR / dataset
    os.makedirs(out_dir_dataset, exist_ok=True)

    run_occlusion_eval(
        model,
        test_moving,
        out_dir=out_dir_dataset,
        device=device,
        eval_batch_size=DEFAULT_EVAL_BATCH_SIZE,
        num_workers=DEFAULT_NUM_WORKERS,
        progress_bar=progress_bar
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run occlusion evaluation on a trained model."
    )
    parser.add_argument(
        "checkpoint_dir",
        type=str,
        help="Path to model checkpoint directory, absolute or relative to repo root.",
    )
    parser.add_argument(
        "--subset-size",
        type=int,
        default=128,
        help="Number of test videos to evaluate.",
    )
    parser.add_argument(
        "--dataset",
        type=parse_dataset,
        default=DatasetType.MNIST,
        help="MNIST variant: <fashion|mnist_old (cleaner)>.",
    )
    args = parser.parse_args()
    # find relevant checkpoint for dataset
    checkpoint_path = Path(args.checkpoint_dir) / f"{args.dataset}.pt"
    main(
        checkpoint_path=checkpoint_path,
        subset_size=args.subset_size,
        dataset=args.dataset,
    )
