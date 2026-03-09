import os
import torch
from torch.utils.data import DataLoader, Subset
import argparse
from data import build_mnist_splits, MovingMNIST, MovingMNISTFrames
from models.small_cnn import SmallCNN
from training.config import TrainConfig
from training.trainer import train_with_early_stopping
from training.evaluation import evaluate_occlusion_sweep, load_best_model
from occluders import EvalConfig
from utils import seed_everything, get_project_root, plot_occlusion_results

OUT_DIR = get_project_root() / "outputs"


def build_loaders(seed: int = 0, batch_size: int = 256, num_workers: int = 2, data_dir=None):
    if data_dir is None:
        data_dir = get_project_root() / "data"
    train_base, val_base, test_base = build_mnist_splits(data_dir)
    train_moving = MovingMNIST(train_base, T=20, canvas=64, seed=1000 + seed)
    val_moving = MovingMNIST(val_base, T=20, canvas=64, seed=2000 + seed)
    test_moving = MovingMNIST(test_base, T=20, canvas=64, seed=3000 + seed)

    # Use only a single batch of test videos to keep evaluation manageable
    test_subset = Subset(test_moving, range(min(batch_size, len(test_moving))))

    train_frames = MovingMNISTFrames(train_moving, frame_seed=10_000 + seed)
    val_frames = MovingMNISTFrames(val_moving, frame_seed=20_000 + seed)

    train_loader = DataLoader(
        train_frames, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_frames, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    return train_loader, val_loader, test_subset


def main(train=False):
    os.makedirs(OUT_DIR, exist_ok=True)
    seed_everything(42)
    train_cfg = TrainConfig()
    train_loader, val_loader, test_moving = build_loaders(
        seed=42,
        batch_size=train_cfg.batch_size,
        num_workers=train_cfg.num_workers,
        data_dir=get_project_root() / "data"
    )

    model = SmallCNN()
    if train:
        model, history = train_with_early_stopping(model, train_loader, val_loader, train_cfg)
        history_path = OUT_DIR / "train_history.csv"
        history.to_csv(history_path, index=False)
    else:
        model = load_best_model(SmallCNN, train_cfg.ckpt_path)

    p_values = torch.linspace(0.1, 0.9, 9)
    evaluate_occlusion_sweep(
        model,
        test_moving,
        p_values=p_values,
        out_csv=OUT_DIR / "occlusion_eval.csv",
        eval_cfg=EvalConfig(k_frames=5, n_mask_seeds=3, static_masks=True),
    )
    plot_occlusion_results(
        per_video_csv=OUT_DIR / "occlusion_eval_per_video.csv",
        out_dir=OUT_DIR / "plots",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train the model from scratch")
    args = parser.parse_args()
    main(train=args.train)
