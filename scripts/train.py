import os
import argparse

from data.mnist_base import DatasetType
from models.small_cnn import SmallCNN
from training.config import TrainConfig, LOG_DIR
from training.trainer import train_with_early_stopping
from utils import seed_everything, get_project_root
from data.loaders import parse_dataset, build_loaders


def main(dataset=DatasetType.MNIST):
    os.makedirs(LOG_DIR, exist_ok=True)
    seed_everything(42)

    train_cfg = TrainConfig()
    print(f"Dataset type: {dataset}")
    ckpt_path = train_cfg.ckpt_dir / f"{dataset}.pt"

    train_loader, val_loader, test_moving = build_loaders(
        seed=42,
        batch_size=train_cfg.batch_size,
        num_workers=train_cfg.num_workers,
        data_dir=get_project_root() / "data",
        dataset=dataset,
    )

    model = SmallCNN()
    model, history = train_with_early_stopping(
        model,
        train_loader,
        val_loader,
        train_cfg,
        ckpt_path
    )
    history.to_csv(LOG_DIR / "train_history.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train/evaluate SmallCNN on moving data built from MNIST or FashionMNIST."
    )
    parser.add_argument(
        "--dataset",
        type=parse_dataset,
        default=DatasetType.MNIST,
        help="Dataset variant: <fashion|mnist_old (cleaner)>.",
    )
    args = parser.parse_args()
    main(dataset=args.dataset)
