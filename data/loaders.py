import argparse
from torch.utils.data import DataLoader, Subset

from data import build_mnist_splits, MovingMNIST, MovingMNISTFrames
from data.mnist_base import DatasetType
from utils import get_project_root

OUT_DIR = get_project_root() / "outputs"

DEFAULT_LOADER_BATCH_SIZE = 64
DEFAULT_EVAL_BATCH_SIZE = 16
DEFAULT_NUM_WORKERS = 4


def parse_dataset(s: str) -> DatasetType:
    try:
        return DatasetType(s.lower())
    except ValueError:
        valid = ", ".join(e.value for e in DatasetType)
        raise argparse.ArgumentTypeError(
            f"Invalid dataset '{s}'. Valid options: {valid}"
        )


def build_loaders(
        seed=0,
        batch_size=256,
        num_workers=2,
        test_subset_size=64,
        data_dir=None,
        dataset=DatasetType.MNIST,
):
    if data_dir is None:
        data_dir = get_project_root() / "data"

    train_base, val_base, test_base = build_mnist_splits(data_dir, dataset=dataset)

    train_moving = MovingMNIST(train_base, T=20, canvas=64, seed=1000 + seed)
    val_moving = MovingMNIST(val_base, T=20, canvas=64, seed=2000 + seed)
    test_moving = MovingMNIST(test_base, T=20, canvas=64, seed=3000 + seed)

    test_subset = Subset(test_moving, range(min(test_subset_size, len(test_moving))))
    train_frames = MovingMNISTFrames(train_moving, frame_seed=10_000 + seed)
    val_frames = MovingMNISTFrames(val_moving, frame_seed=20_000 + seed)

    train_loader = DataLoader(
        train_frames,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_frames,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_subset


def build_test_dataset(
        seed=42,
        subset_size=64,
        num_workers=DEFAULT_NUM_WORKERS,
        data_dir=None,
        dataset=DatasetType.MNIST,
):
    _, _, test_moving = build_loaders(
        seed=seed,
        batch_size=DEFAULT_LOADER_BATCH_SIZE,
        num_workers=num_workers,
        test_subset_size=subset_size,
        data_dir=data_dir,
        dataset=dataset,
    )
    return test_moving
