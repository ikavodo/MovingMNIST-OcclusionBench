import torchvision
from torchvision import transforms
from torch.utils.data import Subset
from enum import StrEnum, auto


class DatasetType(StrEnum):
    FASHION = auto()
    MNIST = auto()  # DEFAULT


def build_mnist_splits(data_dir, dataset=DatasetType.MNIST):
    if dataset == DatasetType.MNIST:
        tfm = transforms.ToTensor()

        train_full = torchvision.datasets.MNIST(root=data_dir, train=True, download=True, transform=tfm)
        test = torchvision.datasets.MNIST(root=data_dir, train=False, download=True, transform=tfm)
    elif dataset == DatasetType.FASHION:
        tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        train_full = torchvision.datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=tfm)
        test = torchvision.datasets.FashionMNIST(root=data_dir, train=False, download=True, transform=tfm)
    else:
        raise ValueError(f"Unknown dataset type: {dataset}, options are <fashion|mnist>")

    n_total = len(train_full)
    n_val = len(test)
    train_idx = list(range(0, n_total - n_val))
    val_idx = list(range(n_total - n_val, n_total))
    train = Subset(train_full, train_idx)
    val = Subset(train_full, val_idx)
    return train, val, test
