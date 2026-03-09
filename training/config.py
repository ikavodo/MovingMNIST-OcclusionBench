from dataclasses import dataclass
from pathlib import Path
import torch
from utils import get_project_root

ROOT = get_project_root()
LOG_DIR = ROOT / "runs" / "moving_mnist"
CKPT_DIR = LOG_DIR / "checkpoints"


@dataclass
class TrainConfig:
    batch_size: int = 256
    lr: float = 1e-3
    max_epochs: int = 30
    patience: int = 5
    num_workers: int = 2
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    log_dir: Path = LOG_DIR
    ckpt_path: Path = CKPT_DIR / "best_smallcnn.pt"
