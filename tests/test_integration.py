import pytest
from torch.utils.data import Subset
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.mnist_base import build_mnist_splits
from data.moving_mnist import MovingMNIST
from models.small_cnn import SmallCNN
from training.evaluation import evaluate_occlusion_sweep
from occluders import Occ
from utils import get_project_root
from training.config import EvalConfig


@pytest.fixture(scope="module")
def small_setup():
    root = get_project_root() / "data"
    train, _, test = build_mnist_splits(root)
    # Use a tiny test set (first 2 videos)
    test_small = Subset(test, range(2))
    moving_test = MovingMNIST(test_small, T=10, canvas=64, seed=42)
    # Create a quick untrained model
    model = SmallCNN()
    return model, moving_test


def test_evaluate_sweep_smoke(small_setup, tmp_path):
    model, moving_test = small_setup
    device = "cpu"
    p_values = [0.3, 0.5]  # just two p values
    eval_cfg = EvalConfig(k_frames=3, n_mask_seeds=1, static_masks=True)
    out_csv = tmp_path / "test_eval.csv"
    df, summary = evaluate_occlusion_sweep(
        model, moving_test, p_values,
        out_csv=out_csv,
        device=device,
        eval_cfg=eval_cfg,
        batch_size=2,  # small batch
        num_workers=0
    )
    # Check that we got results
    assert len(df) > 0
    expected_cols = {"video_idx", "occ", "coverage_target", "video_correct",
                     "sem_cos_conv1", "mask_coverage_realized"}
    assert expected_cols.issubset(df.columns)
    # Check summary
    assert "video_acc" in summary.columns
    assert summary.shape[0] == len(p_values) * len(Occ)  # Occ is the Enum, not an attribute of EvalConfig
    # Verify CSV was written
    assert out_csv.with_name(out_csv.stem + "_per_video.csv").exists()
    assert out_csv.with_name(out_csv.stem + "_summary.csv").exists()
