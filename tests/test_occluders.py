import pytest
import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from occluders import (
    Occ, apply_mask_to_video, build_mask_within_bounds, compute_motion_bounds
)
from data.moving_mnist import MovingMNIST
from data.mnist_base import build_mnist_splits
from utils import get_project_root

TOL = 2e-2
P_VALUES = [0.1, 0.3, 0.5, 0.7, 0.9]


@pytest.fixture(scope="module")
def dummy_video():
    """Single video with batch dimension."""
    root = get_project_root() / "data"
    train, _, _ = build_mnist_splits(root)
    ds = MovingMNIST(train, T=20, canvas=64, seed=42)
    video, label, _, _ = ds[0]
    return video.unsqueeze(0)  # [1,T,1,64,64]


def test_compute_motion_bounds(dummy_video):
    video = dummy_video
    bounds = compute_motion_bounds(video)  # [B,4]
    assert bounds.shape == (1, 4)
    h0, h1, w0, w1 = bounds[0].tolist()
    # Bounds should be within canvas and non‑empty
    assert 0 <= h0 <= h1 < 64
    assert 0 <= w0 <= w1 < 64
    # The digit should be inside these bounds (digit not empty)
    region = video[0, 0, 0, h0:h1 + 1, w0:w1 + 1]
    assert region.mean() > 0


@pytest.mark.parametrize("occ", list(Occ))
def test_occluder_density_within_bounds(dummy_video, occ):
    video = dummy_video
    for p in P_VALUES:
        masked, mask = apply_mask_to_video(video, occ, p, seed=123)
        bounds = compute_motion_bounds(video)
        h0, h1, w0, w1 = bounds[0].tolist()
        region = mask[0, 0, 0, h0:h1 + 1, w0:w1 + 1]
        realized = region.float().mean().item()
        assert abs(realized - p) <= 2 * TOL, f"{occ.name} p={p} gave realized={realized:.4f}"


@pytest.mark.parametrize("occ", list(Occ))
def test_occluder_mask_binary(dummy_video, occ):
    video = dummy_video
    for p in P_VALUES:
        masked, mask = apply_mask_to_video(video, occ, p, seed=123)
        # Mask should be 0 or 1
        unique = mask.unique().cpu().tolist()
        assert all(u in [0, 1] for u in unique), f"{occ.name} mask has values {unique}"


def test_build_mask_within_bounds_no_motion():
    """Edge case: video with no motion (digit stuck)."""
    # Create a video where digit is always at same location (vx=vy=0)
    # We'll manually create a simple video
    H, W = 64, 64
    T = 10
    img = torch.zeros(1, H, W)
    img[0, 20:48, 20:48] = 0.5  # fixed digit
    video = img.unsqueeze(0).repeat(T, 1, 1, 1).unsqueeze(0)  # [1,T,1,H,W]
    # compute_motion_bounds should return non‑empty
    bounds = compute_motion_bounds(video)
    h0, h1, w0, w1 = bounds[0].tolist()
    assert h0 < h1 and w0 < w1
    # Build mask within bounds
    mask2d = build_mask_within_bounds(video, Occ.BERNOULLI, 0.3, seed=7)
    # Check that mask is only inside bounds (outside should be zero)
    outside_top = mask2d[0, :h0, :].sum()
    outside_bottom = mask2d[0, h1 + 1:, :].sum()
    outside_left = mask2d[0, :, :w0].sum()
    outside_right = mask2d[0, :, w1 + 1:].sum()
    assert outside_top == 0
    assert outside_bottom == 0
    assert outside_left == 0
    assert outside_right == 0
