import pytest
import torch
from torch.utils.data import Subset
import sys
from pathlib import Path

from training.evaluation import take_frames_batched

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.mnist_base import build_mnist_splits
from data.moving_mnist import MovingMNIST, MovingMNISTFrames
from utils import get_project_root, take_frames, integrate_frames


@pytest.fixture(scope="module")
def mnist_small():
    """Get a tiny MNIST subset (first 10 train + 10 test) to speed up tests."""
    root = get_project_root() / "data"
    train, val, test = build_mnist_splits(root)
    # Use only first 10 images of each split
    train_small = Subset(train, range(10))
    val_small = Subset(val, range(10))
    test_small = Subset(test, range(10))
    return train_small, val_small, test_small


def test_rolling_mnist_basic(mnist_small):
    train, _, _ = mnist_small
    ds = MovingMNIST(train, T=20, canvas=64, vmin=-3, vmax=3, seed=42)
    video, label, shifts, _ = ds[0]
    # Check shapes
    assert video.shape == (20, 1, 64, 64)
    assert isinstance(label, int)
    assert shifts.shape == (19, 2)
    # Check video values are in [0,1]
    assert video.min() >= 0.0 and video.max() <= 1.0
    # Check shifts are integers within velocity range
    assert shifts.dtype == torch.long
    assert shifts.abs().max() <= 3


def test_rolling_mnist_deterministic(mnist_small):
    train, _, _ = mnist_small
    ds1 = MovingMNIST(train, T=20, canvas=64, seed=42)
    ds2 = MovingMNIST(train, T=20, canvas=64, seed=42)
    v1, l1, s1, _ = ds1[5]
    v2, l2, s2, _ = ds2[5]
    assert torch.allclose(v1, v2)
    assert l1 == l2
    assert torch.all(s1 == s2)


def test_moving_mnist_frames(mnist_small):
    train, _, _ = mnist_small
    moving_ds = MovingMNIST(train, T=20, canvas=64, seed=42)
    frame_ds = MovingMNISTFrames(moving_ds, frame_seed=0)
    x, y, meta = frame_ds[3]
    assert x.shape == (1, 64, 64)
    assert isinstance(y, int)
    assert "base_idx" in meta and "frame_index" in meta
    assert 0 <= meta["frame_index"] < 20


def test_take_frames():
    video = torch.randn(10, 3, 32, 32)
    k = 4
    out, idx = take_frames(video, k)
    assert out.shape == (k, 3, 32, 32)
    assert idx.shape == (k,)
    # Check indices are evenly spaced
    expected = torch.linspace(0, 9, steps=k).long()
    assert torch.all(idx == expected)


def test_take_frames_batched():
    video = torch.randn(2, 10, 3, 32, 32)  # B,T,C,H,W
    k = 4
    out, idx = take_frames_batched(video, k)
    assert out.shape == (2, k, 3, 32, 32)
    assert idx.shape == (2, k)
    # First row should be same indices
    assert torch.all(idx[0] == idx[1])


def test_align_video(mnist_small):
    train, _, _ = mnist_small

    ds = MovingMNIST(train, T=10, canvas=64, seed=42)
    video, label, shifts, _ = ds[0]

    aligned = MovingMNIST.align_video(video, shifts)

    ref = aligned[0]

    # All frames should match frame 0 after alignment
    for t in range(aligned.shape[0]):
        diff = (aligned[t] - ref).abs().max()
        assert diff < 1e-5, f"Frame {t} misaligned (max diff {diff})"

    integrated = integrate_frames(video, shifts)
    diff = (integrated - ref).abs().max()
    assert diff < 1e-5, f"integrate_frames mismatch (max diff {diff:.2e})"


def test_align_video_subsampled(mnist_small):
    train, _, _ = mnist_small

    ds = MovingMNIST(train, T=10, canvas=64, seed=42)
    video, label, shifts, _ = ds[0]  # video: [T, C, H, W], shifts: [T-1, 2]

    # --- sample frames ---
    k = 4
    video_batched = video.unsqueeze(0)  # [1, T, C, H, W]
    videos_k, frame_indices = take_frames_batched(video_batched, k)

    videos_k = videos_k[0]  # [k, C, H, W]
    frame_indices = frame_indices[0]  # [k]

    # --- construct shifts for sampled video ---
    # cumulative shifts over full sequence
    cum = torch.zeros(video.shape[0], 2, dtype=shifts.dtype)
    cum[1:] = torch.cumsum(shifts, dim=0)

    # pick cumulative shifts at sampled indices
    cum_sampled = cum[frame_indices]  # [k, 2]

    # convert back to relative shifts between sampled frames
    sampled_shifts = cum_sampled[1:] - cum_sampled[:-1]  # [k-1, 2]

    # --- align sampled video ---
    aligned = MovingMNIST.align_video(videos_k, sampled_shifts)

    ref = aligned[0]

    # --- same invariant: all frames align to first ---
    for t in range(aligned.shape[0]):
        diff = (aligned[t] - ref).abs().max()
        assert diff < 1e-5, f"Subsampled frame {t} misaligned (max diff {diff})"

    # test integration
    integrated = integrate_frames(videos_k, sampled_shifts)
    diff = (integrated - ref).abs().max()
    assert diff < 1e-5, f"integrate_frames (subsampled) mismatch (max diff {diff:.2e})"


def test_integrate_occluded_k_batched(mnist_small):
    """Temporal integration of aligned occluded frames beats a single occluded frame."""
    from occluders import apply_mask_to_video, Occ

    train, _, _ = mnist_small
    T, k, p = 20, 8, 0.4

    ds = MovingMNIST(train, T=T, canvas=64, seed=7)
    B = 4
    videos     = torch.stack([ds[i][0] for i in range(B)])   # [B, T, 1, 64, 64]
    all_shifts = torch.stack([ds[i][2] for i in range(B)])   # [B, T-1, 2]

    # k-subsample frames
    videos_k, frame_indices = take_frames_batched(videos, k)  # [B, k, 1, 64, 64]

    # Apply static Bernoulli occlusion to the k-batch
    occluded_k, _ = apply_mask_to_video(videos_k, Occ.BERNOULLI, p, seed=42)

    ref     = videos[:, 0]       # [B, 1, 64, 64]  clean frame 0
    occ_ref = occluded_k[:, 0]   # [B, 1, 64, 64]  occluded frame 0

    # Align + integrate each batch element's occluded k-frames
    integrated = []
    for b in range(B):
        cum = torch.zeros(T, 2, dtype=all_shifts.dtype)
        cum[1:] = torch.cumsum(all_shifts[b], dim=0)
        fi = frame_indices[b]                               # [k]
        sampled_shifts_b = cum[fi[1:]] - cum[fi[:-1]]      # [k-1, 2]
        integrated.append(integrate_frames(occluded_k[b], sampled_shifts_b))

    integrated = torch.stack(integrated)                    # [B, 1, 64, 64]

    mse_integrated = (integrated - ref).pow(2).mean().item()
    mse_occ_ref    = (occ_ref    - ref).pow(2).mean().item()

    assert mse_integrated < mse_occ_ref, (
        f"Integration MSE {mse_integrated:.4f} should be < "
        f"single-occluded-frame MSE {mse_occ_ref:.4f}"
    )

def test_saliency_weighted_integration(mnist_small):
    """Score-CAM heatmaps as spatial weights improve integration over a single occluded frame."""
    from occluders import apply_mask_to_video, Occ
    from models.score_cam import ScoreCAM
    from models.small_cnn import SmallCNN
    from training.evaluation import load_best_model

    CKPT = get_project_root() / "data" / "checkpoints" / "mnist.pt"
    if not CKPT.exists():
        pytest.skip("MNIST checkpoint not found")

    train, _, _ = mnist_small
    T, k, p = 20, 8, 0.4

    ds = MovingMNIST(train, T=T, canvas=64, seed=7)
    B = 4
    videos     = torch.stack([ds[i][0] for i in range(B)])   # [B, T, 1, 64, 64]
    all_shifts = torch.stack([ds[i][2] for i in range(B)])   # [B, T-1, 2]

    videos_k, frame_indices = take_frames_batched(videos, k)
    occluded_k, _ = apply_mask_to_video(videos_k, Occ.BERNOULLI, p, seed=42)

    ref     = videos[:, 0]       # [B, 1, 64, 64]  clean frame 0
    occ_ref = occluded_k[:, 0]   # [B, 1, 64, 64]  occluded frame 0

    model = load_best_model(SmallCNN, CKPT)

    weighted_integrated = []
    with ScoreCAM(model) as cam:
        for b in range(B):
            cum = torch.zeros(T, 2, dtype=all_shifts.dtype)
            cum[1:] = torch.cumsum(all_shifts[b], dim=0)
            fi = frame_indices[b]
            sampled_shifts_b = cum[fi[1:]] - cum[fi[:-1]]      # [k-1, 2]

            # Score-CAM heatmap per occluded frame → [k, H, W]
            heatmaps = torch.stack([
                cam(occluded_k[b, t].unsqueeze(0))[0]          # [H, W]
                for t in range(k)
            ])

            weighted_integrated.append(
                integrate_frames(occluded_k[b], sampled_shifts_b, weights=heatmaps)
            )

    weighted_integrated = torch.stack(weighted_integrated)      # [B, 1, 64, 64]

    mse_weighted = (weighted_integrated - ref).pow(2).mean().item()
    mse_occ_ref  = (occ_ref - ref).pow(2).mean().item()

    assert mse_weighted < mse_occ_ref, (
        f"Saliency-weighted MSE {mse_weighted:.4f} should be < "
        f"occluded-frame MSE {mse_occ_ref:.4f}"
    )
