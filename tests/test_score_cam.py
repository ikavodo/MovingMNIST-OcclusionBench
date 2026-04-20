import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.small_cnn import SmallCNN
from models.score_cam import ScoreCAM
from utils import get_project_root

ROOT = get_project_root()
CKPT_MNIST = ROOT / "data" / "checkpoints" / "mnist.pt"
CKPT_FASHION = ROOT / "data" / "checkpoints" / "fashion.pt"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_model(ckpt_path: Path) -> SmallCNN:
    model = SmallCNN()
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    return model


def _assert_valid_saliency(saliency: torch.Tensor, H: int = 64, W: int = 64) -> None:
    assert saliency.shape == (H, W), f"expected shape ({H},{W}), got {saliency.shape}"
    assert float(saliency.min()) >= 0.0,          "saliency has negative values"
    assert float(saliency.max()) <= 1.0 + 1e-6,   "saliency exceeds 1.0"
    assert float(saliency.max()) > 0.0,            "saliency map is all zeros"
    assert float(saliency.std()) > 1e-3,           "saliency map is spatially uniform"


def _get_moving_frame(dataset_type, seed: int = 42) -> tuple[torch.Tensor, int]:
    """Return a single 64×64 frame and its label from MovingMNISTFrames."""
    from data.mnist_base import build_mnist_splits
    from data.moving_mnist import MovingMNIST, MovingMNISTFrames

    _, _, test_base = build_mnist_splits(ROOT / "data", dataset=dataset_type)
    moving = MovingMNIST(test_base, T=20, canvas=64, seed=seed)
    frames_ds = MovingMNISTFrames(moving, frame_seed=0)
    frame, label, _meta = frames_ds[0]   # frame: [1, 64, 64]
    return frame.unsqueeze(0), int(label) # [1, 1, 64, 64]


# ---------------------------------------------------------------------------
# Structural / unit tests  (no checkpoint required)
# ---------------------------------------------------------------------------

def test_score_cam_output_shape():
    model = SmallCNN()
    model.eval()
    x = torch.randn(1, 1, 64, 64)
    with ScoreCAM(model) as cam:
        saliency, cls = cam(x)
    _assert_valid_saliency(saliency)
    assert 0 <= cls < 10


def test_score_cam_all_layers():
    """ScoreCAM runs without error on every supported layer."""
    model = SmallCNN()
    model.eval()
    x = torch.randn(1, 1, 64, 64)
    for layer in ("conv1", "conv2", "conv3"):
        with ScoreCAM(model, target_layer=layer) as cam:
            saliency, _ = cam(x)
        _assert_valid_saliency(saliency)


def test_score_cam_explicit_class():
    model = SmallCNN()
    model.eval()
    x = torch.randn(1, 1, 64, 64)
    with ScoreCAM(model) as cam:
        saliency, cls = cam(x, class_idx=7)
    _assert_valid_saliency(saliency)
    assert cls == 7


def test_score_cam_invalid_layer_raises():
    model = SmallCNN()
    with pytest.raises(ValueError, match="target_layer"):
        ScoreCAM(model, target_layer="conv99")


def test_score_cam_hook_removed_after_close():
    """Verify that hook teardown does not break subsequent model inference."""
    model = SmallCNN()
    model.eval()
    x = torch.randn(1, 1, 64, 64)
    cam = ScoreCAM(model)
    cam(x)
    cam.close()
    # Model should still run correctly after hook removal.
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 10)


# ---------------------------------------------------------------------------
# Trained-checkpoint tests — MNIST
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not CKPT_MNIST.exists(), reason="MNIST checkpoint not found")
def test_score_cam_mnist_real_sample():
    from data.mnist_base import DatasetType

    model = _load_model(CKPT_MNIST)
    x, label = _get_moving_frame(DatasetType.MNIST)

    with ScoreCAM(model) as cam:
        saliency, cls = cam(x)

    _assert_valid_saliency(saliency)
    # Trained model should predict the correct digit on a clean frame.
    assert cls == label, (
        f"MNIST: Score-CAM target class {cls} does not match true label {label}"
    )


@pytest.mark.skipif(not CKPT_MNIST.exists(), reason="MNIST checkpoint not found")
def test_score_cam_mnist_all_layers():
    from data.mnist_base import DatasetType

    model = _load_model(CKPT_MNIST)
    x, _ = _get_moving_frame(DatasetType.MNIST)

    for layer in ("conv1", "conv2", "conv3"):
        with ScoreCAM(model, target_layer=layer) as cam:
            saliency, _ = cam(x)
        _assert_valid_saliency(saliency), f"layer={layer}"


# ---------------------------------------------------------------------------
# Trained-checkpoint tests — FashionMNIST
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not CKPT_FASHION.exists(), reason="Fashion checkpoint not found")
def test_score_cam_fashion_real_sample():
    from data.mnist_base import DatasetType

    model = _load_model(CKPT_FASHION)
    x, label = _get_moving_frame(DatasetType.FASHION)

    with ScoreCAM(model) as cam:
        saliency, cls = cam(x)

    _assert_valid_saliency(saliency)
    assert cls == label, (
        f"Fashion: Score-CAM target class {cls} does not match true label {label}"
    )


@pytest.mark.skipif(not CKPT_FASHION.exists(), reason="Fashion checkpoint not found")
def test_score_cam_fashion_all_layers():
    from data.mnist_base import DatasetType

    model = _load_model(CKPT_FASHION)
    x, _ = _get_moving_frame(DatasetType.FASHION)

    for layer in ("conv1", "conv2", "conv3"):
        with ScoreCAM(model, target_layer=layer) as cam:
            saliency, _ = cam(x)
        _assert_valid_saliency(saliency), f"layer={layer}"


# ---------------------------------------------------------------------------
# Visualization tests  (optional — run with: pytest -m viz)
# ---------------------------------------------------------------------------

VIZ_DIR = Path(__file__).parent / "visualizations"


def _make_overlay_figure(model, x, label, dataset_name, out_path):
    """Save a 1×4 figure: original frame + Score-CAM overlay for each conv layer."""
    import numpy as np
    import matplotlib.pyplot as plt

    layers = ("conv1", "conv2", "conv3")
    saliencies = {}
    pred_cls = None
    for layer in layers:
        with ScoreCAM(model, target_layer=layer) as cam:
            sal, cls = cam(x)
            saliencies[layer] = sal.numpy()
            if pred_cls is None:
                pred_cls = cls

    frame_np = x[0, 0].numpy()  # [64, 64], values in [0, 1] or normalised

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    fig.suptitle(
        f"{dataset_name}  |  true label: {label}  |  pred (conv3): {pred_cls}",
        fontsize=13,
    )

    # Panel 0: original frame
    axes[0].imshow(frame_np, cmap="gray", vmin=frame_np.min(), vmax=frame_np.max())
    axes[0].set_title("original")

    # Panels 1-3: heatmap overlaid on original
    cmap = plt.get_cmap("jet")
    for ax, layer in zip(axes[1:], layers):
        sal = saliencies[layer]
        rgba_heat = np.asarray(cmap(sal))    # [H, W, 4], values in [0,1]
        rgba_heat[..., 3] = sal * 0.7        # alpha proportional to saliency

        # Normalise frame to [0,1] for display
        f_min, f_max = frame_np.min(), frame_np.max()
        frame_01 = (frame_np - f_min) / max(f_max - f_min, 1e-8)
        frame_rgb = np.stack([frame_01] * 3, axis=-1)  # [H, W, 3] grayscale-as-RGB

        ax.imshow(frame_rgb)
        ax.imshow(rgba_heat)
        ax.set_title(layer)

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


@pytest.mark.viz
@pytest.mark.skipif(not CKPT_MNIST.exists(), reason="MNIST checkpoint not found")
def test_score_cam_viz_mnist():
    pytest.importorskip("matplotlib")
    from data.mnist_base import DatasetType

    model = _load_model(CKPT_MNIST)
    x, label = _get_moving_frame(DatasetType.MNIST)
    out = VIZ_DIR / "score_cam_mnist.png"
    _make_overlay_figure(model, x, label, "MNIST", out)
    assert out.exists(), f"expected output at {out}"


@pytest.mark.viz
@pytest.mark.skipif(not CKPT_FASHION.exists(), reason="Fashion checkpoint not found")
def test_score_cam_viz_fashion():
    pytest.importorskip("matplotlib")
    from data.mnist_base import DatasetType

    model = _load_model(CKPT_FASHION)
    x, label = _get_moving_frame(DatasetType.FASHION)
    out = VIZ_DIR / "score_cam_fashion.png"
    _make_overlay_figure(model, x, label, "FashionMNIST", out)
    assert out.exists(), f"expected output at {out}"