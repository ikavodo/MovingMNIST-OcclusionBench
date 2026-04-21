"""Overlay integration — evaluation + optional visualisation.

One MNIST sample and one FashionMNIST sample are overlaid on a shared canvas.
Bernoulli occlusion is swept over P_VALUES.  For each (sample, coverage) pair,
the occluded overlay is recovered by three temporal-integration methods:

  baseline  — no integration; first occluded frame only
  plain     — uniform-weight aligned mean
  opponent  — weights = 1 − opponent_heatmap  (two-model oracle)
  self      — weights = own_heatmap × (1 − clean_pixel)  (one-model oracle)

Heatmaps are computed once per sample on the *clean* overlay.

Optional visualisation (VISUALIZE=True) generates per-sample figures for the
p=0 case only:
  overlay_integration_s{idx}.png   — 2 rows × 6 cols summary
  overlay_sequence_s{idx}.png      — 4 rows × K cols sequence + heatmaps

Quantitative outputs (always produced):
  overlay_eval_raw.csv             — one row per (sample, alignment, p)
  overlay_eval.png                 — 2×2 MSE + accuracy grid

Usage:
    python scripts/overlay_integration.py
"""

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.mnist_base import build_mnist_splits, DatasetType
from data.moving_mnist import MovingMNIST
from models.score_cam import ScoreCAM
from models.small_cnn import SmallCNN
from occluders import apply_mask_to_video, Occ
from training.evaluation import load_best_model, take_frames_batched
from utils import get_project_root, integrate_frames

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ROOT      = get_project_root()
DATA_DIR  = ROOT / "data"
OUT_DIR   = ROOT / "outputs"

SEED_MNIST    = 42
SEED_FASHION  = 99
N_EVAL        = 50          # total samples to evaluate
N_VIZ         = 2           # first N_VIZ samples also get detailed figures (p=0)
VISUALIZE     = True
T             = 20
K             = T // 3      # subsampled frames per sequence
CANVAS        = 64
MIN_VEL_DIST  = 3.0         # min ||vel_m - vel_f||_2 for sample selection
MAX_SEARCH    = 2000
P_VALUES      = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
OCC_SEED_BASE = 7777        # occlusion seed = OCC_SEED_BASE + sample_idx

# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _mse(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).pow(2).mean())


def _infer(model, x: torch.Tensor) -> int:
    with torch.no_grad():
        return int(model(x.unsqueeze(0)).argmax(1).item())


def _sampled_shifts(shifts: torch.Tensor, fi: torch.Tensor) -> torch.Tensor:
    cum = torch.zeros(T, 2, dtype=shifts.dtype)
    cum[1:] = torch.cumsum(shifts, dim=0)
    return cum[fi[1:]] - cum[fi[:-1]]  # [K-1, 2]


def _vel_dist(meta_m: dict, meta_f: dict) -> float:
    return math.hypot(meta_m["vx"] - meta_f["vx"], meta_m["vy"] - meta_f["vy"])


def _check(v: bool) -> str:
    return "✓" if v else "✗"


def _to_display(x: torch.Tensor) -> np.ndarray:
    arr = x.squeeze().float().numpy()
    lo, hi = arr.min(), arr.max()
    return (arr - lo) / max(hi - lo, 1e-8)


def _heatmap_overlay(frame: np.ndarray, heat: np.ndarray) -> np.ndarray:
    cmap  = plt.get_cmap("jet")
    rgba  = np.asarray(cmap(heat))
    alpha = (heat * 0.65)[..., np.newaxis]
    gray3 = np.stack([frame] * 3, axis=-1)
    return np.clip(gray3 * (1 - alpha) + rgba[..., :3] * alpha, 0, 1)


# ---------------------------------------------------------------------------
# Sample selection
# ---------------------------------------------------------------------------

def find_samples(ds_m, ds_f, n: int, min_dist: float, max_search: int):
    """Return (idx, vel_dist) pairs for the first n samples with ||vel||₂ ≥ min_dist."""
    found = []
    for idx in range(min(max_search, len(ds_m))):
        _, _, _, meta_m = ds_m[idx]
        _, _, _, meta_f = ds_f[idx]
        d = _vel_dist(meta_m, meta_f)
        if d >= min_dist:
            found.append((idx, d))
            if len(found) == n:
                break
    if len(found) < n:
        raise RuntimeError(
            f"Only {len(found)}/{n} qualifying samples in first {max_search} indices; "
            "lower MIN_VEL_DIST or increase MAX_SEARCH."
        )
    return found


# ---------------------------------------------------------------------------
# Core integration
# ---------------------------------------------------------------------------

def _compute_heatmaps(model, frames: torch.Tensor) -> torch.Tensor:
    """ScoreCAM on each frame; returns [K, H, W]."""
    with ScoreCAM(model) as cam:
        return torch.stack([cam(frames[t].unsqueeze(0))[0] for t in range(frames.shape[0])])


def _run_integrations(
    occ_k,      # [K, 1, H, W] — occluded (or clean) overlay
    clean_px,   # [K, H, W]   — clean overlay pixels (oracle, for self-weights)
    fi,         # [K]
    heatmaps_m, heatmaps_f,     # [K, H, W] each
    shifts_m, shifts_f,
    label_m, label_f,
    ref_m, ref_f,
    model_m, model_f,
    meta_m, meta_f,
) -> list[dict]:
    """Run all methods for both alignments; return a list of two row dicts."""
    rows = []
    for name, alignment, shifts, label, ref, model, own_hm, opp_hm in [
        ("MNIST",   "mnist",   shifts_m, label_m, ref_m, model_m, heatmaps_m, heatmaps_f),
        ("Fashion", "fashion", shifts_f, label_f, ref_f, model_f, heatmaps_f, heatmaps_m),
    ]:
        ss       = _sampled_shifts(shifts, fi)
        plain    = integrate_frames(occ_k, ss)
        opponent = integrate_frames(occ_k, ss, weights=1.0 - opp_hm)
        self_    = integrate_frames(occ_k, ss, weights=own_hm * (1.0 - clean_px))

        rows.append({
            # display fields
            "name":           name,
            "alignment":      alignment,
            "label":          label,
            "ref":            ref,
            "heatmaps":       own_hm,   # [K, H, W]
            "sampled_shifts": ss,       # [K-1, 2]
            # integration results
            "plain":          plain,
            "weighted":       opponent,
            "self_weighted":  self_,
            # metrics
            "mse_baseline":   _mse(occ_k[0],  ref),
            "mse_plain":      _mse(plain,      ref),
            "mse_weighted":   _mse(opponent,   ref),
            "mse_self":       _mse(self_,      ref),
            "pred_ov":        _infer(model, occ_k[0]),
            "pred_pl":        _infer(model, plain),
            "pred_wt":        _infer(model, opponent),
            "pred_sw":        _infer(model, self_),
            # metadata
            "meta_m":         meta_m,
            "meta_f":         meta_f,
        })
    return rows


def _rows_to_eval(viz_rows, sample_idx, vel_dist, label_m, label_f, p) -> list[dict]:
    """Flatten viz_rows to lightweight eval dicts for the DataFrame."""
    return [
        {
            "sample_idx":   sample_idx,
            "vel_dist":     round(vel_dist, 3),
            "label_m":      label_m,
            "label_f":      label_f,
            "alignment":    r["alignment"],
            "p":            float(p),
            "mse_baseline": r["mse_baseline"],
            "mse_plain":    r["mse_plain"],
            "mse_opponent": r["mse_weighted"],
            "mse_self":     r["mse_self"],
            "acc_baseline": int(r["pred_ov"] == r["label"]),
            "acc_plain":    int(r["pred_pl"] == r["label"]),
            "acc_opponent": int(r["pred_wt"] == r["label"]),
            "acc_self":     int(r["pred_sw"] == r["label"]),
        }
        for r in viz_rows
    ]


# ---------------------------------------------------------------------------
# Visualisation (p = 0 only)
# ---------------------------------------------------------------------------

def make_figure(overlay_k, rows, sample_idx):
    fig, axes = plt.subplots(2, 6, figsize=(26, 9))
    col_titles = [
        "Reference\n(clean)",
        "Overlay frame 0\n(MSE | pred)",
        "Score-CAM\n(frame 0)",
        "Plain\n(MSE | pred)",
        "Opponent-suppressed\n(1−opp heatmap)",
        "Self-suppressed\n(own×(1−px))",
    ]
    for ax, t in zip(axes[0], col_titles):
        ax.set_title(t, fontsize=9, fontweight="bold")

    mm, mf = rows[0]["meta_m"], rows[0]["meta_f"]
    fig.suptitle(
        f"Sample {sample_idx}  |  "
        f"MNIST vx={mm['vx']:+d} vy={mm['vy']:+d}  ·  "
        f"Fashion vx={mf['vx']:+d} vy={mf['vy']:+d}",
        fontsize=12,
    )

    ov_disp = _to_display(overlay_k[0])
    for ri, r in enumerate(rows):
        ax  = axes[ri]
        lbl = r["label"]
        ax[0].set_ylabel(f"{r['name']} alignment\n(label={lbl})", fontsize=9)

        ax[0].imshow(_to_display(r["ref"]),  cmap="gray", vmin=0, vmax=1)
        ax[0].set_xlabel("(clean target)", fontsize=8)

        ax[1].imshow(ov_disp, cmap="gray", vmin=0, vmax=1)
        ax[1].set_xlabel(
            f"MSE={r['mse_baseline']:.4f}  pred={r['pred_ov']}{_check(r['pred_ov']==lbl)}",
            fontsize=8)

        ax[2].imshow(_heatmap_overlay(ov_disp, r["heatmaps"][0].numpy()))
        ax[2].set_xlabel("conv3 saliency", fontsize=8)

        ax[3].imshow(_to_display(r["plain"]), cmap="gray", vmin=0, vmax=1)
        ax[3].set_xlabel(
            f"MSE={r['mse_plain']:.4f}  pred={r['pred_pl']}{_check(r['pred_pl']==lbl)}",
            fontsize=8)

        ax[4].imshow(_to_display(r["weighted"]), cmap="gray", vmin=0, vmax=1)
        ax[4].set_xlabel(
            f"MSE={r['mse_weighted']:.4f}  pred={r['pred_wt']}{_check(r['pred_wt']==lbl)}",
            fontsize=8)

        ax[5].imshow(_to_display(r["self_weighted"]), cmap="gray", vmin=0, vmax=1)
        ax[5].set_xlabel(
            f"MSE={r['mse_self']:.4f}  pred={r['pred_sw']}{_check(r['pred_sw']==lbl)}",
            fontsize=8)

    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])

    fig.tight_layout()
    out = OUT_DIR / f"overlay_integration_s{sample_idx}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out}")


def make_sequence_figure(overlay_k, rows, sample_idx):
    n_frames = overlay_k.shape[0]
    fig, axes = plt.subplots(4, n_frames, figsize=(n_frames * 2.8, 11))

    aligned_m = MovingMNIST.align_video(overlay_k, rows[0]["sampled_shifts"])
    row_labels = [
        "Overlay",
        f"MNIST saliency (label={rows[0]['label']})",
        f"Fashion saliency (label={rows[1]['label']})",
        "MNIST-aligned",
    ]

    for t in range(n_frames):
        ov_disp = _to_display(overlay_k[t])
        axes[0, t].imshow(ov_disp, cmap="gray", vmin=0, vmax=1)
        axes[1, t].imshow(_heatmap_overlay(ov_disp, rows[0]["heatmaps"][t].numpy()))
        axes[2, t].imshow(_heatmap_overlay(ov_disp, rows[1]["heatmaps"][t].numpy()))
        axes[3, t].imshow(_to_display(aligned_m[t]), cmap="gray", vmin=0, vmax=1)
        axes[0, t].set_title(f"f{t}", fontsize=8)

    for i, rl in enumerate(row_labels):
        axes[i, 0].set_ylabel(rl, fontsize=8)
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])

    mm, mf = rows[0]["meta_m"], rows[0]["meta_f"]
    fig.suptitle(
        f"Sequence  sample {sample_idx}  |  "
        f"MNIST vx={mm['vx']:+d} vy={mm['vy']:+d}  ·  "
        f"Fashion vx={mf['vx']:+d} vy={mf['vy']:+d}",
        fontsize=11,
    )
    fig.tight_layout()
    out = OUT_DIR / f"overlay_sequence_s{sample_idx}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out}")


# ---------------------------------------------------------------------------
# Metrics plot
# ---------------------------------------------------------------------------

def _agg_ci(series: pd.Series):
    arr = series.dropna().to_numpy()
    m   = arr.mean()
    se  = arr.std(ddof=1) / math.sqrt(len(arr)) if len(arr) > 1 else 0.0
    return pd.Series({"mean": m, "lo": m - 1.96 * se, "hi": m + 1.96 * se})


def plot_eval_results(df: pd.DataFrame, out_path: Path):
    methods = [
        ("baseline", "Baseline (frame 0)", "dimgray",  "--"),
        ("plain",    "Plain",              "steelblue", "-"),
        ("opponent", "Opponent-suppressed","firebrick", "-"),
        ("self",     "Self-suppressed",    "seagreen",  "-"),
    ]
    alignments = [("mnist", "MNIST alignment"), ("fashion", "Fashion alignment")]
    metrics    = [("mse", "MSE vs clean ref"), ("acc", "Accuracy")]

    fig, axes = plt.subplots(
        len(metrics), len(alignments),
        figsize=(13, 9), sharey="row",
    )

    for col, (alignment, align_title) in enumerate(alignments):
        sub = df[df["alignment"] == alignment]
        for row, (metric, metric_title) in enumerate(metrics):
            ax = axes[row, col]
            for key, label, color, ls in methods:
                agg = (
                    sub.groupby("p")[f"{metric}_{key}"]
                    .apply(_agg_ci)
                    .unstack()
                    .reset_index()
                )
                ax.plot(agg["p"], agg["mean"], marker="o",
                        label=label, color=color, linestyle=ls)
                ax.fill_between(agg["p"], agg["lo"], agg["hi"],
                                alpha=0.12, color=color)

            if row == 0:
                ax.set_title(align_title, fontweight="bold", fontsize=11)
            ax.set_xlabel("Bernoulli coverage" if row == len(metrics) - 1 else "")
            ax.set_ylabel(metric_title, fontsize=9)
            ax.grid(True, alpha=0.3)
            if col == 0:
                ax.legend(fontsize=8)

    fig.suptitle(
        f"Overlay integration  ·  N={df['sample_idx'].nunique()} samples  "
        f"K={K} frames  min_vel_dist={MIN_VEL_DIST}",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ckpt_m = DATA_DIR / "checkpoints" / "mnist.pt"
    ckpt_f = DATA_DIR / "checkpoints" / "fashion.pt"
    for ckpt in (ckpt_m, ckpt_f):
        if not ckpt.exists():
            raise FileNotFoundError(ckpt)

    _, _, mnist_test   = build_mnist_splits(DATA_DIR, dataset=DatasetType.MNIST)
    _, _, fashion_test = build_mnist_splits(DATA_DIR, dataset=DatasetType.FASHION)

    ds_m = MovingMNIST(mnist_test,   T=T, canvas=CANVAS, seed=SEED_MNIST)
    ds_f = MovingMNIST(fashion_test, T=T, canvas=CANVAS, seed=SEED_FASHION)

    model_m = load_best_model(SmallCNN, ckpt_m)
    model_f = load_best_model(SmallCNN, ckpt_f)

    print(f"Selecting {N_EVAL} samples (vel_dist ≥ {MIN_VEL_DIST}) ...")
    samples = find_samples(ds_m, ds_f, N_EVAL, MIN_VEL_DIST, MAX_SEARCH)

    all_rows = []
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for sample_no, (idx, vel_dist) in enumerate(tqdm(samples, desc="samples")):
        video_m, label_m, shifts_m, meta_m = ds_m[idx]
        video_f, label_f, shifts_f, meta_f = ds_f[idx]

        overlay   = (video_m + video_f).clamp(0.0, 1.0)
        ov_b, fi  = take_frames_batched(overlay.unsqueeze(0), K)
        overlay_k = ov_b[0]   # [K, 1, H, W]
        fi        = fi[0]      # [K]
        clean_px  = overlay_k.squeeze(1)  # [K, H, W]

        heatmaps_m = _compute_heatmaps(model_m, overlay_k)
        heatmaps_f = _compute_heatmaps(model_f, overlay_k)

        for p in P_VALUES:
            occ_k, _ = apply_mask_to_video(
                overlay_k.unsqueeze(0), Occ.BERNOULLI, p,
                seed=OCC_SEED_BASE + idx,
            )
            occ_k = occ_k[0]  # [K, 1, H, W]

            viz_rows = _run_integrations(
                occ_k, clean_px, fi,
                heatmaps_m, heatmaps_f,
                shifts_m, shifts_f,
                label_m, label_f,
                video_m[0], video_f[0],
                model_m, model_f,
                meta_m, meta_f,
            )
            all_rows.extend(_rows_to_eval(viz_rows, idx, vel_dist, label_m, label_f, p))

            if VISUALIZE and p == 0.0 and sample_no < N_VIZ:
                tqdm.write(f"\n  viz sample {idx}  vel_dist={vel_dist:.2f}")
                make_figure(overlay_k, viz_rows, idx)
                make_sequence_figure(overlay_k, viz_rows, idx)

    df = pd.DataFrame(all_rows)
    csv_path = OUT_DIR / "overlay_eval_raw.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved → {csv_path}")

    plot_eval_results(df, OUT_DIR / "overlay_eval.png")

    print("\n--- Mean MSE by method and coverage ---")
    print(
        df.groupby(["alignment", "p"])[
            ["mse_baseline", "mse_plain", "mse_opponent", "mse_self"]
        ].mean().round(5).to_string()
    )
