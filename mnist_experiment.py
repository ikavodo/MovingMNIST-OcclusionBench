import random
import copy
from dataclasses import dataclass, asdict
from enum import Enum, auto
from typing import Dict, List, Optional

import cv2
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.tensorboard import SummaryWriter

import torchvision
from torchvision import transforms
import os

# ============================================================
# plotters
# ============================================================
from pathlib import Path
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scripts.utils import get_project_root, seed_everything, ensure_dir
from data.utils import compute_motion_bounds

#
ROOT = get_project_root()  # save stuff in top-level designated subdirs
DATA_DIR = ROOT / "data"
LOG_DIR = ROOT / "runs" / "moving_mnist"
CKPT_DIR = LOG_DIR / "checkpoints"
OUT_DIR = ROOT / "output"
IMG_SIZE = 28  # MNIST
_LOCAL_TRI = np.array([[1.0, 0.0], [0.0, -1.0], [-1.0, 0.0]], dtype=np.float32)  # for occluders


# ============================================================
# Data
# ============================================================

def build_mnist_splits():
    tfm = transforms.ToTensor()
    train_full = torchvision.datasets.MNIST(root=DATA_DIR, train=True, download=True, transform=tfm)
    test = torchvision.datasets.MNIST(root=DATA_DIR, train=False, download=True, transform=tfm)

    # deterministic train/val split over base images
    n_total = len(train_full)
    n_val = 10000
    train_idx = list(range(0, n_total - n_val))
    val_idx = list(range(n_total - n_val, n_total))
    train = Subset(train_full, train_idx)
    val = Subset(train_full, val_idx)
    return train, val, test


class RollingMovingMNIST(Dataset):
    """
    Deterministic per-index MovingMNIST generator.
    Each index corresponds to a base digit image and a deterministic motion seed.
    This makes validation/test reproducible and training compatible with shuffling.
    Returns:
      video:  (T, 1, H, W)
      label:  int
      shifts: (T-1, 2) each row is (dy, dx)
      meta:   dict-like payload for downstream bookkeeping
    """

    def __init__(
            self,
            base_mnist,
            T: int = 20,
            canvas: int = 64,
            vmin: int = -3,
            vmax: int = 3,
            seed: int = 0,
    ):
        assert canvas >= IMG_SIZE
        self.base = base_mnist
        self.T = T
        self.canvas = canvas
        self.vmin = vmin
        self.vmax = vmax
        self.seed = seed

    def __len__(self):
        return len(self.base)

    @staticmethod
    def _bounce(pos, v, lo, hi):
        nxt = pos + v
        if nxt < lo:
            return lo, abs(v)
        if nxt > hi:
            return hi, -abs(v)
        return nxt, v

    @staticmethod
    def _zero_wrap(frame, dy, dx):
        if dy > 0:
            frame[:, :dy, :] = 0
        elif dy < 0:
            frame[:, dy:, :] = 0
        if dx > 0:
            frame[:, :, :dx] = 0
        elif dx < 0:
            frame[:, :, dx:] = 0
        return frame

    def _sample_vxy(self, g: torch.Generator):
        while True:
            vx = torch.randint(self.vmin, self.vmax + 1, (1,), generator=g).item()
            vy = torch.randint(self.vmin, self.vmax + 1, (1,), generator=g).item()
            if vx != 0 or vy != 0:
                return vx, vy

    def __getitem__(self, idx):
        img, label = self.base[idx]
        H = W = self.canvas
        xmax = W - IMG_SIZE
        ymax = H - IMG_SIZE

        g = torch.Generator()
        g.manual_seed(self.seed + idx)

        x = torch.randint(0, xmax + 1, (1,), generator=g).item()
        y = torch.randint(0, ymax + 1, (1,), generator=g).item()
        vx, vy = self._sample_vxy(g)

        frame = torch.zeros(1, H, W, dtype=img.dtype)
        frame[:, y:y + IMG_SIZE, x:x + IMG_SIZE] = img

        video = [frame.clone()]
        shifts = []
        for _ in range(self.T - 1):
            nx, vx = self._bounce(x, vx, 0, xmax)
            ny, vy = self._bounce(y, vy, 0, ymax)
            dx, dy = nx - x, ny - y
            shifts.append((dy, dx))
            frame = torch.roll(frame, shifts=(dy, dx), dims=(1, 2))
            frame = self._zero_wrap(frame, dy, dx)
            x, y = nx, ny
            video.append(frame.clone())

        meta = {"base_idx": int(idx), "seed": int(self.seed + idx)}
        return torch.stack(video, dim=0), int(label), torch.tensor(shifts, dtype=torch.long), meta


class MovingMNISTFrames(Dataset):
    """
    Deterministic single-frame dataset from deterministic videos.
    Frame chosen by idx-dependent seed. This avoids per-epoch stochastic drift in val/test.
    For train, you can re-instantiate with a different base seed if desired.
    """

    def __init__(self, moving_ds: Dataset, frame_seed: int = 0):
        self.ds = moving_ds
        self.frame_seed = frame_seed

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        video, y, _, meta = self.ds[idx]
        rng = random.Random(self.frame_seed + idx)
        t = rng.randrange(video.shape[0])
        x = video[t]
        out_meta = dict(meta)
        out_meta["frame_index"] = int(t)
        return x, y, out_meta


# ============================================================
# Occlusion families
# ============================================================

class Occ(Enum):
    BERNOULLI = auto()
    CGOL = auto()
    TRIANGULAR = auto()
    PERLIN = auto()
    BRANCHES = auto()


@dataclass
class MaskConfig:
    family: str
    coverage: float
    seed: int
    static: bool = True


@dataclass
class EvalConfig:
    k_frames: int = 5
    n_mask_seeds: int = 3
    static_masks: bool = True


# CGOL
def _life_step(x_bool):
    x = x_bool.float().unsqueeze(0).unsqueeze(0)
    kernel = torch.tensor([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=torch.float32, device=x.device).view(1, 1, 3, 3)
    n = F.conv2d(x, kernel, padding=1).squeeze(0).squeeze(0)
    alive = x_bool
    born = (~alive) & (n == 3)
    survive = alive & ((n == 2) | (n == 3))
    return born | survive


def cgol_mask(H, W, *, density=0.3, seed=0, device="cpu", controlled=True, tol=0.02, max_rounds=20, init_density=0.25,
              iters=40):
    target = float(max(0.0, min(1.0, density)))
    if target <= 0:
        return torch.zeros((H, W), dtype=torch.float32, device=device)
    g = torch.Generator(device=device)
    g.manual_seed(int(seed))
    acc = torch.zeros((H, W), dtype=torch.bool, device=device)

    def occ(m):
        return float(m.float().mean().item())

    rounds = max_rounds if controlled else 1
    for _ in range(rounds):
        cur = occ(acc)
        if controlled and abs(cur - target) <= tol:
            break
        if cur >= target:
            break
        remaining = max(0.0, target - cur)
        init = min(0.6, max(0.02, init_density + 0.8 * remaining))
        x = (torch.rand((H, W), generator=g, device=device) < init)
        for _ in range(int(iters)):
            x = _life_step(x)
        acc |= x
    return acc.float()


# TRIANGULAR
def _rot2(theta):
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=np.float32)


def sample_points_min_dist(H, W, *, d=8, border=2, seed=0, max_points=10_000):
    rng = np.random.default_rng(seed)
    pts = []
    d2 = d * d
    for _ in range(max_points):
        x = int(rng.integers(border, W - border))
        y = int(rng.integers(border, H - border))
        ok = True
        for (px, py) in pts[-400:]:
            if (x - px) ** 2 + (y - py) ** 2 < d2:
                ok = False
                break
        if ok:
            pts.append((x, y))
    return pts


def triangles_mask_luxor(H, W, pts, rots, scale, *, border=0):
    mask = np.zeros((H, W), dtype=np.uint8)
    polys = []
    for (x, y), r in zip(pts, rots):
        R = _rot2(r * 2.0 * math.pi)
        verts = (_LOCAL_TRI @ R.T) * float(scale)
        verts[:, 0] += x
        verts[:, 1] += y
        poly = np.round(verts).astype(np.int32)
        polys.append(poly.reshape(-1, 1, 2))
    if polys:
        cv2.fillPoly(mask, polys, 255)
    if border > 0:
        mask[:border, :] = 0
        mask[-border:, :] = 0
        mask[:, :border] = 0
        mask[:, -border:] = 0
    return mask


def triangular_mask_luxor_controlled_any(
        H, W, *,
        density,
        tol=0.02,
        seed=0,
        device="cpu",
        d_schedule=(7, 5, 3),
        border=1,
):
    target = float(max(0.0, min(1.0, density)))
    best = None

    for d in d_schedule:
        pts = sample_points_min_dist(H, W, d=d, border=border, seed=seed)
        rng = np.random.default_rng(seed + 1337)
        rots = rng.random(len(pts)).tolist()
        s_min, s_max, max_iter = 1.0, 0.45 * min(H, W), 14

        def occ_for_scale(s):
            m = triangles_mask_luxor(H, W, pts, rots, s, border=border)
            return (m > 0).mean(), m

        occ_hi, m_hi = occ_for_scale(s_max)
        if occ_hi < target - tol:
            best_mask = torch.from_numpy((m_hi > 0).astype(np.float32)).to(device=device).contiguous()
            best = (
                best_mask,
                {"density": float(occ_hi), "d_used": d, "scale": float(s_max)},
            )
            continue

        lo, hi = s_min, s_max
        best_local = None
        for _ in range(max_iter):
            mid = 0.5 * (lo + hi)
            occ_mid, m_mid = occ_for_scale(mid)
            best_local = (mid, occ_mid, m_mid)
            if abs(occ_mid - target) <= tol:
                break
            if occ_mid < target:
                lo = mid
            else:
                hi = mid

        s_best, occ_best, m_best = best_local
        meta = {"density": float(occ_best), "d_used": d, "scale": float(s_best)}
        best_mask = torch.from_numpy((m_best > 0).astype(np.float32)).to(device=device).contiguous()
        best = (best_mask, meta)

        if abs(occ_best - target) <= tol:
            return best

    return best


# PERLIN
def perlin_field_torch(
        H: int,
        W: int,
        *,
        scale: float = 32.0,
        octaves: int = 4,
        persistence: float = 0.5,
        lacunarity: float = 2.0,
        seed: int = 0,
        device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Pure PyTorch 2D Perlin-style gradient noise.
    Returns a tensor of shape (H, W) with values in [0, 1].
    """

    def fade(t: torch.Tensor) -> torch.Tensor:
        return t * t * t * (t * (t * 6 - 15) + 10)

    def single_octave(H: int, W: int, scale: float, seed: int) -> torch.Tensor:
        rng = torch.Generator(device=device)
        rng.manual_seed(seed)

        gH = int(math.ceil(H / scale)) + 2
        gW = int(math.ceil(W / scale)) + 2

        angles = torch.rand(gH, gW, generator=rng, device=device) * (2 * math.pi)
        gx = torch.cos(angles)
        gy = torch.sin(angles)

        y = torch.arange(H, device=device, dtype=torch.float32) / scale
        x = torch.arange(W, device=device, dtype=torch.float32) / scale
        y_grid, x_grid = torch.meshgrid(y, x, indexing="ij")

        x0 = torch.floor(x_grid).long()
        y0 = torch.floor(y_grid).long()
        x1 = x0 + 1
        y1 = y0 + 1

        sx = x_grid - x0.float()
        sy = y_grid - y0.float()

        u = fade(sx)
        v = fade(sy)

        # Dot products between gradients and local offset vectors
        n00 = gx[y0, x0] * sx + gy[y0, x0] * sy
        n10 = gx[y0, x1] * (sx - 1) + gy[y0, x1] * sy
        n01 = gx[y1, x0] * sx + gy[y1, x0] * (sy - 1)
        n11 = gx[y1, x1] * (sx - 1) + gy[y1, x1] * (sy - 1)

        nx0 = n00 + u * (n10 - n00)
        nx1 = n01 + u * (n11 - n01)
        out = nx0 + v * (nx1 - nx0)

        # Roughly normalize to [-1, 1]
        out = out / (math.sqrt(2) / 2)
        return out.clamp(-1, 1)

    total = torch.zeros((H, W), device=device, dtype=torch.float32)
    amplitude = 1.0
    frequency = 1.0
    amp_sum = 0.0

    for i in range(octaves):
        octave_scale = scale / frequency
        octave = single_octave(H, W, octave_scale, seed + i)
        total += amplitude * octave
        amp_sum += amplitude
        amplitude *= persistence
        frequency *= lacunarity

    total = total / max(amp_sum, 1e-8)  # back to roughly [-1, 1]
    total = (total + 1.0) / 2.0  # map to [0, 1]
    return total.clamp(0, 1)


def perlin_mask_torch(
        H: int,
        W: int,
        *,
        coverage: float = 0.3,
        seed: int = 0,
        scale: float = 32.0,
        octaves: int = 4,
        persistence: float = 0.5,
        lacunarity: float = 2.0,
        device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    f = perlin_field_torch(
        H, W,
        scale=scale,
        octaves=octaves,
        persistence=persistence,
        lacunarity=lacunarity,
        seed=seed,
        device=device,
    )
    thr = torch.quantile(f.flatten(), 1.0 - float(coverage))
    return (f >= thr).float()


# BRANCHES
def sample_branch_specs(H, W, *, seed=0, num_specs=400, thickness=2, movement=0,
                        y_start_band=50, dx_range=50, dy_range=(-300, 100)):
    """
    Pre-sample branch line segments (static template).
    Returns a list of (pt1, pt2, thick).
    """
    rng = np.random.default_rng(seed)
    y_start_band = int(np.clip(y_start_band, 1, H))
    thickness = max(2, int(thickness))

    specs = []
    for _ in range(int(num_specs)):
        x1 = int(rng.integers(0, W))
        y1 = int(rng.integers(H - y_start_band, H))
        x2 = int(np.clip(x1 + rng.integers(-dx_range, dx_range + 1) - int(movement), 0, W - 1))
        y2 = int(np.clip(y1 + rng.integers(dy_range[0], dy_range[1] + 1), 0, H - 1))
        thick = int(rng.integers(thickness - 1, thickness + 2))
        specs.append(((x1, y1), (x2, y2), thick))
    return specs


def render_branches_mask(H, W, specs, k, *, line_type=cv2.LINE_AA):
    """
    Render first k branches from specs into a uint8 image, then binarize.
    """
    canvas = np.zeros((H, W), dtype=np.uint8)
    for (pt1, pt2, thick) in specs[:k]:
        cv2.line(canvas, pt1, pt2, color=255, thickness=int(thick), lineType=line_type)
    mask = (canvas > 0).astype(np.uint8)  # binary footprint
    return mask


def _refine_mask_to_density(mask_u8: np.ndarray, target: float, tol: float, max_steps: int = 12):
    """
    mask_u8: uint8 {0,1} mask
    Adjusts density via small boundary-only dilate/erode steps.
    Keeps branch texture.
    """
    H, W = mask_u8.shape
    HW = H * W

    def dens(m):
        return float(m.mean())

    m = mask_u8.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    for _ in range(max_steps):
        d = dens(m)
        if abs(d - target) <= tol:
            break

        if d > target:
            # Too dense: erode slightly (shrinks strokes)
            m2 = cv2.erode(m, kernel, iterations=1)
            # If erosion collapses too much, stop
            if dens(m2) >= d:
                break
            m = m2
        else:
            # Too sparse: dilate slightly (thickens strokes)
            m2 = cv2.dilate(m, kernel, iterations=1)
            if dens(m2) <= d:
                break
            m = m2

    return m


def branches_mask(
        H, W, *,
        coverage=0.3,
        tol=2e-2,
        seed=0,
        thickness=2,
        movement=0,
        num_specs=900,
        y_start_band=25,
        dx_range=25,
        dy_range=(-600, 30),
        line_type=cv2.LINE_AA,
        device="cpu",
        refine=True,
):
    target = float(np.clip(coverage, 0.0, 1.0))
    if target <= 0.0:
        return torch.zeros((H, W), dtype=torch.float32, device=device), {"k_used": 0, "density": 0.0}
    if target >= 1.0:
        return torch.ones((H, W), dtype=torch.float32, device=device), {"k_used": num_specs, "density": 1.0}

    specs = sample_branch_specs(
        H, W,
        seed=seed,
        num_specs=num_specs,
        thickness=thickness,
        movement=movement,
        y_start_band=y_start_band,
        dx_range=dx_range,
        dy_range=dy_range,
    )

    def occ_for_k(k):
        m = render_branches_mask(H, W, specs, k, line_type=line_type)  # uint8 {0,1}
        return float(m.mean()), m

    occ_hi, m_hi = occ_for_k(num_specs)
    if occ_hi < target - tol:
        mask = torch.from_numpy(m_hi.astype(np.float32)).to(device)
        return mask, {"k_used": num_specs, "density": occ_hi, "note": "hit max branches"}

    lo, hi = 1, int(num_specs)
    best = None
    while lo <= hi:
        mid = (lo + hi) // 2
        occ_mid, m_mid = occ_for_k(mid)
        best = (mid, occ_mid, m_mid)

        if abs(occ_mid - target) <= tol:
            break
        if occ_mid < target:
            lo = mid + 1
        else:
            hi = mid - 1

    k_best, occ_best, m_best = best

    if refine and abs(occ_best - target) > tol:
        m_best = _refine_mask_to_density(m_best.astype(np.uint8), target, tol, max_steps=10)
        occ_best = float(m_best.mean())

    mask = torch.from_numpy(m_best.astype(np.float32)).to(device)
    return mask, {"k_used": int(k_best), "density": float(occ_best)}


# BERNOULLI
def bernoulli_mask(H, W, coverage=0.10, seed=0, device=None):
    """
    Generate a 2D Bernoulli mask with the specified coverage.

    Args:
        H, W: Height and width of the mask
        coverage: Probability of a pixel being masked (1 - keep probability)
        seed: Random seed for reproducibility
        device: Device to create the mask on

    Returns:
        A 2D binary mask of shape (H, W) with values 0 (keep) or 1 (mask)
    """
    if device is None:
        device = torch.device('cpu')

    g = torch.Generator(device=device)
    g.manual_seed(int(seed))
    # Generate random values and threshold at coverage to create mask
    # Values < coverage become 1 (masked), values >= coverage become 0 (kept)
    mask = (torch.rand(H, W, generator=g, device=device) < coverage).float()

    return mask


def apply_structured_occlusion_static(video, mask2d):
    """
    video:  [T,1,H,W] or [B,T,1,H,W]
    mask2d: [H,W] or [B,H,W] float {0,1}
    returns: masked_video, mask (same shape as video)
    """
    mask2d = mask2d.to(device=video.device, dtype=video.dtype)

    if video.dim() == 4:
        T, C, H, W = video.shape
        if mask2d.dim() != 2:
            raise ValueError("For 4D video, mask2d must be [H,W].")
        mask = mask2d.view(1, 1, H, W).repeat(T, 1, 1, 1)
        return video * (1.0 - mask), mask

    if video.dim() == 5:
        B, T, C, H, W = video.shape
        if mask2d.dim() == 2:
            mask = mask2d.view(1, 1, 1, H, W).repeat(B, T, 1, 1, 1)
        elif mask2d.dim() == 3 and mask2d.shape[0] == B:
            mask = mask2d.view(B, 1, 1, H, W).repeat(1, T, 1, 1, 1)
        else:
            raise ValueError("For 5D video, mask2d must be [H,W] or [B,H,W].")
        return video * (1.0 - mask), mask

    raise ValueError("video must be [T,1,H,W] or [B,T,1,H,W]")


def build_mask_within_bounds(video, occ, p, *, seed=7, device=None, tol=2e-2):
    """
    Returns mask2d_full: [B,H,W] (or [H,W] if input video is 4D and we squeeze later),
    where each sample's mask is ONLY inside its motion bounds.
    Density within bounds is approximately p (depends on occluder tolerance).
    """
    if device is None:
        device = video.device

    # ensure batch dimension
    squeeze_B = False
    if video.dim() == 4:
        video_b = video.unsqueeze(0)  # [1,T,1,H,W]
        squeeze_B = True
    else:
        video_b = video

    B, T, C, H, W = video_b.shape
    bounds = compute_motion_bounds(video_b)  # (B,4) on CPU currently; ok

    mask_full = torch.zeros((B, H, W), device=device, dtype=torch.float32)

    for b in range(B):
        h0, h1, w0, w1 = [int(x) for x in bounds[b].tolist()]

        # no motion case (or empty bounds)
        if h1 < h0 or w1 < w0:
            continue

        hb = h1 - h0 + 1
        wb = w1 - w0 + 1

        # local mask (hb,wb) at coverage p
        # IMPORTANT: use a per-sample seed so different samples differ
        s = int(seed) + 1000 * b

        if occ == Occ.BERNOULLI:
            # bernoulli 2D inside bounds
            local = (torch.rand((hb, wb), device=device) < float(p)).float()

        elif occ == Occ.PERLIN:
            local = perlin_mask_torch(
                hb, wb,
                coverage=float(p),
                seed=s,
                device=device,
                scale=14.0, octaves=7, persistence=0.60, lacunarity=2.1
            )

        elif occ == Occ.CGOL:
            local = cgol_mask(
                hb, wb,
                density=float(p),
                seed=s,
                device=device,
                controlled=True, tol=0.02, iters=40
            )

        elif occ == Occ.TRIANGULAR:
            # if triangular uses cv2/numpy, it will return CPU -> move to device
            local, _ = triangular_mask_luxor_controlled_any(
                hb, wb,
                density=float(p),
                tol=0.02,
                seed=s,
                device="cpu",
            )
            local = local.to(device=device)

        elif occ == Occ.BRANCHES:
            local, _ = branches_mask(
                hb, wb,
                coverage=float(p),
                tol=2e-2,
                seed=s,
                thickness=2,
                movement=0,
                num_specs=900,
                y_start_band=min(25, hb),  # keep band sensible for small hb
                dx_range=25,
                dy_range=(-600, 30),
                refine=True,
                device="cpu",
            )
            local = local.to(device=device)

        else:
            raise ValueError(f"Unknown occlusion type: {occ}")

        # paste into full canvas
        mask_full[b, h0:h1 + 1, w0:w1 + 1] = local

        # optional: assert density only within bounds (debug)
        # dens = float(local.mean().item())
        # assert abs(dens - float(p)) <= 2*tol, f"{occ} b={b} dens={dens:.3f} p={float(p):.3f}"

    if squeeze_B:
        return mask_full[0]  # [H,W]
    return mask_full  # [B,H,W]


def apply_mask_to_video(video, occ, p, *, seed=7, device=None):
    if device is None:
        device = video.device

    if float(p) < 0.02:
        return video, torch.zeros_like(video)

    mask2d = build_mask_within_bounds(video, occ, p, seed=seed, device=device, tol=2e-2)
    return apply_structured_occlusion_static(video, mask2d)


def mask_density(mask):
    return float(mask.float().mean().item())


def mask_component_stats(mask2d: torch.Tensor) -> Dict[str, float]:
    """
    Simple structural descriptors for a 2D binary mask.
    This is important because 'structuredness' should not be only coverage.
    """
    x = np.ascontiguousarray(mask2d.detach().cpu().numpy().astype(np.uint8))
    if x.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape {x.shape}")

    if x.size == 0 or x.max() == 0:
        return {
            "mask_coverage_realized": 0.0,
            "mask_n_components": 0,
            "mask_largest_component_frac": 0.0,
            "mask_mean_component_frac": 0.0,
            "mask_boundary_frac": 0.0,
        }

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(x, connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA] if n_labels > 1 else np.array([], dtype=np.int64)
    largest = float(areas.max()) if areas.size else 0.0
    n_comp = int(areas.size)
    mean_area = float(areas.mean()) if areas.size else 0.0
    # crude boundary estimate
    kernel = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.uint8)
    dil = cv2.dilate(x, kernel, iterations=1)
    ero = cv2.erode(x, kernel, iterations=1)
    boundary = float((dil != ero).mean())
    return {
        "mask_coverage_realized": float(x.mean()),
        "mask_n_components": n_comp,
        "mask_largest_component_frac": largest / float(x.size),
        "mask_mean_component_frac": mean_area / float(x.size),
        "mask_boundary_frac": boundary,
    }


# ============================================================
# Model
# ============================================================

class SmallCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        z = self.net(x).flatten(1)
        return self.fc(z)


import torch
import torch.nn.functional as F


class FeatureHook:
    def __init__(self, module):
        self.out = None
        self.h = module.register_forward_hook(self._hook)

    def _hook(self, module, inp, out):
        self.out = out

    def close(self):
        self.h.remove()


@torch.no_grad()
def semantic_feature_loss(model, video_clean, video_occ, device=None, layers=("conv1", "conv2", "conv3"),
                          metric="cosine"):
    """
    video_*: [K,1,H,W]
    Returns dict: {layer: scalar loss}
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()

    # map names -> modules inside your Sequential
    # net = [conv1, relu, pool, conv2, relu, pool, conv3, relu, avgpool]
    name_to_module = {
        "conv1": model.net[0],
        "conv2": model.net[3],
        "conv3": model.net[6],
        "pool": model.net[8],  # AdaptiveAvgPool2d
    }

    hooks = {name: FeatureHook(name_to_module[name]) for name in layers}

    x0 = video_clean.to(device)
    x1 = video_occ.to(device)

    # forward clean, save hook outputs
    _ = model(x0)
    feats_clean = {k: hooks[k].out.detach() for k in layers}

    # forward occluded
    _ = model(x1)
    feats_occ = {k: hooks[k].out.detach() for k in layers}

    for h in hooks.values():
        h.close()

    losses = {}
    for k in layers:
        a = feats_clean[k].flatten(1)  # [K,D]
        b = feats_occ[k].flatten(1)
        if metric == "cosine":
            # cosine distance = 1 - cosine similarity
            sim = F.cosine_similarity(a, b, dim=1)  # [K]
            losses[k] = float((1.0 - sim).mean().item())
        elif metric == "l2":
            losses[k] = float(((a - b) ** 2).mean().sqrt().item())
        else:
            raise ValueError("metric must be 'cosine' or 'l2'")
    return losses


# ============================================================
# Train / validation / early stopping
# ============================================================

@dataclass
class TrainConfig:
    batch_size: int = 256
    lr: float = 1e-3
    max_epochs: int = 30
    patience: int = 5
    num_workers: int = 2
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    log_dir: Optional[str] = LOG_DIR
    ckpt_path: str = CKPT_DIR


@torch.no_grad()
def evaluate_classifier(model, loader, device: str):
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    for x, y, _ in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss_sum += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
    return {
        "loss": loss_sum / max(total, 1),
        "acc": correct / max(total, 1),
    }


def train_with_early_stopping(model, train_loader, val_loader, cfg: TrainConfig):
    device = cfg.device
    model.to(device)
    opt = Adam(model.parameters(), lr=cfg.lr)
    writer = SummaryWriter(cfg.log_dir) if cfg.log_dir else None

    best_val_loss = float("inf")
    best_state = None
    bad_epochs = 0
    history = []
    Path(cfg.ckpt_path).parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        total = 0
        correct = 0
        loss_sum = 0.0
        pbar = tqdm(train_loader, desc=f"train epoch {epoch}/{cfg.max_epochs}", leave=False)
        for x, y, _ in pbar:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += x.size(0)
            pbar.set_postfix(loss=f"{loss_sum / max(total, 1):.4f}", acc=f"{correct / max(total, 1):.3f}")

        train_metrics = {"loss": loss_sum / max(total, 1), "acc": correct / max(total, 1)}
        val_metrics = evaluate_classifier(model, val_loader, device)
        history.append({"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()},
                        **{f"val_{k}": v for k, v in val_metrics.items()}})

        if writer:
            writer.add_scalar("train/loss", train_metrics["loss"], epoch)
            writer.add_scalar("train/acc", train_metrics["acc"], epoch)
            writer.add_scalar("val/loss", val_metrics["loss"], epoch)
            writer.add_scalar("val/acc", val_metrics["acc"], epoch)

        improved = val_metrics["loss"] < best_val_loss - 1e-4
        if improved:
            best_val_loss = val_metrics["loss"]
            best_state = copy.deepcopy(model.state_dict())
            torch.save({"model_state": best_state, "history": history, "config": asdict(cfg)}, cfg.ckpt_path)
            bad_epochs = 0
        else:
            bad_epochs += 1

        print(
            f"epoch {epoch:02d} | train loss {train_metrics['loss']:.4f} acc {train_metrics['acc']:.3f} "
            f"| val loss {val_metrics['loss']:.4f} acc {val_metrics['acc']:.3f}"
        )

        if bad_epochs >= cfg.patience:
            print(f"Early stopping triggered after {epoch} epochs.")
            break

    if writer:
        writer.close()
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, pd.DataFrame(history)


def load_best_model(model_class, checkpoint_path=ROOT / "runs/moving_mnist/checkpoints/best_smallcnn.pt",
                    device=None):
    """
    Load the best trained model from a checkpoint that contains a 'model_state' key.

    Args:
        model_class: The model class (e.g., SmallCNN) – an instance will be created.
        checkpoint_path: Path to the .pt file containing the checkpoint dict.
        device: Device to load the model onto (default: auto-detect).

    Returns:
        model: Loaded model in evaluation mode.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate the model (assumes num_classes=10 default, adjust if needed)
    model = model_class().to(device)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract the model state dict – handle both direct state dict and wrapped dict
    if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
        state_dict = checkpoint['model_state']
    else:
        # Fallback: assume checkpoint is the state dict itself
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()
    return model


# ============================================================
# Video-level evaluation
# ============================================================


def take_frames(video, k=5):
    T = video.shape[0]
    if k >= T:
        return video, torch.arange(T)
    idx = torch.linspace(0, T - 1, steps=k).long()
    return video[idx], idx


@torch.no_grad()
def video_metrics_from_logits(logits, y, eps=1e-8):
    probs = F.softmax(logits, dim=1)
    pred = probs.argmax(1)
    conf = probs.max(1).values
    y = y.long()
    p_true = probs[:, y]
    avg_logp = F.log_softmax(logits, dim=1).mean(dim=0)
    video_probs = torch.softmax(avg_logp, dim=0)
    video_pred = int(video_probs.argmax().item())
    video_true = float(video_probs[y].item())
    video_conf = float(video_probs.max().item())
    nll = float(-torch.log(video_probs[y] + eps).item())
    brier = float(torch.sum((video_probs - F.one_hot(y, num_classes=10).float()) ** 2).item())
    return {
        "frame_acc": float((pred == y).float().mean().item()),
        "frame_conf_mean": float(conf.mean().item()),
        "frame_p_true_mean": float(p_true.mean().item()),
        "video_pred": video_pred,
        "video_correct": bool(video_pred == int(y.item())),
        "video_conf": video_conf,
        "video_p_true": video_true,
        "video_nll": nll,
        "video_brier": brier,
        "overconfident_error": float(video_conf - float(video_pred == int(y.item()))),
    }


@torch.no_grad()
def evaluate_occlusion_sweep(
        model,
        moving_test_ds,
        p_values: List[float] | torch.Tensor,
        out_csv: Optional[str] = None,
        device: Optional[str] = None,
        eval_cfg: EvalConfig = EvalConfig(),
):
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    rows = []

    for idx in tqdm(range(len(moving_test_ds)), desc="occlusion sweep"):
        video, label, _, meta = moving_test_ds[idx]
        video = video.to(device)
        video_k, frame_idx = take_frames(video, k=eval_cfg.k_frames)
        label_t = torch.tensor(label, device=device)

        for p in p_values:
            for occ in Occ:
                for mask_seed in range(eval_cfg.n_mask_seeds):
                    # print(f"[DBG] idx={idx} occ={occ.name} p={p} mask_seed={mask_seed}")
                    masked, mask = apply_mask_to_video(video_k, occ, float(p), seed=10_000 * idx + 100 * mask_seed + 7,
                                                       device=device)
                    # compute semantic loss
                    feat_loss = semantic_feature_loss(
                        model, video_k, masked,
                        layers=("conv1", "conv2", "conv3"),
                        metric="cosine",
                    )
                    logits = model(masked)
                    m = video_metrics_from_logits(logits, label_t)
                    mask2d = mask[0, 0] if mask.dim() == 4 else mask[0, 0, 0]
                    struct = mask_component_stats(mask2d)
                    rows.append({
                        "video_idx": idx,
                        "label": int(label),
                        "base_idx": int(meta["base_idx"]),
                        "coverage_target": float(p),
                        "occ": occ.name.lower(),
                        "mask_seed": mask_seed,
                        "k_frames": int(video_k.shape[0]),
                        "sem_cos_conv1": feat_loss["conv1"],
                        "sem_cos_conv2": feat_loss["conv2"],
                        "sem_cos_conv3": feat_loss["conv3"],
                        **struct,
                        **m,
                    })

    df = pd.DataFrame(rows)
    # add after df is created
    sem_cols = [c for c in df.columns if c.startswith("sem_")]

    agg_dict = dict(
        video_acc=("video_correct", "mean"),
        video_p_true=("video_p_true", "mean"),
        video_conf=("video_conf", "mean"),
        video_nll=("video_nll", "mean"),
        video_brier=("video_brier", "mean"),
        frame_acc=("frame_acc", "mean"),
        frame_p_true=("frame_p_true_mean", "mean"),
        realized_cov=("mask_coverage_realized", "mean"),
        largest_comp=("mask_largest_component_frac", "mean"),
        n_components=("mask_n_components", "mean"),
    )

    # semantic loss (mean over samples)
    for c in sem_cols:
        agg_dict[c] = (c, "mean")

    summary = (
        df.groupby(["occ", "coverage_target"])
        .agg(**agg_dict)
        .reset_index()
    )

    if out_csv:
        out_csv = Path(out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv.with_name(out_csv.stem + "_per_video.csv"), index=False)
        summary.to_csv(out_csv.with_name(out_csv.stem + "_summary.csv"), index=False)
    return df, summary


def ensure_parent_dir(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _mean_ci(x, z=1.96):
    x = pd.Series(x).dropna().to_numpy()
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    m = float(np.mean(x))
    if len(x) == 1:
        return m, m, m
    se = float(np.std(x, ddof=1) / math.sqrt(len(x)))
    return m, m - z * se, m + z * se


def _aggregate_with_ci(df, group_cols, value_col):
    rows = []
    for keys, g in df.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        mean, lo, hi = _mean_ci(g[value_col])
        row = {col: key for col, key in zip(group_cols, keys)}
        row.update({
            value_col: mean,
            f"{value_col}_lo": lo,
            f"{value_col}_hi": hi,
            "n": len(g),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def reliability_table(df, prob_col="video_conf", correct_col="video_correct", n_bins=10):
    x = df[[prob_col, correct_col]].dropna().copy()
    x["bin"] = pd.cut(
        x[prob_col],
        bins=np.linspace(0.0, 1.0, n_bins + 1),
        include_lowest=True,
        duplicates="drop",
    )
    out = (
        x.groupby("bin", observed=False)
        .agg(
            conf=(prob_col, "mean"),
            acc=(correct_col, "mean"),
            count=(correct_col, "size"),
        )
        .reset_index()
    )
    return out


def plot_semantic_loss_curves(df, out_dir, coverage_col="coverage_target", occ_col="occ"):
    sem_cols = [c for c in df.columns if c.startswith("sem_")]
    if not sem_cols:
        return

    # one row per semantic metric
    n = len(sem_cols)
    fig, axes = plt.subplots(n, 1, figsize=(10, 3.5 * n), constrained_layout=True)
    if n == 1:
        axes = [axes]

    for ax, sem in zip(axes, sem_cols):
        agg = _aggregate_with_ci(df, [occ_col, coverage_col], sem)
        for occ, g in agg.groupby(occ_col):
            g = g.sort_values(coverage_col)
            ax.plot(g[coverage_col], g[sem], marker="o", label=occ)
            ax.fill_between(g[coverage_col], g[f"{sem}_lo"], g[f"{sem}_hi"], alpha=0.2)

        ax.set_title(f"Semantic feature loss vs coverage ({sem})")
        ax.set_xlabel("Target occlusion coverage")
        ax.set_ylabel(sem)
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.savefig(Path(out_dir) / "semantic_loss_vs_coverage.png", dpi=160)
    plt.close(fig)


def plot_semantic_vs_accuracy(df, out_dir, occ_col="occ"):
    sem_cols = [c for c in df.columns if c.startswith("sem_")]
    if not sem_cols:
        return

    # pick a representative layer (conv3 tends to be most "semantic")
    sem = "sem_cos_conv3" if "sem_cos_conv3" in sem_cols else sem_cols[-1]

    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    for occ, g in df.groupby(occ_col):
        ax.scatter(g[sem], g["video_correct"], alpha=0.15, s=12, label=occ)

    ax.set_xlabel(sem)
    ax.set_ylabel("Video correct (0/1)")
    ax.set_title(f"Accuracy vs semantic feature loss ({sem})")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(Path(out_dir) / "semantic_loss_vs_accuracy.png", dpi=160)
    plt.close(fig)


def expected_calibration_error(df, prob_col="video_conf", correct_col="video_correct", n_bins=10):
    tab = reliability_table(df, prob_col=prob_col, correct_col=correct_col, n_bins=n_bins)
    n = tab["count"].sum()
    if n == 0:
        return np.nan
    return float(np.sum((tab["count"] / n) * np.abs(tab["acc"] - tab["conf"])))


def plot_occlusion_results(
        per_video_csv,
        out_dir="outputs/plots",
        coverage_col="coverage_target",
        occ_col="occ",
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(per_video_csv)
    # Defensive typing
    bool_cols = ["video_correct"]
    for c in bool_cols:
        if c in df.columns:
            df[c] = df[c].astype(float)

    metric_specs = [
        ("video_correct", "Video accuracy"),
        ("video_conf", "Video confidence"),
        ("video_p_true", "True-class probability"),
        ("video_nll", "Video NLL"),
        ("video_brier", "Video Brier score"),
        ("frame_acc", "Frame accuracy"),
        ("mask_coverage_realized", "Realized coverage"),
        ("mask_largest_component_frac", "Largest occluder component fraction"),
        ("mask_n_components", "Number of occluder components"),
    ]
    sem_cols = [c for c in df.columns if c.startswith("sem_")]
    for c in sem_cols:
        metric_specs.append((c, f"Semantic loss: {c.replace('sem_', '')}"))

    # 1) Main coverage curves
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    axes = axes.ravel()

    main_metrics = [
        ("video_correct", "Video accuracy"),
        ("video_conf", "Video confidence"),
        ("video_nll", "Video NLL"),
        ("video_brier", "Video Brier"),
    ]

    for ax, (metric, title) in zip(axes, main_metrics):
        agg = _aggregate_with_ci(df, [occ_col, coverage_col], metric)
        for occ, g in agg.groupby(occ_col):
            g = g.sort_values(coverage_col)
            ax.plot(g[coverage_col], g[metric], marker="o", label=occ)
            ax.fill_between(
                g[coverage_col],
                g[f"{metric}_lo"],
                g[f"{metric}_hi"],
                alpha=0.2,
            )
        ax.set_title(title)
        ax.set_xlabel("Target occlusion coverage")
        ax.set_ylabel(title)
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.suptitle("Occlusion robustness summary", fontsize=14)
    fig.savefig(out_dir / "coverage_summary.png", dpi=160)
    plt.close(fig)

    # 2) Confidence vs accuracy
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    agg_acc = _aggregate_with_ci(df, [occ_col, coverage_col], "video_correct")
    agg_conf = _aggregate_with_ci(df, [occ_col, coverage_col], "video_conf")
    merged = agg_acc.merge(agg_conf, on=[occ_col, coverage_col], suffixes=("_acc", "_conf"))

    for occ, g in merged.groupby(occ_col):
        g = g.sort_values(coverage_col)
        ax.plot(g["video_conf"], g["video_correct"], marker="o", label=occ)
        for _, row in g.iterrows():
            ax.annotate(f"{row[coverage_col]:.1f}", (row["video_conf"], row["video_correct"]), fontsize=8)

    lims = [0, 1]
    ax.plot(lims, lims, linestyle="--")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Mean confidence")
    ax.set_ylabel("Mean accuracy")
    ax.set_title("Confidence vs accuracy by mask family\n(labels are occlusion coverage)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(out_dir / "confidence_vs_accuracy.png", dpi=160)
    plt.close(fig)

    # 3) Structural explanation plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    ax = axes[0]
    for occ, g in df.groupby(occ_col):
        ax.scatter(
            g["mask_largest_component_frac"],
            g["video_correct"],
            alpha=0.15,
            s=12,
            label=occ,
        )
    ax.set_xlabel("Largest occluder component fraction")
    ax.set_ylabel("Video correct (0/1)")
    ax.set_title("Performance vs largest occluder chunk")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for occ, g in df.groupby(occ_col):
        ax.scatter(
            g["mask_n_components"],
            g["video_correct"],
            alpha=0.15,
            s=12,
            label=occ,
        )
    ax.set_xlabel("Number of occluder components")
    ax.set_ylabel("Video correct (0/1)")
    ax.set_title("Performance vs mask fragmentation")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.savefig(out_dir / "structure_vs_performance.png", dpi=160)
    plt.close(fig)

    # 4) Reliability diagrams by family
    families = sorted(df[occ_col].dropna().unique().tolist())
    n = len(families)
    ncols = min(2, max(1, n))
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows), constrained_layout=True)
    if n == 1:
        axes = np.array([axes])
    axes = np.array(axes).reshape(-1)

    ece_rows = []
    for ax, occ in zip(axes, families):
        g = df[df[occ_col] == occ].copy()
        tab = reliability_table(g, prob_col="video_conf", correct_col="video_correct", n_bins=10)
        ece = expected_calibration_error(g, prob_col="video_conf", correct_col="video_correct", n_bins=10)
        ece_rows.append({"occ": occ, "ece": ece})

        ax.plot([0, 1], [0, 1], linestyle="--")
        ax.plot(tab["conf"], tab["acc"], marker="o")
        ax.set_title(f"{occ} (ECE={ece:.3f})")
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Accuracy")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)

    for j in range(len(families), len(axes)):
        axes[j].axis("off")

    fig.savefig(out_dir / "reliability_by_family.png", dpi=160)
    plt.close(fig)
    plot_semantic_loss_curves(df, out_dir, coverage_col=coverage_col, occ_col=occ_col)
    plot_semantic_vs_accuracy(df, out_dir, occ_col=occ_col)
    pd.DataFrame(ece_rows).to_csv(out_dir / "ece_by_family.csv", index=False)

    # 5) Save compact aggregated tables
    summary_tables = {}
    for metric, _ in metric_specs:
        if metric in df.columns:
            summary_tables[metric] = _aggregate_with_ci(df, [occ_col, coverage_col], metric)

    for metric, table in summary_tables.items():
        table.to_csv(out_dir / f"summary_{metric}.csv", index=False)

    return {
        "n_rows": len(df),
        "families": families,
        "out_dir": str(out_dir),
    }


# ============================================================
# Main entry point
# ============================================================

def build_loaders(seed: int = 0, batch_size: int = 256, num_workers: int = 2):
    train_base, val_base, test_base = build_mnist_splits()
    train_moving = RollingMovingMNIST(train_base, T=20, canvas=64, seed=1000 + seed)
    val_moving = RollingMovingMNIST(val_base, T=20, canvas=64, seed=2000 + seed)
    test_moving = RollingMovingMNIST(test_base, T=20, canvas=64, seed=3000 + seed)

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
    train_loader, val_loader, test_moving = build_loaders(seed=42, batch_size=train_cfg.batch_size,
                                                          num_workers=train_cfg.num_workers)

    model = SmallCNN()
    if train:
        model, history = train_with_early_stopping(model, train_loader, val_loader, train_cfg)
        # After training loop, before history.to_csv(...)
        history_path = OUT_DIR / "train_history.csv"
        ensure_dir(history_path)
        history.to_csv(history_path, index=False)
    else:
        # Load the best model
        model = load_best_model(SmallCNN, CKPT_DIR / "best_smallcnn.pt")

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
    main()
