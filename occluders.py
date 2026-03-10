import math
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from enum import Enum, auto
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


# ----------------------------------------------------------------------
# Enums and configs
# ----------------------------------------------------------------------

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


# ----------------------------------------------------------------------
# CGOL
# ----------------------------------------------------------------------

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


# ----------------------------------------------------------------------
# Triangular (uses OpenCV)
# ----------------------------------------------------------------------

_LOCAL_TRI = np.array([[1.0, 0.0], [0.0, -1.0], [-1.0, 0.0]], dtype=np.float32)


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


# ----------------------------------------------------------------------
# Perlin
# ----------------------------------------------------------------------

def perlin_field(
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

        n00 = gx[y0, x0] * sx + gy[y0, x0] * sy
        n10 = gx[y0, x1] * (sx - 1) + gy[y0, x1] * sy
        n01 = gx[y1, x0] * sx + gy[y1, x0] * (sy - 1)
        n11 = gx[y1, x1] * (sx - 1) + gy[y1, x1] * (sy - 1)

        nx0 = n00 + u * (n10 - n00)
        nx1 = n01 + u * (n11 - n01)
        out = nx0 + v * (nx1 - nx0)

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

    total = total / max(amp_sum, 1e-8)
    total = (total + 1.0) / 2.0
    return total.clamp(0, 1)


def perlin_mask(
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
    f = perlin_field(
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


# ----------------------------------------------------------------------
# Branches (uses OpenCV)
# ----------------------------------------------------------------------

def sample_branch_specs(H, W, *, seed=0, num_specs=400, thickness=1, movement=0,
                        y_start_band=50, dx_range=50, dy_range=(-300, 100)):
    rng = np.random.default_rng(seed)
    y_start_band = int(np.clip(y_start_band, 1, H))

    specs = []
    for _ in range(int(num_specs)):
        x1 = int(rng.integers(0, W))
        y1 = int(rng.integers(H - y_start_band, H))
        x2 = int(np.clip(x1 + rng.integers(-dx_range, dx_range + 1) - int(movement), 0, W - 1))
        y2 = int(np.clip(y1 + rng.integers(dy_range[0], dy_range[1] + 1), 0, H - 1))
        thick = int(rng.integers(thickness, thickness + 2))
        specs.append(((x1, y1), (x2, y2), thick))
    return specs


def render_branches_mask(H, W, specs, k, *, line_type=cv2.LINE_AA):
    canvas = np.zeros((H, W), dtype=np.uint8)
    for (pt1, pt2, thick) in specs[:k]:
        cv2.line(canvas, pt1, pt2, color=255, thickness=int(thick), lineType=line_type)
    mask = (canvas > 0).astype(np.uint8)
    return mask


def branches_mask(
        H, W, *,
        coverage=0.3,
        tol=2e-2,
        seed=0,
        thickness=1,
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

    # Binary search for k
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

    # ---- Local search around k_best (cheap fine-tuning) ----
    candidates = [k_best]
    if k_best > 1:
        candidates.append(k_best - 1)
    if k_best < num_specs:
        candidates.append(k_best + 1)

    best_err = abs(occ_best - target)
    for k_cand in candidates:
        if k_cand == k_best:
            continue
        occ_cand, m_cand = occ_for_k(k_cand)
        err = abs(occ_cand - target)
        if err < best_err:
            best_err = err
            k_best, occ_best, m_best = k_cand, occ_cand, m_cand
            if best_err <= tol:
                break

    # ---- Limited refinement if still needed ----
    if refine and best_err > tol:
        # Quick refinement (max 5 steps) – early stop as soon as tolerance is met
        m_best = _refine_mask_to_density(m_best.astype(np.uint8), target, tol, max_steps=5)
        occ_best = float(m_best.mean())

    mask = torch.from_numpy(m_best.astype(np.float32)).to(device)
    return mask, {"k_used": int(k_best), "density": float(occ_best)}


def _refine_mask_to_density(mask_u8: np.ndarray, target: float, tol: float, max_steps: int = 5):
    """Quick refinement – stops as soon as tolerance is met or no change."""
    m = mask_u8.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    for _ in range(max_steps):
        d = m.mean()
        if abs(d - target) <= tol:
            break
        if d > target:
            m_new = cv2.erode(m, kernel, iterations=1)
        else:
            m_new = cv2.dilate(m, kernel, iterations=1)
        if np.array_equal(m, m_new):
            break
        m = m_new
    return m


# ----------------------------------------------------------------------
# Bernoulli
# ----------------------------------------------------------------------

def bernoulli_mask(H, W, coverage=0.10, seed=0, device=None):
    if device is None:
        device = torch.device('cpu')

    g = torch.Generator(device=device)
    g.manual_seed(int(seed))
    mask = (torch.rand(H, W, generator=g, device=device) < coverage).float()
    return mask


# ----------------------------------------------------------------------
# Application and utilities
# ----------------------------------------------------------------------

def compute_motion_bounds(video_b):
    """
    video_b: [B,T,1,H,W]  (assumes one digit moving)
    returns (B,4) tensor with [h0, h1, w0, w1] inclusive bounds.
    """
    B, T, C, H, W = video_b.shape
    bounds = []
    for b in range(B):
        nonzero = (video_b[b] > 0.1).float()  # [T,1,H,W]
        h_idx = torch.where(nonzero.sum(dim=(0, 1, 3)) > 0)[0]  # over T,C,W -> H
        w_idx = torch.where(nonzero.sum(dim=(0, 1, 2)) > 0)[0]  # over T,C,H -> W
        if len(h_idx) == 0 or len(w_idx) == 0:
            bounds.append([0, 0, 0, 0])
        else:
            h0, h1 = h_idx.min().item(), h_idx.max().item()
            w0, w1 = w_idx.min().item(), w_idx.max().item()
            bounds.append([h0, h1, w0, w1])
    return torch.tensor(bounds, dtype=torch.long)


def apply_structured_occlusion_static(video, mask2d):
    """
    video: [T,1,H,W] or [B,T,1,H,W]
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
    if device is None:
        device = video.device

    squeeze_B = False
    if video.dim() == 4:
        video_b = video.unsqueeze(0)
        squeeze_B = True
    else:
        video_b = video

    B, T, C, H, W = video_b.shape
    bounds = compute_motion_bounds(video_b)

    mask_full = torch.zeros((B, H, W), device=device, dtype=torch.float32)

    for b in range(B):
        h0, h1, w0, w1 = [int(x) for x in bounds[b].tolist()]
        if h1 < h0 or w1 < w0:
            continue

        hb = h1 - h0 + 1
        wb = w1 - w0 + 1
        s = int(seed) + 1000 * b

        if occ == Occ.BERNOULLI:
            local = (torch.rand((hb, wb), device=device) < float(p)).float()
        elif occ == Occ.PERLIN:
            local = perlin_mask(
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
                thickness=1,
                movement=0,
                num_specs=900,
                y_start_band=min(25, hb),
                dx_range=25,
                dy_range=(-600, 30),
                refine=True,
                device="cpu",
            )
            local = local.to(device=device)
        else:
            raise ValueError(f"Unknown occlusion type: {occ}")

        mask_full[b, h0:h1 + 1, w0:w1 + 1] = local

    if squeeze_B:
        return mask_full[0]
    return mask_full


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
