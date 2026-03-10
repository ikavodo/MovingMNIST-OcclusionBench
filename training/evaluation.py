import torch
import torch.nn.functional as F
import pandas as pd
from tqdm.auto import tqdm
from pathlib import Path

from torch.utils.data import DataLoader

from occluders import (
    Occ, apply_mask_to_video, mask_component_stats
)
from loss.metrics import semantic_feature_loss
from training.config import EvalConfig


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
        p_values,
        out_csv=None,
        device=None,
        eval_cfg: EvalConfig = EvalConfig(),
        batch_size: int = 64,
        num_workers: int = 4,
):
    if device is None:
        device = next(model.parameters()).device
    else:
        device = torch.device(device)

    model.eval()
    model = model.to(device)

    # Use DataLoader for batching
    loader = DataLoader(
        moving_test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    rows = []
    k = eval_cfg.k_frames
    n_mask_seeds = eval_cfg.n_mask_seeds
    n_occ = len(Occ)
    n_p = len(p_values)
    n_batches = len(loader)
    total_steps = n_batches * n_p * n_occ * n_mask_seeds

    use_autocast = (device.type == "cuda")

    pbar = tqdm(total=total_steps, desc="occlusion eval", unit="combo")

    for batch_idx, (videos, labels, _, metas) in enumerate(loader):
        videos = videos.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        videos_k, frame_indices = take_frames_batched(videos, k)

        for p in p_values:
            p_float = float(p)

            for occ in Occ:
                for mask_seed in range(n_mask_seeds):
                    base_seed = (
                            10_000 * (batch_idx * batch_size) +
                            100 * mask_seed + 7
                    )

                    masked, mask = apply_mask_to_video(
                        videos_k, occ, p_float,
                        seed=base_seed, device=device
                    )

                    B, k, C, H, W = masked.shape
                    flat = masked.view(B * k, C, H, W)

                    with torch.autocast(device_type="cuda", enabled=use_autocast):
                        logits_flat = model(flat)

                    logits = logits_flat.view(B, k, 10)

                    for b in range(B):
                        video_logits = logits[b]
                        label = labels[b]
                        meta = {
                            key: value[b] if isinstance(value, torch.Tensor) else value[b]
                            for key, value in metas.items()
                        }

                        m = video_metrics_from_logits(video_logits, label)

                        feat_loss = semantic_feature_loss(
                            model, videos_k[b], masked[b],
                            layers=("conv1", "conv2", "conv3"),
                            metric="cosine",
                        )

                        mask2d = mask[b, 0, 0]
                        struct = mask_component_stats(mask2d)

                        rows.append({
                            "video_idx": batch_idx * batch_size + b,
                            "label": label.item(),
                            "base_idx": int(meta.get("base_idx", -1)),
                            "coverage_target": p_float,
                            "occ": occ.name.lower(),
                            "mask_seed": mask_seed,
                            "k_frames": k,
                            "sem_cos_conv1": feat_loss["conv1"],
                            "sem_cos_conv2": feat_loss["conv2"],
                            "sem_cos_conv3": feat_loss["conv3"],
                            **struct,
                            **m,
                        })

                    pbar.update(1)
                    pbar.set_postfix(
                        batch=f"{batch_idx + 1}/{n_batches}",
                        occ=occ.name.lower(),
                        p=f"{p_float:.1f}",
                        seed=mask_seed,
                        rows=len(rows),
                    )

    pbar.close()

    # Create DataFrame and summaries (same as before)
    df = pd.DataFrame(rows)
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


def take_frames_batched(videos, k):
    """
    videos: [B, T, C, H, W]
    returns: [B, k, C, H, W], indices
    """
    B, T, C, H, W = videos.shape
    if k >= T:
        return videos, torch.arange(T, device=videos.device).expand(B, -1)
    # Uniform sampling
    indices = torch.linspace(0, T - 1, steps=k, device=videos.device).long()  # [k]
    indices = indices.view(1, -1).expand(B, -1)  # [B, k]
    # Gather along T dimension
    videos_k = torch.gather(
        videos, 1,
        indices.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, C, H, W)
    )  # [B, k, C, H, W]
    return videos_k, indices


def load_best_model(model_class, checkpoint_path, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_class().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
        state_dict = checkpoint['model_state']
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return model
