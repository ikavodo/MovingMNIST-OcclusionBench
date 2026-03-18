import copy
from dataclasses import asdict
from pathlib import Path
import pandas as pd
from tqdm.auto import tqdm
import torch
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from torch.utils.tensorboard import SummaryWriter
from .config import TrainConfig


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
    opt = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
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
