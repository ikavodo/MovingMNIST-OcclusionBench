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
    if device is None:
        device = next(model.parameters()).device
    model.eval()

    name_to_module = {
        "conv1": model.features[0],
        "conv2": model.features[4],
        "conv3": model.features[8],
        "pool": model.features[11],
    }

    hooks = {name: FeatureHook(name_to_module[name]) for name in layers}

    x0 = video_clean.to(device)
    x1 = video_occ.to(device)

    _ = model(x0)
    feats_clean = {k: hooks[k].out.detach() for k in layers}
    _ = model(x1)
    feats_occ = {k: hooks[k].out.detach() for k in layers}

    for h in hooks.values():
        h.close()

    losses = {}
    for k in layers:
        a = feats_clean[k].flatten(1)
        b = feats_occ[k].flatten(1)
        if metric == "cosine":
            sim = F.cosine_similarity(a, b, dim=1)
            losses[k] = float((1.0 - sim).mean().item())
        elif metric == "l2":
            losses[k] = float(((a - b) ** 2).mean().sqrt().item())
        else:
            raise ValueError("metric must be 'cosine' or 'l2'")
    return losses
