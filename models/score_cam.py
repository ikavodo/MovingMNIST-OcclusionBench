"""Score-CAM: Score-Weighted Visual Explanations for Convolutional Neural Networks.

Wang et al., CVPR Workshops 2020.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Hook after ReLU to capture non-negative post-activation feature maps.
# Indices into SmallCNN.features: [Conv,BN,ReLU,Pool, Conv,BN,ReLU,Pool, Conv,BN,ReLU, AvgPool]
_LAYER_IDX = {
    "conv1": 2,   # ReLU output: [B, 32, 64, 64]
    "conv2": 6,   # ReLU output: [B, 64, 32, 32]
    "conv3": 10,  # ReLU output: [B, 128, 16, 16]
}


class ScoreCAM:
    """Score-CAM saliency maps for SmallCNN.

    Usage::

        with ScoreCAM(model) as cam:
            saliency, class_idx = cam(x)   # x: [1, 1, H, W]
    """

    def __init__(self, model: nn.Module, target_layer: str = "conv3"):
        if target_layer not in _LAYER_IDX:
            raise ValueError(f"target_layer must be one of {list(_LAYER_IDX)}")
        self.model = model
        self._activations: torch.Tensor | None = None
        self._hook = model.features[_LAYER_IDX[target_layer]].register_forward_hook(
            lambda _m, _i, o: setattr(self, "_activations", o.detach())
        )

    # ------------------------------------------------------------------
    # Core computation
    # ------------------------------------------------------------------

    def __call__(
        self,
        x: torch.Tensor,
        class_idx: int | None = None,
    ) -> tuple[torch.Tensor, int]:
        """Compute Score-CAM saliency map.

        Args:
            x: Input tensor ``[1, 1, H, W]``.
            class_idx: Target class index. Defaults to argmax of model output.

        Returns:
            saliency: ``[H, W]`` tensor with values in ``[0, 1]``.
            class_idx: Target class used.
        """
        self.model.eval()
        device = next(self.model.parameters()).device
        x = x.to(device)

        with torch.no_grad():
            logits = self.model(x)
        if class_idx is None:
            class_idx = int(logits.argmax(dim=1))

        acts = self._activations          # [1, C, h, w]
        C, h, w = acts.shape[1], acts.shape[2], acts.shape[3]
        H, W = x.shape[2], x.shape[3]

        # Upsample every activation channel to input resolution and normalise to [0, 1].
        acts_up = F.interpolate(
            acts[0].unsqueeze(1),         # [C, 1, h, w]
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)                       # [C, H, W]

        a_min = acts_up.flatten(1).min(1).values.view(C, 1, 1)
        a_max = acts_up.flatten(1).max(1).values.view(C, 1, 1)
        acts_norm = (acts_up - a_min) / (a_max - a_min).clamp(min=1e-8)  # [C, H, W]

        # One batched forward pass through all C masked inputs.
        masked = x * acts_norm.unsqueeze(1)   # [C, 1, H, W]
        with torch.no_grad():
            scores = torch.softmax(self.model(masked), dim=1)[:, class_idx]  # [C]

        # Weighted sum of (unnormalised) activation maps → upsample → ReLU → normalise.
        saliency = (scores.view(C, 1, 1) * acts[0]).sum(0)  # [h, w]
        saliency = F.interpolate(
            saliency.unsqueeze(0).unsqueeze(0),
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        saliency = saliency.relu()
        s_max = saliency.max()
        if s_max > 0:
            saliency = saliency / s_max

        return saliency.cpu(), class_idx

    # ------------------------------------------------------------------
    # Resource management
    # ------------------------------------------------------------------

    def close(self) -> None:
        self._hook.remove()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()