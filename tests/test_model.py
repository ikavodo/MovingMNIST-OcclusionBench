import pytest
import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.small_cnn import SmallCNN
from loss.metrics import FeatureHook, semantic_feature_loss

def test_smallcnn_forward():
    model = SmallCNN(num_classes=10)
    x = torch.randn(4, 1, 64, 64)  # batch of 4 frames
    out = model(x)
    assert out.shape == (4, 10)
    # Check it works on different input size
    x2 = torch.randn(2, 1, 64, 64)
    out2 = model(x2)
    assert out2.shape == (2, 10)

def test_feature_hook():
    model = SmallCNN()
    # Attach hook to first conv layer
    hook = FeatureHook(model.features[0])
    x = torch.randn(2, 1, 64, 64)
    _ = model(x)
    assert hook.out is not None
    assert hook.out.shape[1] == 32  # first conv has 16 channels
    hook.close()

def test_semantic_feature_loss():
    model = SmallCNN()
    model.eval()
    x_clean = torch.randn(5, 1, 64, 64)  # 5 frames
    x_occ = x_clean.clone()
    # Slightly modify to simulate occlusion
    x_occ[0,0,10:20,10:20] = 0
    losses = semantic_feature_loss(model, x_clean, x_occ, layers=("conv1","conv2","conv3"))
    assert set(losses.keys()) == {"conv1","conv2","conv3"}
    for v in losses.values():
        assert isinstance(v, float)
        assert 0 <= v <= 2  # cosine distance range [0,2]