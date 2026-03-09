import pytest
import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.evaluation import video_metrics_from_logits

def test_video_metrics_from_logits():
    # Create dummy logits for 5 frames, 10 classes
    logits = torch.randn(5, 10)
    # Make class 3 the true label
    label = torch.tensor(3)
    # Set logits so that class 3 is highest for all frames
    logits[:, 3] += 5.0
    metrics = video_metrics_from_logits(logits, label)
    # Frame accuracy should be 1.0
    assert metrics["frame_acc"] == 1.0
    # Video correct should be True
    assert metrics["video_correct"] is True
    # Confidence should be high (close to 1)
    assert metrics["video_conf"] > 0.9
    # Check other keys exist
    expected_keys = {"frame_acc", "frame_conf_mean", "frame_p_true_mean",
                     "video_pred", "video_correct", "video_conf",
                     "video_p_true", "video_nll", "video_brier", "overconfident_error"}
    assert expected_keys.issubset(metrics.keys())