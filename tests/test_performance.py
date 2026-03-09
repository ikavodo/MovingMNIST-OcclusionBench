import pytest
import time
from occluders import Occ, apply_mask_to_video
from test_occluders import dummy_video

@pytest.mark.slow
def test_occluder_speed(dummy_video):
    video = dummy_video
    p_values = [0.1, 0.3, 0.5, 0.7, 0.9]
    repeats = 3
    for occ in Occ:
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            for p in p_values:
                _, _ = apply_mask_to_video(video, occ, p, seed=123)
            t1 = time.perf_counter()
            times.append(t1 - t0)
        avg = sum(times) / repeats
        print(f"{occ.name}: {avg:.3f}s for {len(p_values)} p values")
        # No assertion, just informational