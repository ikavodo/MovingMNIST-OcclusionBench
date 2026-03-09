import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import _mean_ci, _aggregate_with_ci, reliability_table, expected_calibration_error

def test_mean_ci():
    data = [1,2,3,4,5]
    m, lo, hi = _mean_ci(data)
    assert m == 3.0
    assert lo < m < hi
    # Single element
    m, lo, hi = _mean_ci([10])
    assert m == 10
    assert lo == 10
    assert hi == 10
    # Empty
    m, lo, hi = _mean_ci([])
    assert np.isnan(m)

def test_aggregate_with_ci():
    df = pd.DataFrame({
        "group": ["A","A","B","B"],
        "value": [1,3,5,7]
    })
    agg = _aggregate_with_ci(df, ["group"], "value")
    assert agg.shape[0] == 2
    a_row = agg[agg["group"]=="A"].iloc[0]
    assert a_row["value"] == 2.0
    assert "value_lo" in a_row and "value_hi" in a_row
    assert "n" in a_row and a_row["n"] == 2

def test_reliability_table():
    df = pd.DataFrame({
        "video_conf": [0.1,0.2,0.8,0.9],
        "video_correct": [0,0,1,1]
    })
    tab = reliability_table(df, n_bins=2)
    assert tab.shape[0] == 2
    assert "conf" in tab.columns
    assert "acc" in tab.columns
    assert "count" in tab.columns

def test_expected_calibration_error():
    df = pd.DataFrame({
        "video_conf": [0.1,0.2,0.8,0.9],
        "video_correct": [0,0,1,1]
    })
    ece = expected_calibration_error(df, n_bins=2)
    assert 0 <= ece <= 1