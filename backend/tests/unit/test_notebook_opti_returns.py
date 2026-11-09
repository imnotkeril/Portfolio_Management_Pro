"""Notebook-style constant-weight portfolio returns on a return matrix."""

import pandas as pd

from services.optimization_ui_bundle import _notebook_constant_weight_returns


def test_notebook_constant_weight_returns_dot_product() -> None:
    idx = pd.date_range("2024-01-01", periods=2, freq="D")
    df = pd.DataFrame({"A": [0.01, -0.02], "B": [0.0, 0.01]}, index=idx)
    w = {"A": 0.5, "B": 0.5}
    s = _notebook_constant_weight_returns(df, w, ["A", "B"])
    assert len(s) == 2
    assert abs(float(s.iloc[0]) - 0.005) < 1e-12
    assert abs(float(s.iloc[1]) - (-0.005)) < 1e-12


def test_notebook_constant_weight_returns_renormalizes_weights() -> None:
    idx = pd.date_range("2024-01-01", periods=1, freq="D")
    df = pd.DataFrame({"A": [0.04]}, index=idx)
    w = {"A": 0.25, "B": 0.25}
    s = _notebook_constant_weight_returns(df, w, ["A", "B"])
    assert len(s) == 1
    assert abs(float(s.iloc[0]) - 0.04) < 1e-12
