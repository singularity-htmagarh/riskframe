"""Benchmark-relative statistics (ex-post)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from riskframe.conventions import Conventions
from riskframe.metrics.performance import annualized_return, is_effectively_zero


def active_returns(r: pd.Series, b: pd.Series) -> pd.Series:
    """Per-period arithmetic active return r - b (inputs must already be aligned)."""
    if not r.index.equals(b.index):
        raise ValueError("active_returns requires aligned indices; use riskframe.data.align")
    return r - b


def active_return(r: pd.Series, b: pd.Series, conv: Conventions) -> float:
    """Annualised active return: geometric -> ann(r) - ann(b); arithmetic -> mean(r-b)*ppy."""
    if conv.linking == "geometric":
        return annualized_return(r, conv) - annualized_return(b, conv)
    return float(active_returns(r, b).mean() * conv.periods_per_year)


def tracking_error(r: pd.Series, b: pd.Series, conv: Conventions) -> float:
    a = active_returns(r, b).to_numpy()
    return float(a.std(ddof=conv.ddof) * np.sqrt(conv.periods_per_year))


def information_ratio(r: pd.Series, b: pd.Series, conv: Conventions) -> float:
    te = tracking_error(r, b, conv)
    a = active_returns(r, b).to_numpy()
    if is_effectively_zero(te / np.sqrt(conv.periods_per_year), np.concatenate([a, r.to_numpy()])):
        return float("nan")
    return active_return(r, b, conv) / te


def beta(r: pd.Series, b: pd.Series, conv: Conventions) -> float:
    rr, bb = r.to_numpy(), b.to_numpy()
    var_b = bb.var(ddof=conv.ddof)
    if is_effectively_zero(float(np.sqrt(var_b)), bb):
        return float("nan")
    cov = ((rr - rr.mean()) * (bb - bb.mean())).sum() / (len(rr) - conv.ddof)
    return float(cov / var_b)
