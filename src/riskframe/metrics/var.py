"""Value-at-Risk and Expected Shortfall.

All functions return POSITIVE loss magnitudes (fraction of NAV) for the horizon in
``conv.var_horizon_periods``. Horizon scaling is sqrt-of-time (i.i.d. assumption) and
is stated in the result metadata by the engine.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import norm

from riskframe.conventions import Conventions


def _arr(r: pd.Series | np.ndarray) -> np.ndarray:
    return np.asarray(r, dtype="float64")


def tail_observations(n_obs: int, confidence: float) -> float:
    """Expected number of observations beyond the VaR quantile."""
    return n_obs * (1.0 - confidence)


def historical_var(r: pd.Series | np.ndarray, conv: Conventions) -> float:
    x = _arr(r)
    q = np.quantile(x, 1.0 - conv.var_confidence)
    return float(-q * math.sqrt(conv.var_horizon_periods))


def historical_es(r: pd.Series | np.ndarray, conv: Conventions) -> float:
    x = _arr(r)
    q = np.quantile(x, 1.0 - conv.var_confidence)
    tol = 1e-12 * max(1.0, abs(q))  # guard against float noise at the quantile boundary
    tail = x[x <= q + tol]
    return float(-tail.mean() * math.sqrt(conv.var_horizon_periods))


def _normal_moments(r: pd.Series | np.ndarray, conv: Conventions) -> tuple[float, float]:
    x = _arr(r)
    h = conv.var_horizon_periods
    mu = x.mean() * h if conv.var_use_mean else 0.0
    sigma = x.std(ddof=conv.ddof) * math.sqrt(h)
    return float(mu), float(sigma)


def normal_var(mu: float, sigma: float, confidence: float) -> float:
    return float(norm.ppf(confidence) * sigma - mu)


def normal_es(mu: float, sigma: float, confidence: float) -> float:
    z = norm.ppf(confidence)
    return float(sigma * norm.pdf(z) / (1.0 - confidence) - mu)


def parametric_var(r: pd.Series | np.ndarray, conv: Conventions) -> float:
    mu, sigma = _normal_moments(r, conv)
    return normal_var(mu, sigma, conv.var_confidence)


def parametric_es(r: pd.Series | np.ndarray, conv: Conventions) -> float:
    mu, sigma = _normal_moments(r, conv)
    return normal_es(mu, sigma, conv.var_confidence)
