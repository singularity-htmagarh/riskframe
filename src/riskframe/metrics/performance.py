"""Absolute performance / risk statistics. Pure functions over return series."""
from __future__ import annotations

import numpy as np
import pandas as pd

from riskframe.conventions import Conventions


def _arr(r: pd.Series | np.ndarray) -> np.ndarray:
    return np.asarray(r, dtype="float64")


def is_effectively_zero(dispersion: float, values: np.ndarray, rel: float = 1e-12) -> bool:
    """True if a dispersion measure is zero up to floating-point noise at the data's scale.

    The std of a constant array is ~1e-19, not 0; dividing by it would return garbage.
    """
    scale = max(float(np.max(np.abs(values))), 1e-12)
    return bool(dispersion <= rel * scale)


def annualized_return(r: pd.Series | np.ndarray, conv: Conventions) -> float:
    x = _arr(r)
    if conv.linking == "geometric":
        growth = np.prod(1.0 + x)
        if growth <= 0:
            return float("nan")
        return float(growth ** (conv.periods_per_year / len(x)) - 1.0)
    return float(x.mean() * conv.periods_per_year)


def annualized_volatility(r: pd.Series | np.ndarray, conv: Conventions) -> float:
    x = _arr(r)
    return float(x.std(ddof=conv.ddof) * np.sqrt(conv.periods_per_year))


def sharpe_ratio(r: pd.Series | np.ndarray, conv: Conventions) -> float:
    """Mean excess return / std of excess return, annualised by sqrt(periods_per_year)."""
    ex = _arr(r) - conv.rf_per_period
    sd = ex.std(ddof=conv.ddof)
    if is_effectively_zero(sd, ex):
        return float("nan")
    return float(ex.mean() / sd * np.sqrt(conv.periods_per_year))


def downside_deviation(r: pd.Series | np.ndarray, conv: Conventions, mar_annual: float = 0.0) -> float:
    """Annualised root-mean-square shortfall below the minimum acceptable return (MAR)."""
    shortfall = np.minimum(_arr(r) - mar_annual / conv.periods_per_year, 0.0)
    return float(np.sqrt(np.mean(shortfall**2)) * np.sqrt(conv.periods_per_year))


def sortino_ratio(r: pd.Series | np.ndarray, conv: Conventions, mar_annual: float = 0.0) -> float:
    dd = downside_deviation(r, conv, mar_annual)
    if is_effectively_zero(dd / np.sqrt(conv.periods_per_year), _arr(r)):
        return float("nan")
    return float((_arr(r).mean() * conv.periods_per_year - mar_annual) / dd)


def drawdown_series(r: pd.Series) -> pd.Series:
    """Drawdown from running peak of cumulative wealth (starting at 1.0); values <= 0."""
    wealth = np.concatenate([[1.0], np.cumprod(1.0 + _arr(r))])
    peak = np.maximum.accumulate(wealth)
    dd = wealth / peak - 1.0
    return pd.Series(dd[1:], index=r.index, name="drawdown")


def max_drawdown(r: pd.Series | np.ndarray) -> float:
    """Maximum peak-to-trough decline, as a POSITIVE fraction."""
    wealth = np.concatenate([[1.0], np.cumprod(1.0 + _arr(r))])
    peak = np.maximum.accumulate(wealth)
    return float(-np.min(wealth / peak - 1.0))
