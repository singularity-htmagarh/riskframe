"""Covariance estimators. All return PER-PERIOD covariance as a labelled DataFrame."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

from riskframe.errors import DataValidationError


@runtime_checkable
class CovarianceEstimator(Protocol):
    @property
    def name(self) -> str: ...

    def estimate(self, returns: pd.DataFrame) -> pd.DataFrame: ...


def is_psd(cov: np.ndarray | pd.DataFrame, tol: float = 1e-10) -> bool:
    a = np.asarray(cov, dtype="float64")
    return bool(np.linalg.eigvalsh((a + a.T) / 2).min() >= -tol)


def nearest_psd(cov: pd.DataFrame, eps: float = 0.0) -> pd.DataFrame:
    """Project a symmetric matrix onto the PSD cone by clipping negative eigenvalues."""
    a = cov.to_numpy(dtype="float64")
    a = (a + a.T) / 2
    vals, vecs = np.linalg.eigh(a)
    fixed = (vecs * np.clip(vals, eps, None)) @ vecs.T
    fixed = (fixed + fixed.T) / 2
    result: pd.DataFrame = pd.DataFrame(fixed, index=cov.index, columns=cov.columns)
    return result


def _frame(mat: np.ndarray, like: pd.DataFrame) -> pd.DataFrame:
    result: pd.DataFrame = pd.DataFrame(mat, index=like.columns, columns=like.columns)
    return result


def _check(returns: pd.DataFrame) -> np.ndarray:
    if returns.shape[0] < 2:
        raise DataValidationError("covariance needs at least 2 observations")
    x = returns.to_numpy(dtype="float64")
    if not np.isfinite(x).all():
        raise DataValidationError("covariance input contains NaN/inf")
    return x


@dataclass(frozen=True)
class SampleCovariance:
    ddof: int = 1
    name: str = "sample"

    def estimate(self, returns: pd.DataFrame) -> pd.DataFrame:
        x = _check(returns)
        xc = x - x.mean(axis=0)
        return _frame(xc.T @ xc / (len(x) - self.ddof), returns)


@dataclass(frozen=True)
class EWMACovariance:
    """RiskMetrics-style exponentially weighted covariance (zero-mean by default)."""

    decay: float = 0.94
    demean: bool = False
    name: str = "ewma"

    def __post_init__(self) -> None:
        if not 0 < self.decay < 1:
            raise ValueError("decay must be in (0, 1)")

    def estimate(self, returns: pd.DataFrame) -> pd.DataFrame:
        x = _check(returns)
        if self.demean:
            x = x - x.mean(axis=0)
        t = len(x)
        w = self.decay ** np.arange(t - 1, -1, -1, dtype="float64")
        w /= w.sum()
        return _frame((x * w[:, None]).T @ x, returns)


@dataclass(frozen=True)
class LedoitWolfCovariance:
    """Ledoit-Wolf (2004) shrinkage towards a scaled identity.

    Positive definite even when assets > observations. Note: the shrinkage intensity
    is estimated on the supplied universe, so results for a sub-portfolio differ
    slightly from slicing a covariance estimated on a larger universe.
    """

    name: str = "ledoit_wolf"

    def estimate(self, returns: pd.DataFrame) -> pd.DataFrame:
        x = _check(returns)
        t, n = x.shape
        xc = x - x.mean(axis=0)
        s = xc.T @ xc / t
        mu = np.trace(s) / n
        d2 = np.sum((s - mu * np.eye(n)) ** 2) / n
        if d2 == 0:
            return _frame(s, returns)
        b2bar = (np.sum(np.sum(xc**2, axis=1) ** 2) - t * np.sum(s**2)) / (t**2 * n)
        b2 = min(b2bar, d2)
        shrink = float(np.clip(b2 / d2, 0.0, 1.0))
        out = shrink * mu * np.eye(n) + (1 - shrink) * s
        res = _frame(out, returns)
        res.attrs["shrinkage"] = shrink
        return res
