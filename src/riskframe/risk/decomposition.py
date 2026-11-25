"""Ex-ante risk from a covariance matrix: volatility and Euler risk decomposition."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from riskframe.errors import DataValidationError


def _aligned(w: pd.Series, cov: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    missing = w.index.difference(cov.index)
    if len(missing):
        raise DataValidationError(f"no covariance for assets: {list(missing)[:5]}")
    c = cov.loc[w.index, w.index].to_numpy(dtype="float64")
    return w.to_numpy(dtype="float64"), c


def portfolio_vol(w: pd.Series, cov: pd.DataFrame) -> float:
    """sqrt(w' S w) in the units of ``cov`` (pass an annualised covariance for annual vol)."""
    wv, c = _aligned(w, cov)
    return float(np.sqrt(max(wv @ c @ wv, 0.0)))


def batch_portfolio_vol(weights: pd.DataFrame, cov: pd.DataFrame) -> pd.Series:
    """Vectorised vol for many portfolios at once. ``weights``: rows = portfolios, cols = assets."""
    missing = weights.columns.difference(cov.index)
    if len(missing):
        raise DataValidationError(f"no covariance for assets: {list(missing)[:5]}")
    c = cov.loc[weights.columns, weights.columns].to_numpy(dtype="float64")
    w = weights.to_numpy(dtype="float64")
    var = np.einsum("pi,ij,pj->p", w, c, w)
    vol: pd.Series = pd.Series(np.sqrt(np.clip(var, 0.0, None)), index=weights.index, name="vol")
    return vol


def risk_contributions(w: pd.Series, cov: pd.DataFrame, confidence: float | None = None) -> pd.DataFrame:
    """Euler decomposition of portfolio volatility.

    Columns: weight, marginal (d sigma / d w), component (w * marginal; sums to sigma),
    pct (component / sigma). With ``confidence``, adds ``component_var`` (zero-mean
    normal VaR contribution = z * component; sums to parametric VaR).
    """
    wv, c = _aligned(w, cov)
    sigma = float(np.sqrt(max(wv @ c @ wv, 0.0)))
    if sigma == 0.0:
        marginal = np.zeros_like(wv)
    else:
        marginal = (c @ wv) / sigma
    component = wv * marginal
    out: pd.DataFrame = pd.DataFrame(
        {
            "weight": wv,
            "marginal": marginal,
            "component": component,
            "pct": component / sigma if sigma > 0 else np.zeros_like(wv),
        },
        index=w.index,
    )
    if confidence is not None:
        out["component_var"] = norm.ppf(confidence) * out["component"]
    return out


def group_contributions(contribs: pd.DataFrame, groups: pd.Series) -> pd.DataFrame:
    """Aggregate component contributions (and weights) by a group label per asset."""
    g = groups.reindex(contribs.index)
    if g.isna().any():
        raise DataValidationError("group labels missing for some assets")
    cols = [c for c in ("weight", "component", "pct", "component_var") if c in contribs.columns]
    grouped: pd.DataFrame = contribs[cols].groupby(g).sum()
    return grouped
