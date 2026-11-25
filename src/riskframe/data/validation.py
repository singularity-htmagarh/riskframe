"""Input validation. Fails loudly with specific messages; never silently fills NaNs."""
from __future__ import annotations

from typing import TypeVar

import numpy as np
import pandas as pd

from riskframe.errors import DataValidationError, InsufficientDataError

T = TypeVar("T", pd.Series, pd.DataFrame)


def validate_returns(returns: T, *, name: str = "returns", min_obs: int = 2) -> T:
    """Validate a return Series/DataFrame and return a float64 copy.

    Requirements: pandas object, numeric, unique and strictly increasing index,
    no NaN/inf, at least ``min_obs`` rows.
    """
    if not isinstance(returns, (pd.Series, pd.DataFrame)):
        raise DataValidationError(f"{name} must be a pandas Series or DataFrame")
    if len(returns) < min_obs:
        raise InsufficientDataError(f"{name}: {len(returns)} observations, need >= {min_obs}")
    try:
        out: T = returns.astype("float64")
    except (TypeError, ValueError) as exc:
        raise DataValidationError(f"{name} must be numeric: {exc}") from exc
    if not out.index.is_unique:
        raise DataValidationError(f"{name}: index contains duplicate dates")
    if not out.index.is_monotonic_increasing:
        raise DataValidationError(f"{name}: index must be sorted ascending")
    values = out.to_numpy()
    bad = ~np.isfinite(values)
    if bad.any():
        if isinstance(out, pd.DataFrame):
            per_col = pd.Series(bad.sum(axis=0), index=out.columns)
            detail = ", ".join(f"{c}: {n}" for c, n in per_col[per_col > 0].items())
            raise DataValidationError(f"{name}: {int(bad.sum())} NaN/inf values ({detail})")
        raise DataValidationError(f"{name}: {int(bad.sum())} NaN/inf values")
    return out


def align(a: pd.Series, b: pd.Series, *, min_obs: int = 2) -> tuple[pd.Series, pd.Series, list[str]]:
    """Inner-join two series on their index; returns (a, b, warnings)."""
    idx = a.index.intersection(b.index)
    if len(idx) < min_obs:
        raise InsufficientDataError(f"only {len(idx)} overlapping observations, need >= {min_obs}")
    warnings: list[str] = []
    dropped = max(len(a), len(b)) - len(idx)
    if dropped:
        warnings.append(f"alignment dropped {dropped} non-overlapping observations")
    return a.loc[idx], b.loc[idx], warnings
