"""Portfolio and Benchmark containers with strict validation."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from riskframe.errors import DataValidationError


@dataclass(frozen=True, eq=False)
class Portfolio:
    """Holdings as weights (fraction of NAV) per asset id.

    Weights need not sum to 1 (cash, leverage, long/short are all legitimate); a
    deviation beyond ``weight_sum_tol`` is surfaced as a warning, not an error.
    ``classification`` optionally maps assets to groups (e.g. sector, country).
    """

    name: str
    weights: pd.Series
    base_currency: str = "USD"
    classification: pd.DataFrame | None = None
    weight_sum_tol: float = 1e-6
    expected_weight_sum: float = 1.0

    def __post_init__(self) -> None:
        w = self.weights
        if not isinstance(w, pd.Series) or w.empty:
            raise DataValidationError(f"{self.name}: weights must be a non-empty pandas Series")
        if not w.index.is_unique:
            raise DataValidationError(f"{self.name}: duplicate asset identifiers in weights")
        w = w.astype("float64")
        if not np.isfinite(w.to_numpy()).all():
            raise DataValidationError(f"{self.name}: weights contain NaN/inf")
        object.__setattr__(self, "weights", w)
        if self.classification is not None:
            missing = w.index.difference(self.classification.index)
            if len(missing):
                raise DataValidationError(
                    f"{self.name}: classification missing for assets {list(missing)[:5]}"
                )

    def warnings(self) -> list[str]:
        out: list[str] = []
        total = float(self.weights.sum())
        if abs(total - self.expected_weight_sum) > self.weight_sum_tol:
            out.append(
                f"weights sum to {total:.6f} (expected {self.expected_weight_sum:.6f}); "
                "residual is treated as zero-return cash"
            )
        return out


@dataclass(frozen=True, eq=False)
class Benchmark:
    """Benchmark identity plus optional data: a return series and/or constituent weights."""

    name: str
    returns: pd.Series | None = None
    weights: pd.Series | None = field(default=None)

    def __post_init__(self) -> None:
        if self.weights is not None:
            if not self.weights.index.is_unique:
                raise DataValidationError(f"{self.name}: duplicate benchmark constituents")
            if not np.isfinite(self.weights.to_numpy(dtype="float64")).all():
                raise DataValidationError(f"{self.name}: benchmark weights contain NaN/inf")
