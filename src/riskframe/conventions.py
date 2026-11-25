"""Explicit, documented conventions.

The same-named metric can differ materially between shops (annualisation factor,
linking, VaR sign, ...). riskframe never hides these: every result echoes the
conventions it was computed under.

Sign conventions (fixed, not configurable):
    * VaR, ES, volatility, tracking error, max drawdown are reported as POSITIVE numbers
      (a loss / risk magnitude), expressed as a fraction of NAV (0.02 == 2%).
    * Returns keep their natural sign.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

from riskframe.errors import MandateError


@dataclass(frozen=True)
class Conventions:
    periods_per_year: int = 252
    """Annualisation factor (252 daily, 52 weekly, 12 monthly)."""
    risk_free_annual: float = 0.0
    """Annual risk-free rate, de-annualised arithmetically (rf / periods_per_year) per period."""
    linking: Literal["geometric", "arithmetic"] = "geometric"
    """How annualised (and active) returns are computed."""
    var_confidence: float = 0.95
    var_horizon_periods: int = 1
    """Horizon in return periods. Multi-period VaR/ES scale by sqrt(h) (i.i.d. assumption)."""
    var_use_mean: bool = False
    """Parametric VaR: include the sample mean (False = zero-mean, the common risk convention)."""
    ddof: int = 1
    """Degrees of freedom for standard deviations / covariances."""
    min_observations: int = 20
    warn_tail_observations: int = 5
    """Warn when historical VaR rests on fewer than this many tail observations."""

    def __post_init__(self) -> None:
        if self.periods_per_year <= 0:
            raise MandateError("periods_per_year must be positive")
        if not 0.5 < self.var_confidence < 1:
            raise MandateError("var_confidence must be in (0.5, 1)")
        if self.var_horizon_periods < 1:
            raise MandateError("var_horizon_periods must be >= 1")
        if self.ddof not in (0, 1):
            raise MandateError("ddof must be 0 or 1")
        if self.linking not in ("geometric", "arithmetic"):
            raise MandateError("linking must be 'geometric' or 'arithmetic'")

    @property
    def rf_per_period(self) -> float:
        return self.risk_free_annual / self.periods_per_year

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
