"""Mandate: what the portfolio is for, what it is measured against, and its limits."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

from riskframe.core.portfolio import Benchmark
from riskframe.errors import MandateError
from riskframe.limits import Limit


class Objective(str, Enum):
    ABSOLUTE_RETURN = "absolute_return"
    RELATIVE_RETURN = "relative_return"
    CAPITAL_PRESERVATION = "capital_preservation"
    INCOME = "income"
    LIABILITY_DRIVEN = "liability_driven"


@dataclass(frozen=True)
class RiskBudget:
    """Risk limits expressed as annualised fractions of NAV (VaR at the engine's
    confidence/horizon). Evaluated ex-ante (covariance based) where possible."""

    volatility: float | None = None
    tracking_error: float | None = None
    var: float | None = None
    max_drawdown: float | None = None  # ex-post, positive fraction

    def __post_init__(self) -> None:
        for k, v in self.__dict__.items():
            if v is not None and not v > 0:
                raise MandateError(f"RiskBudget.{k} must be positive")


@dataclass(frozen=True)
class Mandate:
    name: str
    objective: Objective
    benchmark: Benchmark | None = None
    risk_budget: RiskBudget = RiskBudget()
    constraints: Sequence[Limit] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.objective, Objective):
            raise MandateError("objective must be an Objective")
        if self.risk_budget.tracking_error is not None and self.benchmark is None:
            raise MandateError(f"{self.name}: a tracking-error budget needs a benchmark")
        object.__setattr__(self, "constraints", tuple(self.constraints))
