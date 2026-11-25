"""Limits: constraints evaluated against a portfolio (and its computed metrics).

Every limit is an upper bound on a non-negative measure. Evaluation yields value,
utilisation (value / limit), headroom and a status. A limit that cannot be evaluated
because an input is missing is reported as NOT_EVALUATED with a reason, never skipped.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

import pandas as pd

from riskframe.core.portfolio import Benchmark, Portfolio
from riskframe.errors import MandateError, MissingInputError


class LimitStatus(str, Enum):
    OK = "OK"
    WARNING = "WARNING"
    BREACH = "BREACH"
    NOT_EVALUATED = "NOT_EVALUATED"


_SEVERITY = {
    LimitStatus.OK: 0,
    LimitStatus.NOT_EVALUATED: 1,
    LimitStatus.WARNING: 2,
    LimitStatus.BREACH: 3,
}


def worst_status(statuses: list[LimitStatus]) -> LimitStatus:
    return max(statuses, key=_SEVERITY.__getitem__, default=LimitStatus.OK)


@dataclass(frozen=True)
class LimitContext:
    portfolio: Portfolio
    benchmark: Benchmark | None
    metrics: Mapping[str, float]


@dataclass(frozen=True)
class LimitResult:
    name: str
    limit: float
    value: float
    utilization: float
    headroom: float
    status: LimitStatus
    detail: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name, "limit": self.limit, "value": self.value,
            "utilization": self.utilization, "headroom": self.headroom,
            "status": self.status.value, "detail": self.detail,
        }


@dataclass(frozen=True, kw_only=True)
class Limit(ABC):
    limit: float
    warn_at: float = 0.9
    """Utilisation at or above which status becomes WARNING."""

    def __post_init__(self) -> None:
        if not self.limit > 0 or not math.isfinite(self.limit):
            raise MandateError(f"{type(self).__name__}: limit must be a positive finite number")
        if not 0 < self.warn_at <= 1:
            raise MandateError(f"{type(self).__name__}: warn_at must be in (0, 1]")

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def measure(self, ctx: LimitContext) -> tuple[float, str]:
        """Return (value, detail). Raise MissingInputError if an input is unavailable."""

    def evaluate(self, ctx: LimitContext) -> LimitResult:
        try:
            value, detail = self.measure(ctx)
        except MissingInputError as exc:
            return LimitResult(self.name, self.limit, math.nan, math.nan, math.nan,
                               LimitStatus.NOT_EVALUATED, str(exc))
        if not math.isfinite(value):
            return LimitResult(self.name, self.limit, value, math.nan, math.nan,
                               LimitStatus.NOT_EVALUATED, "measure is not finite")
        util = value / self.limit
        if value > self.limit * (1 + 1e-9):
            status = LimitStatus.BREACH
        elif util >= self.warn_at:
            status = LimitStatus.WARNING
        else:
            status = LimitStatus.OK
        return LimitResult(self.name, self.limit, value, util, self.limit - value, status, detail)


@dataclass(frozen=True, kw_only=True)
class MaxPosition(Limit):
    """Largest absolute single-asset weight."""

    @property
    def name(self) -> str:
        return "max_position"

    def measure(self, ctx: LimitContext) -> tuple[float, str]:
        w = ctx.portfolio.weights.abs()
        return float(w.max()), f"largest: {w.idxmax()}"


@dataclass(frozen=True, kw_only=True)
class MaxGrossExposure(Limit):
    @property
    def name(self) -> str:
        return "max_gross_exposure"

    def measure(self, ctx: LimitContext) -> tuple[float, str]:
        return float(ctx.portfolio.weights.abs().sum()), ""


@dataclass(frozen=True, kw_only=True)
class MaxNetExposure(Limit):
    @property
    def name(self) -> str:
        return "max_net_exposure"

    def measure(self, ctx: LimitContext) -> tuple[float, str]:
        return float(ctx.portfolio.weights.sum()), ""


@dataclass(frozen=True, kw_only=True)
class MaxActiveWeight(Limit):
    """Largest absolute single-asset deviation from benchmark weights."""

    @property
    def name(self) -> str:
        return "max_active_weight"

    def measure(self, ctx: LimitContext) -> tuple[float, str]:
        if ctx.benchmark is None or ctx.benchmark.weights is None:
            raise MissingInputError("benchmark constituent weights not supplied")
        active = ctx.portfolio.weights.sub(ctx.benchmark.weights, fill_value=0.0).abs()
        return float(active.max()), f"largest: {active.idxmax()}"


@dataclass(frozen=True, kw_only=True)
class MaxGroupWeight(Limit):
    """Largest net weight in any group of a classification column (e.g. sector)."""

    group_field: str

    @property
    def name(self) -> str:
        return f"max_{self.group_field}_weight"

    def measure(self, ctx: LimitContext) -> tuple[float, str]:
        cls = ctx.portfolio.classification
        if cls is None or self.group_field not in cls.columns:
            raise MissingInputError(f"classification column '{self.group_field}' not supplied")
        grouped = ctx.portfolio.weights.groupby(cls[self.group_field].reindex(ctx.portfolio.weights.index)).sum()
        return float(grouped.max()), f"largest: {grouped.idxmax()}"


@dataclass(frozen=True, kw_only=True)
class MaxMetric(Limit):
    """Upper bound on any metric the engine computed (e.g. 'ex_ante_tracking_error')."""

    metric: str

    @property
    def name(self) -> str:
        return f"max_{self.metric}"

    def measure(self, ctx: LimitContext) -> tuple[float, str]:
        if self.metric not in ctx.metrics:
            raise MissingInputError(f"metric '{self.metric}' was not computed")
        return float(ctx.metrics[self.metric]), ""


def limits_frame(results: list[LimitResult]) -> pd.DataFrame:
    cols = ["name", "limit", "value", "utilization", "headroom", "status", "detail"]
    frame: pd.DataFrame = pd.DataFrame([r.to_dict() for r in results], columns=cols)
    return frame
