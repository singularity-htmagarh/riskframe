"""Structured, self-describing results."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from riskframe.limits import LimitResult, LimitStatus, limits_frame, worst_status


@dataclass(frozen=True)
class MetricResult:
    name: str
    value: float
    unit: str
    basis: str  # "ex_post" | "ex_ante"
    as_of: pd.Timestamp | None = None
    n_obs: int | None = None
    warnings: tuple[str, ...] = ()
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "value": self.value, "unit": self.unit, "basis": self.basis,
            "as_of": None if self.as_of is None else str(pd.Timestamp(self.as_of).date()),
            "n_obs": self.n_obs, "warnings": list(self.warnings), "notes": self.notes,
        }


@dataclass
class RiskReport:
    portfolio: str
    mandate: str
    objective: str
    as_of: pd.Timestamp | None
    metrics: dict[str, MetricResult]
    limits: list[LimitResult]
    decomposition: pd.DataFrame | None
    group_decomposition: dict[str, pd.DataFrame]
    conventions: dict[str, Any]
    covariance_model: str
    warnings: list[str] = field(default_factory=list)

    @property
    def status(self) -> LimitStatus:
        return worst_status([lim.status for lim in self.limits])

    def metrics_frame(self) -> pd.DataFrame:
        frame: pd.DataFrame = pd.DataFrame([m.to_dict() for m in self.metrics.values()]).set_index("name")
        return frame

    def limits_frame(self) -> pd.DataFrame:
        return limits_frame(self.limits)

    def to_dict(self) -> dict[str, Any]:
        return {
            "portfolio": self.portfolio, "mandate": self.mandate, "objective": self.objective,
            "as_of": None if self.as_of is None else str(pd.Timestamp(self.as_of).date()),
            "status": self.status.value, "covariance_model": self.covariance_model,
            "conventions": self.conventions,
            "metrics": {k: m.to_dict() for k, m in self.metrics.items()},
            "limits": [lim.to_dict() for lim in self.limits],
            "warnings": list(self.warnings),
        }

    def to_json(self, **kwargs: Any) -> str:
        def default(o: Any) -> Any:
            if isinstance(o, float) and not math.isfinite(o):
                return None
            raise TypeError(type(o))

        return json.dumps(_nan_to_none(self.to_dict()), default=default, **kwargs)


def _nan_to_none(o: Any) -> Any:
    if isinstance(o, float) and not math.isfinite(o):
        return None
    if isinstance(o, dict):
        return {k: _nan_to_none(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_nan_to_none(v) for v in o]
    return o
