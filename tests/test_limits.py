import math

import pandas as pd
import pytest

from riskframe import (
    Benchmark,
    LimitStatus,
    MandateError,
    MaxActiveWeight,
    MaxGrossExposure,
    MaxGroupWeight,
    MaxMetric,
    MaxNetExposure,
    MaxPosition,
    Portfolio,
)
from riskframe.limits import LimitContext, worst_status


def ctx(weights, bm_weights=None, classification=None, metrics=None):
    p = Portfolio("p", pd.Series(weights), classification=classification)
    bm = Benchmark("bm", weights=pd.Series(bm_weights)) if bm_weights is not None else None
    return LimitContext(p, bm, metrics or {})


def test_status_ok_warning_breach():
    c = ctx({"A": 0.5, "B": 0.5})
    assert MaxPosition(limit=0.8).evaluate(c).status is LimitStatus.OK
    assert MaxPosition(limit=0.55).evaluate(c).status is LimitStatus.WARNING   # 0.5/0.55 = 91%
    r = MaxPosition(limit=0.4).evaluate(c)
    assert r.status is LimitStatus.BREACH
    assert r.utilization == pytest.approx(1.25)
    assert r.headroom == pytest.approx(-0.1)


def test_exposures():
    c = ctx({"A": 0.8, "B": -0.3})
    assert MaxGrossExposure(limit=1.5).evaluate(c).value == pytest.approx(1.1)
    assert MaxNetExposure(limit=1.0).evaluate(c).value == pytest.approx(0.5)


def test_active_weight_uses_union_of_assets():
    c = ctx({"A": 0.6, "B": 0.4}, bm_weights={"A": 0.5, "C": 0.5})
    r = MaxActiveWeight(limit=0.6).evaluate(c)
    assert r.value == pytest.approx(0.5)  # C is held at 0 vs 0.5 in benchmark
    assert "C" in r.detail


def test_group_weight():
    cls = pd.DataFrame({"sector": ["tech", "tech", "fin"]}, index=["A", "B", "C"])
    r = MaxGroupWeight(group_field="sector", limit=0.5).evaluate(
        ctx({"A": 0.3, "B": 0.3, "C": 0.4}, classification=cls))
    assert r.value == pytest.approx(0.6) and r.status is LimitStatus.BREACH and "tech" in r.detail


def test_missing_inputs_are_reported_not_skipped():
    assert MaxActiveWeight(limit=0.1).evaluate(ctx({"A": 1.0})).status is LimitStatus.NOT_EVALUATED
    r = MaxGroupWeight(group_field="sector", limit=0.5).evaluate(ctx({"A": 1.0}))
    assert r.status is LimitStatus.NOT_EVALUATED and "sector" in r.detail
    assert MaxMetric(metric="nope", limit=1.0).evaluate(ctx({"A": 1.0})).status is LimitStatus.NOT_EVALUATED


def test_nan_metric_is_not_evaluated_rather_than_ok():
    r = MaxMetric(metric="x", limit=1.0).evaluate(ctx({"A": 1.0}, metrics={"x": math.nan}))
    assert r.status is LimitStatus.NOT_EVALUATED


def test_invalid_limits_rejected():
    for bad in (0, -1, math.inf, math.nan):
        with pytest.raises(MandateError):
            MaxPosition(limit=bad)
    with pytest.raises(MandateError):
        MaxPosition(limit=0.1, warn_at=1.5)


def test_worst_status_ordering():
    S = LimitStatus
    assert worst_status([S.OK, S.WARNING, S.BREACH]) is S.BREACH
    assert worst_status([S.OK, S.NOT_EVALUATED]) is S.NOT_EVALUATED
    assert worst_status([]) is S.OK
