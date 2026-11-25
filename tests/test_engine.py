import json
import math

import numpy as np
import pandas as pd
import pytest

from riskframe import (
    Benchmark,
    Conventions,
    DataValidationError,
    LimitStatus,
    Mandate,
    MandateError,
    MaxGroupWeight,
    MaxPosition,
    Objective,
    Portfolio,
    RiskBudget,
    RiskEngine,
    SampleCovariance,
    run_book,
    summarize,
)


def pf(cols, name="P", w=None, cls=True):
    w = pd.Series(w if w is not None else np.full(len(cols), 1 / len(cols)), index=cols)
    classification = pd.DataFrame({"sector": ["x", "x", "y", "y", "y"][: len(cols)]}, index=cols) if cls else None
    return Portfolio(name, w, classification=classification)


def bm_from(asset_returns, name="BM"):
    w = pd.Series(1 / asset_returns.shape[1], index=asset_returns.columns)
    return Benchmark(name, returns=asset_returns @ w, weights=w)


def test_report_contents_relative_mandate(asset_returns):
    mand = Mandate("Core", Objective.RELATIVE_RETURN, bm_from(asset_returns),
                   RiskBudget(tracking_error=0.05, var=0.05),
                   [MaxPosition(limit=0.5), MaxGroupWeight(group_field="sector", limit=0.7)])
    p = pf(asset_returns.columns, w=[0.3, 0.3, 0.2, 0.1, 0.1])
    rep = RiskEngine(mandate=mand).run(p, asset_returns)
    for key in ("annualized_return", "sharpe_ratio", "historical_var", "parametric_es",
                "ex_ante_volatility", "ex_ante_var", "tracking_error", "information_ratio",
                "beta", "ex_ante_tracking_error"):
        assert key in rep.metrics, key
    assert rep.status in set(LimitStatus)
    names = [lim.name for lim in rep.limits]
    assert "max_ex_ante_tracking_error" in names  # ex-ante preferred when benchmark weights exist
    assert set(rep.group_decomposition) == {"sector"}
    assert rep.decomposition["component"].sum() == pytest.approx(rep.metrics["ex_ante_volatility"].value)
    # component VaR contributions sum to ex-ante VaR
    assert rep.decomposition["component_var"].sum() == pytest.approx(rep.metrics["ex_ante_var"].value)
    json.loads(rep.to_json())  # serialisable, NaN-safe
    assert "value" in rep.metrics_frame().columns


def test_fully_replicating_benchmark_has_zero_ex_ante_te(asset_returns):
    mand = Mandate("Idx", Objective.RELATIVE_RETURN, bm_from(asset_returns))
    rep = RiskEngine(mand).run(pf(asset_returns.columns), asset_returns)
    assert rep.metrics["ex_ante_tracking_error"].value == pytest.approx(0.0, abs=1e-12)
    assert rep.metrics["beta"].value == pytest.approx(1.0)
    assert math.isnan(rep.metrics["information_ratio"].value)
    assert rep.metrics["information_ratio"].warnings  # NaN is explained, not silent


def test_breach_detected(asset_returns):
    mand = Mandate("Tight", Objective.ABSOLUTE_RETURN, risk_budget=RiskBudget(volatility=0.001))
    rep = RiskEngine(mand).run(pf(asset_returns.columns), asset_returns)
    assert rep.status is LimitStatus.BREACH
    assert "downside_deviation" in rep.metrics  # objective-driven metric set


def test_relative_mandate_without_benchmark_data_warns(asset_returns):
    mand = Mandate("NoBM", Objective.RELATIVE_RETURN, Benchmark("BM"))
    rep = RiskEngine(mand).run(pf(asset_returns.columns), asset_returns)
    assert any("no benchmark data" in w for w in rep.warnings)
    assert "tracking_error" not in rep.metrics


def test_tracking_error_budget_requires_benchmark():
    with pytest.raises(MandateError):
        Mandate("bad", Objective.RELATIVE_RETURN, None, RiskBudget(tracking_error=0.03))


def test_unspecialised_objective_is_flagged(asset_returns):
    rep = RiskEngine(Mandate("Inc", Objective.INCOME)).run(pf(asset_returns.columns), asset_returns)
    assert any("not yet implemented" in w for w in rep.warnings)


def test_weights_not_summing_to_one_warns(asset_returns):
    p = pf(asset_returns.columns, w=[0.3, 0.3, 0.3, 0.3, 0.3])
    rep = RiskEngine(Mandate("m", Objective.ABSOLUTE_RETURN)).run(p, asset_returns)
    assert any("weights sum to" in w for w in rep.warnings)


def test_bad_inputs_fail_loudly(asset_returns):
    eng = RiskEngine(Mandate("m", Objective.ABSOLUTE_RETURN))
    nan = asset_returns.copy()
    nan.iloc[10, 0] = np.nan
    with pytest.raises(DataValidationError, match="NaN"):
        eng.run(pf(asset_returns.columns), nan)
    with pytest.raises(DataValidationError, match="no column"):
        eng.run(pf(["AAA", "ZZZ"], cls=False), asset_returns)
    dup = pd.concat([asset_returns.iloc[:5], asset_returns.iloc[:5], asset_returns.iloc[5:]])
    with pytest.raises(DataValidationError, match="duplicate"):
        eng.run(pf(asset_returns.columns), dup)
    with pytest.raises(DataValidationError):
        Portfolio("dupe", pd.Series([0.5, 0.5], index=["A", "A"]))
    with pytest.raises(DataValidationError):
        Portfolio("nan", pd.Series([np.nan, 1.0], index=["A", "B"]))


def test_run_book_matches_single_runs_with_sample_cov(asset_returns):
    conv = Conventions()
    mand = Mandate("m", Objective.ABSOLUTE_RETURN)
    portfolios = [pf(asset_returns.columns, f"P{i}", w=w, cls=False) for i, w in enumerate(
        [[.2] * 5, [.5, .5, 0, 0, 0], [0, 0, .4, .3, .3]])]
    book = run_book([(p, mand) for p in portfolios], asset_returns, conv, SampleCovariance())
    for p in portfolios:
        single = RiskEngine(mand, conv, SampleCovariance()).run(p, asset_returns)
        for k in ("ex_ante_volatility", "ex_ante_var", "annualized_return", "historical_var"):
            assert book[p.name].metrics[k].value == pytest.approx(single.metrics[k].value)
    summary = summarize(book)
    assert list(summary.index) == ["P0", "P1", "P2"] and "status" in summary.columns


def test_run_book_rejects_duplicate_names(asset_returns):
    mand = Mandate("m", Objective.ABSOLUTE_RETURN)
    p = pf(asset_returns.columns, cls=False)
    with pytest.raises(MandateError):
        run_book([(p, mand), (p, mand)], asset_returns)


def test_conventions_are_echoed_and_change_results(asset_returns):
    mand = Mandate("m", Objective.ABSOLUTE_RETURN)
    p = pf(asset_returns.columns, cls=False)
    a = RiskEngine(mand, Conventions(var_confidence=0.95)).run(p, asset_returns)
    b = RiskEngine(mand, Conventions(var_confidence=0.99, var_horizon_periods=10)).run(p, asset_returns)
    assert b.conventions["var_confidence"] == 0.99
    assert b.metrics["parametric_var"].value > a.metrics["parametric_var"].value


def test_small_sample_tail_warning(asset_returns):
    rep = RiskEngine(Mandate("m", Objective.ABSOLUTE_RETURN), Conventions(var_confidence=0.999)).run(
        pf(asset_returns.columns, cls=False), asset_returns.iloc[:200])
    assert rep.metrics["historical_var"].warnings
