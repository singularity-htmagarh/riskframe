import math

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from riskframe import Conventions
from riskframe import metrics as m

C = Conventions()


def test_max_drawdown_known_value():
    r = pd.Series([0.10, -0.20, 0.05])  # wealth 1.1 -> 0.88 -> 0.924; DD = 0.2
    assert m.max_drawdown(r) == pytest.approx(0.2)


def test_drawdown_series_nonpositive_and_matches_max():
    r = pd.Series([0.01, -0.03, 0.02, -0.01, 0.04])
    dd = m.drawdown_series(r)
    assert (dd <= 0).all()
    assert -dd.min() == pytest.approx(m.max_drawdown(r))


def test_annualized_return_geometric_constant():
    r = pd.Series([0.001] * 252)
    assert m.annualized_return(r, C) == pytest.approx(1.001**252 - 1)


def test_annualized_return_arithmetic():
    c = Conventions(linking="arithmetic")
    r = pd.Series([0.001, 0.003])
    assert m.annualized_return(r, c) == pytest.approx(0.002 * 252)


def test_sharpe_matches_formula_and_rf():
    r = np.array([0.01, -0.005, 0.007, 0.002, -0.001, 0.004])
    c = Conventions(risk_free_annual=0.02)
    ex = r - 0.02 / 252
    expected = ex.mean() / ex.std(ddof=1) * math.sqrt(252)
    assert m.sharpe_ratio(r, c) == pytest.approx(expected)


def test_sharpe_zero_variance_is_nan_not_inf():
    # constant series: float noise makes std ~1e-19, not exactly 0 -- must still be NaN
    assert math.isnan(m.sharpe_ratio(np.full(30, 0.001), C))
    assert math.isnan(m.sortino_ratio(np.full(30, 0.001), C))


def test_downside_deviation_known_value():
    r = np.array([-0.01, 0.02, -0.03, 0.01])
    assert m.downside_deviation(r, C) == pytest.approx(math.sqrt(2.5e-4) * math.sqrt(252))


def test_relative_metrics():
    idx = pd.bdate_range("2023-01-02", periods=5)
    r = pd.Series([0.01, 0.00, 0.02, -0.01, 0.01], index=idx)
    b = pd.Series([0.005, 0.001, 0.01, -0.005, 0.004], index=idx)
    a = r - b
    assert m.tracking_error(r, b, C) == pytest.approx(a.std(ddof=1) * math.sqrt(252))
    assert m.beta(r, b, C) == pytest.approx(np.cov(r, b, ddof=1)[0, 1] / b.var(ddof=1))
    # beta of a series against itself is 1, TE is 0 (IR undefined -> NaN)
    assert m.beta(b, b, C) == pytest.approx(1.0)
    assert m.tracking_error(b, b, C) == 0.0
    assert math.isnan(m.information_ratio(b, b, C))


def test_active_returns_requires_alignment():
    a = pd.Series([0.1, 0.2], index=[1, 2])
    b = pd.Series([0.1, 0.2], index=[2, 3])
    with pytest.raises(ValueError):
        m.active_returns(a, b)


# ---- VaR / ES ------------------------------------------------------------------------------

def test_historical_var_es_known_values():
    r = np.linspace(-0.05, 0.05, 101)
    assert m.historical_var(r, C) == pytest.approx(0.045)
    assert m.historical_es(r, C) == pytest.approx(0.0475)


def test_parametric_var_es_normal_closed_form():
    r = np.random.default_rng(0).normal(0, 0.01, 5000)
    sigma = r.std(ddof=1)
    assert m.parametric_var(r, C) == pytest.approx(1.6448536269514722 * sigma)
    assert m.parametric_es(r, C) == pytest.approx(2.0627128075074257 * sigma)


def test_horizon_scales_by_sqrt_time():
    r = np.random.default_rng(1).normal(0, 0.01, 500)
    c10 = Conventions(var_horizon_periods=10)
    assert m.parametric_var(r, c10) == pytest.approx(m.parametric_var(r, C) * math.sqrt(10))
    assert m.historical_var(r, c10) == pytest.approx(m.historical_var(r, C) * math.sqrt(10))


returns_st = arrays(np.float64, st.integers(60, 300), elements=st.floats(-0.2, 0.2))


@settings(max_examples=60, deadline=None)
@given(returns_st)
def test_property_es_geq_var(r):
    assert m.historical_es(r, C) >= m.historical_var(r, C) - 1e-12
    if r.std(ddof=1) > 0:
        assert m.parametric_es(r, C) >= m.parametric_var(r, C) - 1e-12


@settings(max_examples=60, deadline=None)
@given(returns_st)
def test_property_var_monotone_in_confidence(r):
    lo = m.historical_var(r, Conventions(var_confidence=0.90))
    hi = m.historical_var(r, Conventions(var_confidence=0.99))
    assert hi >= lo - 1e-12


@settings(max_examples=60, deadline=None)
@given(returns_st, st.floats(0.1, 5.0))
def test_property_vol_scales_linearly(r, k):
    assert m.annualized_volatility(r * k, C) == pytest.approx(
        k * m.annualized_volatility(r, C), rel=1e-9, abs=1e-12)


@settings(max_examples=60, deadline=None)
@given(returns_st)
def test_property_max_drawdown_in_unit_interval(r):
    assert 0.0 <= m.max_drawdown(r) <= 1.0
