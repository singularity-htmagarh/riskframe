import numpy as np
import pandas as pd
import pytest

from riskframe import EWMACovariance, LedoitWolfCovariance, SampleCovariance
from riskframe.errors import DataValidationError
from riskframe.models import is_psd, nearest_psd
from riskframe.risk import (
    batch_portfolio_vol,
    group_contributions,
    portfolio_vol,
    risk_contributions,
)


def test_sample_matches_numpy(asset_returns):
    est = SampleCovariance().estimate(asset_returns)
    np.testing.assert_allclose(est.to_numpy(), np.cov(asset_returns.to_numpy().T, ddof=1), atol=1e-14)
    assert list(est.index) == list(asset_returns.columns)


def test_ewma_symmetric_psd(asset_returns):
    c = EWMACovariance(0.94).estimate(asset_returns)
    np.testing.assert_allclose(c.to_numpy(), c.to_numpy().T, atol=1e-15)
    assert is_psd(c)


def test_ledoit_wolf_shrinks_and_is_pd_when_assets_exceed_obs():
    rng = np.random.default_rng(3)
    x = pd.DataFrame(rng.normal(0, 0.01, (30, 60)), columns=[f"A{i}" for i in range(60)])
    sample = SampleCovariance().estimate(x)
    lw = LedoitWolfCovariance().estimate(x)
    assert np.linalg.eigvalsh(sample.to_numpy()).min() < 1e-12  # singular (rank deficient)
    assert np.linalg.eigvalsh(lw.to_numpy()).min() > 0           # shrinkage restores PD
    assert 0.0 < lw.attrs["shrinkage"] <= 1.0


def test_ledoit_wolf_shrinks_little_when_data_is_plentiful_and_structured():
    # Unequal vols + strong common factor: the identity target is a poor fit, T >> N,
    # so the optimal intensity should be small and the estimate close to the sample.
    rng = np.random.default_rng(4)
    t, n = 20000, 4
    f = rng.normal(0, 0.01, (t, 1))
    vols = np.array([0.002, 0.006, 0.012, 0.02])
    x = pd.DataFrame(f * np.array([0.5, 1.0, 1.5, 2.0]) + rng.normal(0, 1, (t, n)) * vols,
                     columns=list("ABCD"))
    lw = LedoitWolfCovariance().estimate(x)
    assert lw.attrs["shrinkage"] < 0.01
    np.testing.assert_allclose(lw.to_numpy(), SampleCovariance(ddof=0).estimate(x).to_numpy(),
                               rtol=0.02, atol=1e-7)


def test_ledoit_wolf_full_shrinkage_when_truth_is_the_target():
    # i.i.d. equal-variance data: truth == scaled identity, so intensity ~ 1 is correct.
    rng = np.random.default_rng(4)
    x = pd.DataFrame(rng.normal(0, 0.01, (20000, 4)), columns=list("ABCD"))
    assert LedoitWolfCovariance().estimate(x).attrs["shrinkage"] > 0.9


def test_nearest_psd_repairs_negative_eigenvalue():
    bad = pd.DataFrame([[1.0, 0.99, -0.99], [0.99, 1.0, 0.99], [-0.99, 0.99, 1.0]],
                       index=list("abc"), columns=list("abc"))
    assert not is_psd(bad)
    assert is_psd(nearest_psd(bad))


def test_nan_input_rejected(asset_returns):
    bad = asset_returns.copy()
    bad.iloc[3, 1] = np.nan
    with pytest.raises(DataValidationError):
        SampleCovariance().estimate(bad)


def _w(cols):
    return pd.Series(np.linspace(0.1, 0.3, len(cols)) / np.linspace(0.1, 0.3, len(cols)).sum(), index=cols)


def test_euler_contributions_sum_to_total_vol(asset_returns):
    cov = SampleCovariance().estimate(asset_returns) * 252
    w = _w(asset_returns.columns)
    rc = risk_contributions(w, cov, confidence=0.95)
    assert rc["component"].sum() == pytest.approx(portfolio_vol(w, cov))
    assert rc["pct"].sum() == pytest.approx(1.0)
    assert rc["component_var"].sum() == pytest.approx(1.6448536269514722 * portfolio_vol(w, cov))


def test_vol_scales_linearly_with_weights(asset_returns):
    cov = SampleCovariance().estimate(asset_returns) * 252
    w = _w(asset_returns.columns)
    assert portfolio_vol(3 * w, cov) == pytest.approx(3 * portfolio_vol(w, cov))


def test_batch_vol_equals_loop(asset_returns):
    cov = SampleCovariance().estimate(asset_returns) * 252
    rng = np.random.default_rng(5)
    W = pd.DataFrame(rng.dirichlet(np.ones(5), size=40), columns=asset_returns.columns,
                     index=[f"P{i}" for i in range(40)])
    batch = batch_portfolio_vol(W, cov)
    loop = pd.Series({p: portfolio_vol(W.loc[p], cov) for p in W.index})
    np.testing.assert_allclose(batch.to_numpy(), loop.to_numpy(), rtol=1e-12)


def test_group_contributions_sum(asset_returns):
    cov = SampleCovariance().estimate(asset_returns) * 252
    w = _w(asset_returns.columns)
    rc = risk_contributions(w, cov)
    groups = pd.Series(["x", "x", "y", "y", "y"], index=asset_returns.columns)
    g = group_contributions(rc, groups)
    assert g["component"].sum() == pytest.approx(rc["component"].sum())
    assert g["weight"].sum() == pytest.approx(1.0)
