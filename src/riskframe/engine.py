"""RiskEngine: runs a mandate-aware analysis for one portfolio, a batch, or a whole book."""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import pandas as pd
from scipy.stats import norm

from riskframe import metrics as m
from riskframe.conventions import Conventions
from riskframe.core.portfolio import Benchmark, Portfolio
from riskframe.data.validation import align, validate_returns
from riskframe.errors import DataValidationError, MandateError
from riskframe.limits import Limit, LimitContext, LimitResult, MaxMetric
from riskframe.mandate import Mandate, Objective
from riskframe.models.covariance import (
    CovarianceEstimator,
    LedoitWolfCovariance,
    is_psd,
    nearest_psd,
)
from riskframe.results import MetricResult, RiskReport
from riskframe.risk.decomposition import group_contributions, portfolio_vol, risk_contributions

_ABSOLUTE_FOCUS = {Objective.ABSOLUTE_RETURN, Objective.CAPITAL_PRESERVATION}
_NOT_YET_SPECIALISED = {Objective.INCOME, Objective.LIABILITY_DRIVEN}


class RiskEngine:
    """Analyse portfolios against a single mandate.

    >>> engine = RiskEngine(mandate)
    >>> report = engine.run(portfolio, asset_returns)
    >>> reports = engine.run_batch([p1, p2, p3], asset_returns)   # shared covariance
    """

    def __init__(
        self,
        mandate: Mandate,
        conventions: Conventions | None = None,
        covariance: CovarianceEstimator | None = None,
    ) -> None:
        self.mandate = mandate
        self.conventions = conventions or Conventions()
        self.covariance: CovarianceEstimator = covariance or LedoitWolfCovariance()

    def run(
        self,
        portfolio: Portfolio,
        asset_returns: pd.DataFrame,
        benchmark: Benchmark | None = None,
    ) -> RiskReport:
        mandate = self.mandate
        if benchmark is not None:
            mandate = Mandate(mandate.name, mandate.objective, benchmark, mandate.risk_budget,
                              mandate.constraints)
        return run_book([(portfolio, mandate)], asset_returns, self.conventions,
                        self.covariance)[portfolio.name]

    def run_batch(self, portfolios: Sequence[Portfolio], asset_returns: pd.DataFrame) -> dict[str, RiskReport]:
        return run_book([(p, self.mandate) for p in portfolios], asset_returns,
                        self.conventions, self.covariance)


def run_book(
    assignments: Sequence[tuple[Portfolio, Mandate]],
    asset_returns: pd.DataFrame,
    conventions: Conventions | None = None,
    covariance: CovarianceEstimator | None = None,
) -> dict[str, RiskReport]:
    """Analyse many (portfolio, mandate) pairs, estimating the covariance matrix ONCE.

    The covariance is estimated on the union of all assets involved; for shrinkage
    estimators the intensity is therefore estimated on that shared universe.
    """
    names = [p.name for p, _ in assignments]
    if len(set(names)) != len(names):
        raise MandateError("portfolio names must be unique within a book")
    estimator = covariance or LedoitWolfCovariance()
    conv = conventions or Conventions()
    R = validate_returns(asset_returns, name="asset_returns", min_obs=conv.min_observations)

    universe: set[str] = set()
    for p, mand in assignments:
        universe |= set(p.weights.index)
        if mand.benchmark is not None and mand.benchmark.weights is not None:
            universe |= set(mand.benchmark.weights.index)
    missing = sorted(map(str, universe - set(R.columns)))
    if missing:
        raise DataValidationError(f"asset_returns has no column for: {missing[:10]}")
    cols = [c for c in R.columns if c in universe]

    cov_period = estimator.estimate(R[cols])
    cov_annual = cov_period * conv.periods_per_year
    shared_warnings: list[str] = []
    if not is_psd(cov_annual):
        cov_annual = nearest_psd(cov_annual)
        shared_warnings.append("covariance was not PSD; projected to nearest PSD matrix")

    return {
        p.name: _analyze(p, mand, R, cov_annual, conv, estimator.name, list(shared_warnings))
        for p, mand in assignments
    }


def _analyze(
    portfolio: Portfolio,
    mandate: Mandate,
    R: pd.DataFrame,
    cov_annual: pd.DataFrame,
    conv: Conventions,
    cov_name: str,
    warnings: list[str],
) -> RiskReport:
    warnings += portfolio.warnings()
    w = portfolio.weights
    rp = pd.Series(R[w.index].to_numpy() @ w.to_numpy(), index=R.index, name=portfolio.name)
    n, as_of = len(rp), rp.index[-1]
    if n < conv.min_observations:
        warnings.append(f"only {n} observations (< {conv.min_observations})")
    sqrt_h = math.sqrt(conv.var_horizon_periods)
    z = float(norm.ppf(conv.var_confidence))

    out: dict[str, MetricResult] = {}

    def add(name: str, value: float, unit: str, basis: str, n_obs: int,
            warns: Sequence[str] = (), notes: str = "") -> None:
        ws = list(warns)
        if not math.isfinite(value):
            ws.append("undefined (zero variance or non-positive growth)")
        out[name] = MetricResult(name, float(value), unit, basis, as_of, n_obs, tuple(ws), notes)

    # --- absolute, ex-post --------------------------------------------------------------
    add("annualized_return", m.annualized_return(rp, conv), "fraction/yr", "ex_post", n,
        notes=f"{conv.linking} linking")
    add("annualized_volatility", m.annualized_volatility(rp, conv), "fraction/yr", "ex_post", n)
    add("sharpe_ratio", m.sharpe_ratio(rp, conv), "ratio", "ex_post", n,
        notes=f"rf={conv.risk_free_annual:.4f}/yr")
    add("max_drawdown", m.max_drawdown(rp), "fraction (positive=loss)", "ex_post", n)
    if mandate.objective in _ABSOLUTE_FOCUS:
        add("downside_deviation", m.downside_deviation(rp, conv), "fraction/yr", "ex_post", n,
            notes="MAR=0")
        add("sortino_ratio", m.sortino_ratio(rp, conv), "ratio", "ex_post", n, notes="MAR=0")

    horizon_note = f"{conv.var_confidence:.0%}, {conv.var_horizon_periods}-period, sqrt-time scaling"
    tail = m.tail_observations(n, conv.var_confidence)
    tail_w = [f"only ~{tail:.1f} tail observations"] if tail < conv.warn_tail_observations else []
    hv = m.historical_var(rp, conv)
    hv_w = tail_w + (["negative VaR: no losses at this quantile"] if hv < 0 else [])
    add("historical_var", hv, "fraction of NAV (positive=loss)", "ex_post", n, hv_w, horizon_note)
    add("historical_es", m.historical_es(rp, conv), "fraction of NAV (positive=loss)", "ex_post", n,
        tail_w, horizon_note)
    add("parametric_var", m.parametric_var(rp, conv), "fraction of NAV (positive=loss)", "ex_post", n,
        notes=horizon_note + ("; mean included" if conv.var_use_mean else "; zero-mean"))
    add("parametric_es", m.parametric_es(rp, conv), "fraction of NAV (positive=loss)", "ex_post", n,
        notes=horizon_note + ("; mean included" if conv.var_use_mean else "; zero-mean"))

    # --- ex-ante, covariance based -----------------------------------------------------
    ex_vol = portfolio_vol(w, cov_annual)
    add("ex_ante_volatility", ex_vol, "fraction/yr", "ex_ante", n, notes=f"covariance: {cov_name}")
    ex_var = z * ex_vol / math.sqrt(conv.periods_per_year) * sqrt_h
    add("ex_ante_var", ex_var, "fraction of NAV (positive=loss)", "ex_ante", n,
        notes=horizon_note + "; zero-mean normal")

    # --- benchmark-relative ------------------------------------------------------------
    bm = mandate.benchmark
    has_returns = bm is not None and bm.returns is not None
    has_weights = bm is not None and bm.weights is not None
    if mandate.objective is Objective.RELATIVE_RETURN and not (has_returns or has_weights):
        warnings.append("relative-return mandate but no benchmark data supplied; relative metrics skipped")
    if mandate.objective in _NOT_YET_SPECIALISED:
        warnings.append(f"objective-specific metrics for '{mandate.objective.value}' are not yet "
                        "implemented; showing the general metric set")
    if has_returns:
        assert bm is not None and bm.returns is not None
        b = validate_returns(bm.returns, name=f"benchmark '{bm.name}' returns")
        rp_a, b_a, aw = align(rp, b, min_obs=conv.min_observations)
        warnings += aw
        k = len(rp_a)
        add("active_return", m.active_return(rp_a, b_a, conv), "fraction/yr", "ex_post", k,
            notes=f"{conv.linking} linking vs {bm.name}")
        add("tracking_error", m.tracking_error(rp_a, b_a, conv), "fraction/yr", "ex_post", k,
            notes=f"vs {bm.name}")
        add("information_ratio", m.information_ratio(rp_a, b_a, conv), "ratio", "ex_post", k)
        add("beta", m.beta(rp_a, b_a, conv), "ratio", "ex_post", k, notes=f"vs {bm.name}")
    if has_weights:
        assert bm is not None and bm.weights is not None
        active = w.sub(bm.weights, fill_value=0.0)
        add("ex_ante_tracking_error", portfolio_vol(active, cov_annual), "fraction/yr", "ex_ante", n,
            notes=f"active weights vs {bm.name}; covariance: {cov_name}")

    # --- decomposition -----------------------------------------------------------------
    contribs = risk_contributions(w, cov_annual, conv.var_confidence)
    contribs["component_var"] *= sqrt_h / math.sqrt(conv.periods_per_year)
    groups: dict[str, pd.DataFrame] = {}
    if portfolio.classification is not None:
        for col in portfolio.classification.columns:
            groups[str(col)] = group_contributions(contribs, portfolio.classification[col])

    # --- limits --------------------------------------------------------------------------
    metric_values = {k: v.value for k, v in out.items()}
    limits: list[Limit] = list(mandate.constraints)
    rb = mandate.risk_budget
    if rb.volatility is not None:
        limits.append(MaxMetric(metric="ex_ante_volatility", limit=rb.volatility))
    if rb.tracking_error is not None:
        key = "ex_ante_tracking_error" if "ex_ante_tracking_error" in out else "tracking_error"
        limits.append(MaxMetric(metric=key, limit=rb.tracking_error))
    if rb.var is not None:
        limits.append(MaxMetric(metric="ex_ante_var", limit=rb.var))
    if rb.max_drawdown is not None:
        limits.append(MaxMetric(metric="max_drawdown", limit=rb.max_drawdown))
    ctx = LimitContext(portfolio, bm, metric_values)
    results: list[LimitResult] = [lim.evaluate(ctx) for lim in limits]

    return RiskReport(
        portfolio=portfolio.name, mandate=mandate.name, objective=mandate.objective.value,
        as_of=as_of, metrics=out, limits=results, decomposition=contribs,
        group_decomposition=groups, conventions=conv.to_dict(), covariance_model=cov_name,
        warnings=warnings,
    )


def summarize(reports: Mapping[str, RiskReport]) -> pd.DataFrame:
    """One row per portfolio: mandate, worst limit status, and every metric value."""
    rows = []
    for name, r in reports.items():
        row: dict[str, object] = {"portfolio": name, "mandate": r.mandate, "status": r.status.value}
        row.update({k: v.value for k, v in r.metrics.items()})
        rows.append(row)
    summary: pd.DataFrame = pd.DataFrame(rows).set_index("portfolio")
    return summary
