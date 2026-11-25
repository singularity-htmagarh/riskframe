# riskframe

Mandate-aware portfolio risk and performance analytics for Python.

Most risk libraries treat a portfolio as a return series. `riskframe` measures risk
**against the mandate**: its benchmark, objective, risk budget and constraints, and
reports every result with the conventions it was computed under.

> **Status: 0.1 pre-release.** The API will change. This is an analytics toolkit,
> not a compliance system of record.

## Install (from source)

```bash
pip install -e ".[dev]"
pytest
```

## Quickstart

```python
from riskframe import (Benchmark, Mandate, MaxGroupWeight, MaxPosition, Objective,
                       Portfolio, RiskBudget, RiskEngine)

mandate = Mandate(
    name="US Large-Cap Core",
    objective=Objective.RELATIVE_RETURN,
    benchmark=Benchmark("BM", returns=bm_returns, weights=bm_weights),
    risk_budget=RiskBudget(tracking_error=0.03, var=0.02),
    constraints=[MaxPosition(limit=0.35), MaxGroupWeight(group_field="sector", limit=0.60)],
)
report = RiskEngine(mandate).run(portfolio, asset_returns)

report.status             # worst limit status: OK / WARNING / BREACH / NOT_EVALUATED
report.metrics_frame()    # every metric with unit, basis (ex_post/ex_ante), warnings
report.limits_frame()     # limit, value, utilisation, headroom, status
report.decomposition      # Euler risk contributions by asset (vol and VaR)
report.to_json()          # structured, NaN-safe
```

Many portfolios, one covariance estimate:

```python
from riskframe import run_book
reports = run_book([(portfolio_a, mandate_1), (portfolio_b, mandate_2)], asset_returns)
```

See `examples/quickstart.py` and `examples/benchmark_book.py`.

## Conventions

Fixed sign convention: VaR, ES, volatility, tracking error and max drawdown are
reported as **positive** magnitudes, as fractions of NAV. Everything else is explicit
and configurable through `Conventions`, and echoed in every report.

| Item | Default | Notes |
|---|---|---|
| Annualisation | 252 | `periods_per_year` |
| Risk-free | 0 | de-annualised arithmetically per period |
| Return linking | geometric | or arithmetic |
| VaR | 95%, 1 period | sqrt-time scaling for longer horizons (i.i.d. assumption) |
| Parametric VaR | zero-mean normal | `var_use_mean=True` to include the mean |
| Historical VaR | linear-interpolated quantile | warns on thin tails |
| Sharpe | mean excess / std excess * sqrt(ppy) | NaN, never inf, on zero variance |
| Max drawdown | peak-to-trough on wealth starting at 1.0 | positive |
| Covariance | Ledoit-Wolf (default), sample, EWMA | per-period, annualised by engine |

Ex-ante measures (volatility, VaR, tracking error) use the covariance model; risk
budgets are evaluated ex-ante where possible, and the limit name says which metric.

## Design rules

- Inputs are validated and fail loudly. NaNs are never silently filled.
- Undefined results are `NaN` with an explanation attached, not a misleading number.
- A limit that can't be evaluated is reported `NOT_EVALUATED` with a reason, never skipped.
- Metric functions are pure; the engine adds structure, limits and warnings.

## What is implemented in 0.1

Returns/risk statistics, historical and parametric VaR/ES, benchmark-relative metrics
(ex-post and ex-ante), covariance estimators (sample, EWMA, Ledoit-Wolf, nearest-PSD),
Euler risk decomposition by asset and group, limits and risk budgets, batch execution.

## Roadmap

Factor models and attribution (Brinson, factor), stress testing and VaR backtesting
(Kupiec, Christoffersen), liquidity risk, rates (DV01, key-rate duration), credit
(spread duration, DTS), multi-currency, look-through, objective-specific metric sets
for income and liability-driven mandates, Excel/HTML reporting.
