"""Quickstart: one portfolio, one mandate, on synthetic data."""
import numpy as np
import pandas as pd

from riskframe import (Benchmark, Mandate, MaxGroupWeight, MaxPosition, Objective, Portfolio,
                       RiskBudget, RiskEngine)

rng = np.random.default_rng(42)
idx = pd.bdate_range("2023-01-02", periods=750)
assets = ["AAPL", "MSFT", "JPM", "XOM", "JNJ"]
mkt = rng.normal(0.0003, 0.008, len(idx))
returns = pd.DataFrame({a: 0.8 * mkt + rng.normal(0, 0.007, len(idx)) for a in assets}, index=idx)

bench_w = pd.Series(0.2, index=assets)
benchmark = Benchmark("EqualWeight5", returns=returns @ bench_w, weights=bench_w)

mandate = Mandate(
    name="US Large-Cap Core",
    objective=Objective.RELATIVE_RETURN,
    benchmark=benchmark,
    risk_budget=RiskBudget(tracking_error=0.03, var=0.02),
    constraints=[MaxPosition(limit=0.35), MaxGroupWeight(group_field="sector", limit=0.60)],
)
portfolio = Portfolio(
    "Fund A",
    pd.Series([0.30, 0.30, 0.15, 0.10, 0.15], index=assets),
    classification=pd.DataFrame(
        {"sector": ["Tech", "Tech", "Fin", "Energy", "Health"]}, index=assets),
)

report = RiskEngine(mandate).run(portfolio, returns)
pd.set_option("display.width", 120)
print(f"{report.portfolio} | {report.mandate} | as of {report.as_of:%Y-%m-%d} | status: {report.status.value}\n")
print(report.metrics_frame()[["value", "unit", "basis"]].round(4).to_string(), "\n")
print(report.limits_frame().round(4).to_string(index=False), "\n")
print("Risk contribution by sector:\n", report.group_decomposition["sector"].round(4).to_string())
for w in report.warnings:
    print("WARNING:", w)
