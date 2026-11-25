"""Rough scale check: many portfolios, one shared covariance estimate."""
import time

import numpy as np
import pandas as pd

from riskframe import Mandate, Objective, Portfolio, RiskBudget, run_book

N_PORT, N_ASSETS, N_DAYS = 200, 500, 1000
rng = np.random.default_rng(0)
idx = pd.bdate_range("2020-01-01", periods=N_DAYS)
assets = [f"A{i:04d}" for i in range(N_ASSETS)]
factor = rng.normal(0, 0.01, (N_DAYS, 1))
R = pd.DataFrame(factor * rng.uniform(0.5, 1.5, N_ASSETS) + rng.normal(0, 0.01, (N_DAYS, N_ASSETS)),
                 index=idx, columns=assets)
mandate = Mandate("Core", Objective.ABSOLUTE_RETURN, risk_budget=RiskBudget(volatility=0.20))
book = []
for i in range(N_PORT):
    held = rng.choice(assets, size=rng.integers(40, 200), replace=False)
    book.append((Portfolio(f"P{i:03d}", pd.Series(rng.dirichlet(np.ones(len(held))), index=held)), mandate))

t0 = time.perf_counter()
reports = run_book(book, R)
dt = time.perf_counter() - t0
print(f"{N_PORT} portfolios x {N_ASSETS}-asset universe x {N_DAYS} days: {dt:.2f}s "
      f"({dt / N_PORT * 1000:.0f} ms/portfolio)")
