import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def asset_returns() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    idx = pd.bdate_range("2022-01-03", periods=750)
    cols = ["AAA", "BBB", "CCC", "DDD", "EEE"]
    market = rng.normal(0.0003, 0.008, len(idx))
    data = {c: 0.8 * market + rng.normal(0, 0.007 + 0.001 * i, len(idx)) for i, c in enumerate(cols)}
    return pd.DataFrame(data, index=idx)
