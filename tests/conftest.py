"""A small synthetic dataset in the real on-disk layout (``ml4trading.data``)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

TZ = "America/New_York"


def make_bars(symbol_seed: int, start: str, end: str, price0: float = 100.0) -> pd.DataFrame:
    rng = np.random.default_rng(symbol_seed)
    days = pd.bdate_range(start, end)
    stamps = []
    for d in days:
        stamps.extend(pd.date_range(d + pd.Timedelta("9h30min"), periods=13, freq="30min"))
    ts = pd.DatetimeIndex(stamps).tz_localize(TZ)
    steps = rng.normal(0.0002, 0.003, size=len(ts))
    close = price0 * np.exp(np.cumsum(steps))
    open_ = np.concatenate(([price0], close[:-1])) * (1 + rng.normal(0, 0.0005, len(ts)))
    high = np.maximum(open_, close) * 1.001
    low = np.minimum(open_, close) * 0.999
    volume = rng.integers(50_000, 150_000, size=len(ts)).astype(float)
    return pd.DataFrame(
        {"ts": ts, "open": open_, "high": high, "low": low, "close": close, "volume": volume}
    )


def write_dataset(root: Path, start="2021-01-04", end="2021-12-31") -> Path:
    (root / "bars_30min").mkdir(parents=True)
    symbols = {"AAA": (1, 50.0), "BBB": (2, 120.0), "CCC": (3, 30.0), "VOO": (4, 350.0)}
    files = {}
    for sym, (seed, p0) in symbols.items():
        path = root / "bars_30min" / f"{sym}.parquet"
        make_bars(seed, start, end, p0).to_parquet(path, index=False)
        files[f"bars_30min/{sym}.parquet"] = path
    divs = pd.DataFrame(
        {
            "symbol": ["VOO", "VOO", "AAA"],
            "ex_date": pd.to_datetime(["2021-03-26", "2021-06-25", "2021-05-10"]),
            "div_ret": [0.004, 0.003, 0.01],
        }
    )
    divs.to_parquet(root / "dividends.parquet", index=False)
    files["dividends.parquet"] = root / "dividends.parquet"
    universe = {
        "as_of": "2021-12-31",
        "symbols": ["AAA", "BBB", "CCC"],
        "etfs": ["VOO"],
        "market_cap_usd": {"AAA": 3e11, "BBB": 2e11, "CCC": 1e11},
    }
    (root / "universe.json").write_text(json.dumps(universe))
    files["universe.json"] = root / "universe.json"
    manifest = {
        "files": {
            rel: {"sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for rel, p in files.items()
        }
    }
    (root / "MANIFEST.json").write_text(json.dumps(manifest))
    return root


@pytest.fixture
def data_dir(tmp_path) -> Path:
    return write_dataset(tmp_path / "data")
