"""The dataset on disk (layout and provenance: DATA.md) and its readers.

    <data_dir>/bars_30min/<SYMBOL>.parquet   ts, open, high, low, close, volume
    <data_dir>/dividends.parquet             symbol, ex_date, div_ret
    <data_dir>/universe.json                 as_of, symbols, etfs, market_cap_usd
    <data_dir>/MANIFEST.json                 sha256 and row count of every file

``ts`` is the bar's START, tz-aware America/New_York, regular session only.
``python -m ml4trading.data verify [--data-dir DIR]`` checks every file against the manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import pandas as pd

from ml4trading.config import TIMEZONE

BAR_COLUMNS = ["ts", "open", "high", "low", "close", "volume"]


def default_data_dir() -> Path:
    return Path(os.environ.get("ML4T_DATA_DIR", "data"))


def load_universe(data_dir: Path) -> dict:
    return json.loads((Path(data_dir) / "universe.json").read_text())


def all_symbols(data_dir: Path) -> list[str]:
    u = load_universe(data_dir)
    return list(u["symbols"]) + list(u["etfs"])


def load_bars(data_dir: Path, symbols: list[str], start=None, end=None) -> pd.DataFrame:
    """Long-form 30-minute bars of ``symbols`` in ``[start, end)`` (dates or timestamps,
    interpreted in New York time), sorted by (symbol, ts). A symbol with no file is refused."""
    lo = _ny(start)
    hi = _ny(end)
    frames = []
    for sym in symbols:
        path = Path(data_dir) / "bars_30min" / f"{sym}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"no bars for {sym!r} at {path}")
        df = pd.read_parquet(path, columns=BAR_COLUMNS)
        if lo is not None:
            df = df[df["ts"] >= lo]
        if hi is not None:
            df = df[df["ts"] < hi]
        df.insert(1, "symbol", sym)
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["ts", "symbol", *BAR_COLUMNS[1:]])
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["symbol", "ts"], kind="stable").reset_index(drop=True)


def load_dividends(data_dir: Path, symbols: list[str] | None = None) -> pd.DataFrame:
    path = Path(data_dir) / "dividends.parquet"
    divs = pd.read_parquet(path)
    if symbols is not None:
        divs = divs[divs["symbol"].isin(set(symbols))]
    return divs.reset_index(drop=True)


def _ny(t):
    if t is None:
        return None
    ts = pd.Timestamp(t)
    return ts.tz_localize(TIMEZONE) if ts.tz is None else ts.tz_convert(TIMEZONE)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(data_dir: Path) -> list[str]:
    """Every manifest entry whose file is missing or whose sha256 differs (empty = intact)."""
    manifest = json.loads((Path(data_dir) / "MANIFEST.json").read_text())
    bad = []
    for rel, meta in sorted(manifest["files"].items()):
        path = Path(data_dir) / rel
        if not path.exists() or _sha256(path) != meta["sha256"]:
            bad.append(rel)
    return bad


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m ml4trading.data")
    ap.add_argument("command", choices=["verify"])
    ap.add_argument("--data-dir", type=Path, default=default_data_dir())
    args = ap.parse_args()
    bad = verify(args.data_dir)
    if bad:
        raise SystemExit(f"{len(bad)} file(s) missing or corrupt: {bad[:10]}")
    print(f"{args.data_dir}: every manifest file present and intact")


if __name__ == "__main__":
    main()
