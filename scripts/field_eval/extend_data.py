"""Extend the dataset past its last session with Alpaca SIP 30-minute bars, for the field pilot.

    python scripts/field_eval/extend_data.py --src data --dst data_live [--end YYYY-MM-DD]

Credentials: ALPACA_API_KEY and ALPACA_SECRET_KEY from the environment, or from a KEY=VALUE file
given with --env-file. <dst> holds licensed data: it is git-ignored (data_*/) and stays local.

- Copies <src> to <dst> (APFS clone), never touches <src>.
- Fetches split-adjusted SIP 30-min bars 2026-08-03 .. end for every symbol, keeps regular-session
  bars (start 09:30..15:30 ET), and appends the bars after the source's last session.
- Overlap check (2026-08-03 .. 2026-08-31): per symbol, median Alpaca/source close ratio and the
  share of bars whose close matches within 0.1 %. A ratio far from 1 means a split after 08-31
  (Alpaca adjusts to today): the source rows are rescaled by it (price x r, volume / r).
- Cash dividends with ex-date after the source's last session come from Alpaca corporate actions;
  div_ret = rate / previous session's last close.
- Writes <dst>/live_extension.json (per-symbol overlap stats, rescales, row counts) and a fresh
  MANIFEST.json for <dst>.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

TZ = "America/New_York"
OVERLAP_START = "2026-08-24"


def alpaca_headers(env_file: Path | None = None) -> dict:
    env = dict(os.environ)
    if env_file is not None:
        for line in open(env_file):
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.strip().split("=", 1)
                env[k] = v.strip().strip('"').strip("'")
    missing = [k for k in ("ALPACA_API_KEY", "ALPACA_SECRET_KEY") if not env.get(k)]
    if missing:
        raise SystemExit(f"set {' and '.join(missing)} in the environment or pass --env-file")
    return {
        "APCA-API-KEY-ID": env["ALPACA_API_KEY"],
        "APCA-API-SECRET-KEY": env["ALPACA_SECRET_KEY"],
    }


N_429 = 0


def get_json(url: str, headers: dict, tries: int = 200) -> dict:
    """The key is shared with other jobs that keep its 200/min budget near 0: retry 429s
    patiently (2-10 s), give up only on other errors after 5 tries."""
    global N_429
    other = 0
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=headers)
            return json.load(urllib.request.urlopen(req, timeout=60))
        except urllib.error.HTTPError as e:
            if e.code == 429:
                N_429 += 1
                time.sleep(0.5 + 0.5 * (i % 4))
                continue
            other += 1
            if other >= 5:
                raise
            time.sleep(5)
        except Exception:
            other += 1
            if other >= 5:
                raise
            time.sleep(5)
    raise RuntimeError(f"gave up after {tries} tries: {url[:120]}")


def fetch_bars(symbols: list[str], start: str, end: str, headers: dict) -> pd.DataFrame:
    """Regular-session bars only: one request series per (NYSE session, 100-symbol chunk), so
    Alpaca's ~400-bar pages carry no extended/overnight bars; 4 threads."""
    import threading
    from concurrent.futures import ThreadPoolExecutor

    import pandas_market_calendars as mcal

    sched = mcal.get_calendar("NYSE").schedule(start_date=start[:10], end_date=end[:10])
    end_cap = pd.Timestamp(end)
    days = [
        (r.market_open, min(r.market_close - pd.Timedelta(minutes=1), end_cap))
        for r in sched.itertuples()
        if r.market_open < end_cap
    ]
    chunks = [symbols[i : i + 100] for i in range(0, len(symbols), 100)]
    tasks = [(c, o, e) for (o, e) in days for c in chunks]
    rows, lock, done = [], threading.Lock(), [0]

    def run(task):
        chunk, o, e = task
        token, got = None, []
        while True:
            q = {
                "symbols": ",".join(chunk),
                "timeframe": "30Min",
                "start": o.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "end": e.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "adjustment": "split",
                "feed": "sip",
                "limit": 10000,
            }
            if token:
                q["page_token"] = token
            r = get_json(
                "https://data.alpaca.markets/v2/stocks/bars?" + urllib.parse.urlencode(q), headers
            )
            for sym, bars in (r.get("bars") or {}).items():
                for b in bars:
                    got.append((sym, b["t"], b["o"], b["h"], b["l"], b["c"], b["v"]))
            token = r.get("next_page_token")
            if not token:
                break
        with lock:
            rows.extend(got)
            done[0] += 1
            if done[0] % 25 == 0 or done[0] == len(tasks):
                print(
                    f"bars: {done[0]}/{len(tasks)} tasks ({len(days)} sessions), {len(rows)} bars, "
                    f"{N_429} x 429",
                    flush=True,
                )

    with ThreadPoolExecutor(4) as ex:
        list(ex.map(run, tasks))
    df = pd.DataFrame(rows, columns=["symbol", "t", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["t"], utc=True).dt.tz_convert(TZ).astype(f"datetime64[us, {TZ}]")
    hm = df["ts"].dt.hour * 60 + df["ts"].dt.minute
    df = df[(hm >= 9 * 60 + 30) & (hm <= 15 * 60 + 30)]
    df["volume"] = df["volume"].astype(float)
    df = df.drop(columns="t").drop_duplicates(["symbol", "ts"]).sort_values(["symbol", "ts"])
    return df.reset_index(drop=True)


def fetch_dividends(symbols: list[str], start: str, end: str, headers: dict) -> pd.DataFrame:
    rows = []
    for i in range(0, len(symbols), 50):
        token = None
        while True:
            q = {
                "symbols": ",".join(symbols[i : i + 50]),
                "types": "cash_dividend",
                "start": start,
                "end": end,
                "limit": 1000,
            }
            if token:
                q["page_token"] = token
            r = get_json(
                "https://data.alpaca.markets/v1/corporate-actions?" + urllib.parse.urlencode(q),
                headers,
            )
            for d in (r.get("corporate_actions") or {}).get("cash_dividends", []):
                rows.append((d["symbol"], d["ex_date"], float(d["rate"])))
            token = r.get("next_page_token")
            if not token:
                break
    return pd.DataFrame(rows, columns=["symbol", "ex_date", "rate"]).drop_duplicates()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=Path("data"))
    ap.add_argument("--dst", type=Path, default=Path("data_live"))
    ap.add_argument("--end", default=pd.Timestamp.now(tz=TZ).strftime("%Y-%m-%d"))
    ap.add_argument("--env-file", type=Path, default=None)
    args = ap.parse_args()
    src, dst = args.src, args.dst
    if dst.exists():
        shutil.rmtree(dst)
    try:  # an APFS clone on macOS costs no disk space; a plain copy elsewhere
        subprocess.run(["cp", "-cR", str(src), str(dst)], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        shutil.copytree(src, dst)

    u = json.loads((src / "universe.json").read_text())
    symbols = list(u["symbols"]) + list(u["etfs"])
    headers = alpaca_headers(args.env_file)
    end_utc = (pd.Timestamp(args.end, tz=TZ) + pd.Timedelta(hours=20)).tz_convert("UTC")
    end_utc = min(end_utc, pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=16))
    new = fetch_bars(
        symbols, f"{OVERLAP_START}T00:00:00Z", end_utc.strftime("%Y-%m-%dT%H:%M:%SZ"), headers
    )

    report, last_close = {}, {}
    src_last_session = None
    for sym in symbols:
        old = pd.read_parquet(src / "bars_30min" / f"{sym}.parquet")
        a = new[new["symbol"] == sym].drop(columns="symbol")
        info = {"src_rows": len(old)}
        if len(old) == 0 or len(a) == 0:
            info["status"] = "no alpaca bars" if len(a) == 0 else "empty source"
            report[sym] = info
            continue
        last_ts = old["ts"].max()
        src_last_session = (
            last_ts.normalize()
            if src_last_session is None
            else max(src_last_session, last_ts.normalize())
        )
        ov = old[old["ts"] >= pd.Timestamp(OVERLAP_START, tz=TZ)].merge(
            a, on="ts", suffixes=("_s", "_a")
        )
        ratio = float(np.median(ov["close_a"] / ov["close_s"])) if len(ov) else float("nan")
        info.update(overlap_bars=len(ov), median_ratio=ratio)
        r = 1.0
        if len(ov) and abs(ratio - 1) > 0.02:  # a split (or ratio change) after the source ends
            r = ratio
            for c in ("open", "high", "low", "close"):
                old[c] = old[c] * r
            old["volume"] = old["volume"] / r
            info["rescaled_source_by"] = r
            ov = old[old["ts"] >= pd.Timestamp(OVERLAP_START, tz=TZ)].merge(
                a, on="ts", suffixes=("_s", "_a")
            )
        if len(ov):
            rel = (ov["close_a"] / ov["close_s"] - 1).abs()
            info["close_match_0.1pct"] = float((rel < 1e-3).mean())
            info["close_absdev_median"] = float(rel.median())
            info["volume_ratio_median"] = float(
                np.median(ov["volume_a"] / ov["volume_s"].replace(0, np.nan))
            )
        add = a[a["ts"] > last_ts].copy()
        vr = info.get("volume_ratio_median")
        # Alpaca's SIP volume, rescaled to the dataset vendor's volume definition
        if vr is not None and np.isfinite(vr) and vr > 0:
            add["volume"] = add["volume"] / vr
            info["volume_rescaled_by"] = 1 / vr
        out = (
            pd.concat([old, add[old.columns]], ignore_index=True)
            .sort_values("ts")
            .reset_index(drop=True)
        )
        out.to_parquet(dst / "bars_30min" / f"{sym}.parquet", index=False)
        info.update(added_rows=len(add), new_last_ts=str(out["ts"].max()))
        report[sym] = info
        # previous-session last close for dividend returns
        out["d"] = out["ts"].dt.tz_localize(None).dt.normalize()
        last_close[sym] = out.groupby("d")["close"].last()

    divs = pd.read_parquet(src / "dividends.parquet")
    start_div = (src_last_session + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    nd = fetch_dividends(symbols, start_div, args.end, headers)
    added_divs = []
    for row in nd.itertuples():
        lc = last_close.get(row.symbol)
        ex = pd.Timestamp(row.ex_date)
        if lc is None or ex > lc.index.max():
            continue
        prev = lc[lc.index < ex]
        if len(prev) == 0:
            continue
        added_divs.append((row.symbol, ex, row.rate / float(prev.iloc[-1])))
    if added_divs:
        add = pd.DataFrame(added_divs, columns=["symbol", "ex_date", "div_ret"])
        add["ex_date"] = add["ex_date"].astype(divs["ex_date"].dtype)
        divs = pd.concat([divs, add], ignore_index=True)
    divs.to_parquet(dst / "dividends.parquet", index=False)

    manifest = json.loads((src / "MANIFEST.json").read_text())
    manifest["live_extension"] = {
        "source": "Alpaca SIP 30Min, adjustment=split",
        "end": args.end,
        "dividends_added": len(added_divs),
        "volume_rescaled_to_source": True,
    }
    for rel in list(manifest["files"]):
        p = dst / rel
        manifest["files"][rel] = {
            "rows": len(pd.read_parquet(p)) if p.suffix == ".parquet" else None,
            "sha256": sha256(p),
        }
    (dst / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (dst / "live_extension.json").write_text(
        json.dumps(
            {"symbols": report, "dividends_added": len(added_divs), "volume_rescaled": True},
            indent=2,
            default=str,
        )
        + "\n"
    )
    st = pd.DataFrame(report).T
    print(
        "symbols:", len(st), "| no alpaca bars:", int((st.get("status") == "no alpaca bars").sum())
    )
    print(
        "rescaled:",
        {k: v.get("rescaled_source_by") for k, v in report.items() if "rescaled_source_by" in v},
    )
    m = pd.to_numeric(st["close_match_0.1pct"], errors="coerce")
    print(
        f"overlap close-match share: median {m.median():.3f}, p10 {m.quantile(0.1):.3f}, "
        f"min {m.min():.3f}"
    )
    print("worst 8:", m.nsmallest(8).round(3).to_dict())
    added = pd.to_numeric(st["added_rows"], errors="coerce")
    print(
        f"added rows per symbol: median {added.median():.0f}, min {added.min():.0f}, "
        f"max {added.max():.0f}"
    )
    print("dividends added:", len(added_divs))


if __name__ == "__main__":
    main()
