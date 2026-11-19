"""The next session's recommended portfolio from the frozen v29 configuration, for the field pilot.

    python scripts/field_eval/recommend.py --data-dir data_live [--date YYYY-MM-DD]
        [--capital 20000]

1. D = the next NYSE session after the data's last session (or --date).
2. data_pred = a clone of <data-dir> plus one placeholder 09:30 bar on D per symbol (OHLC = the
   last close). Every feature of D's row reads the close of D-1 or earlier (a per-symbol
   shift(1), checked by assert_causal), so the placeholder's values never reach a prediction;
   it only makes D's row exist.
3. Train v29 (runs/report/v29/run.json's config, unchanged) for the fold of D's month only:
   deploy fit on the 36 months before the month start, ending 5 sessions before it, exactly as
   in the backtest.
4. D's predictions -> the backtest's own trader (target_weights) and share sizing (size_shares),
   entry price estimated by D-1's last close, on a fresh account of --capital.

Writes results/field_eval/recs/<D>.csv (the portfolio), <D>_all.csv (all 100 universe predictions)
and <D>.md (the message for participants); the fold is trained into runs/live/v29_<D>/.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import pandas as pd
import pandas_market_calendars as mcal

from ml4trading.config import RunConfig
from ml4trading.sizing import size_shares
from ml4trading.trader import target_weights
from ml4trading.train import train

TZ = "America/New_York"
ROOT = Path(__file__).resolve().parents[2]


def flat_overrides(cfg: dict) -> dict:
    out = {}
    for sec in ("model", "selector", "trader", "folds", "costs"):
        for k, v in cfg[sec].items():
            out[f"{sec}.{k}"] = v
    for k in ("K", "capital", "reinvest"):
        out[k] = cfg[k]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data_live")
    ap.add_argument("--date", default=None)
    ap.add_argument("--capital", type=float, default=20000.0)
    ap.add_argument("--frozen", type=Path, default=ROOT / "runs/report/v29/run.json")
    args = ap.parse_args()
    # The feature build sizes its process pool from os.cpu_count() when it runs; cap it
    # (ML4T_WORKERS, default 4) so the nightly run does not take over a shared machine.
    # The features do not depend on the number of workers.
    workers = int(os.environ.get("ML4T_WORKERS", "4"))
    os.cpu_count = lambda: workers

    u = json.loads((args.data_dir / "universe.json").read_text())
    symbols = list(u["symbols"]) + list(u["etfs"])
    last_close, last_session = {}, None
    for s in symbols:
        b = pd.read_parquet(args.data_dir / "bars_30min" / f"{s}.parquet")
        if len(b):
            last_close[s] = (
                b["ts"].iloc[-1],
                float(b["close"].iloc[-1]),
                float(b["volume"].iloc[-1]),
            )
            d = b["ts"].iloc[-1].normalize()
            last_session = d if last_session is None else max(last_session, d)
    sched = mcal.get_calendar("NYSE").schedule(
        start_date=last_session.date(), end_date=(last_session + pd.Timedelta(days=10)).date()
    )
    nxt = [pd.Timestamp(x) for x in sched.index if pd.Timestamp(x) > last_session.tz_localize(None)]
    D = pd.Timestamp(args.date) if args.date else nxt[0]
    D = D.tz_localize(None) if D.tz else D
    assert D == nxt[0], (
        f"--date {D.date()} is not the next session after the data ({nxt[0].date()})"
    )
    print(f"data ends {last_session.date()}, predicting {D.date()}", flush=True)

    pred_dir = ROOT / "data_pred"
    subprocess.run(["rm", "-rf", str(pred_dir)], check=True)
    subprocess.run(["cp", "-cR", str(args.data_dir), str(pred_dir)], check=True)
    open_ts = pd.Timestamp(f"{D.date()} 09:30", tz=TZ)
    n_ph = 0
    for s, (ts, c, v) in last_close.items():
        if ts.normalize() != last_session:  # no bar on the last session: no placeholder either
            continue
        p = pred_dir / "bars_30min" / f"{s}.parquet"
        b = pd.read_parquet(p)
        row = pd.DataFrame(
            {"ts": [open_ts], "open": [c], "high": [c], "low": [c], "close": [c], "volume": [v]}
        )
        row["ts"] = row["ts"].astype(b["ts"].dtype)
        pd.concat([b, row], ignore_index=True).to_parquet(p, index=False)
        n_ph += 1
    print(f"placeholder bars on {D.date()}: {n_ph}", flush=True)

    frozen = json.loads(args.frozen.read_text())
    ov = flat_overrides(frozen["config"])
    month_start = D.replace(day=1)
    month_end = month_start + pd.offsets.MonthEnd(0)
    ov["folds.path_start"] = str(month_start.date())
    ov["folds.path_end"] = str(month_end.date())
    ov["capital"] = args.capital
    out = ROOT / "runs" / "live" / f"v29_{D.date()}"
    train(frozen["method"], out, pred_dir, overrides=ov)
    run = json.loads((out / "run.json").read_text())
    cfg = RunConfig.from_dict(run["config"])
    fold = run["folds"][-1]
    fj = json.loads((out / "folds" / fold / "fold.json").read_text())
    preds = pd.read_parquet(out / "folds" / fold / "predictions.parquet")
    today = preds[preds["period"] == D.tz_localize(None)].copy()
    today = today.dropna(subset=["prediction"])
    today["open_px"] = today["symbol"].map(lambda s: last_close[s][1])
    print(
        f"fold {fold}: fit_end {fj['fit_end']}, universe {fj['n_universe']}, "
        f"{len(today)} predictions on {D.date()}",
        flush=True,
    )

    w = target_weights(
        today[["symbol", "prediction", "open_px"]],
        trader=cfg.trader,
        costs=cfg.costs,
        capital=args.capital,
        hold_overnight=True,
    )
    prices = dict(zip(today["symbol"], today["open_px"], strict=True))
    shares = size_shares(w, prices, args.capital, {}, {}, trader=cfg.trader, costs=cfg.costs)
    rows = []
    for s, wt in sorted(w.items(), key=lambda kv: -abs(kv[1])):
        pr = float(today.loc[today["symbol"] == s, "prediction"].iloc[0])
        sh = int(shares.get(s, 0))
        rows.append(
            {
                "date": str(D.date()),
                "symbol": s,
                "side": "BUY" if wt > 0 else "SHORT",
                "weight": round(wt, 4),
                "dollars": round(wt * args.capital, 2),
                "shares": sh,
                "ref_price_prev_close": prices[s],
                "pred_return_bp": round(pr * 1e4, 1),
            }
        )
    book = pd.DataFrame(rows)
    rec = ROOT / "results" / "field_eval" / "recs"
    rec.mkdir(parents=True, exist_ok=True)
    book.to_csv(rec / f"{D.date()}.csv", index=False)
    today.sort_values("prediction").to_csv(rec / f"{D.date()}_all.csv", index=False)

    lines = [
        f"# Recommendations for {D.date()} (enter at the 9:30 open, hold to the next open)",
        "",
        f"Account: ${args.capital:,.0f} virtual. Prices are {last_session.date()} closes; "
        "shares = whole shares at that price.",
        "",
        "| # | Ticker | Action | Weight | $ | Shares | Predicted 1-day return |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for i, r in enumerate(rows, 1):
        lines.append(
            f"| {i} | {r['symbol']} | {r['side']} | {abs(r['weight']):.1%} | "
            f"${abs(r['dollars']):,.0f} | {abs(r['shares'])} | {r['pred_return_bp']:+.0f} bp |"
        )
    gross = sum(abs(r["weight"]) for r in rows)
    long_w = sum(r["weight"] for r in rows if r["weight"] > 0)
    short_w = -sum(r["weight"] for r in rows if r["weight"] < 0)
    lines += [
        "",
        f"Gross exposure {gross:.0%}; long {long_w:.0%}, short {short_w:.0%}.",
    ]
    (rec / f"{D.date()}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
