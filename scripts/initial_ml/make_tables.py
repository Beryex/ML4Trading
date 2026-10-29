"""The Initial ML Solution's tables, from saved run outputs only (no training, no backtest).

    python scripts/initial_ml/make_tables.py [--runs runs] [--data-dir D] [--out results/initial_ml]

Reads each run's backtest/ (metrics.json, by_year.csv, daily.csv, book.csv, positions.csv) and
folds/ (fold.json, predictions.parquet, universe.json, model/) under ``--runs``/report/ -- the
layout ``scripts/initial_ml/README.md`` builds -- and the dataset's bars for the realized labels
and the liquidity ranks. Writes CSVs to ``--out``/tables/ and ``--out``/daily_equity.csv. Every
definition follows ml4trading.metrics (Sharpe = mean / population std of daily open-to-open
account returns x sqrt(252), no risk-free rate).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ml4trading.config import RunConfig
from ml4trading.costs import leg_amounts
from ml4trading.data import all_symbols, default_data_dir, load_bars, load_dividends, load_universe
from ml4trading.diagnostics import ic_summary, leg_pnl, period_ic
from ml4trading.folds import embargoed_end
from ml4trading.metrics import sharpe
from ml4trading.models.learned.risk_adjust import target_volatility
from ml4trading.periods import build_panel
from ml4trading.trader import apply_caps, cost_gate, decision_metric, select

# set by configure() from the command line
DATA: Path = Path()
OUT: Path = Path()
TAB: Path = Path()
R: Path = Path()

BASELINES = {
    "Buy-and-hold VOO": "voo",
    "Buy-and-hold QQQ": "qqq",
    "Daily momentum": "daily_mom",
    "Daily reversal": "daily_rev",
    "Weekly momentum": "weekly_mom",
    "Weekly reversal": "weekly_rev",
}
RULES = ["Daily momentum", "Daily reversal", "Weekly momentum", "Weekly reversal"]


def configure(runs: Path, data_dir: Path, out: Path) -> None:
    global DATA, OUT, TAB, R, ML, NEW, OLD, ALL_NEW
    DATA, OUT, R = Path(data_dir), Path(out), Path(runs) / "report"
    TAB = OUT / "tables"
    ML = {
        "Three-feature linear regression (alpha_ratio 1e-6)": R / "lin3_ols",
        "Three-feature linear (per-fold grid)": R / "lin3",
        "linear, all 188 features (per-fold grid)": R / "linear_grid",
        "nonlinear LightGBM, all 189 features (per-fold grid)": R / "nonlinear_grid",
        "Validation-selected: linear vs nonlinear grids": R / "sel_grid",
        "linear v29 (frozen)": R / "v29",
        "nonlinear P3b hybrid (frozen)": R / "p3b",
        "Validation-selected: v29 vs P3b": R / "sel_final",
    }
    NEW = {k: R / "new" / v for k, v in BASELINES.items()}
    OLD = {k: R / "old" / v for k, v in BASELINES.items()}
    ALL_NEW = {**NEW, **ML}


COST_COLS = [
    "entry_spread",
    "entry_commission",
    "exit_spread",
    "exit_commission",
    "close_spread",
    "close_commission",
    "borrow",
]
YEARS = list(range(2020, 2027))


def load_run(d: Path) -> dict:
    bt = d / "backtest"
    run = json.loads((d / "run.json").read_text())
    folds = [json.loads((d / "folds" / f / "fold.json").read_text()) for f in run["folds"]]
    pos = pd.read_csv(bt / "positions.csv", parse_dates=["period"])
    pos["net"] = pos["gross"] - pos[COST_COLS].sum(axis=1)
    return {
        "dir": d,
        "run": run,
        "cfg": RunConfig.from_dict(run["config"]),
        "folds": folds,
        "metrics": json.loads((bt / "metrics.json").read_text()),
        "by_year": pd.read_csv(bt / "by_year.csv"),
        "daily": pd.read_csv(bt / "daily.csv", parse_dates=["session"]),
        "book": pd.read_csv(bt / "book.csv", parse_dates=["period", "session"]),
        "pos": pos,
    }


def preds_of(r: dict) -> pd.DataFrame:
    d = r["dir"]
    return pd.concat(
        [pd.read_parquet(d / "folds" / f / "predictions.parquet") for f in r["run"]["folds"]],
        ignore_index=True,
    )


def fold_of(dates: pd.Series, folds: list[dict]) -> pd.Series:
    """The fold whose test month holds each date."""
    out = pd.Series(index=dates.index, dtype=object)
    for f in folds:
        lo, hi = pd.Timestamp(f["test"][0]), pd.Timestamp(f["test"][1])
        out[(dates >= lo) & (dates <= hi)] = f["name"]
    return out


def n_orders(pos: pd.DataFrame) -> np.ndarray:
    """Orders per row: one per leg (an entry, an exit; a sign flip is both)."""
    out = []
    for prev, cur in zip(
        pos["prev_shares"].to_numpy(float), pos["shares"].to_numpy(float), strict=True
    ):
        e, x = leg_amounts(prev, cur)
        out.append(int(e > 0) + int(x > 0))
    return np.asarray(out)


def held_per_session(r: dict) -> pd.Series:
    """Names held during each valued session (the book period covering it)."""
    pos, daily = r["pos"], r["daily"]
    held = pos[pos["shares"] != 0].groupby("period")["symbol"].nunique()
    periods = pd.DataFrame({"period": sorted(r["book"]["period"].unique())})
    periods["held"] = periods["period"].map(held).fillna(0).astype(int)
    m = pd.merge_asof(
        daily[["session"]].sort_values("session"), periods, left_on="session", right_on="period"
    )
    return m.set_index("session")["held"]


def whole_period(r: dict) -> dict:
    m = r["metrics"]
    costs = m["total_spread_cost"] + m["total_commission"] + m["total_borrow"]
    held = held_per_session(r)
    orders = n_orders(r["pos"]).sum()
    return {
        "net_sharpe": m["net_sharpe"],
        "gross_sharpe": m["gross_sharpe"],
        "net_profit_usd": m["net_profit"],
        "net_profit_pct": 100 * m["cumulative_return"],
        "alpha_pct": 100 * m["alpha"],
        "alpha_t": m["alpha_t"],
        "beta": m["beta"],
        "max_drawdown_pct": 100 * m["max_drawdown"],
        "trade_hit_rate_pct": None
        if m.get("trade_hit_rate") is None
        else 100 * m["trade_hit_rate"],
        "completed_trades": m["n_trades"],
        "costs_paid_usd": costs,
        "spread_usd": m["total_spread_cost"],
        "commission_usd": m["total_commission"],
        "borrow_usd": m["total_borrow"],
        "avg_names_held_per_session": float(held.mean()),
        "share_sessions_all_cash_pct": 100 * float((held == 0).mean()),
        "avg_orders_per_session": orders / len(r["daily"]),
        "n_sessions_valued": len(r["daily"]),
        "start": m["start"],
        "end": m["end"],
    }


def by_year(runs: dict, col: str, scale: float = 1.0) -> pd.DataFrame:
    rows = {}
    for name, r in runs.items():
        y = r["by_year"].set_index("year")
        rows[name] = {
            yr: (None if pd.isna(y[col].get(yr)) else scale * y[col].get(yr)) for yr in YEARS
        }
    return pd.DataFrame(rows).T


def monthly(r: dict) -> pd.Series:
    d = r["daily"].set_index("session")["ret"]
    return (1 + d).groupby(d.index.to_period("M")).prod() - 1


def write(df: pd.DataFrame, name: str, index=True) -> None:
    df.to_csv(TAB / name, index=index, float_format="%.6g")
    print("wrote", name)


def main() -> None:
    ap = argparse.ArgumentParser(prog="python scripts/initial_ml/make_tables.py")
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    ap.add_argument("--data-dir", type=Path, default=default_data_dir())
    ap.add_argument("--out", type=Path, default=Path("results/initial_ml"))
    args = ap.parse_args()
    configure(args.runs, args.data_dir, args.out)
    TAB.mkdir(parents=True, exist_ok=True)
    runs = {k: load_run(v) for k, v in ALL_NEW.items()}
    old = {k: load_run(v) for k, v in OLD.items()}
    for name, r in {**runs, **{f"OLD {k}": v for k, v in old.items()}}.items():
        total = r["pos"]["net"].sum()
        assert abs(total - r["metrics"]["net_profit"]) < 1e-3 * max(1, abs(total)), name

    # ---- B1-B3: baselines, OLD vs NEW universe
    rows = []
    for k in BASELINES:
        rows.append({"baseline": k, "universe": "OLD (502)", **whole_period(old[k])})
        rows.append({"baseline": k, "universe": "NEW (552)", **whole_period(runs[k])})
    write(pd.DataFrame(rows), "B1_baselines_whole_period_old_vs_new.csv", index=False)
    for tag, col, sc in (
        ("B2_baselines_sharpe_by_year", "net_sharpe", 1),
        ("B3_baselines_hit_rate_by_year", "trade_hit_rate", 100),
    ):
        frames = []
        for k in BASELINES:
            frames.append(by_year({f"{k} | OLD": old[k]}, col, sc))
            frames.append(by_year({f"{k} | NEW": runs[k]}, col, sc))
        out = pd.concat(frames)
        if col == "trade_hit_rate":
            n = pd.concat(
                [by_year({f"{k} | OLD": old[k]}, "n_trades") for k in BASELINES]
                + [by_year({f"{k} | NEW": runs[k]}, "n_trades") for k in BASELINES]
            )
            out = out.join(n.add_suffix(" trades"))
        write(out, f"{tag}_old_vs_new.csv")

    # ---- G1: everything on the NEW universe
    g1 = pd.DataFrame({k: whole_period(r) for k, r in runs.items()}).T
    write(g1, "G1_whole_period_new_universe.csv")
    write(by_year(runs, "net_sharpe"), "G1b_sharpe_by_year_new_universe.csv")
    hit = by_year(runs, "trade_hit_rate", 100)
    write(
        hit.join(by_year(runs, "n_trades").add_suffix(" trades")),
        "G1c_hit_rate_by_year_new_universe.csv",
    )
    write(by_year(runs, "gross_sharpe"), "G1d_gross_sharpe_by_year_new_universe.csv")

    # ---- G2: deployment bar (best rule = highest whole-period net Sharpe among the four rules)
    best_rule = max(RULES, key=lambda k: runs[k]["metrics"]["net_sharpe"])
    sh, ht = by_year(runs, "net_sharpe"), by_year(runs, "trade_hit_rate", 100)
    voo_dd = runs["Buy-and-hold VOO"]["metrics"]["max_drawdown"]
    bh = max(
        runs["Buy-and-hold VOO"]["metrics"]["net_sharpe"],
        runs["Buy-and-hold QQQ"]["metrics"]["net_sharpe"],
    )
    rows = []
    for k in ML:
        row = {"model": k, "best_rule": best_rule}
        passed = 0
        for yr in range(2020, 2026):
            ds, dh = sh.loc[k, yr] - sh.loc[best_rule, yr], ht.loc[k, yr] - ht.loc[best_rule, yr]
            ok = bool(ds >= 0.3 and dh >= 3.0)
            passed += ok
            row[f"{yr}_dSharpe"], row[f"{yr}_dHit_pts"], row[f"{yr}_pass"] = ds, dh, ok
        m = runs[k]["metrics"]
        row.update(
            years_passed_of_6=passed,
            maxdd_pct=100 * m["max_drawdown"],
            maxdd_within_voo=bool(m["max_drawdown"] >= voo_dd),
            whole_sharpe=m["net_sharpe"],
            beats_both_buy_and_holds=bool(m["net_sharpe"] > bh),
            clears_bar=bool(passed >= 4 and m["max_drawdown"] >= voo_dd and m["net_sharpe"] > bh),
        )
        rows.append(row)
    write(pd.DataFrame(rows), "G2_deployment_bar.csv", index=False)

    # ---- G3: losing months
    mon = pd.DataFrame({k: monthly(r) for k, r in runs.items()})
    write(mon, "G3a_monthly_net_returns.csv")
    rows, worst = [], []
    for k in ML:
        for ref in ("Weekly momentum", "Buy-and-hold VOO", best_rule):
            below = mon[k] < mon[ref]
            row = {"model": k, "vs": ref, "months_below": int(below.sum()), "of": int(below.size)}
            for yr in YEARS:
                sel = below[below.index.year == yr]
                row[str(yr)] = f"{int(sel.sum())}/{sel.size}"
            rows.append(row)
        for p in mon[k].nsmallest(5).index:
            worst.append(
                {"model": k, "month": str(p), **{c: mon.loc[p, c] for c in [k, *BASELINES]}}
            )
    write(
        pd.DataFrame(rows).drop_duplicates(["model", "vs"]),
        "G3b_months_below_reference.csv",
        index=False,
    )
    write(pd.DataFrame(worst), "G3c_worst5_months.csv", index=False)

    # ---- panels for labels and liquidity ranks
    u = load_universe(DATA)
    dropouts = set(u["dropouts"])
    syms = all_symbols(DATA)
    bars = load_bars(DATA, syms)
    divs = load_dividends(DATA, syms)
    panel = build_panel(bars, 390, True, divs)
    panels = {390: panel, 1950: build_panel(bars, 1950, True, divs)}
    del bars

    def liquidity_ranks(r: dict) -> dict:
        """(fold, symbol) -> rank 1..100 in the fold's universe by median dollar volume over the
        deploy window, on the run's own K panel (the selector's ranking)."""
        p = panels[r["cfg"].K]
        out = {}
        for f in r["folds"]:
            uni = json.loads((r["dir"] / "folds" / f["name"] / "universe.json").read_text())
            if not uni:
                continue
            lo, hi = pd.Timestamp(f["deploy"][0]), pd.Timestamp(f["deploy"][1])
            w = p[(p["session"] >= lo) & (p["session"] <= hi) & p["symbol"].isin(uni)]
            med = (w["close"] * w["volume"]).groupby(w["symbol"], observed=True).median()
            ranked = sorted(med.items(), key=lambda kv: (-kv[1], kv[0]))
            for i, (s, _) in enumerate(ranked, 1):
                out[(f["name"], s)] = i
        return out

    # ---- G4 / B2: subsets
    subset_runs = {k: runs[k] for k in [*ML, "Weekly momentum", "Daily reversal"]}
    rows, tops = [], []
    for k, r in subset_runs.items():
        pos = r["pos"].copy()
        legs = leg_pnl(pos)
        pos["fold"] = fold_of(pos["period"], r["folds"])
        ranks = liquidity_ranks(r)
        pos["liq_rank"] = [
            ranks.get((f, s)) for f, s in zip(pos["fold"], pos["symbol"], strict=True)
        ]
        tercile = pd.cut(pos["liq_rank"], [0, 33, 66, 100], labels=["1-33", "34-66", "67-100"])
        by_sym = pos.groupby("symbol")["net"].sum().sort_values(ascending=False)
        total = pos["net"].sum()
        row = {
            "method": k,
            "total_net_pnl": total,
            **legs,
            "dropouts50_net_pnl": pos.loc[pos["symbol"].isin(dropouts), "net"].sum(),
            "original_net_pnl": pos.loc[~pos["symbol"].isin(dropouts), "net"].sum(),
            "dropouts50_share_of_name_periods_held_pct": 100
            * float(pos.loc[pos["shares"] != 0, "symbol"].isin(dropouts).mean()),
            "etf_net_pnl": pos.loc[pos["symbol"].isin(["VOO", "QQQ"]), "net"].sum(),
        }
        for t in ["1-33", "34-66", "67-100"]:
            row[f"liq_{t}_net_pnl"] = pos.loc[tercile == t, "net"].sum()
        row["liq_unranked_net_pnl"] = pos.loc[pos["liq_rank"].isna(), "net"].sum()
        for yr in YEARS:
            row[f"net_pnl_{yr}"] = pos.loc[pos["period"].dt.year == yr, "net"].sum()
        top5 = by_sym.head(5)
        row["top5_names"] = " ".join(top5.index)
        row["top5_net_pnl"] = top5.sum()
        row["top5_share_of_total_pct"] = 100 * top5.sum() / total if total else None
        rows.append(row)
        for s, v in top5.items():
            tops.append({"method": k, "symbol": s, "net_pnl": v, "dropout": s in dropouts})
        for s, v in by_sym.tail(5).items():
            tops.append(
                {"method": k, "symbol": s, "net_pnl": v, "dropout": s in dropouts, "bottom5": True}
            )
    write(pd.DataFrame(rows).set_index("method").T, "G4_subsets.csv")
    write(pd.DataFrame(tops), "G4b_top_bottom_names.csv", index=False)

    # ---- C3: test-path IC and top-10 sign accuracy (daily methods only)
    lab = panel[["period", "symbol", "y", "open_px"]]
    ic_rows, scored_by = [], {}
    for k in [*ML, "Daily momentum", "Daily reversal"]:
        r = runs[k]
        s = preds_of(r).merge(lab, on=["period", "symbol"], how="left")
        s = s[np.isfinite(s["y"]) & np.isfinite(s["prediction"])]
        scored_by[k] = s
        summ = ic_summary(period_ic(s))
        hits, n = 0, 0
        for _, g in s.groupby("period"):
            g = g[g["prediction"] != 0].sort_values("symbol", kind="stable")
            pos_ = select(g["prediction"].to_numpy(float), g["symbol"].to_numpy(), 10, "BOTH")
            gg = g.iloc[pos_]
            gg = gg[gg["y"] != 0]
            hits += int((np.sign(gg["prediction"]) == np.sign(gg["y"])).sum())
            n += len(gg)
        ic_rows.append(
            {
                "method": k,
                **summ,
                "top10_sign_accuracy_pct": 100 * hits / n,
                "top10_name_days": n,
                "rows_scored": len(s),
            }
        )
    write(pd.DataFrame(ic_rows), "C3_test_ic_and_sign_accuracy.csv", index=False)

    # ---- G5: prediction scale vs the cost gate, replayed on the test path
    g5 = []
    for k in [*ML, "Daily momentum", "Daily reversal"]:
        r, s = runs[k], scored_by[k]
        cfg = r["cfg"]
        s = s[s["open_px"] > 0]
        by_p = dict(tuple(s.groupby("period")))
        sel_n, pass_n, gates, ratios = [], [], [], []
        for b in r["book"].itertuples(index=False):
            g = by_p.get(b.period)
            if g is None or b.equity_before <= 0:
                continue
            g = g.sort_values("symbol", kind="stable")
            pred, px = g["prediction"].to_numpy(float), g["open_px"].to_numpy(float)
            metric = decision_metric(pred, True, cfg.trader, cfg.costs)
            p_ = select(metric, g["symbol"].to_numpy(), cfg.trader.number_of_symbols_to_buy, "BOTH")
            sel_n.append(len(p_))
            if not len(p_):
                pass_n.append(0)
                continue
            w = apply_caps(
                np.abs(metric[p_]) / np.abs(metric[p_]).sum(),
                cfg.trader.max_symbol_weight,
                cfg.trader.gross_exposure_cap,
            )
            gate = cost_gate(
                w,
                px[p_],
                trader=cfg.trader,
                costs=cfg.costs,
                capital=b.equity_before if cfg.reinvest else cfg.capital,
            )
            pass_n.append(int((np.abs(pred[p_]) > gate).sum()))
            gates.extend(gate)
            ratios.extend(np.abs(pred[p_]) / gate)
        a = np.abs(s["prediction"].to_numpy(float))
        g5.append(
            {
                "method": k,
                "abs_pred_median": np.median(a),
                "abs_pred_p90": np.quantile(a, 0.9),
                "abs_pred_p99": np.quantile(a, 0.99),
                "top10_abs_pred_median": np.nan,
                "gate_cost_median": np.median(gates),
                "gate_cost_p10": np.quantile(gates, 0.1),
                "gate_cost_p90": np.quantile(gates, 0.9),
                "selected_per_session": np.mean(sel_n),
                "passed_per_session": np.mean(pass_n),
                "share_top10_passed_pct": 100 * np.sum(pass_n) / np.sum(sel_n),
                "ratio_pred_to_gate_median": np.median(ratios),
                "ratio_pred_to_gate_p10": np.quantile(ratios, 0.1),
            }
        )
    write(
        pd.DataFrame(g5).drop(columns="top10_abs_pred_median"),
        "G5_prediction_scale_vs_cost.csv",
        index=False,
    )

    # ---- C4: validation vs test, per fold
    c4, c4s = [], []
    for k in ML:
        r = runs[k]
        mon_k = monthly(r)
        if (r["dir"] / "choice.json").exists():
            ch = json.loads((r["dir"] / "choice.json").read_text())
            val = [max(v for kk, v in c.items() if kk.startswith("val_sharpe")) for c in ch]
        else:
            val = [f["candidate_val_net_sharpe"][f["chosen"]] for f in r["folds"]]
        for f, v in zip(r["folds"], val, strict=True):
            p = pd.Period(f["test"][0], "M")
            c4.append(
                {
                    "model": k,
                    "fold": f["name"],
                    "val_net_sharpe": v,
                    "test_month_net_return": mon_k.get(p),
                }
            )
        sub = pd.DataFrame(c4)
        sub = sub[sub["model"] == k]
        c4s.append(
            {
                "model": k,
                "pearson": sub["val_net_sharpe"].corr(sub["test_month_net_return"]),
                "spearman": sub["val_net_sharpe"].corr(
                    sub["test_month_net_return"], method="spearman"
                ),
                "mean_val_sharpe": sub["val_net_sharpe"].mean(),
                "median_val_sharpe": sub["val_net_sharpe"].median(),
                "n_folds": len(sub),
            }
        )
    write(pd.DataFrame(c4), "C4a_val_vs_test_per_fold.csv", index=False)
    write(pd.DataFrame(c4s), "C4b_val_vs_test_correlation.csv", index=False)

    # ---- F2: configuration choices
    f2, sw = [], []
    for k in [
        "Three-feature linear (per-fold grid)",
        "linear, all 188 features (per-fold grid)",
        "nonlinear LightGBM, all 189 features (per-fold grid)",
    ]:
        r = runs[k]
        keys = [g.removeprefix("model.") for g in r["run"]["grid"]]
        chosen = [tuple((kk, f["model_params"][kk]) for kk in keys) for f in r["folds"]]
        vc = pd.Series([str(dict(c)) for c in chosen]).value_counts()
        for cfg_, n in vc.items():
            f2.append({"model": k, "config": cfg_, "folds_chosen": int(n)})
        changes = sum(a != b for a, b in zip(chosen, chosen[1:], strict=False))
        sw.append(
            {
                "model": k,
                "n_candidates": int(np.prod([len(v) for v in r["run"]["grid"].values()])),
                "distinct_configs_chosen": len(vc),
                "month_to_month_changes": changes,
                "of_transitions": len(chosen) - 1,
            }
        )
    write(pd.DataFrame(f2), "F2a_config_choice_counts.csv", index=False)
    write(pd.DataFrame(sw), "F2b_config_switching.csv", index=False)
    sel = []
    for k in ["Validation-selected: linear vs nonlinear grids", "Validation-selected: v29 vs P3b"]:
        ch = pd.DataFrame(json.loads((runs[k]["dir"] / "choice.json").read_text()))
        ch["year"] = ch["fold"].str[1:5].astype(int)
        t = ch.groupby(["year", "chosen"]).size().unstack(fill_value=0)
        t["model"] = k
        sel.append(t.reset_index())
    write(pd.concat(sel), "F2c_selected_model_family_by_year.csv", index=False)

    # ---- G6: interpretability
    import lightgbm as lgb

    gain_rows = []
    for k, sub in [
        ("nonlinear LightGBM, all 189 features (per-fold grid)", "model/model.txt"),
        ("nonlinear P3b hybrid (frozen)", "model/model.txt"),
    ]:
        r = runs[k]
        shares = []
        for f in r["run"]["folds"]:
            b = lgb.Booster(model_file=str(r["dir"] / "folds" / f / sub))
            g = pd.Series(b.feature_importance("gain"), index=b.feature_name(), dtype=float)
            shares.append(g / g.sum() if g.sum() > 0 else g)
        m = pd.concat(shares, axis=1).fillna(0.0)
        avg = m.mean(axis=1).sort_values(ascending=False)
        for i, (feat, v) in enumerate(avg.head(15).items(), 1):
            gain_rows.append(
                {
                    "model": k,
                    "rank": i,
                    "feature": feat,
                    "mean_gain_share_pct": 100 * v,
                    "folds_in_top10": int(
                        (m.rank(ascending=False)[m.columns].loc[feat] <= 10).sum()
                    ),
                }
            )
    write(pd.DataFrame(gain_rows), "G6a_lightgbm_gain_importance.csv", index=False)
    coef_rows = []
    for k, sub in [
        ("linear v29 (frozen)", "model/model.npz"),
        ("linear, all 188 features (per-fold grid)", "model/model.npz"),
        ("Three-feature linear (per-fold grid)", "model/model.npz"),
        ("Three-feature linear regression (alpha_ratio 1e-6)", "model/model.npz"),
        ("nonlinear P3b hybrid (frozen)", "model/init/model.npz"),
    ]:
        r = runs[k]
        cs = []
        for f in r["run"]["folds"]:
            with np.load(r["dir"] / "folds" / f / sub) as z:
                cs.append(pd.Series(z["coef"], index=[str(n) for n in z["names"]]))
        m = pd.concat(cs, axis=1).fillna(0.0)
        mean = m.mean(axis=1)
        sign_holds = (np.sign(m).T == np.sign(mean)).T.mean(axis=1)
        nonzero = (m != 0).mean(axis=1)
        order = mean.abs().sort_values(ascending=False).index
        for i, feat in enumerate(order[:20], 1):
            coef_rows.append(
                {
                    "model": k,
                    "rank": i,
                    "feature": feat,
                    "mean_std_coef": mean[feat],
                    "sign_holds_share_pct": 100 * sign_holds[feat],
                    "nonzero_share_pct": 100 * nonzero[feat],
                }
            )
        coef_rows.append(
            {
                "model": k,
                "rank": None,
                "feature": "(features with nonzero mean coef)",
                "mean_std_coef": float((mean != 0).sum()),
                "sign_holds_share_pct": None,
                "nonzero_share_pct": float(len(mean)),
            }
        )
    write(pd.DataFrame(coef_rows), "G6b_linear_coefficients.csv", index=False)

    # ---- G7: monthly block bootstrap of Sharpe differences
    rng = np.random.default_rng(0)
    pairs = [
        ("Validation-selected: v29 vs P3b", "Buy-and-hold VOO"),
        ("Validation-selected: v29 vs P3b", best_rule),
        ("Validation-selected: v29 vs P3b", "Buy-and-hold QQQ"),
        ("linear v29 (frozen)", "Buy-and-hold QQQ"),
        ("nonlinear P3b hybrid (frozen)", "Buy-and-hold QQQ"),
    ]
    g7 = []
    for a, b in pairs:
        da = runs[a]["daily"].set_index("session")["ret"]
        db = runs[b]["daily"].set_index("session")["ret"]
        j = pd.concat([da.rename("a"), db.rename("b")], axis=1, join="inner")
        months = j.index.to_period("M")
        blocks = [j[months == p].to_numpy() for p in months.unique()]
        diffs = []
        for _ in range(10000):
            pick = rng.integers(0, len(blocks), len(blocks))
            x = np.concatenate([blocks[i] for i in pick])
            diffs.append(sharpe(x[:, 0]) - sharpe(x[:, 1]))
        g7.append(
            {
                "model": a,
                "vs": b,
                "sharpe_diff": sharpe(j["a"]) - sharpe(j["b"]),
                "ci95_lo": np.quantile(diffs, 0.025),
                "ci95_hi": np.quantile(diffs, 0.975),
                "share_boot_diff_gt0_pct": 100 * float(np.mean(np.asarray(diffs) > 0)),
                "n_days": len(j),
                "n_months": len(blocks),
                "reps": 10000,
            }
        )
    write(pd.DataFrame(g7), "G7_block_bootstrap_sharpe_diff.csv", index=False)

    # ---- D6: training rows per fold (deploy fits; pool = the 550 stocks)
    pool = [s for s in syms if s not in ("VOO", "QQQ")]
    pp = panel[panel["symbol"].isin(pool)]
    sigma = target_volatility(panel, "vol", 63)
    finite_sig = pd.Series(np.isfinite(sigma), index=panel.index)[pp.index]
    sessions = sorted(panel["session"].unique())
    d6 = []
    for f in runs["linear v29 (frozen)"]["folds"]:
        lo = pd.Timestamp(f["deploy"][0])
        hi = pd.Timestamp(f["fit_end"])
        w = (pp["session"] >= lo) & (pp["session"] <= hi) & np.isfinite(pp["y"])
        vlo = pd.Timestamp(f["train"][0])
        vhi = pd.Timestamp(embargoed_end(f["train"][1], sessions, 5))
        vw = (pp["session"] >= vlo) & (pp["session"] <= vhi) & np.isfinite(pp["y"])
        d6.append(
            {
                "fold": f["name"],
                "deploy_fit_rows_raw_label": int(w.sum()),
                "deploy_fit_rows_vol_label": int((w & finite_sig).sum()),
                "deploy_fit_symbols": int(pp.loc[w, "symbol"].nunique()),
                "val_fit_rows_raw_label": int(vw.sum()),
            }
        )
    write(pd.DataFrame(d6), "D6_training_rows_per_fold.csv", index=False)

    # ---- G8: daily equity at each session's open
    eq = pd.DataFrame(
        {k: r["daily"].set_index("session")["equity_before"] for k, r in runs.items()}
    )
    eq.index.name = "session"
    eq.to_csv(OUT / "daily_equity.csv", float_format="%.4f")
    print("wrote daily_equity.csv", eq.shape)

    # ---- universes: are they the same across methods?
    drop_rows = []
    for k in ["linear v29 (frozen)", "Daily momentum", "Weekly momentum"]:
        r = runs[k]
        counts = {}
        for f in r["run"]["folds"]:
            for s_ in json.loads((r["dir"] / "folds" / f / "universe.json").read_text()):
                if s_ in dropouts:
                    counts[s_] = counts.get(s_, 0) + 1
        drop_rows.append(
            {
                "universe_of": k,
                "name_folds": sum(counts.values()),
                "per_fold": sum(counts.values()) / len(r["run"]["folds"]),
                "names": " ".join(
                    f"{s_}:{n}" for s_, n in sorted(counts.items(), key=lambda kv: -kv[1])
                ),
            }
        )
    write(pd.DataFrame(drop_rows), "A7c_dropouts_in_universe_by_method.csv", index=False)
    a7 = []
    for f in runs["linear v29 (frozen)"]["run"]["folds"]:
        uni = json.loads(
            (runs["linear v29 (frozen)"]["dir"] / "folds" / f / "universe.json").read_text()
        )
        dn = [s_ for s_ in uni if s_ in dropouts]
        a7.append(
            {
                "fold": f,
                "n_universe": len(uni),
                "n_dropouts": len(dn),
                "dropout_names": ";".join(dn),
            }
        )
    write(pd.DataFrame(a7), "A7_dropouts_in_universe.csv", index=False)
    uv = {}
    for k in ["linear v29 (frozen)", "Daily momentum", "Weekly momentum"]:
        r = runs[k]
        uv[k] = {
            f: json.loads((r["dir"] / "folds" / f / "universe.json").read_text())
            for f in r["run"]["folds"]
        }
    rows = []
    for f in uv["linear v29 (frozen)"]:
        a, b, c = (set(uv[k][f]) for k in uv)
        rows.append(
            {
                "fold": f,
                "v29_vs_daily_mom_differ": len(a ^ b),
                "v29_vs_weekly_mom_differ": len(a ^ c),
                "etfs_in_daily_mom": " ".join(sorted(b & {"VOO", "QQQ"})),
                "etfs_in_weekly_mom": " ".join(sorted(c & {"VOO", "QQQ"})),
            }
        )
    write(pd.DataFrame(rows), "A7b_universe_differences_by_method.csv", index=False)


if __name__ == "__main__":
    main()
