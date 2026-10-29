"""Aggregate a run's per-fold validation outputs into one report.

    python -m ml4trading.diagnose --run runs/dev/v00-defaults

One fold's validation book is one draw: its Sharpe moves with the month, and a feature's
importance with the window. ``train`` keeps, for every fold, the chosen candidate's validation
breakdown (``fold.json``'s ``val_diagnostics``) and, with ``--importance``, its per-feature
importance (``val/importance.parquet``); this module reads them all and reports their average,
their spread over the years of the path and how stable the importance is from one fold to the
next, so a change to a model is judged on every fold rather than on one. It reads only the run
directory -- no dataset -- and recomputes nothing: every number is a file's value or a plain
statistic of them.

Writes ``<run>/diagnose/``:

* ``folds.csv``: one row per fold -- ``fold``, ``test_year`` (the year its test month falls in),
  ``val_net_sharpe`` (the chosen candidate's score, ``candidate_val_net_sharpe[chosen]``; 0.0 for a
  fold whose universe is empty, which has no ``val_diagnostics.net_sharpe``) -- then every
  ``val_diagnostics`` key. For a run with a search (``fold.json`` has an ``hpo`` block), two more
  columns follow: ``hpo_n_trials`` (the trials the fold's search completed) and
  ``hpo_best_iteration`` (the round the winning trial kept, ``trials[best_number]``; empty for a
  fold that was not searched). A run without a search has neither column.
* ``by_test_year.csv``: the mean of each of those columns over the folds of a test year, and
  ``n_folds``.
* ``importance.csv`` and ``importance_stability.csv``, when any fold wrote an importance table
  (see below).
* ``summary.md``: the development score (the mean and median ``val_net_sharpe``), the means of the
  headline diagnostics and, with importance, the features at the top of the ranking.

Every mean skips a value that is undefined (None in ``fold.json``, NaN in a table), so a fold
without an IC does not pull ``ic_mean`` toward zero; ``summary.md`` says how many folds each mean
covers.

Importance. A fold without ``val/importance.parquet`` (the run did not ask, the model fits no
estimator, or the fold scored no rows) is left out. For each measure of the tables
(``gain, split, shap, perm_mse, perm_ic`` for the nonlinear model, the last two for the linear
one) and each feature:

* ``<m>_mean``: its mean value over the folds;
* ``<m>_mean_rank``: its mean rank over the folds, where a fold ranks its features on the signed
  value, rank 1 the largest (``perm_mse`` and ``perm_ic`` can be negative; a negative value ranks
  below zero), tied features sharing their average rank;
* ``<m>_top20_share``: the share of the folds in which it is among that fold's 20 largest values,
  a tie at the cut going to the feature name that sorts first.

``importance_stability.csv`` has, per measure, the mean Spearman correlation and the mean top-20
overlap (the size of the intersection over 20) between the importance vectors of two consecutive
folds. Only folds adjacent in the run are paired -- a fold left out breaks the chain rather than
being bridged -- and ``n_pairs`` counts the pairs used. A fold where a measure is undefined for
every feature takes no part in that measure's shares or pairs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ml4trading.diagnostics import _spearman

#: How many top-ranked features the share, the overlap and the summary look at.
TOP_N = 20

#: The ``val_diagnostics`` means ``summary.md`` reports next to the development score.
SUMMARY_METRICS = [
    "ic_mean",
    "gross_sharpe",
    "beta",
    "long_net",
    "short_net",
    "same_sign_share",
    "gate_pass_mean",
    "r2_mse",
]

#: The importance rankings ``summary.md`` lists, as (measure, heading).
SUMMARY_RANKINGS = [("shap", "SHAP"), ("perm_mse", "permutation MSE")]


def _hpo_columns(hpo: dict | None) -> dict:
    """A fold's search as columns: its trial count and the round its winner kept (None for a
    fold that was not searched, which has no winner). Nothing for a fold without a search."""
    if hpo is None:
        return {}
    best = hpo["best_number"]
    rounds = {t["number"]: t["best_iteration"] for t in hpo["trials"]}
    return {
        "hpo_n_trials": hpo["n_trials"],
        "hpo_best_iteration": None if best is None else rounds[best],
    }


def _read_folds(run_dir: Path, names: list[str]) -> pd.DataFrame:
    """One row per fold, in run order: ``fold, test_year, val_net_sharpe`` then every
    ``val_diagnostics`` key and, for a fold with an ``hpo`` block, ``hpo_n_trials`` and
    ``hpo_best_iteration``; all numeric (a None becomes NaN)."""
    rows = []
    for name in names:
        meta = json.loads((run_dir / "folds" / name / "fold.json").read_text())
        if meta.get("val_diagnostics") is None:
            raise ValueError(f"{name}: fold.json has no val_diagnostics; train the run again")
        rows.append(
            {
                "fold": name,
                "test_year": pd.Timestamp(meta["test"][0]).year,
                "val_net_sharpe": float(meta["candidate_val_net_sharpe"][meta["chosen"]]),
                **meta["val_diagnostics"],
                **_hpo_columns(meta.get("hpo")),
            }
        )
    folds = pd.DataFrame(rows)
    for column in folds.columns[3:]:  # a key that is None in every fold would stay object
        folds[column] = pd.to_numeric(folds[column])
    return folds


def _by_test_year(folds: pd.DataFrame) -> pd.DataFrame:
    grouped = folds.drop(columns="fold").groupby("test_year")
    table = grouped.mean()
    table["n_folds"] = grouped.size()
    return table.reset_index()


def _read_importance(run_dir: Path, names: list[str]) -> dict[str, pd.DataFrame]:
    """Each fold's importance table indexed by feature, for the folds that wrote one."""
    tables = {}
    for name in names:
        path = run_dir / "folds" / name / "val" / "importance.parquet"
        if path.exists():
            tables[name] = pd.read_parquet(path).set_index("feature")
    return tables


def _top(values: pd.Series) -> set[str]:
    """The ``TOP_N`` features with the largest finite values (ties by feature name)."""
    finite = values.dropna().rename("value").rename_axis("feature").reset_index()
    ordered = finite.sort_values(["value", "feature"], ascending=[False, True])
    return set(ordered["feature"].iloc[:TOP_N])


def _importance_summary(
    tables: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, dict[str, dict[str, set[str]]]]:
    """The per-feature table of ``importance.csv`` (``feature`` then, per measure, its mean, mean
    rank and top share) and, per measure, each fold's top set (only the folds where the measure
    has a finite value)."""
    stacked = pd.concat(tables, names=["fold", "feature"])
    means = stacked.groupby(level="feature").mean()
    ranks = stacked.groupby(level="fold").rank(ascending=False).groupby(level="feature").mean()
    tops = {
        measure: {
            fold: _top(t[measure])
            for fold, t in tables.items()
            if measure in t and t[measure].notna().any()
        }
        for measure in stacked.columns
    }
    columns = {}
    for measure in stacked.columns:
        sets = list(tops[measure].values())
        counts = pd.Series([f for s in sets for f in s], dtype=object).value_counts()
        columns[f"{measure}_mean"] = means[measure]
        columns[f"{measure}_mean_rank"] = ranks[measure]
        columns[f"{measure}_top{TOP_N}_share"] = (
            counts.reindex(means.index, fill_value=0) / len(sets) if sets else np.nan
        )
    return pd.DataFrame(columns, index=means.index).reset_index(), tops


def _stability(
    names: list[str], tables: dict[str, pd.DataFrame], tops: dict[str, dict[str, set[str]]]
) -> pd.DataFrame:
    """Per measure, the mean Spearman correlation and top-20 overlap between the importance
    vectors of consecutive folds."""
    rows = []
    for measure, top in tops.items():
        rho, overlap = [], []
        for a, b in zip(names, names[1:], strict=False):
            if a in top and b in top:  # adjacent folds, the measure defined in both
                x, y = tables[a][measure], tables[b][measure]
                both = x.index.intersection(y.index)
                rho.append(_spearman(x[both].to_numpy(float), y[both].to_numpy(float)))
                overlap.append(len(top[a] & top[b]) / TOP_N)
        rows.append(
            {
                "measure": measure,
                "mean_spearman": pd.Series(rho, dtype=float).mean(),
                f"mean_top{TOP_N}_overlap": pd.Series(overlap, dtype=float).mean(),
                "n_pairs": len(overlap),
            }
        )
    return pd.DataFrame(rows)


def _fmt(value: float) -> str:
    return "n/a" if pd.isna(value) else f"{value:.4g}"


def _summary(run_dir: Path, method: str, folds: pd.DataFrame, importance) -> str:
    score = folds["val_net_sharpe"]
    lines = [
        f"# Validation diagnostics: {run_dir}",
        "",
        f"Method `{method}`, {len(folds)} folds ({folds['fold'].iloc[0]} to "
        f"{folds['fold'].iloc[-1]}).",
        "",
        "## Development score",
        "",
        f"- mean `val_net_sharpe`: {_fmt(score.mean())}",
        f"- median `val_net_sharpe`: {_fmt(score.median())}",
        f"- `n_folds`: {len(folds)}",
        "",
        "`val_net_sharpe` is the chosen candidate's validation net Sharpe as `train` recorded it "
        "(0.0 for a fold whose universe is empty).",
        "",
        "## Means over folds",
        "",
        "| metric | mean | folds defined |",
        "|---|---:|---:|",
    ]
    lines += [
        f"| `{m}` | {_fmt(folds[m].mean())} | {int(folds[m].notna().sum())} |"
        for m in SUMMARY_METRICS
    ]
    for measure, heading in SUMMARY_RANKINGS:
        rank = f"{measure}_mean_rank"
        # a measure no fold defines (``shap`` of linear trees) has no ranking to list
        if importance is None or rank not in importance or importance[rank].isna().all():
            continue
        top = importance.dropna(subset=[rank]).sort_values([rank, "feature"]).head(TOP_N)
        lines += [
            "",
            f"## Top {TOP_N} features by `{rank}` ({heading})",
            "",
            f"| feature | mean rank | mean {measure} | top-{TOP_N} share |",
            "|---|---:|---:|---:|",
        ]
        lines += [
            f"| `{r['feature']}` | {_fmt(r[rank])} | {_fmt(r[f'{measure}_mean'])} | "
            f"{_fmt(r[f'{measure}_top{TOP_N}_share'])} |"
            for _, r in top.iterrows()
        ]
    return "".join(f"{line}\n" for line in lines)


def diagnose(run_dir: Path) -> Path:
    """Write the report of ``run_dir`` to ``<run_dir>/diagnose/`` and return that directory."""
    run_dir = Path(run_dir)
    run = json.loads((run_dir / "run.json").read_text())
    names = run["folds"]
    if not names:
        raise ValueError(f"{run_dir}: the run has no folds")
    folds = _read_folds(run_dir, names)
    out = run_dir / "diagnose"
    out.mkdir(exist_ok=True)
    folds.to_csv(out / "folds.csv", index=False)
    _by_test_year(folds).to_csv(out / "by_test_year.csv", index=False)

    tables = _read_importance(run_dir, names)
    importance = None
    if tables:
        importance, tops = _importance_summary(tables)
        importance.to_csv(out / "importance.csv", index=False)
        _stability(names, tables, tops).to_csv(out / "importance_stability.csv", index=False)
    else:  # a report of an earlier run of this directory must not outlive its tables
        (out / "importance.csv").unlink(missing_ok=True)
        (out / "importance_stability.csv").unlink(missing_ok=True)
    (out / "summary.md").write_text(_summary(run_dir, run["method"], folds, importance))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m ml4trading.diagnose")
    ap.add_argument("--run", type=Path, required=True)
    args = ap.parse_args()
    out = diagnose(args.run)
    print((out / "summary.md").read_text(), end="")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
