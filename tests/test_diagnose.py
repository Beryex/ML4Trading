import json

import numpy as np
import pandas as pd
import pytest
from conftest import write_dataset

from ml4trading.diagnose import diagnose
from ml4trading.train import train

SHORT_FOLDS = {
    "folds.train_months": 3,
    "folds.val_months": 1,
    "folds.path_start": "2021-06-01",
    "folds.path_end": "2021-08-31",
}
DIAGNOSTIC_KEYS = [
    "net_sharpe",
    "gross_sharpe",
    "alpha",
    "beta",
    "alpha_t",
    "ic_mean",
    "ic_std",
    "ic_t",
    "ic_positive_share",
    "ic_periods",
    "long_gross",
    "long_net",
    "short_gross",
    "short_net",
    "same_sign_share",
    "mean_net_exposure",
    "mean_gross_exposure",
    "holding_share",
    "gate_selected_mean",
    "gate_pass_mean",
    "gate_ratio_q10",
    "gate_ratio_q50",
    "gate_ratio_q90",
    "pred_xs_std",
    "label_xs_std",
    "r2_mse",
]
EVERY_FILE = [
    "folds.csv",
    "by_test_year.csv",
    "importance.csv",
    "importance_stability.csv",
    "summary.md",
]


@pytest.fixture(scope="module")
def learned_data(tmp_path_factory):
    return write_dataset(tmp_path_factory.mktemp("diagnose") / "data")


def _read(path):
    return pd.read_csv(path)


def _exact(path, column):
    """One CSV column parsed by Python's float(), which is exact (the table reader may round the
    last digit); an empty cell is NaN."""
    cells = pd.read_csv(path, dtype=str, usecols=[column])[column]
    return np.array([np.nan if pd.isna(c) else float(c) for c in cells])


def _fold_meta(run, name):
    return json.loads((run / "folds" / name / "fold.json").read_text())


def test_diagnose_aggregates_folds(learned_data, tmp_path):
    run = train("nonlinear", tmp_path / "run", learned_data, overrides=SHORT_FOLDS, importance=True)
    names = json.loads((run / "run.json").read_text())["folds"]

    out = diagnose(run)

    assert out == run / "diagnose"
    for file in EVERY_FILE:
        assert (out / file).exists(), file
    folds = _read(out / "folds.csv")
    metas = [_fold_meta(run, n) for n in names]
    assert folds["fold"].tolist() == names and len(folds) == 3
    chosen_scores = [m["candidate_val_net_sharpe"][m["chosen"]] for m in metas]
    assert _exact(out / "folds.csv", "val_net_sharpe").tolist() == chosen_scores
    assert folds["test_year"].tolist() == [2021, 2021, 2021]
    # every val_diagnostics value lands in its column as the file has it (None as an empty cell)
    assert list(folds.columns) == [
        "fold",
        "test_year",
        "val_net_sharpe",
        *metas[0]["val_diagnostics"],
    ]
    for key in metas[0]["val_diagnostics"]:
        wanted = [
            np.nan if m["val_diagnostics"][key] is None else m["val_diagnostics"][key]
            for m in metas
        ]
        np.testing.assert_array_equal(_exact(out / "folds.csv", key), np.array(wanted, dtype=float))

    importance = _read(out / "importance.csv")
    n_features = len(pd.read_parquet(run / "folds" / names[0] / "val" / "importance.parquet"))
    assert len(importance) == n_features
    assert importance["shap_mean_rank"].between(1, n_features).all()
    stability = _read(out / "importance_stability.csv")
    assert stability["measure"].tolist() == ["gain", "split", "shap", "perm_mse", "perm_ic"]
    assert (stability["n_pairs"] == 2).all()
    summary = (out / "summary.md").read_text()
    assert "`n_folds`: 3" in summary and "`shap_mean_rank`" in summary


def test_diagnose_without_importance(learned_data, tmp_path):
    run = train("investment", tmp_path / "run", learned_data, overrides=SHORT_FOLDS)

    out = diagnose(run)

    assert {p.name for p in out.iterdir()} == {"folds.csv", "by_test_year.csv", "summary.md"}
    folds = _read(out / "folds.csv")
    assert len(folds) == 3
    # the constant forecast has no defined IC: an undefined value is an empty cell, not a zero
    assert folds["ic_mean"].isna().all()
    assert "importance" not in (out / "summary.md").read_text()


# -- aggregation, pinned on hand-written runs ---------------------------------------------------


def _write_run(root, folds, method="nonlinear"):
    """A run directory of hand-written folds. A fold is a dict with ``name``, ``test_start``,
    ``scores``, ``chosen``, ``diagnostics`` (the keys it sets; the rest are None) and an optional
    ``importance`` frame and ``hpo`` block."""
    (root / "folds").mkdir(parents=True)
    (root / "run.json").write_text(
        json.dumps({"method": method, "folds": [f["name"] for f in folds]})
    )
    for fold in folds:
        fdir = root / "folds" / fold["name"]
        (fdir / "val").mkdir(parents=True)
        diagnostics = {k: fold["diagnostics"].get(k) for k in DIAGNOSTIC_KEYS}
        meta = {
            "name": fold["name"],
            "test": [fold["test_start"], fold["test_start"]],
            "candidate_val_net_sharpe": fold["scores"],
            "chosen": fold["chosen"],
            "val_diagnostics": diagnostics,
        }
        if "hpo" in fold:
            meta["hpo"] = fold["hpo"]
        (fdir / "fold.json").write_text(json.dumps(meta))
        if fold.get("importance") is not None:
            fold["importance"].to_parquet(fdir / "val" / "importance.parquet", index=False)
    return root


def _fold(name, test_start, sharpe, **diagnostics):
    return {
        "name": name,
        "test_start": test_start,
        "scores": [-9.0, sharpe, 9.0],
        "chosen": 1,
        "diagnostics": diagnostics,
    }


def test_headline_numbers_follow_the_files(tmp_path):
    run = _write_run(
        tmp_path / "run",
        [
            _fold("F2021-12", "2021-12-01", 1.0, ic_mean=0.02, beta=0.5),
            _fold("F2022-01", "2022-01-03", 2.0, ic_mean=None, beta=0.7),
            _fold("F2022-02", "2022-02-01", 6.0, ic_mean=0.04, beta=None),
            _fold("F2022-03", "2022-03-01", 3.0, ic_mean=0.06, beta=0.9),
        ],
    )

    out = diagnose(run)

    folds = _read(out / "folds.csv")
    assert folds["test_year"].tolist() == [2021, 2022, 2022, 2022]
    assert folds["val_net_sharpe"].tolist() == [1.0, 2.0, 6.0, 3.0]  # candidate[chosen]
    by_year = _read(out / "by_test_year.csv")
    assert by_year["test_year"].tolist() == [2021, 2022]
    assert by_year["n_folds"].tolist() == [1, 3]
    assert by_year["val_net_sharpe"].tolist() == [1.0, pytest.approx(11.0 / 3.0)]
    # the undefined one is skipped
    assert by_year["ic_mean"].tolist() == [pytest.approx(0.02), pytest.approx(0.05)]
    assert by_year["beta"].tolist() == [0.5, pytest.approx(0.8)]
    assert by_year["r2_mse"].isna().all()  # undefined in every fold
    summary = (out / "summary.md").read_text()
    assert "- mean `val_net_sharpe`: 3" in summary  # (1 + 2 + 6 + 3) / 4 = 3
    assert "- median `val_net_sharpe`: 2.5" in summary
    assert "- `n_folds`: 4" in summary
    assert "| `ic_mean` | 0.04 | 3 |" in summary
    assert "| `beta` | 0.7 | 3 |" in summary
    assert "| `r2_mse` | n/a | 0 |" in summary


def test_an_hpo_runs_trial_counts_and_winning_rounds_are_columns(tmp_path):
    def block(rounds, best):
        trials = [{"number": i, "best_iteration": n} for i, n in enumerate(rounds)]
        return {"n_trials": len(rounds), "best_number": best, "trials": trials}

    searched = _write_run(
        tmp_path / "searched",
        [
            {**_fold("F2021-12", "2021-12-01", 1.0), "hpo": block([40, 25, 70], 1)},
            {**_fold("F2022-01", "2022-01-03", 2.0), "hpo": block([55], 0)},
            # a fold with nothing to fit is not searched: no winner, no round
            {**_fold("F2022-02", "2022-02-01", 0.0), "hpo": block([], None)},
        ],
    )

    folds = _read(diagnose(searched) / "folds.csv")

    assert list(folds.columns[-2:]) == ["hpo_n_trials", "hpo_best_iteration"]
    assert folds["hpo_n_trials"].tolist() == [3, 1, 0]
    assert folds["hpo_best_iteration"].iloc[:2].tolist() == [25, 55]  # trials[best_number]
    assert np.isnan(folds["hpo_best_iteration"].iloc[2])

    plain = _write_run(tmp_path / "plain", [_fold("F2021-12", "2021-12-01", 1.0)])
    assert not [c for c in _read(diagnose(plain) / "folds.csv").columns if "hpo" in c]


def test_an_empty_universe_fold_keeps_its_recorded_score(tmp_path):
    # train records 0.0 as the candidate score and None as val_diagnostics.net_sharpe
    run = _write_run(
        tmp_path / "run",
        [
            _fold("F2021-12", "2021-12-01", 0.0, net_sharpe=None),
            _fold("F2022-01", "2022-01-03", 2.0, net_sharpe=2.0),
        ],
    )

    folds = _read(diagnose(run) / "folds.csv")

    assert folds["val_net_sharpe"].tolist() == [0.0, 2.0]
    assert folds["net_sharpe"].isna().tolist() == [True, False]


def test_a_run_without_val_diagnostics_is_refused(tmp_path):
    run = _write_run(tmp_path / "run", [_fold("F2021-12", "2021-12-01", 1.0)])
    path = run / "folds" / "F2021-12" / "fold.json"
    meta = json.loads(path.read_text())
    path.write_text(json.dumps({k: v for k, v in meta.items() if k != "val_diagnostics"}))

    with pytest.raises(ValueError, match="F2021-12.*val_diagnostics"):
        diagnose(run)


FEATURES = [f"f{i:02d}" for i in range(22)]  # two more than the 20 the top sets keep


def _importance(perm_mse, shap=None):
    table = pd.DataFrame({"feature": FEATURES, "perm_mse": np.asarray(perm_mse, dtype=float)})
    if shap is not None:
        table.insert(1, "shap", np.asarray(shap, dtype=float))
    return table


def _importance_run(root, tables, method="nonlinear"):
    folds = [
        {**_fold(f"F2022-{i + 1:02d}", f"2022-{i + 1:02d}-01", 1.0), "importance": table}
        for i, table in enumerate(tables)
    ]
    return _write_run(root, folds, method)


def test_importance_ranks_means_and_top_shares(tmp_path):
    rising = np.arange(22.0) - 5.0  # f21 largest; f00..f04 negative, below zero
    run = _importance_run(
        tmp_path / "run",
        [_importance(rising), _importance(rising), _importance(-rising)],
    )

    out = diagnose(run)

    table = _read(out / "importance.csv").set_index("feature")
    assert list(table.columns) == ["perm_mse_mean", "perm_mse_mean_rank", "perm_mse_top20_share"]
    assert table.index.tolist() == FEATURES
    assert table.loc["f21", "perm_mse_mean"] == pytest.approx(16.0 / 3)  # 16, 16, -16
    # rank 1 is the largest signed value: f21 is first in two folds and last in the third
    assert table.loc["f21", "perm_mse_mean_rank"] == pytest.approx((1 + 1 + 22) / 3)
    assert table.loc["f00", "perm_mse_mean_rank"] == pytest.approx((22 + 22 + 1) / 3)
    # a fold's top 20 of `rising` is f02..f21, of `-rising` it is f00..f19
    assert table.loc["f21", "perm_mse_top20_share"] == pytest.approx(2 / 3)
    assert table.loc["f00", "perm_mse_top20_share"] == pytest.approx(1 / 3)
    assert table.loc["f10", "perm_mse_top20_share"] == 1.0
    stability = _read(out / "importance_stability.csv")
    assert stability["measure"].tolist() == ["perm_mse"]
    assert stability["n_pairs"].tolist() == [2]
    # identical vectors, then reversed ones: rho = 1 then -1; overlap 20/20 then 18/20
    assert stability["mean_spearman"].tolist() == [pytest.approx(0.0)]
    assert stability["mean_top20_overlap"].tolist() == [pytest.approx((1.0 + 0.9) / 2)]
    summary = (out / "summary.md").read_text()
    assert "## Top 20 features by `perm_mse_mean_rank`" in summary
    assert "`shap_mean_rank`" not in summary  # a measure the tables lack is absent
    listed = [line for line in summary.splitlines() if line.startswith("| `f")]
    assert len(listed) == 20


def test_tied_values_share_their_average_rank(tmp_path):
    shap = np.zeros(22)
    shap[0] = 1.0  # f00 first; the other 21 tie for ranks 2..22
    run = _importance_run(tmp_path / "run", [_importance(np.arange(22.0), shap)])

    table = _read(diagnose(run) / "importance.csv").set_index("feature")

    assert table.loc["f00", "shap_mean_rank"] == 1.0
    assert table.loc["f05", "shap_mean_rank"] == pytest.approx(12.0)  # mean of 2..22
    assert table["shap_mean_rank"].between(1, 22).all()


def test_a_fold_without_importance_is_skipped_and_breaks_the_chain(tmp_path):
    rising = np.arange(22.0)
    run = _importance_run(
        tmp_path / "run", [_importance(rising), None, _importance(rising), _importance(-rising)]
    )

    out = diagnose(run)

    table = _read(out / "importance.csv").set_index("feature")
    assert table.loc["f21", "perm_mse_mean"] == pytest.approx(21.0 / 3)  # 21, 21, -21
    assert table.loc["f21", "perm_mse_top20_share"] == pytest.approx(2 / 3)  # of the 3 with tables
    stability = _read(out / "importance_stability.csv")
    # F2022-01 and F2022-03 are not consecutive; only F2022-03 -> F2022-04 forms a pair
    assert stability["n_pairs"].tolist() == [1]
    assert stability["mean_spearman"].tolist() == [pytest.approx(-1.0)]
    assert len(_read(out / "folds.csv")) == 4


def test_an_undefined_measure_takes_no_part_in_the_tops(tmp_path):
    # perm_ic is NaN in the second fold (no IC defined there)
    tables = []
    for ic in [np.arange(22.0), np.full(22, np.nan), np.arange(22.0)]:
        table = _importance(np.arange(22.0))
        table["perm_ic"] = ic
        tables.append(table)
    run = _importance_run(tmp_path / "run", tables)

    out = diagnose(run)

    table = _read(out / "importance.csv").set_index("feature")
    assert table.loc["f21", "perm_ic_mean"] == 21.0  # the NaN fold is not a zero
    assert table.loc["f21", "perm_ic_mean_rank"] == 1.0
    assert table.loc["f21", "perm_ic_top20_share"] == 1.0  # 2 of the 2 folds that define it
    stability = _read(out / "importance_stability.csv").set_index("measure")
    assert stability.loc["perm_mse", "n_pairs"] == 2
    assert stability.loc["perm_ic", "n_pairs"] == 0  # no two adjacent folds define it
    assert pd.isna(stability.loc["perm_ic", "mean_spearman"])


def test_a_rerun_removes_importance_files_the_run_no_longer_has(tmp_path):
    rising = np.arange(22.0)
    run = _importance_run(tmp_path / "run", [_importance(rising), _importance(rising)])
    out = diagnose(run)
    assert (out / "importance.csv").exists() and (out / "importance_stability.csv").exists()

    for path in (run / "folds").glob("*/val/importance.parquet"):
        path.unlink()
    out = diagnose(run)

    assert {p.name for p in out.iterdir()} == {"folds.csv", "by_test_year.csv", "summary.md"}
