import json

import pandas as pd
import pytest
from conftest import write_dataset

from ml4trading import config, hpo
from ml4trading.backtest import backtest
from ml4trading.diagnose import diagnose
from ml4trading.hpo import MAX_ROUNDS, PATIENCE, SPACE, baseline_point, search_fold
from ml4trading.models import get_model
from ml4trading.models.base import rows_in
from ml4trading.models.learned.base import EarlyStop
from ml4trading.pipeline import load_frames
from ml4trading.train import train

FIT = (pd.Timestamp("2021-03-01"), pd.Timestamp("2021-05-21"))
JUNE = (pd.Timestamp("2021-06-01"), pd.Timestamp("2021-06-30"))
POOL = ["AAA", "BBB", "CCC"]
SHORT_FOLDS = {
    "folds.train_months": 3,
    "folds.val_months": 1,
    "folds.path_start": "2021-06-01",
    "folds.path_end": "2021-08-31",
}


@pytest.fixture(scope="module")
def learned_data(tmp_path_factory):
    return write_dataset(tmp_path_factory.mktemp("hpo") / "data")


@pytest.fixture(scope="module")
def frames(learned_data):
    return load_frames(config.resolve("nonlinear"), learned_data)[1]


@pytest.fixture(scope="module")
def eval_periods(frames):
    return [pd.Timestamp(p) for p in sorted(rows_in(frames.panel, JUNE)["period"].unique())]


def _search(frames, eval_periods, *, timeout=None, n_trials=4, **overrides):
    cfg = config.resolve("nonlinear", {f"model.{k}": v for k, v in overrides.items()})
    return cfg, search_fold(
        cfg, frames, FIT, eval_periods, POOL, timeout=timeout, n_trials=n_trials
    )


def test_the_first_trial_is_the_clamped_baseline(frames, eval_periods):
    cfg, found = _search(frames, eval_periods, n_trials=2)
    # the defaults are lambda_l1 = lambda_l2 = 0.0, below the log-scale search space
    assert cfg.model["lambda_l1"] == 0.0
    assert found.trials[0].params == baseline_point(cfg.model)
    assert found.trials[0].params == {
        "num_leaves": 31,
        "learning_rate": 0.1,
        "min_data_in_leaf": 20,
        "lambda_l1": 1e-8,
        "lambda_l2": 1e-8,
    }


def test_the_baseline_point_is_clamped_into_the_space():
    point = baseline_point(
        {
            "num_leaves": 1000,
            "learning_rate": 0.001,
            "min_data_in_leaf": 7,
            "lambda_l1": 0.0,
            "lambda_l2": 5,
            "hold_overnight": True,
        }
    )
    assert point == {
        "num_leaves": 255,
        "learning_rate": 0.005,
        "min_data_in_leaf": 20,
        "lambda_l1": 1e-8,
        "lambda_l2": 1.0,
    }
    assert list(point) == list(SPACE)


def test_a_seeded_study_reproduces(frames, eval_periods):
    a = _search(frames, eval_periods)[1].trials
    b = _search(frames, eval_periods)[1].trials
    assert len(a) == len(b) == 4
    assert [t.params for t in a] == [t.params for t in b]
    assert [t.value for t in a] == [t.value for t in b]
    assert [t.best_iteration for t in a] == [t.best_iteration for t in b]
    # the later trials are sampled, not the baseline over again
    assert len({tuple(t.params.values()) for t in a}) == 4


def test_a_seeded_study_reproduces_once_tpe_takes_over(frames, eval_periods):
    # Optuna's first 10 completed trials are random (the baseline included); the 11th on are TPE's
    a = _search(frames, eval_periods, n_trials=12)[1]
    b = _search(frames, eval_periods, n_trials=12)[1]
    assert [t.params for t in a.trials] == [t.params for t in b.trials]
    assert [t.value for t in a.trials] == [t.value for t in b.trials]
    assert (a.best.number, a.best.params, a.best.value) == (
        b.best.number,
        b.best.params,
        b.best.value,
    )


def test_every_trial_stays_in_the_space(frames, eval_periods):
    found = _search(frames, eval_periods, n_trials=6)[1]
    assert [t.number for t in found.trials] == list(range(6))
    for trial in found.trials:
        assert list(trial.params) == list(SPACE)
        for name, (kind, low, high) in SPACE.items():
            value = trial.params[name]
            assert low <= value <= high
            assert isinstance(value, int if kind == "int" else float)
        assert trial.seconds > 0
        assert 1 <= trial.best_iteration <= 1000


def test_the_winner_is_the_lowest_value_and_its_fit_is_kept(frames, eval_periods):
    # a baseline that overfits at once (learning_rate 0.5), so that a sampled trial beats it
    cfg, found = _search(frames, eval_periods, n_trials=6, learning_rate=0.5)
    values = [t.value for t in found.trials]
    assert found.best.value == min(values)
    assert found.best.number > 0
    # equal values (here several trials stop on the same constant) go to the earliest
    assert found.best == found.trials[values.index(min(values))]
    assert found.result.eval_mse == found.best.value
    assert found.result.best_iteration == found.best.best_iteration
    # the kept fit is the winner's: predictions over exactly the evaluation sessions, and the
    # same as fitting the winner's point again
    assert set(found.result.predictions["period"]) == set(eval_periods)
    stop = EarlyStop(tuple(eval_periods), tuple(POOL), MAX_ROUNDS, PATIENCE)
    refit = get_model("nonlinear")(**{**cfg.model, **found.best.params}).fit_predict_fold(
        frames, FIT, (min(eval_periods), max(eval_periods)), stop
    )
    pd.testing.assert_frame_equal(found.result.predictions, refit.predictions)


# Optuna checks the deadline before each trial and times at microsecond resolution, so on its own
# it runs the baseline under 1e-9 only by chance, and never under 0
@pytest.mark.parametrize("timeout", [1e-9, 0.0])
def test_a_tiny_timeout_still_runs_the_baseline(frames, eval_periods, timeout):
    cfg, found = _search(frames, eval_periods, timeout=timeout, n_trials=None)
    assert len(found.trials) == 1
    assert found.best == found.trials[0]
    assert found.trials[0].params == baseline_point(cfg.model)
    assert found.result.eval_mse == found.best.value


def test_a_timeout_spends_what_the_baseline_left(frames, eval_periods):
    # the baseline takes well over a millisecond, so no sampled trial follows it ...
    assert len(_search(frames, eval_periods, timeout=1e-3, n_trials=50)[1].trials) == 1
    # ... and the rest of a long budget goes to the trial limit
    assert len(_search(frames, eval_periods, timeout=3600.0, n_trials=3)[1].trials) == 3


def test_a_search_needs_a_limit(frames, eval_periods):
    with pytest.raises(ValueError, match="timeout or n_trials"):
        _search(frames, eval_periods, timeout=None, n_trials=None)
    with pytest.raises(ValueError, match="n_trials"):
        _search(frames, eval_periods, n_trials=0)


def test_an_empty_fit_window_is_refused(frames, eval_periods):
    cfg = config.resolve("nonlinear")
    with pytest.raises(ValueError, match="no fit rows"):
        search_fold(cfg, frames, None, eval_periods, POOL, timeout=None, n_trials=1)


def _folds(run):
    """(fold.json, fold directory) of every fold of a run, in order."""
    return [
        (json.loads((fdir / "fold.json").read_text()), fdir)
        for fdir in sorted((run / "folds").iterdir())
    ]


def _run_meta(run):
    return json.loads((run / "run.json").read_text())


def test_train_with_hpo(learned_data, tmp_path, monkeypatch):
    found = []
    real = hpo.search_fold

    def spy(*args, **kwargs):
        found.append(real(*args, **kwargs))
        return found[-1]

    monkeypatch.setattr("ml4trading.hpo.search_fold", spy)
    run = train(
        "nonlinear",
        tmp_path / "run",
        learned_data,
        overrides=SHORT_FOLDS,
        hpo_trials=3,
        importance=True,
    )

    folds = _folds(run)
    assert len(folds) == len(found) == 3
    for (fold, fdir), search in zip(folds, found, strict=True):
        block = fold["hpo"]
        assert list(block) == ["timeout_s", "n_trials", "best_number", "trials"]
        assert block["timeout_s"] is None and block["n_trials"] == 3
        assert [t["number"] for t in block["trials"]] == [0, 1, 2]
        winner = block["trials"][block["best_number"]]
        assert block["best_number"] == search.best.number
        assert list(winner) == ["number", "params", "value", "best_iteration", "seconds"]
        # the deploy fit takes the winner's five parameters and the round it kept
        assert list(winner["params"]) == list(SPACE)
        for name, value in winner["params"].items():
            assert fold["model_params"][name] == value
        assert fold["model_params"]["n_estimators"] == winner["best_iteration"]
        # one candidate, the winner, whose validation is what the fold keeps
        assert len(fold["candidate_val_net_sharpe"]) == len(fold["candidate_val_r2"]) == 1
        assert fold["chosen"] == 0
        assert fold["val_diagnostics"]["net_sharpe"] == fold["candidate_val_net_sharpe"][0]
        val = pd.read_parquet(fdir / "val" / "predictions.parquet")
        assert len(val) > 0
        assert set(val["symbol"]) <= set(POOL)
        # ... namely the winner's own predictions on the rows it was searched on
        won = search.result.predictions.merge(
            val, on=["period", "symbol"], suffixes=("_search", "")
        )
        assert len(won) == len(val)
        assert won["prediction"].equals(won["prediction_search"])
        # ... which is also the fit the importance is scored on
        assert len(pd.read_parquet(fdir / "val" / "importance.parquet")) > 0
    assert backtest(run, learned_data)["n_periods"] > 0
    # the run records the request; a fold's own count is in its fold.json
    assert _run_meta(run)["hpo"] == {"hours": None, "trials": 3, "timeout_s": None}

    # diagnose carries each fold's trial count and the winner's round into folds.csv
    table = pd.read_csv(diagnose(run) / "folds.csv")
    assert list(table.columns[-2:]) == ["hpo_n_trials", "hpo_best_iteration"]
    assert table["hpo_n_trials"].tolist() == [3, 3, 3]
    assert table["hpo_best_iteration"].tolist() == [
        fold["hpo"]["trials"][fold["hpo"]["best_number"]]["best_iteration"] for fold, _ in folds
    ]


def test_the_budget_is_split_evenly(learned_data, tmp_path, monkeypatch):
    seen = []
    real = hpo.search_fold

    def spy(*args, timeout, n_trials):
        seen.append((timeout, n_trials))
        return real(*args, timeout=0.0, n_trials=1)  # the baseline alone, to keep the test short

    monkeypatch.setattr("ml4trading.hpo.search_fold", spy)
    run = train("nonlinear", tmp_path / "run", learned_data, overrides=SHORT_FOLDS, hpo_hours=0.003)
    assert [n for _, n in seen] == [None, None, None]  # three folds, no trial limit
    assert [t for t, _ in seen] == [pytest.approx(3.6)] * 3  # 0.003 h = 10.8 s, a third each
    for fold, _ in _folds(run):
        assert fold["hpo"]["timeout_s"] == pytest.approx(3.6)
        assert fold["hpo"]["n_trials"] == 1
    assert _run_meta(run)["hpo"] == {
        "hours": 0.003,
        "trials": None,
        "timeout_s": pytest.approx(3.6),
    }


def test_a_run_without_a_search_records_none(learned_data, tmp_path):
    run = train("linear", tmp_path / "run", learned_data, overrides=SHORT_FOLDS)
    assert "hpo" not in _run_meta(run)
    assert all("hpo" not in fold for fold, _ in _folds(run))
    assert not [c for c in pd.read_csv(diagnose(run) / "folds.csv").columns if "hpo" in c]


def test_hpo_refusals(learned_data, tmp_path, monkeypatch):
    def no_loading(*args, **kwargs):
        raise AssertionError("the data was loaded before the HPO request was checked")

    monkeypatch.setattr("ml4trading.train.load_frames", no_loading)
    out = tmp_path / "run"
    with pytest.raises(ValueError, match="grid"):
        train("nonlinear", out, learned_data, grid={"model.num_leaves": [7]}, hpo_trials=2)
    with pytest.raises(ValueError, match="grid"):  # the CLI's --hpo-hours with --grid
        train("nonlinear", out, learned_data, grid={"model.num_leaves": [7]}, hpo_hours=1.0)
    with pytest.raises(ValueError, match="nonlinear"):
        train("linear", out, learned_data, hpo_trials=2)
    with pytest.raises(ValueError, match="hpo_hours"):
        train("nonlinear", out, learned_data, hpo_hours=0)
    with pytest.raises(ValueError, match="hpo_hours"):
        train("nonlinear", out, learned_data, hpo_hours=-1.0)
    with pytest.raises(ValueError, match="hpo_trials"):
        train("nonlinear", out, learned_data, hpo_trials=0)


@pytest.mark.parametrize(
    "overrides",
    [
        # nobody is liquid enough to trade
        {**SHORT_FOLDS, "selector.min_dollar_volume": 1e12},
        # the train window (January, the dataset's first month) ends within the embargo of its
        # first session, so the validation fit has no window; the universe is not empty
        {
            "folds.train_months": 2,
            "folds.val_months": 3,
            "folds.path_start": "2021-05-01",
            "folds.path_end": "2021-05-31",
            "folds.embargo_days": 25,
            "selector.min_coverage_sessions": 5,
        },
    ],
)
def test_a_fold_with_nothing_to_fit_is_not_searched(learned_data, tmp_path, monkeypatch, overrides):
    def no_search(*args, **kwargs):
        raise AssertionError("a fold with nothing to fit was searched")

    monkeypatch.setattr("ml4trading.hpo.search_fold", no_search)
    run = train("nonlinear", tmp_path / "run", learned_data, overrides=overrides, hpo_trials=2)
    for fold, fdir in _folds(run):
        assert fold["hpo"] == {"timeout_s": None, "n_trials": 0, "best_number": None, "trials": []}
        # as any fold with an empty validation fit: scored 0, run's own configuration deployed
        assert fold["candidate_val_net_sharpe"] == [0.0] and fold["chosen"] == 0
        assert fold["model_params"] == config.resolve("nonlinear", overrides).model
        assert pd.read_parquet(fdir / "val" / "predictions.parquet").empty


def test_a_rerun_clears_the_old_validation_outputs(learned_data, tmp_path):
    out = tmp_path / "run"
    asked = train("nonlinear", out, learned_data, overrides=SHORT_FOLDS, importance=True)
    assert all((fdir / "val" / "importance.parquet").exists() for _, fdir in _folds(asked))
    plain = train("nonlinear", out, learned_data, overrides=SHORT_FOLDS)
    for _, fdir in _folds(plain):
        names = sorted(p.name for p in (fdir / "val").iterdir())
        assert names == ["daily.parquet", "predictions.parquet"]  # no importance.parquet left


def test_a_rerun_clears_the_old_fitted_model(learned_data, tmp_path):
    out = tmp_path / "run"
    first = train("linear", out, learned_data, overrides=SHORT_FOLDS)
    assert all(
        sorted(p.name for p in (d / "model").iterdir()) == ["model.npz"] for _, d in _folds(first)
    )
    # a different estimator into the same --out: nothing of the linear fit stays beside it
    again = train("nonlinear", out, learned_data, overrides=SHORT_FOLDS)
    for _, fdir in _folds(again):
        assert sorted(p.name for p in (fdir / "model").iterdir()) == ["model.txt"]
    # a fold that now fits nothing (nobody is liquid enough) has no model at all
    empty = train(
        "nonlinear",
        out,
        learned_data,
        overrides={**SHORT_FOLDS, "selector.min_dollar_volume": 1e12},
    )
    for _, fdir in _folds(empty):
        assert not (fdir / "model").exists()


def test_the_first_of_equal_candidates_is_chosen(learned_data, tmp_path):
    # one rule picks both the choice and the validation that is kept: two candidates that tie
    # leave the first
    grid = {"model.alpha_ratio": [0.1, 0.1]}
    run = train("linear", tmp_path / "run", learned_data, grid=grid, overrides=SHORT_FOLDS)
    for fold, _ in _folds(run):
        first, second = fold["candidate_val_net_sharpe"]
        assert first == second and fold["chosen"] == 0
