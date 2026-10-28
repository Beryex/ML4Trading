import pandas as pd
import pytest
from conftest import write_dataset

from ml4trading import config
from ml4trading.hpo import MAX_ROUNDS, PATIENCE, SPACE, baseline_point, search_fold
from ml4trading.models import get_model
from ml4trading.models.base import rows_in
from ml4trading.models.learned.base import EarlyStop
from ml4trading.pipeline import load_frames

FIT = (pd.Timestamp("2021-03-01"), pd.Timestamp("2021-05-21"))
JUNE = (pd.Timestamp("2021-06-01"), pd.Timestamp("2021-06-30"))
POOL = ["AAA", "BBB", "CCC"]


@pytest.fixture(scope="module")
def frames(tmp_path_factory):
    data = write_dataset(tmp_path_factory.mktemp("hpo") / "data")
    return load_frames(config.resolve("nonlinear"), data)[1]


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
