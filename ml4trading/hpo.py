"""A seeded hyperparameter search for ``nonlinear``, one study per walk-forward fold.

A trial fits the fold's booster with one point of ``SPACE`` and stops it early on a validation
set (``EarlyStop``): the number of rounds is found by the fit, not searched, so the search
space is the five parameters left. The trial's value is the squared error on that set at the
round the fit kept, which makes a value comparable across trials whatever their learning rate.
The study minimizes it with Optuna's TPE sampler, seeded, and the winner is the study's own
best trial.

The first trial is always the run's own configuration (clamped into the space), enqueued
before the sampler draws anything, so a fold has a winner however short its budget: the
baseline runs whatever ``timeout`` says, and the search can only improve on it. Memory stays
flat across trials: a trial's ``FoldResult`` (a booster and its predictions) is kept only while
it is the best so far.

Optuna is imported only when a search runs, as lightgbm is, so the other methods do not need it.
"""

from __future__ import annotations

import logging
import time
import warnings
from dataclasses import dataclass

from ml4trading.config import RunConfig
from ml4trading.models import get_model
from ml4trading.models.base import FoldResult, Frames
from ml4trading.models.learned.base import EarlyStop

#: The sampler's seed, hence the whole study's. ML4Investment seeds its TPESampler too, with its
#: ``SEED`` of 42 (``optimize_model_hyperparameters`` in ml4investment/utils/model_training.py,
#: ml4investment/config/global_settings.py); 0 is kept here as the seed the project's fits
#: already use (nonlinear's LightGBM ``seed=0``).
SEED = 0
#: Most boosting rounds a trial may train: ML4Investment's ``NUM_ROUNDS`` default
#: (ml4investment/config/global_settings.py).
MAX_ROUNDS = 1000
#: Rounds without a better validation error before a trial stops. This project's own choice (no
#: public source fixes one): a tenth of ``MAX_ROUNDS``.
PATIENCE = 100

#: Name -> (kind, low, high), all searched on a log scale. ML4Investment searches these five
#: (``_build_hyperparameter_objective`` in ml4investment/utils/model_training.py) with the same
#: bounds, except ``min_data_in_leaf``; the others it searches (dart's drop rates) do not exist
#: under gbdt, which is what a trial fits.
SPACE: dict[str, tuple[str, float, float]] = {
    "num_leaves": ("int", 15, 255),  # ML4Investment: 15-255, log
    "learning_rate": ("float", 0.005, 0.5),  # ML4Investment: 0.005-0.5, log
    # ML4Investment: 20-100, linear. Widened to 5000 and made log: LightGBM's "Parameters Tuning"
    # guide says "hundreds or thousands is enough for a large dataset", and a fold's fit window
    # is one (``FoldConfig.train_months`` of daily rows for every stock of the pool).
    "min_data_in_leaf": ("int", 20, 5000),
    "lambda_l1": ("float", 1e-8, 1.0),  # ML4Investment: 1e-8-1, log
    "lambda_l2": ("float", 1e-8, 1.0),  # ML4Investment: 1e-8-1, log
}


def baseline_point(model: dict) -> dict:
    """The run config's value of every ``SPACE`` parameter, clamped into its bounds. A log-scale
    parameter cannot take 0, which is the config's default for ``lambda_l1`` / ``lambda_l2``, so
    that becomes the lower bound 1e-8 -- ML4Investment's own baseline value for both
    (``_enqueue_hyperparameter_baseline``), which is practically no penalty."""
    point = {}
    for name, (kind, low, high) in SPACE.items():
        value = max(low, min(high, model[name]))
        point[name] = int(round(value)) if kind == "int" else float(value)
    return point


@dataclass
class Trial:
    """One completed trial: its point of ``SPACE``, its validation squared error (the study's
    objective), the round it kept, and its wall-clock seconds."""

    number: int
    params: dict
    value: float
    best_iteration: int
    seconds: float


@dataclass
class SearchResult:
    """A fold's study: every completed trial, the winner among them, and the winner's fit (with
    its predictions over the evaluation window)."""

    trials: list[Trial]
    best: Trial
    result: FoldResult


def search_fold(
    cfg: RunConfig,
    frames: Frames,
    fit_window,
    eval_periods: list,
    eval_symbols: list[str],
    *,
    timeout: float | None,
    n_trials: int | None,
    predict_window=None,
) -> SearchResult:
    """Search ``SPACE`` for the fold fitted on ``fit_window`` and validated on ``eval_symbols``
    over ``eval_periods`` (K = 390: the periods are the evaluation sessions).

    A trial predicts ``predict_window`` -- the window a grid's fit would predict, so that a
    prediction that reads earlier sessions (``smooth_halflife``) is the one the same parameters
    give in a grid; the early stopping reads only ``eval_periods``. Left out, the window is the
    span of ``eval_periods``.

    The study stops after ``n_trials`` trials, or once ``timeout`` seconds have passed since the
    call, whichever comes first; one of them is required. The baseline trial always runs and
    counts: a trial in progress at the deadline finishes, so the time spent can exceed
    ``timeout`` by up to one fit."""
    if timeout is None and n_trials is None:
        raise ValueError("search_fold: give a timeout or n_trials, or the search never ends")
    if n_trials is not None and n_trials < 1:
        raise ValueError(f"search_fold: n_trials={n_trials}; the baseline trial always runs")
    import optuna

    optuna.logging.set_verbosity(logging.WARNING)  # a trial per line is noise in a run's log
    started = time.perf_counter()
    model_class = get_model(cfg.method)
    window = (min(eval_periods), max(eval_periods)) if predict_window is None else predict_window
    stop = EarlyStop(tuple(eval_periods), tuple(eval_symbols), MAX_ROUNDS, PATIENCE)
    kept: dict = {"value": float("inf")}  # the best trial's fit so far

    def objective(trial: optuna.Trial) -> float:
        params = {
            name: (
                trial.suggest_int(name, int(low), int(high), log=True)
                if kind == "int"
                else trial.suggest_float(name, low, high, log=True)
            )
            for name, (kind, low, high) in SPACE.items()
        }
        model = model_class(**{**cfg.model, **params})
        result = model.fit_predict_fold(frames, fit_window, window, stop)
        if result.eval_mse is None:
            raise ValueError("search_fold: the fit window has no fit rows")
        trial.set_user_attr("best_iteration", result.best_iteration)
        if result.eval_mse < kept["value"]:  # strictly: a tie keeps the earlier trial, as Optuna
            kept.update(value=result.eval_mse, number=trial.number, result=result)
        return result.eval_mse

    with warnings.catch_warnings():
        # ``multivariate=True`` is flagged experimental in Optuna 4; ML4Investment samples with it
        # too (``MODEL_OPTIMIZATION_SAMPLING_MULTIVARIATE``, default true, in
        # ml4investment/config/global_settings.py)
        warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=SEED, multivariate=True),
            pruner=optuna.pruners.NopPruner(),
        )
        study.enqueue_trial(baseline_point(cfg.model))
        # Optuna checks the deadline before a trial starts, so a timeout shorter than one fit
        # would run none: the baseline is run on its own, and the rest of the budget is what is
        # left after it.
        study.optimize(objective, n_trials=1)
        left = None if timeout is None else timeout - (time.perf_counter() - started)
        if (n_trials is None or n_trials > 1) and (left is None or left > 0):
            study.optimize(
                objective, n_trials=None if n_trials is None else n_trials - 1, timeout=left
            )

    trials = [
        Trial(
            t.number,
            dict(t.params),
            float(t.value),
            int(t.user_attrs["best_iteration"]),
            t.duration.total_seconds(),
        )
        for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    ]
    winner = study.best_trial.number
    best = next(t for t in trials if t.number == winner)
    if kept["number"] != best.number:  # cannot happen: both keep the first of equal values
        raise RuntimeError(f"search_fold: kept trial {kept['number']} but best is {best.number}")
    return SearchResult(trials, best, kept["result"])
