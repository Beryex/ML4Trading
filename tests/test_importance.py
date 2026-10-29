import numpy as np
import pandas as pd
import pytest
from conftest import write_dataset
from pandas.testing import assert_frame_equal

from ml4trading import config
from ml4trading.importance import permutation_importance
from ml4trading.models import Frames, get_model
from ml4trading.models.base import rows_in
from ml4trading.models.learned.base import FitArtifact
from ml4trading.pipeline import load_frames
from ml4trading.train import train

FIT = (pd.Timestamp("2021-03-01"), pd.Timestamp("2021-05-21"))
JUNE = (pd.Timestamp("2021-06-01"), pd.Timestamp("2021-06-30"))
SHORT_FOLDS = {
    "folds.train_months": 3,
    "folds.val_months": 1,
    "folds.path_start": "2021-06-01",
    "folds.path_end": "2021-08-31",
}
NONLINEAR_COLUMNS = ["feature", "gain", "split", "shap", "perm_mse", "perm_ic"]
LINEAR_COLUMNS = ["feature", "perm_mse", "perm_ic"]


@pytest.fixture(scope="module")
def learned_data(tmp_path_factory):
    return write_dataset(tmp_path_factory.mktemp("importance") / "data")


def _keyed(frames, window):
    """The features of ``window``'s rows, indexed by (period, symbol) as a model hands them."""
    panel = frames.panel
    keys = pd.MultiIndex.from_arrays([panel["period"], panel["symbol"]], names=["period", "symbol"])
    rows = rows_in(panel, window).index
    return frames.X.set_axis(keys).iloc[panel.index.get_indexer(rows)]


class _TwiceA(FitArtifact):
    """A stub fit that predicts ``2 * a`` and ignores ``b``."""

    FILENAME = "stub"

    def predict(self, X):
        return 2.0 * X["a"].to_numpy()

    def feature_names(self):
        return ["a", "b"]

    def save(self, directory):
        raise NotImplementedError

    @classmethod
    def load(cls, directory):
        raise NotImplementedError


def test_permutation_finds_the_only_informative_column():
    noise = np.random.default_rng(7)
    index = pd.MultiIndex.from_product(
        [pd.bdate_range("2021-06-01", periods=5), list("WXYZ")], names=["period", "symbol"]
    )
    X = pd.DataFrame({"a": noise.normal(size=20), "b": noise.normal(size=20)}, index=index)
    label = y = 2.0 * X["a"].to_numpy()

    table = permutation_importance(_TwiceA(), X, label, y)

    assert list(table.index) == ["a", "b"]
    assert list(table.columns) == ["perm_mse", "perm_ic"]
    assert table.loc["a", "perm_mse"] > 0.0
    assert table.loc["b", "perm_mse"] == 0.0
    assert table.loc["a", "perm_ic"] > 0.0
    assert table.loc["b", "perm_ic"] == 0.0
    # one generator, one seed: the same call gives the same table
    assert_frame_equal(table, permutation_importance(_TwiceA(), X, label, y))


def test_native_importance_columns(learned_data):
    f = load_frames(config.resolve("nonlinear"), learned_data)[1]
    X = f.X.copy()
    X["Return_1d"] = 0.5  # a column the trees cannot split on
    f = Frames(f.panel, X)
    cfg = config.resolve("nonlinear")
    artifact = get_model("nonlinear")(**cfg.model).fit_predict_fold(f, FIT, JUNE).artifact

    native = artifact.native_importance(_keyed(f, JUNE))

    assert list(native.index) == artifact.feature_names()
    assert list(native.columns) == ["gain", "split", "shap"]
    assert (native >= 0).all().all()
    assert native.loc["Return_1d"].tolist() == [0.0, 0.0, 0.0]
    assert native["shap"].sum() > 0.0


@pytest.mark.parametrize("init_model", ["none", "linear"])
def test_linear_trees_report_gain_and_split_without_shap(learned_data, tmp_path, init_model):
    # LightGBM computes no TreeSHAP for linear trees: the booster's own account keeps its gain
    # and split count and leaves shap undefined, plain or boosted from the linear initial score
    f = load_frames(config.resolve("nonlinear"), learned_data)[1]
    params = {"model.linear_tree": True, "model.loss": "mse", "model.init_model": init_model}
    cfg = config.resolve("nonlinear", params)
    artifact = get_model("nonlinear")(**cfg.model).fit_predict_fold(f, FIT, JUNE).artifact
    X = _keyed(f, JUNE)

    native = artifact.native_importance(X)

    assert list(native.index) == artifact.feature_names()
    assert list(native.columns) == ["gain", "split", "shap"]
    assert native["shap"].isna().all()
    assert np.isfinite(native[["gain", "split"]]).all().all() and native["gain"].sum() > 0.0
    # a saved booster, loaded, is still known to hold linear trees
    artifact.save(tmp_path / "model")
    assert_frame_equal(type(artifact).load(tmp_path / "model").native_importance(X), native)


def _fold_dirs(run):
    return sorted((run / "folds").iterdir())


@pytest.mark.parametrize("init_model", ["none", "linear"])
def test_a_linear_tree_run_writes_importance_with_no_shap(learned_data, tmp_path, init_model):
    from ml4trading.diagnose import diagnose

    overrides = {
        **SHORT_FOLDS,
        "model.linear_tree": True,
        "model.loss": "mse",
        "model.init_model": init_model,
    }
    run = train("nonlinear", tmp_path / "run", learned_data, overrides=overrides, importance=True)
    assert len(_fold_dirs(run)) > 0
    for fdir in _fold_dirs(run):
        table = pd.read_parquet(fdir / "val" / "importance.parquet")
        assert list(table.columns) == NONLINEAR_COLUMNS
        assert table["shap"].isna().all()
        assert np.isfinite(table[["gain", "split", "perm_mse"]]).all().all()
    # the run's diagnosis ranks on what the folds have, and lists no empty SHAP ranking
    out = diagnose(run)
    importance = pd.read_csv(out / "importance.csv")
    assert importance["shap_mean"].isna().all() and importance["gain_mean"].notna().all()
    summary = (out / "summary.md").read_text()
    assert "(SHAP)" not in summary and "(permutation MSE)" in summary


def test_train_writes_importance_only_when_asked(learned_data, tmp_path):
    plain = train("nonlinear", tmp_path / "plain", learned_data, overrides=SHORT_FOLDS)
    for fdir in _fold_dirs(plain):
        assert not (fdir / "val" / "importance.parquet").exists()

    asked = train(
        "nonlinear", tmp_path / "asked", learned_data, overrides=SHORT_FOLDS, importance=True
    )
    for fdir in _fold_dirs(asked):
        table = pd.read_parquet(fdir / "val" / "importance.parquet")
        assert list(table.columns) == NONLINEAR_COLUMNS
        assert table["feature"].is_unique and "stock_id" in set(table["feature"])
        assert np.isfinite(table[["gain", "split", "shap", "perm_mse"]]).all().all()
        # the importance does not change what the fold predicts
        assert_frame_equal(
            pd.read_parquet(fdir / "predictions.parquet"),
            pd.read_parquet(plain / "folds" / fdir.name / "predictions.parquet"),
            check_exact=True,
        )

    linear = train(
        "linear", tmp_path / "linear", learned_data, overrides=SHORT_FOLDS, importance=True
    )
    for fdir in _fold_dirs(linear):
        table = pd.read_parquet(fdir / "val" / "importance.parquet")
        assert list(table.columns) == LINEAR_COLUMNS
        assert "stock_id" not in set(table["feature"])


def test_a_fit_free_model_writes_no_importance(learned_data, tmp_path):
    run = train(
        "investment", tmp_path / "run", learned_data, overrides=SHORT_FOLDS, importance=True
    )
    assert len(_fold_dirs(run)) > 0
    for fdir in _fold_dirs(run):
        assert not (fdir / "val" / "importance.parquet").exists()


def test_the_importance_is_scored_on_the_validation_rows_in_order(
    learned_data, tmp_path, monkeypatch
):
    import ml4trading.train as train_mod

    seen = []
    real = train_mod.importance_table

    def spy(artifact, X, label, y):
        seen.append((artifact, X, label, y))
        return real(artifact, X, label, y)

    monkeypatch.setattr(train_mod, "importance_table", spy)
    run = train("nonlinear", tmp_path / "run", learned_data, overrides=SHORT_FOLDS, importance=True)

    assert len(seen) == len(_fold_dirs(run))
    for (artifact, X, label, y), fdir in zip(seen, _fold_dirs(run), strict=True):
        val = pd.read_parquet(fdir / "val" / "predictions.parquet")
        assert X.index.names == ["period", "symbol"]
        assert X.index.tolist() == list(zip(val["period"], val["symbol"], strict=True))
        assert np.array_equal(label, val["label"].to_numpy())
        assert np.array_equal(y, val["y"].to_numpy())
        # the kept fit, scored on these rows, gives the predictions the fold traded
        assert np.array_equal(artifact.predict(X), val["prediction"].to_numpy())
