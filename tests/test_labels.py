import numpy as np
import pandas as pd
import pytest

from ml4trading.models.learned.labels import (
    LABELS,
    WINSOR_QUANTILES,
    LabelTransform,
    fit_label_transform,
)


def test_the_kinds_and_the_quantiles():
    assert LABELS == ("raw", "demean", "winsor", "demean_winsor")
    assert WINSOR_QUANTILES == (0.005, 0.995)  # ML4Investment's clip quantiles


def test_demean():
    transform = LabelTransform("demean")
    got = transform.apply(np.array([0.01, 0.03, -0.02]), np.array([1, 1, 2]))
    assert np.allclose(got, [-0.01, 0.01, 0.0], rtol=0, atol=1e-15)


def test_demean_takes_the_mean_of_each_period_given():
    d1, d2 = pd.Timestamp("2021-01-04"), pd.Timestamp("2021-01-05")
    y = np.array([0.02, 0.04, 0.10, 0.20])
    got = LabelTransform("demean").apply(y, np.array([d1, d1, d2, d2], dtype="datetime64[ns]"))
    assert np.allclose(got, [-0.01, 0.01, -0.05, 0.05], rtol=0, atol=1e-15)


def test_winsor_bounds_come_from_the_fit():
    y = np.arange(1000, dtype=float)
    fitted = fit_label_transform("winsor", y, np.zeros(1000, dtype=int))
    lo, hi = np.quantile(y, 0.005), np.quantile(y, 0.995)
    assert (fitted.kind, fitted.lo, fitted.hi) == ("winsor", lo, hi)
    got = fitted.apply(np.array([-5.0, 2000.0, 500.0]), np.zeros(3, dtype=int))
    assert got.tolist() == [lo, hi, 500.0]


def test_demean_winsor_clips_the_demeaned_label_at_the_demeaned_fit_quantiles():
    period = np.repeat([0, 1], 500)
    y = np.arange(1000, dtype=float) + 1000.0 * period  # period 1 sits 1000 above period 0
    demeaned = y - pd.Series(y).groupby(period).transform("mean").to_numpy()
    fitted = fit_label_transform("demean_winsor", y, period)
    assert fitted.lo == np.quantile(demeaned, 0.005)
    assert fitted.hi == np.quantile(demeaned, 0.995)
    # a later period's outlier is demeaned over the rows given, then clipped to the fit's bounds
    got = fitted.apply(np.array([0.0, 0.0, 10_000.0]), np.array([7, 7, 7]))
    assert got.tolist() == [fitted.lo, fitted.lo, fitted.hi]


def test_only_the_winsor_kinds_have_bounds_and_winsor_alone_does_not_demean():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    period = np.array([1, 1, 2, 2])
    assert fit_label_transform("demean", y, period) == LabelTransform("demean")
    assert fit_label_transform("raw", y, period) == LabelTransform("raw")
    winsor = fit_label_transform("winsor", y, period)
    assert winsor.lo > 1.0 and winsor.hi < 4.0
    assert winsor.apply(np.array([2.5]), np.array([1])).tolist() == [2.5]  # not centred


def test_raw_is_the_identity():
    y = np.array([0.0123456789012345, -0.0, np.nan, 3.0e-17, 1.0 / 3.0])
    got = LabelTransform("raw").apply(y, np.array([1, 1, 2, 2, 3]))
    assert np.array_equal(got, y, equal_nan=True)
    assert got.tobytes() == y.tobytes()  # the same bits, -0.0 and NaN payload included
    fitted = fit_label_transform("raw", y, np.array([1, 1, 2, 2, 3]))
    assert fitted == LabelTransform("raw")


@pytest.mark.parametrize("kind", ["demean", "winsor", "demean_winsor"])
def test_nan_stays_nan_and_is_left_out_of_the_mean(kind):
    y = np.array([0.01, np.nan, 0.03, np.nan, 0.05])
    period = np.array([1, 1, 1, 2, 2])
    fitted = fit_label_transform(kind, y, period)
    got = fitted.apply(y, period)
    assert np.isnan(got).tolist() == [False, True, False, True, False]
    if kind == "demean":
        # period 1's mean is (0.01 + 0.03) / 2: the NaN is not a zero
        assert np.allclose(got[~np.isnan(got)], [-0.01, 0.01, 0.0], rtol=0, atol=1e-15)
        assert fitted.lo is None and fitted.hi is None
    else:
        assert np.isfinite([fitted.lo, fitted.hi]).all()


def test_an_infinite_label_is_left_out_of_the_mean_and_kept():
    got = LabelTransform("demean").apply(np.array([0.01, np.inf, 0.03]), np.array([1, 1, 1]))
    assert np.allclose(got[[0, 2]], [-0.01, 0.01], rtol=0, atol=1e-15) and np.isinf(got[1])


def test_a_bad_transform_is_refused():
    with pytest.raises(ValueError, match="rank"):
        fit_label_transform("rank", np.ones(3), np.ones(3, dtype=int))
    with pytest.raises(ValueError, match="rank"):
        LabelTransform("rank").apply(np.ones(3), np.ones(3, dtype=int))
    with pytest.raises(ValueError, match="bounds"):  # a winsor transform needs its bounds
        LabelTransform("winsor").apply(np.ones(3), np.ones(3, dtype=int))
    with pytest.raises(ValueError, match="quantiles"):  # nothing to take them of
        fit_label_transform("winsor", np.array([np.nan]), np.array([1]))
    with pytest.raises(ValueError, match="labels against"):
        LabelTransform("demean").apply(np.ones(3), np.ones(2, dtype=int))
