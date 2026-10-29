import hashlib
import math

import numpy as np
import pandas as pd
import pytest

from ml4trading.backtest import backtest
from ml4trading.compare import compare, main
from ml4trading.metrics import sharpe
from ml4trading.significance import deflated_sharpe, sharpe_difference_test
from ml4trading.train import train

SHORT_FOLDS = {
    "folds.train_months": 3,
    "folds.val_months": 1,
    "folds.path_start": "2021-06-01",
    "folds.path_end": "2021-11-30",
}
EULER_GAMMA = 0.5772156649015329


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _series(values, start="2021-01-04"):
    return pd.Series(values, index=pd.bdate_range(start, periods=len(values)), dtype=float)


def test_the_memmel_jobson_korkie_statistic_matches_a_hand_computation():
    # r1 = 1, 3, 1, 3 (in %): mean 2, population std 1, SR1 = 2.
    # r2 = 0, 4, 2, 2: mean 2, population std sqrt(2), SR2 = sqrt(2); cov = 1, so rho = 1 / sqrt(2)
    # and rho^2 = 1/2. Per period (T = 4):
    #   diff = 2 - sqrt(2)
    #   T * var = 2 - 2 rho + (SR1^2 + SR2^2 - 2 SR1 SR2 rho^2) / 2 = 5 - 2 sqrt(2)
    r1 = _series([0.01, 0.03, 0.01, 0.03])
    r2 = _series([0.00, 0.04, 0.02, 0.02])
    out = sharpe_difference_test(r1, r2, per_year=252)
    root = math.sqrt(252)
    diff = 2.0 - math.sqrt(2.0)
    se = math.sqrt((5.0 - 2.0 * math.sqrt(2.0)) / 4.0)
    z = diff / se
    assert set(out) == {"sharpe_1", "sharpe_2", "diff", "se", "z", "p_two_sided", "n", "corr"}
    assert out["n"] == 4
    assert out["corr"] == pytest.approx(1.0 / math.sqrt(2.0), rel=1e-12)
    assert out["sharpe_1"] == pytest.approx(2.0 * root, rel=1e-12)  # annualized at the end
    assert out["sharpe_2"] == pytest.approx(math.sqrt(2.0) * root, rel=1e-12)
    assert out["diff"] == pytest.approx(diff * root, rel=1e-12)
    assert out["se"] == pytest.approx(se * root, rel=1e-12)
    assert out["z"] == pytest.approx(z, rel=1e-12)  # a ratio: annualizing leaves it unchanged
    assert out["p_two_sided"] == pytest.approx(2.0 * (1.0 - _normal_cdf(z)), rel=1e-9)
    # the sign flips with the order of the arguments, the evidence does not
    swapped = sharpe_difference_test(r2, r1, per_year=252)
    assert swapped["diff"] == pytest.approx(-out["diff"], rel=1e-12)
    assert swapped["z"] == pytest.approx(-out["z"], rel=1e-12)
    assert swapped["p_two_sided"] == pytest.approx(out["p_two_sided"], rel=1e-12)


def test_the_test_pairs_on_the_common_dates_and_matches_the_metrics_sharpe():
    draws = np.random.default_rng(seed=21)
    a = _series(draws.normal(0.001, 0.01, 60))
    b = _series(draws.normal(0.0005, 0.01, 60)).iloc[5:]  # starts five sessions later
    a.iloc[20] = np.nan  # a missing return drops that date from both
    out = sharpe_difference_test(a, b)
    common = a.index.intersection(b.index).drop(a.index[20])
    assert out["n"] == len(common) == 54
    assert out["sharpe_1"] == pytest.approx(sharpe(a.loc[common]), rel=1e-12)
    assert out["sharpe_2"] == pytest.approx(sharpe(b.loc[common]), rel=1e-12)
    assert out["diff"] == pytest.approx(out["sharpe_1"] - out["sharpe_2"], rel=1e-12)
    assert out["se"] > 0 and 0.0 <= out["p_two_sided"] <= 1.0
    assert out["z"] == pytest.approx(out["diff"] / out["se"], rel=1e-12)


def test_identical_series_have_no_difference():
    r = _series(np.random.default_rng(3).normal(0.001, 0.01, 80))
    out = sharpe_difference_test(r, r.copy())
    assert out["diff"] == 0.0 and out["z"] == 0.0 and out["p_two_sided"] == 1.0
    assert out["corr"] == pytest.approx(1.0, abs=1e-12)
    assert out["n"] == 80


def test_a_series_without_a_sharpe_ratio_is_refused():
    r = _series([0.01, -0.02, 0.03, 0.0])
    with pytest.raises(ValueError) as flat:
        sharpe_difference_test(r, _series([0.0, 0.0, 0.0, 0.0]))
    assert "constant" in str(flat.value)
    with pytest.raises(ValueError, match="at least"):
        sharpe_difference_test(r.iloc[:2], r.iloc[:2])
    with pytest.raises(ValueError, match="at least"):  # no common date at all
        sharpe_difference_test(r, _series([0.01, -0.02, 0.03, 0.0], start="2022-01-03"))


def test_the_expected_maximum_sharpe_matches_a_hand_computation_for_ten_trials():
    # Phi^-1(1 - 1/10) = 1.2815515655446004 and Phi^-1(1 - 1/(10 e)) = 1.7892417645816283
    # (scipy.stats.norm.ppf, pasted so the test does not use the code under test)
    z_n, z_ne = 1.2815515655446004, 1.7892417645816283
    expected = 0.1 * ((1.0 - EULER_GAMMA) * z_n + EULER_GAMMA * z_ne)
    assert expected == pytest.approx(0.157459830134575, rel=1e-12)
    out = deflated_sharpe(0.2, 10, 0.1, 250, 0.0, 3.0)
    assert set(out) == {"expected_max_sr", "dsr"}
    assert out["expected_max_sr"] == pytest.approx(expected, rel=1e-9)


def test_the_deflated_sharpe_ratio_matches_a_hand_computation():
    sr, n_obs, skew, kurt = 0.2, 250, -0.5, 4.0
    sr0 = deflated_sharpe(sr, 10, 0.1, n_obs, skew, kurt)["expected_max_sr"]
    # 1 - skew sr + (kurt - 1) / 4 sr^2 = 1 + 0.1 + 0.03
    stat = (sr - sr0) * math.sqrt(n_obs - 1) / math.sqrt(1.13)
    assert deflated_sharpe(sr, 10, 0.1, n_obs, skew, kurt)["dsr"] == pytest.approx(
        _normal_cdf(stat), rel=1e-9
    )
    # normal returns (skew 0, kurtosis 3) give the 1 + sr^2 / 2 variance of the plain Sharpe
    plain = deflated_sharpe(sr, 10, 0.1, n_obs, 0.0, 3.0)["dsr"]
    assert plain == pytest.approx(
        _normal_cdf((sr - sr0) * math.sqrt(n_obs - 1) / math.sqrt(1.02)), rel=1e-9
    )
    # a Sharpe equal to the expected maximum is a coin flip
    assert deflated_sharpe(sr0, 10, 0.1, n_obs, 0.0, 3.0)["dsr"] == pytest.approx(0.5)


def test_the_deflated_sharpe_ratio_falls_as_more_trials_are_tried():
    outs = [deflated_sharpe(0.2, n, 0.1, 250, -0.3, 4.0) for n in (1, 2, 5, 10, 100, 1000)]
    maxima = [o["expected_max_sr"] for o in outs]
    dsrs = [o["dsr"] for o in outs]
    assert maxima[0] == 0.0  # one trial: no selection to deflate for
    assert maxima == sorted(maxima) and len(set(maxima)) == len(maxima)
    assert dsrs == sorted(dsrs, reverse=True) and len(set(dsrs)) == len(dsrs)
    assert all(0.0 <= d <= 1.0 for d in dsrs)


def test_deflated_sharpe_inputs_that_cannot_be_deflated_are_refused():
    with pytest.raises(ValueError, match="n_trials"):
        deflated_sharpe(0.2, 0, 0.1, 250, 0.0, 3.0)
    with pytest.raises(ValueError, match="n_obs"):
        deflated_sharpe(0.2, 10, 0.1, 1, 0.0, 3.0)
    with pytest.raises(ValueError, match="variance"):  # 1 - skew sr + (kurt - 1) / 4 sr^2 <= 0
        deflated_sharpe(2.0, 10, 0.1, 250, 5.0, 1.0)


def _tree_digest(root):
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.glob("**/*"))
        if p.is_file()
    }


@pytest.fixture
def two_runs(data_dir, tmp_path):
    runs = []
    for method in ("investment", "investment_tech"):
        run = train(method, tmp_path / method, data_dir, overrides=SHORT_FOLDS)
        backtest(run, data_dir)
        runs.append(run)
    return runs


def test_compare_pairs_two_backtested_runs_and_deflates_the_first(two_runs, tmp_path, capsys):
    a, b = two_runs
    trials = tmp_path / "trials.txt"
    trials.write_text("0.4\n-0.1\n\n0.9\n0.2\n")  # one annualized Sharpe per line
    before = _tree_digest(tmp_path / "investment"), _tree_digest(tmp_path / "investment_tech")

    result = compare(a, b, trials)

    ra = pd.read_csv(a / "backtest" / "daily.csv", parse_dates=["session"]).set_index("session")
    rb = pd.read_csv(b / "backtest" / "daily.csv", parse_dates=["session"]).set_index("session")
    paired = result["paired"]
    assert paired == sharpe_difference_test(ra["ret"], rb["ret"])
    assert paired["n"] == len(ra) == len(rb)
    assert paired["sharpe_1"] == pytest.approx(sharpe(ra["ret"]), rel=1e-12)
    assert paired["sharpe_2"] == pytest.approx(sharpe(rb["ret"]), rel=1e-12)
    # the trials are annualized Sharpes; the deflation works per period
    root = math.sqrt(252)
    per_period = np.array([0.4, -0.1, 0.9, 0.2]) / root
    r = ra["ret"].to_numpy()
    z = (r - r.mean()) / r.std()
    sr_a, skew, kurt = float(r.mean() / r.std()), float((z**3).mean()), float((z**4).mean())
    expected = deflated_sharpe(sr_a, 4, float(per_period.std(ddof=1)), len(r), skew, kurt)
    assert result["deflated"]["dsr"] == pytest.approx(expected["dsr"], rel=1e-9)
    assert result["deflated"]["n_trials"] == 4
    assert result["deflated"]["expected_max_sharpe"] == pytest.approx(
        expected["expected_max_sr"] * root, rel=1e-9
    )
    assert compare(a, b)["deflated"] is None  # no trials file, no deflation
    one = tmp_path / "one.txt"
    one.write_text("0.7\n")  # a single trial: nothing was selected, so nothing to deflate by
    alone = compare(a, b, one)["deflated"]
    assert alone["n_trials"] == 1 and alone["expected_max_sharpe"] == 0.0
    assert alone["dsr"] == pytest.approx(
        deflated_sharpe(sr_a, 1, 0.0, len(r), skew, kurt)["dsr"], rel=1e-9
    )

    main(["--run", str(a), "--run", str(b), "--trials-sharpes", str(trials)])
    printed = capsys.readouterr().out
    assert "z" in printed and "p_two_sided" in printed and "dsr" in printed
    main(["--run", str(a), "--run", str(b)])
    assert "dsr" not in capsys.readouterr().out

    after = _tree_digest(tmp_path / "investment"), _tree_digest(tmp_path / "investment_tech")
    assert after == before  # the comparison writes nothing into a run


def test_compare_needs_two_backtested_runs(two_runs, tmp_path):
    a, _ = two_runs
    (tmp_path / "bare" / "backtest").mkdir(parents=True)
    for second in (None, tmp_path / "missing", tmp_path / "bare"):  # not given, no dir, no file
        argv = ["--run", str(a)] + ([] if second is None else ["--run", str(second)])
        with pytest.raises(SystemExit) as refused:
            main(argv)
        assert refused.value.code == 2
