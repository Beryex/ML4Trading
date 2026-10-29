import subprocess
import sys

import pandas as pd
import pytest

from ml4trading import config
from ml4trading.pipeline import load_frames, load_panel
from ml4trading.train import train


def test_a_grid_on_a_frame_param_is_refused(data_dir, tmp_path):
    with pytest.raises(ValueError, match="frames are built from it"):
        train(
            "momentum",
            tmp_path / "run",
            data_dir,
            grid={"model.hold_overnight": [True, False]},
        )


@pytest.mark.parametrize("method", ["investment", "momentum"])
def test_frames_panel_equals_load_panel(data_dir, method):
    cfg = config.resolve(method)
    pool, frames = load_frames(cfg, data_dir)
    panel_pool, panel = load_panel(cfg, data_dir)
    assert pool == panel_pool
    pd.testing.assert_frame_equal(frames.panel, panel)
    assert frames.X is None


def test_importing_the_package_does_not_load_lightgbm():
    # lightgbm needs an OpenMP runtime some machines lack: only the nonlinear model may load it
    code = "import sys, ml4trading.train, ml4trading.backtest; print('lightgbm' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"
