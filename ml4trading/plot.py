"""Account value over time, one line per backtested run.

    python -m ml4trading.plot --run runs/investment="Buy-and-hold VOO" \
        --run runs/momentum="Weekly momentum" --out equity.png

Each run's ``backtest/daily.csv`` (the account valued every trading day) is drawn as its value
at each day's end, starting from the capital it was given. Every line is labeled at its end
with its final value, so identity never rests on color alone; a dashed line marks the starting
capital. Needs the ``plot`` extra (matplotlib).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

#: Categorical series colors, assigned in this fixed order (validated for color-vision
#: deficiency on a light surface; never cycled past its length).
SERIES_COLORS = [
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#4a3aa7",
    "#e34948",
]
TEXT_PRIMARY, TEXT_SECONDARY, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def account_values(run_dir: Path) -> pd.Series:
    """The run's account value per day: the starting capital on the first session, then the
    value at the end of every session."""
    daily = pd.read_csv(Path(run_dir) / "backtest" / "daily.csv", parse_dates=["session"])
    start = pd.Series([daily["equity_before"].iloc[0]], index=[daily["session"].iloc[0]])
    end = daily.set_index("session")["equity"]
    end.index = list(daily["session"].iloc[1:]) + [daily["session"].iloc[-1] + pd.Timedelta(days=1)]
    return pd.concat([start, end])


def _spread_labels(values: list[float], min_gap: float) -> list[float]:
    """Label heights near ``values``, pushed apart to at least ``min_gap`` (bottom-up)."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    placed = [0.0] * len(values)
    floor = float("-inf")
    for i in order:
        placed[i] = max(values[i], floor + min_gap)
        floor = placed[i]
    return placed


def plot_runs(runs: list[tuple[Path, str]], out: Path, title: str | None = None) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    if len(runs) > len(SERIES_COLORS):
        raise ValueError(f"at most {len(SERIES_COLORS)} runs per chart")
    series = [(label, account_values(path)) for path, label in runs]
    capital = series[0][1].iloc[0]

    fig, ax = plt.subplots(figsize=(12, 6.75), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for (label, values), color in zip(series, SERIES_COLORS, strict=False):
        ax.plot(values.index, values.to_numpy(), color=color, linewidth=2.0, label=label)
    ax.axhline(capital, color=TEXT_SECONDARY, linewidth=1.0, linestyle=(0, (4, 3)))
    ax.text(
        max(v.index[-1] for _, v in series),
        capital,
        f"Start ${capital:,.0f} ",
        color=TEXT_SECONDARY,
        fontsize=9,
        va="bottom",
        ha="right",
        bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1.5},
    )

    top = max(v.max() for _, v in series)
    ax.set_ylim(0, top * 1.08)
    ends = [float(v.iloc[-1]) for _, v in series]
    heights = _spread_labels(ends, min_gap=top * 0.045)
    last_day = max(v.index[-1] for _, v in series)
    for (label, values), color, y in zip(series, SERIES_COLORS, heights, strict=False):
        ax.plot(
            [values.index[-1]],
            [values.iloc[-1]],
            marker="o",
            markersize=5,
            color=color,
            markeredgecolor=SURFACE,
            markeredgewidth=1.5,
        )
        ax.annotate(
            f"{label}  ${values.iloc[-1]:,.0f}",
            xy=(values.index[-1], values.iloc[-1]),
            xytext=(last_day + pd.Timedelta(days=25), y),
            textcoords="data",
            fontsize=9,
            color=TEXT_PRIMARY,
            va="center",
            ha="left",
            arrowprops={"arrowstyle": "-", "color": color, "linewidth": 1.0},
        )

    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v / 1000:,.0f}k"))
    ax.set_ylabel("Account value", color=TEXT_SECONDARY)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    legend = ax.legend(loc="upper left", frameon=False, fontsize=9, ncol=2)
    for text in legend.get_texts():
        text.set_color(TEXT_PRIMARY)
    if title:
        ax.set_title(title, loc="left", fontsize=12, color=TEXT_PRIMARY, pad=12)
    fig.subplots_adjust(left=0.07, right=0.78, top=0.9, bottom=0.08)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m ml4trading.plot")
    ap.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="DIR=LABEL",
        help="a backtested run directory and its legend label",
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--title")
    args = ap.parse_args()
    runs = []
    for item in args.run:
        path, _, label = item.partition("=")
        runs.append((Path(path), label or Path(path).name))
    print(plot_runs(runs, args.out, args.title))


if __name__ == "__main__":
    main()
