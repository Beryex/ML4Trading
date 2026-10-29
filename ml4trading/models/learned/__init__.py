"""Models fitted per fold on ``ml4trading.features``: ``linear`` (an elastic net) and
``nonlinear`` (LightGBM), over the shared ``LearnedModel``. Importing the package registers
both."""

from ml4trading.models.learned import linear, nonlinear  # noqa: F401
