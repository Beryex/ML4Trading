"""Daily features of every stock, computed on the 30-minute bars: ML4Investment's feature set.

Adapted from ML4Investment (https://github.com/Beryex/ML4Investment,
``ml4investment/utils/feature_calculating.py`` at commit 3650552; MIT License, Copyright (c)
2025 Boyao Wang). The computations are ML4Investment's own; the adaptations are:

- the daily aggregation groups a symbol's bars by New York session date (ML4Investment resamples
  on a business-day calendar and forward-fills a day the symbol has no bars on; here that day has
  no row);
- no target column, no ``sector_id`` and no live incomplete-day drop; ``stock_id`` is the
  symbol's position in the sorted pool, a categorical code;
- the ETF columns come from VOO and QQQ, the dataset's two ETFs (ML4Investment embeds eight);
- every column that a stock's price or volume scale moves is dropped (``SCALE_DEPENDENT``): the
  bars are split-adjusted back from the end of the dataset, so a price or volume LEVEL in the
  past reveals which companies split later -- that is, which grew;
- each row is shifted one session: row (d, s) holds what was known at session d's open;
- two pandas 3 renames (``DatetimeIndex.day_of_week`` / ``day_of_year`` for the deprecated
  ``dayofweek`` / ``dayofyear``), the same values.

Public API:

- ``build_features(bars, pool, workers=None)`` -- the pool's float32 feature matrix, indexed by
  (session, symbol);
- ``symbol_daily_features(bars)`` -- one symbol's unshifted daily features;
- ``load_feature_set(name)`` -- the columns of a named subset (``feature_sets/<name>.txt``), or
  None for ``"all"``;
- ``slow_features(bars, pool)`` -- the pool's slow cross-sectional features from the published
  literature (``features/slow.py``): a float32 frame indexed by (session, symbol) like
  ``build_features``, one column per name in ``SLOW_FEATURES``;
- ``CALENDAR``, ``FEATURE_ETFS``, ``SCALE_DEPENDENT``.
"""

from ml4trading.features.build import (
    CALENDAR,
    FEATURE_ETFS,
    SCALE_DEPENDENT,
    build_features,
    load_feature_set,
)
from ml4trading.features.daily import symbol_daily_features
from ml4trading.features.slow import SLOW_FEATURES, slow_features

__all__ = [
    "CALENDAR",
    "FEATURE_ETFS",
    "SCALE_DEPENDENT",
    "SLOW_FEATURES",
    "build_features",
    "load_feature_set",
    "slow_features",
    "symbol_daily_features",
]
