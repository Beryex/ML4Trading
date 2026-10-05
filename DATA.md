# Dataset card

The repository ships **no data**. The dataset is distributed separately to course staff and
team members (ask the maintainer for access) and is placed under `data/` (or any directory
named by `ML4T_DATA_DIR` / `--data-dir`). Do not commit or redistribute it: it is derived from
licensed market data.

## Contents

| | |
|---|---|
| Symbols | 502: the 500 largest companies by market capitalization with a US-tradable listing (ADRs included), plus the ETFs VOO and QQQ |
| Window | 2016-01-04 to 2026-08-31, 2,680 sessions |
| Bars | 30-minute OHLCV, regular trading hours only (09:30–16:00 ET; 13:00 on half days) |
| Prices | split-adjusted |
| Dividends | one cash-dividend return per (symbol, ex-date), as a fraction of the prior close |
| Size | ~420 MB, ~16.2 million bars |

## Layout

```
data/
  bars_30min/<SYMBOL>.parquet   ts, open, high, low, close, volume
  dividends.parquet             symbol, ex_date, div_ret
  universe.json                 as_of, symbols (market-cap order), etfs, market_cap_usd
  MANIFEST.json                 sha256 and row count of every file
```

`ts` is the bar's **start**, timezone-aware `America/New_York`. Check a copy with
`python -m ml4trading.data verify`.

## How it was built

1-minute regular-session bars from a commercial market-data vendor were cleaned and aggregated
to 30 minutes (first open, max high, min low, last close, summed volume). Cleaning removed
duplicate minutes, minutes with no trade, and prints more than 4x away from a rolling median
close. Half days have 7 bars, full days 13.

## Known limitations — read before drawing conclusions

- **Selection with hindsight.** The 500 names are ranked by market capitalization on
  2026-09-22, *after* the end of the window, and that one list is used for the whole decade:
  every name is a company that survived and grew into the top 500. Results on this universe are
  biased upward relative to what a 2016 investor could have chosen.
- **Late listings.** 74 symbols start trading after 2016-01-04 (IPOs, spin-offs, listings) and
  5 more have gaps; 423 have the full history.
- **One large company is missing:** Berkshire Hathaway (its class-B ticker could not be mapped
  when the universe was built).
- **One corrected scale break.** CRWD's bars from 2021-06-24 to 2026-06-22 came from the vendor
  on the pre-split scale of its 4-for-1 split of 2026-06-23, while the bars around them were
  split-adjusted; their prices are divided by 4 and volumes multiplied by 4 (verified against a
  later, consistently adjusted download). `MANIFEST.json` records the correction.
- **Unverified suspect series.** A scan for overnight moves larger than 1.5x flags three series we
  could not check against a second source: HDB (prices double and volume halves on 2020-12-31, the
  signature of a split-scale break), WBD (levels inconsistent between 2018 and 2022), and MRNA
  (an opening price of $35.88 on 2020-02-27, likely a bad print). None of the baselines ever
  holds HDB or WBD, or MRNA around that date; treat these series with care in learned models.
- **No closing auction.** The last bar's close is the last regular trade before 16:00, not the
  official closing-auction price.
- **Noisy dividends.** The dividend series was estimated from adjusted vs. unadjusted daily
  closes and contains spurious small entries on low-priced or heavily split-adjusted histories
  (e.g. NVDA in 2016: 108 entries summing to +50 %). It is used as-is by the overnight labels;
  treat dividend-sensitive results with care.
