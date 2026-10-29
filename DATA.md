# Dataset card

The repository ships **no data**. The dataset is distributed separately to course staff and
team members (ask the maintainer for access) and is placed under `data/` (or any directory
named by `ML4T_DATA_DIR` / `--data-dir`). Do not commit or redistribute it: it is derived from
licensed market data.

## Contents

| | |
|---|---|
| Symbols | 552: the 500 largest companies by market capitalization with a US-tradable listing (ADRs included), 50 former top-500 companies that had left the top 500 by the end of the window (the *drop-outs*, below), and the ETFs VOO and QQQ |
| Window | 2016-01-04 to 2026-08-31, 2,680 sessions |
| Bars | 30-minute OHLCV, regular trading hours only (09:30–16:00 ET; 13:00 on half days) |
| Prices | split-adjusted |
| Dividends | one cash-dividend return per (symbol, ex-date), as a fraction of the prior close |
| Size | ~480 MB, ~17.8 million bars |

## Layout

```
data/
  bars_30min/<SYMBOL>.parquet   ts, open, high, low, close, volume
  dividends.parquet             symbol, ex_date, div_ret
  universe.json                 as_of, symbols, etfs, market_cap_usd, dropouts, dropouts_market_cap_as_of
  MANIFEST.json                 sha256 and row count of every file
```

`ts` is the bar's **start**, timezone-aware `America/New_York`. Check a copy with
`python -m ml4trading.data verify`.

`universe.json`'s `symbols` lists the 500 (market-cap order on `as_of`) followed by the 50
drop-outs (market-cap order on `dropouts_market_cap_as_of`); `dropouts` names the 50 on their own;
`market_cap_usd` covers all 550.

## How it was built

1-minute regular-session bars from a commercial market-data vendor were cleaned and aggregated
to 30 minutes (first open, max high, min low, last close, summed volume). Cleaning removed
duplicate minutes, minutes with no trade, and prints more than 4x away from a rolling median
close. Half days have 7 bars, full days 13.

### The 50 drop-outs

Added to soften the selection-with-hindsight bias below. Chosen from monthly market-cap top-500
snapshots of 2016-01 .. 2026-08, rebuilt from archived public rankings (the US exchanges' company
lists through 2020, a global market-cap ranking from 2021; on the earlier, exchange-only lists a
company counts as top 500 at rank 392 or better, the share of exchange-listed names in the later
ranking's top 500). Of the companies that ranked in the top 500 in some month but not at the end
of the window and still trade today, the 50 with the longest time in the top 500 were taken --
counting only the months the vendor serves data for their current listing.

Their bars are the same vendor's own 30-minute regular-session bars rather than aggregates of
cleaned 1-minute bars; bars with zero volume (no trade) are dropped. On a one-month comparison
over two symbols of the 500, the vendor's 30-minute bars matched the aggregated ones on every
open, high, low, close and volume of 273 bars for one symbol, and on all but one or two bars
(within 0.1 %) for the other. Dividends are derived as for the 500.

## Known limitations — read before drawing conclusions

- **Selection with hindsight.** The 500 names are ranked by market capitalization on
  2026-09-22, *after* the end of the window, and that one list is used for the whole decade:
  every name is a company that survived and grew into the top 500. Results on this universe are
  biased upward relative to what a 2016 investor could have chosen. The 50 drop-outs only soften
  this: about 200 still-listed companies left the top 500 during the window and 50 of them are
  included; companies that stopped trading (acquired, taken private, failed) are absent
  altogether; and nothing restricts a fold to the names that were in the top 500 at its own
  dates -- all 550 are candidates in every fold.
- **Late listings.** 74 of the 500 start trading after 2016-01-04 (IPOs, spin-offs, listings) and
  5 more have gaps; 423 have the full history. Of the 50 drop-outs, 8 start after 2016-01-04
  (DD and DOW at their current listings, 2017-09 and 2019-03: their predecessors' earlier
  years are not included) and FTV and VIV miss 3 and 1 sessions.
- **One large company is missing:** Berkshire Hathaway (its class-B ticker could not be mapped
  when the universe was built).
- **Three corrected scale breaks.** CRWD's bars from 2021-06-24 to 2026-06-22 were stored on the
  pre-split scale of its 4-for-1 split of 2026-06-23, while the bars around them were
  split-adjusted; their prices are divided by 4 and volumes multiplied by 4 (verified against a
  later, consistently adjusted download). Two drop-outs were served without adjustment for a
  share-ratio change: TLK before 2016-10-26 (an ADR-ratio change) and WIT before 2017-06-22 (a
  1:1 bonus issue); their prices are divided by 2 and volumes multiplied by 2 (verified against a
  second, independent 30-minute source that is continuous across both dates). `MANIFEST.json`
  records all three.
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
  treat dividend-sensitive results with care. Three drop-outs, BCE, TU and RCI, declare their
  dividends in Canadian dollars and the vendor's adjusted series priced them in CAD against the
  USD close; their entries are converted to USD at the prior day's exchange rate.
