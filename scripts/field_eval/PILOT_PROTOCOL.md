# Field pilot protocol

Fixed on 2026-11-18, before the first trading day of the pilot.

## Question

Does receiving the daily recommendations change a participant's paper-trading return, compared
with trading on their own over the same days?

## Participants and groups

- Six participants: the four team members and two outside participants with different risk
  appetite and familiarity with finance.
- `assign_groups.py` (seed 20261118) splits each stratum (team, outside) evenly between two groups:
  - **Recommendation group**: receives the recommended portfolio (frozen v29, long and short, up to
    10 stocks) by email every evening and decides, stock by stock, whether to follow it.
  - **Control group**: receives no recommendations and trades on their own.
- Everyone starts with the same $20,000 virtual account in cash. Shorting is allowed.

## Daily timeline (US Eastern)

1. After the close: refresh the data (`extend_data.py`), produce the next session's portfolio
   (`recommend.py`) and email it to the recommendation group. The control group gets a reminder
   email with the decision form link only.
2. By 09:25, every participant records in the decision form the position they want to hold after
   the open: side (buy, short or flat) and number of shares for each stock they change. The
   recommendation group answers for every recommended stock (follow, skip or change); a stock that
   drops out of the recommended portfolio means "go flat". Stocks a participant does not mention
   keep yesterday's position. Answers submitted after 09:30 are left out of the results and
   counted as deviations.
3. Every order fills at that session's 09:30 opening price (the first 30-minute bar's open).
   Positions are held until the participant changes them; there is no intraday trading.
4. Accounts are valued at every open (open-to-open returns, as in the backtest) and, for partial
   reports, also at the close.

## Costs

The backtest's: a 4 bp quoted spread (entries pay none, exits half), $0.0035 per share with a
$0.35 minimum per order, and 1% a year to borrow shares that are shorted.

## Outcomes

- Primary (pilot version): the account's net return over the pilot minus VOO's over the same days.
- Secondary: follow rate (share of recommended stocks held as recommended, with partial follows
  counted separately); the gap between a recommendation-group account and the account that
  follows every recommendation; number of trades and costs; maximum drawdown; decision time;
  the 1-5 confidence given with each decision; an end-of-pilot survey (trust, would use it, why
  recommendations were skipped).
- Process: whether the portfolio was delivered on time each day, completeness of the decision
  form, deviations from this protocol.

## Benchmarks we compute ourselves

- The account that follows every recommendation exactly, with the same fills and costs.
- VOO bought and held over the same days.

## Analysis

Results are reported per participant and per group. With three people per group and a few days,
no comparison has enough power; the pilot checks that every step works end to end and measures
how closely people follow the recommendations, how fast they decide and what data problems appear.
