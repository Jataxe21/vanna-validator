# 0DTE SPX Short Iron Butterfly — Loss Attribution & Tradable Filter Analysis

## Central finding

Losses are **NOT** explained by a "proxy-vanna" / volatility-regime effect.
They **ARE** explained by adverse underlying **movement** and **distance** of
price from the body strike — measured **at exit**.

Key correlations from the current 39-trade sample:

| Metric | Correlation with P/L |
|--------|----------------------|
| Absolute SPX move during trade (exit-measured) | −0.55 |
| Absolute exit distance from center strike | −0.75 |

Wins moved ~7 pts on average; losses moved ~16 pts. Trades exiting within ~15
pts of the body almost never lose.

---

## Critical distinction: biased diagnostics vs. tradable counterfactuals

This repository maintains a strict separation between two types of analysis:

| Type | What it uses | Is it tradable? | Where to find it |
|------|-------------|-----------------|-----------------|
| **Post-hoc diagnostic** | Exit price, exit distance, exit-measured movement | **NO** — look-ahead bias | `MOVEMENT_PRESSURE_REPORT.txt` |
| **Tradable counterfactual** | Entry time, prior-day range, gap, intraday bars up to breach | **YES** | `ENTRY_FILTER_REPORT.txt`, `INTRADAY_STOP_SIMULATION.txt` |

> **Why the distinction matters:** `underlying_move_abs_pts` = abs(underlyingClose −
> underlyingOpen) is measured at the **actual exit** — it is an *outcome*, not
> something you can observe at entry. Flagging trades where this exceeds a
> threshold is a circular re-description of the losers, not predictive skill.
> Any "P/L if skipped" figure derived from exit movement is labeled
> `POST_HOC_BIASED` in the reports and CSVs, and should not be automated.

---

## Data files

```text
data/trades.csv          # completed 0DTE SPX iron butterfly trades
data/spx_daily.csv       # daily SPX OHLCV history
data/spx_intraday.csv    # optional — 1-minute SPX bars; enables first-breach stop simulation
```

### `data/spx_intraday.csv` schema (optional)

| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | datetime | Bar timestamp, e.g. `2026-05-04 10:55:00` (ET assumed) |
| `open` | float | Bar open price |
| `high` | float | Bar high price |
| `low` | float | Bar low price |
| `close` | float | Bar close price |

Frequency: 1-minute bars recommended. ES futures acceptable as a proxy (document source).
Coverage must span all trade windows in `data/trades.csv`.

If the file is **absent**: the analysis runs normally, skips the intraday
simulation, and writes a stub note in `outputs/INTRADAY_STOP_SIMULATION.txt`.
If the file is **present**: the first-breach stop simulation activates
automatically with no other changes needed.

---

## Run it on GitHub

No local setup required.

1. Open this repository on GitHub.
2. Click the **Actions** tab.
3. Select **Run Vanna Proxy Analysis**.
4. Click **Run workflow**.
5. Wait for the workflow to finish.
6. Check the generated `outputs/` files or download the workflow artifact.

---

## Generated outputs

```text
outputs/VANNA_PROXY_REPORT.txt              # proxy-vanna regime analysis
outputs/MOVEMENT_PRESSURE_REPORT.txt        # POST-HOC exit-distance diagnostic (biased — labeled)
outputs/ENTRY_FILTER_REPORT.txt             # TRADABLE entry-time & range/gap filters
outputs/INTRADAY_STOP_SIMULATION.txt        # TRADABLE first-breach stop (requires intraday data)
outputs/trade_proxy_analysis.csv            # trade-by-trade enriched data
outputs/environment_summary.csv            # win rate/P&L by proxy environment
outputs/spx_proxy_regimes.csv              # SPX daily proxy regime features
outputs/movement_threshold_analysis.csv    # threshold table (exit data, biased columns labeled)
outputs/center_strike_distance_analysis.csv # exit distance buckets (exit data)
outputs/entry_filter_time_of_day.csv       # time-of-day filter results
outputs/entry_filter_range_gap.csv         # prior-day range/gap filter results
outputs/entry_filter_trade_features.csv    # per-trade entry-safe features
outputs/intraday_stop_simulation_trades.csv  # per-trade breach detail (if intraday data present)
outputs/intraday_stop_simulation_summary.csv # aggregate stop results by distance
outputs/vanna_proxy_summary.png            # 4-panel chart
outputs/movement_pressure_summary.png      # movement/distance diagnostic charts
```

---

## Analysis modules

| Module | Purpose |
|--------|---------|
| `analysis/vanna_analysis.py` | Main entry point; proxy-vanna & exit-diagnostic analysis |
| `analysis/entry_filters.py` | Tradable entry-time and prior-day range filters (zero look-ahead) |
| `analysis/intraday_sim.py` | Real-bot baseline intraday simulation (30% PT / 1.5x SL / -$1,500 cap / 15:15 timed exit) + alternative distance-stop tests (gated on `data/spx_intraday.csv`) |

Running `python analysis/vanna_analysis.py` triggers all three in sequence.

---

## Sample-size caution

39 trades (9 losses) is a small sample. Every result in this repository should
be treated as **triage / hypothesis generation**, not statistical proof.
Validate any promising filter on the next 50+ forward trades before acting on
it in production.

---

## Interpretation guide

### MOVEMENT_PRESSURE_REPORT.txt (post-hoc, biased)
- All metrics derive from exit data. The "threshold table" and
  `pnl_impact_if_skipped_POST_HOC_BIASED` column are diagnostic only.
- `post_hoc_separation_score (diagnostic only, biased)` replaces the old
  `kill_switch_score` to prevent misuse as a strategy signal.
- **Do not automate a hard stop or entry skip based on these numbers.**

### ENTRY_FILTER_REPORT.txt (tradable)
- Section 1: Time-of-day cutoffs (11:00 / 11:30 / 12:00 ET). Skipping early
  entries is an immediately actionable rule requiring no future data.
- Section 2: Prior-day range, open gap, entry-move-from-open filters.
  These are entry-time-safe because all inputs are observable before entry.
- Weak predictors on a 39-trade sample; forward-test before trusting.

### INTRADAY_STOP_SIMULATION.txt (tradable, requires intraday data)
- Baseline models the real bot exits: 30% profit target, 1.5x-credit stop,
  -$1,500 hard cap, and unconditional 15:15 ET timed exit.
- Alternative distance stops [8, 10, 12, 15] are tested against that real
  baseline (not against a no-stop assumption).
- If `delta_vs_simulated_baseline > 0`: variant helped on this sample.
- If `delta_vs_simulated_baseline < 0`: whipsaw warning — variant hurt overall.
- Intraday option marks are unavailable; mark and P/L are inferred from
  underlying distance to body strike, so results are directional.

---

## Limitations

- No historical option-chain Greeks, IV, OI, volume, or bid/ask data.
  Proxy-vanna is a volatility/range-regime proxy, not true strike-level vanna.
- Daily OHLCV only (without intraday file). No intraday path simulation possible.
- 39-trade sample: results are triage, not proof.
- For true institutional-grade vanna/GEX analysis, collect historical SPX chains
  with: `date, timestamp, expiry, dte, strike, call_put, bid, ask, last, volume,
  open_interest, implied_volatility, delta, gamma, vega, theta`.
