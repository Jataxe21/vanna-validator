# Phase 1: Vanna Proxy Validator

This repository runs a first-pass analysis of your 0DTE SPX short iron butterfly trades using the resources currently available:

```text
data/trades.csv      # completed 0DTE SPX iron butterfly trades
data/spx_daily.csv   # daily SPX OHLCV history
```

## Important limitation

`data/spx_daily.csv` is **daily SPX price history**, not historical SPX 0DTE option-chain data.

So this repo does **not** calculate true strike-level vanna, dealer GEX, charm, IV skew, or open-interest exposure yet.

Instead, it performs a useful **proxy vanna / volatility-regime analysis** using:

- daily SPX range,
- 5-day realized volatility,
- gap/range stress,
- SPX movement during each trade,
- trade P/L and win/loss outcome.

The practical question is:

> Do losses cluster on volatile/range-expansion days enough to justify deeper true options-chain vanna/GEX work?

## Run it on GitHub

No local setup required.

1. Open this repository on GitHub.
2. Click the **Actions** tab.
3. Select **Run Vanna Proxy Analysis**.
4. Click **Run workflow**.
5. Wait for the workflow to finish.
6. Check the generated `outputs/` files or download the workflow artifact.

## Generated outputs

```text
outputs/VANNA_PROXY_REPORT.txt      # executive report
outputs/trade_proxy_analysis.csv    # trade-by-trade enriched data
outputs/environment_summary.csv     # win rate/P&L by proxy environment
outputs/spx_proxy_regimes.csv       # SPX daily proxy regime features
outputs/vanna_proxy_summary.png     # 4-panel chart
```

## Interpretation

Use this as a screening test, not final proof.

- If 5+ of 9 losses cluster in HIGH proxy-vanna / high-volatility regimes, then deeper true vanna/GEX analysis is worth pursuing.
- If losses do not cluster, then the current strategy may not need a vanna filter yet.

For true institutional-grade vanna/GEX analysis, the next dataset needs historical SPX option-chain records with:

```text
date, timestamp, expiry, dte, strike, call_put,
bid, ask, last, volume, open_interest,
implied_volatility, delta, gamma, vega, theta
```
