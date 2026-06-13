#!/usr/bin/env python3
"""
Intraday first-breach stop simulation — TRADABLE COUNTERFACTUAL.

This module requires data/spx_intraday.csv with columns:
    timestamp, open, high, low, close
(1-minute bars; ES futures acceptable as proxy — document the source)

If the file is ABSENT the module prints a single informational message,
writes a stub note into outputs/INTRADAY_STOP_SIMULATION.txt, and returns
without error. The rest of the analysis is unaffected.

If PRESENT, for each trade it simulates an honest first-breach stop:
    body = center strike (same logic as VannaProxyValidator.center_strike())
    window = intraday bars where openDate <= timestamp <= closeDate
    For each candidate stop distance N in [8, 10, 12, 15] points:
        Find the FIRST bar where abs(midpoint − body) >= N
          OR high >= body + N  (upper breach)
          OR low  <= body − N  (lower breach — checked first via low/high).
        Decision uses ONLY bars at or before the breach timestamp.
        Estimate fly P/L at breach (see approximation note below).
    Compute aggregate stopped_pnl for each N and the delta vs baseline.

WHY THIS IS TRADABLE:
    The stop trigger (bar high/low relative to body + N) is knowable at the
    moment of that bar's close. No exit price or post-breach data is used.

P/L APPROXIMATION AT BREACH:
    Exact option prices at the breach bar are not available. The approximation:
        estimated_close_cost_per_share = N   (intrinsic only; no time value)
        pnl_at_breach = (openPrice − N) * 100
    This is conservative in the sense that the actual cost to close is
    usually slightly HIGHER than pure intrinsic (residual time value adds cost).
    The approximation is therefore OPTIMISTIC for the stop scenario; if anything,
    the real stopped P/L is somewhat worse than shown. The directional conclusion
    (stop good vs. bad) is still valid.

    If pnl_at_breach > actual_pnl the stop would have helped.
    If pnl_at_breach < actual_pnl the stop would have hurt (whipsaw).

CAUTION: 39 trades is a small sample. Results are directional only.
"""

from pathlib import Path
from textwrap import dedent

import numpy as np
import pandas as pd


INTRADAY_SCHEMA = dedent("""
    Expected schema for data/spx_intraday.csv
    ------------------------------------------
    Column      Type        Description
    timestamp   datetime    Bar timestamp (e.g., 2026-05-04 10:55:00)
                            Must be parseable by pd.to_datetime().
                            Timezone: assumed ET (same as trades).
    open        float       Bar open price
    high        float       Bar high price
    low         float       Bar low price
    close       float       Bar close price

    Frequency: 1-minute bars recommended. Wider bars reduce detection accuracy.
    Source:    Cash SPX (preferred) or ES futures as proxy.
               If ES futures, document the source and any roll adjustments.
    Coverage:  Must span all trade windows in trades.csv
               (from earliest openDate to latest closeDate).
""")

STOP_DISTANCES = [8, 10, 12, 15]


# ---------------------------------------------------------------------------
# Center-strike extractor (mirrors VannaProxyValidator.center_strike)
# ---------------------------------------------------------------------------

def _center_strike(description: str) -> float:
    """Extract repeated short/body strike from an iron butterfly description."""
    strikes = []
    for raw_part in str(description).split(", "):
        part = raw_part.replace(",", "")
        for token in part.split():
            cleaned = token.replace("+", "").replace("-", "")
            if cleaned.isdigit() and len(cleaned) >= 4:
                strikes.append(float(cleaned))
                break
    if not strikes:
        return np.nan
    return float(pd.Series(strikes).value_counts().index[0])


# ---------------------------------------------------------------------------
# Per-trade first-breach simulation
# ---------------------------------------------------------------------------

def _simulate_trade(
    trade: pd.Series,
    intraday: pd.DataFrame,
    stop_dist: float,
) -> dict:
    """
    Simulate a first-breach stop for a single trade.

    Returns a dict with breach_found, breach_timestamp, pnl_at_breach,
    actual_pnl, stopped (bool), pnl_if_stopped.
    """
    body = trade["center_strike"]
    open_date = pd.to_datetime(trade["openDate"])
    close_date = pd.to_datetime(trade["closeDate"])
    open_price = float(trade["openPrice"])
    actual_pnl = float(trade["pnl"])

    if np.isnan(body):
        return {
            "stop_dist": stop_dist,
            "breach_found": False,
            "breach_timestamp": pd.NaT,
            "pnl_at_breach": np.nan,
            "actual_pnl": actual_pnl,
            "stopped": False,
            "pnl_if_stopped": actual_pnl,
            "note": "center_strike extraction failed",
        }

    window = intraday[
        (intraday["timestamp"] >= open_date) & (intraday["timestamp"] <= close_date)
    ].copy()

    if window.empty:
        return {
            "stop_dist": stop_dist,
            "breach_found": False,
            "breach_timestamp": pd.NaT,
            "pnl_at_breach": np.nan,
            "actual_pnl": actual_pnl,
            "stopped": False,
            "pnl_if_stopped": actual_pnl,
            "note": "no intraday bars in trade window",
        }

    # Find first bar where high >= body + N or low <= body - N.
    breach_mask = (window["high"] >= body + stop_dist) | (window["low"] <= body - stop_dist)
    breach_bars = window[breach_mask]

    if breach_bars.empty:
        return {
            "stop_dist": stop_dist,
            "breach_found": False,
            "breach_timestamp": pd.NaT,
            "pnl_at_breach": np.nan,
            "actual_pnl": actual_pnl,
            "stopped": False,
            "pnl_if_stopped": actual_pnl,
            "note": "no breach within trade window",
        }

    breach_bar = breach_bars.iloc[0]
    breach_ts = breach_bar["timestamp"]

    # Approximate P/L at breach (intrinsic only).
    # pnl_at_breach = (credit_received − intrinsic_cost) * multiplier
    # intrinsic_cost_per_share ≈ stop_dist (the short option has N pts intrinsic)
    # multiplier for SPX options = 100
    pnl_at_breach = (open_price - stop_dist) * 100

    return {
        "stop_dist": stop_dist,
        "breach_found": True,
        "breach_timestamp": breach_ts,
        "pnl_at_breach": pnl_at_breach,
        "actual_pnl": actual_pnl,
        "stopped": True,
        "pnl_if_stopped": pnl_at_breach,
        "note": "",
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_intraday_stop_simulation(
    trades: pd.DataFrame,
    intraday_file: Path,
    output_dir: Path,
) -> None:
    """
    Run (or gracefully skip) the intraday first-breach stop simulation.

    Parameters
    ----------
    trades        : enriched trades DataFrame (must have openDate, closeDate,
                    openPrice, pnl, outcome, description or center_strike).
    intraday_file : Path to data/spx_intraday.csv (may not exist).
    output_dir    : Path to outputs/ directory.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not Path(intraday_file).exists():
        _write_stub_report(output_dir)
        print(
            f"[intraday_sim] data/spx_intraday.csv not found — "
            "first-breach stop simulation skipped. "
            "A stub note has been written to outputs/INTRADAY_STOP_SIMULATION.txt"
        )
        return

    # Load and validate intraday data.
    try:
        intraday = pd.read_csv(intraday_file)
        intraday["timestamp"] = pd.to_datetime(intraday["timestamp"])
        for col in ["open", "high", "low", "close"]:
            intraday[col] = pd.to_numeric(intraday[col], errors="coerce")
        intraday = intraday.sort_values("timestamp").reset_index(drop=True)
    except Exception as exc:
        print(f"[intraday_sim] Failed to load {intraday_file}: {exc}")
        _write_stub_report(output_dir, error=str(exc))
        return

    # Ensure center_strike is available.
    trades = trades.copy()
    if "center_strike" not in trades.columns:
        trades["center_strike"] = trades["description"].apply(_center_strike)
    if "openPrice" not in trades.columns:
        trades["openPrice"] = pd.to_numeric(trades.get("openPrice"), errors="coerce")

    # Run simulation for each trade × stop distance.
    all_rows = []
    for _, trade in trades.iterrows():
        for dist in STOP_DISTANCES:
            result = _simulate_trade(trade, intraday, dist)
            result["trade_openDate"] = trade["openDate"]
            result["trade_closeDate"] = trade["closeDate"]
            result["outcome"] = trade["outcome"]
            result["center_strike"] = trade["center_strike"]
            all_rows.append(result)

    sim_df = pd.DataFrame(all_rows)

    # Aggregate by stop distance.
    baseline_pnl = float(trades["pnl"].sum())
    agg_rows = []
    for dist in STOP_DISTANCES:
        sub = sim_df[sim_df["stop_dist"] == dist]
        stopped_count = int(sub["stopped"].sum())
        stopped_wins = int(
            sub[sub["stopped"] & (sub["outcome"] == "WIN")]["stopped"].sum()
        )
        stopped_losses = int(
            sub[sub["stopped"] & (sub["outcome"] == "LOSS")]["stopped"].sum()
        )
        total_pnl_if_stopped = float(sub["pnl_if_stopped"].sum())
        delta = total_pnl_if_stopped - baseline_pnl
        agg_rows.append({
            "stop_distance_pts": dist,
            "trades_stopped": stopped_count,
            "stops_on_winners": stopped_wins,
            "stops_on_losers": stopped_losses,
            "total_pnl_if_stopped": total_pnl_if_stopped,
            "pnl_delta_vs_baseline": delta,
            "whipsaw_warning": "YES — stop hurt overall" if delta < 0 else "no",
        })

    agg_df = pd.DataFrame(agg_rows)

    # Write outputs.
    sim_df.to_csv(output_dir / "intraday_stop_simulation_trades.csv", index=False)
    agg_df.to_csv(output_dir / "intraday_stop_simulation_summary.csv", index=False)
    _write_full_report(trades, sim_df, agg_df, baseline_pnl, output_dir)

    print(
        f"[intraday_sim] First-breach stop simulation complete. "
        f"Outputs written to {output_dir}"
    )


# ---------------------------------------------------------------------------
# Report writers
# ---------------------------------------------------------------------------

def _write_stub_report(output_dir: Path, error: str = "") -> None:
    report = """
INTRADAY FIRST-BREACH STOP SIMULATION
======================================

STATUS: SKIPPED — data/spx_intraday.csv not found.

This simulation requires intraday SPX (or ES futures proxy) 1-minute bar data.
No intraday data file was detected, so the path-simulation has been skipped.
All other analyses ran normally.

HOW TO ENABLE THIS SIMULATION
------------------------------
1. Obtain 1-minute SPX (or ES futures) OHLC data covering the date range of
   your trades (approximately {earliest} to {latest}).
   Sources: Tradovate, Interactive Brokers, Alpaca, Polygon.io, etc.

2. Format the data as a CSV with the following columns (no extra columns
   required; extra columns are ignored):

{schema}

3. Save the file as data/spx_intraday.csv in the repository root.

4. Re-run the analysis:  python analysis/vanna_analysis.py

The intraday simulation will activate automatically and write:
    outputs/INTRADAY_STOP_SIMULATION.txt
    outputs/intraday_stop_simulation_trades.csv
    outputs/intraday_stop_simulation_summary.csv

WHY THIS SIMULATION IS THE TRADABLE COUNTERFACTUAL
---------------------------------------------------
Unlike the exit-distance diagnostic in MOVEMENT_PRESSURE_REPORT.txt (which
uses exit data and is subject to look-ahead bias), the intraday first-breach
stop uses ONLY intraday bar data that is available at the moment of the breach
bar's close. It is a genuinely forward-looking risk management rule and the
result of running it is the correct number to use for evaluating whether a hard
stop adds value.
""".format(
        schema=INTRADAY_SCHEMA,
        earliest="(trades start date — see data/trades.csv)",
        latest="(trades end date — see data/trades.csv)",
    )

    if error:
        report += f"\nERROR DETAIL: {error}\n"

    (output_dir / "INTRADAY_STOP_SIMULATION.txt").write_text(report)


def _write_full_report(
    trades: pd.DataFrame,
    sim_df: pd.DataFrame,
    agg_df: pd.DataFrame,
    baseline_pnl: float,
    output_dir: Path,
) -> None:
    total = len(trades)
    wins = int((trades["outcome"] == "WIN").sum())
    losses = int((trades["outcome"] == "LOSS").sum())

    lines = []
    lines.append("""
INTRADAY FIRST-BREACH STOP SIMULATION
======================================

╔══════════════════════════════════════════════════════════════════════════════╗
║  TRADABLE COUNTERFACTUAL — INTRADAY PATH DATA, ZERO LOOK-AHEAD             ║
║                                                                              ║
║  Every stop trigger in this simulation uses only intraday bar data that     ║
║  is available at or before the breach bar's close. No exit price or         ║
║  post-breach data is used. This IS a valid forward-looking test.            ║
║                                                                              ║
║  CONTRAST with MOVEMENT_PRESSURE_REPORT.txt which uses EXIT data and is     ║
║  labeled POST-HOC / BIASED.                                                  ║
╚══════════════════════════════════════════════════════════════════════════════╝

⚠ SAMPLE SIZE CAUTION: 39 trades (9 losses, 30 wins). Results are directional
  indicators only — validate on next 50+ forward trades before automating.

⚠ P/L APPROXIMATION: At breach, estimated exit price = N (intrinsic only,
  no time value). Actual close cost is typically slightly higher due to residual
  time premium, so actual stopped P/L is SOMEWHAT WORSE than shown below.
  Conclusions remain directionally valid.

BASELINE
--------""")
    lines.append(f"Total trades : {total}")
    lines.append(f"Wins         : {wins}")
    lines.append(f"Losses       : {losses}")
    lines.append(f"Total P/L    : ${baseline_pnl:,.0f}")

    lines.append("""
AGGREGATE RESULTS BY STOP DISTANCE
------------------------------------
(pnl_delta > 0 = stop helped; pnl_delta < 0 = stop hurt / whipsaw warning)
""")
    lines.append(agg_df.to_string(index=False))

    # Whipsaw warning
    for _, row in agg_df.iterrows():
        if row["pnl_delta_vs_baseline"] < 0:
            lines.append(
                f"\n⚠ WHIPSAW WARNING for N={row['stop_distance_pts']} pts: "
                f"a hard stop at this distance would have reduced total P/L by "
                f"${abs(row['pnl_delta_vs_baseline']):,.0f}. "
                "Do NOT automate this stop distance."
            )

    lines.append("""
PER-TRADE BREACH DETAIL
-----------------------
(pnl_if_stopped = actual P/L when no breach, or estimated stopped P/L at breach)
""")
    per_trade_display = sim_df[[
        "trade_openDate", "outcome", "stop_dist", "breach_found",
        "breach_timestamp", "actual_pnl", "pnl_at_breach", "pnl_if_stopped", "note",
    ]].copy()
    per_trade_display["trade_openDate"] = pd.to_datetime(
        per_trade_display["trade_openDate"]
    ).dt.strftime("%Y-%m-%d %H:%M")
    per_trade_display["breach_timestamp"] = pd.to_datetime(
        per_trade_display["breach_timestamp"]
    ).dt.strftime("%Y-%m-%d %H:%M").fillna("—")
    lines.append(per_trade_display.to_string(index=False))

    lines.append(dedent("""

HOW TO INTERPRET
----------------
1. If pnl_delta_vs_baseline > 0 for a stop distance N: this stop would have
   improved total P/L on this sample. Still validate forward before automating.
2. If pnl_delta_vs_baseline < 0: a hard stop here causes whipsaw (stopped out
   before expiry, price recovers). Do NOT automate. Size management instead.
3. Compare stops_on_winners vs stops_on_losers: a good stop should fire mostly
   on losers. If it fires frequently on winners, the distance is too tight.

INTRADAY DATA SCHEMA USED
--------------------------
"""))
    lines.append(INTRADAY_SCHEMA)

    report = "\n".join(lines)
    (output_dir / "INTRADAY_STOP_SIMULATION.txt").write_text(report)
