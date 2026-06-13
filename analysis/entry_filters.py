#!/usr/bin/env python3
"""
Tradable entry-filter counterfactuals — ZERO LOOK-AHEAD BIAS.

Every filter in this module uses ONLY information available AT OR BEFORE trade
entry time:

  SAFE to use at entry:
    openDate time component     — you know when you are entering the trade
    underlyingOpen              — the SPX price at the moment of entry
    daily Open (spx_daily)      — market open occurs at 09:30 ET, before all
                                   trades in this sample (earliest entry 10:55)
    prior-day High/Low/Close    — fully settled before today's market open

  NOT safe to use as an entry filter (would require knowing the future):
    underlyingClose             — exit price: future at entry time
    daily High / Low / Close    — end-of-day or intraday future values
    underlying_move_abs_pts     — measured at exit: future at entry time
    abs_exit_distance_from_center — measured at exit: future at entry time

The post-hoc diagnostics using exit data are kept in MOVEMENT_PRESSURE_REPORT.txt
with explicit look-ahead bias labels. This file contains only the tradable
counterfactuals, clearly separated.

CAUTION: 39 trades is a small sample. Every result below should be treated
as triage / hypothesis generation, not statistical proof. Any filter found
here should be validated on the NEXT 50+ forward trades before acting on it.
"""

from pathlib import Path
from textwrap import dedent

import numpy as np
import pandas as pd


# Default time-of-day cutoffs (decimal ET hours: 11.0 = 11:00, 11.5 = 11:30, etc.)
TIME_CUTOFFS_ET = [11.0, 11.5, 12.0]


# ---------------------------------------------------------------------------
# Helper: entry time parsing
# ---------------------------------------------------------------------------

def _entry_hour_minute(open_date_series: pd.Series) -> pd.Series:
    """Return decimal hours from the openDate timestamp column.

    Times are assumed to be ET (Eastern Time), consistent with the 09:30–16:00
    SPX trading session. If the timestamps carry timezone info (tz-aware), they
    are used as-is; naive timestamps are treated as ET without conversion.
    """
    ts = pd.to_datetime(open_date_series)
    return ts.dt.hour + ts.dt.minute / 60.0


# ---------------------------------------------------------------------------
# Time-of-day filter counterfactual
# ---------------------------------------------------------------------------

def _time_of_day_counterfactuals(
    trades: pd.DataFrame,
    cutoffs_et: list[float],
) -> pd.DataFrame:
    """
    For each cutoff H (decimal ET hours), compute:
      - trades entered strictly before the cutoff (would be skipped)
      - trades entered at or after the cutoff (would be kept)
      - wins/losses and P/L for each group
      - P/L delta vs. baseline (all 39 trades)

    This IS tradable: you know what time it is before you enter the trade.
    """
    trades = trades.copy()
    trades["entry_hour_et"] = _entry_hour_minute(trades["openDate"])
    baseline_pnl = float(trades["pnl"].sum())
    baseline_wins = int((trades["outcome"] == "WIN").sum())
    baseline_losses = int((trades["outcome"] == "LOSS").sum())
    total = len(trades)

    rows = []
    for cutoff in cutoffs_et:
        h = int(cutoff)
        m = int(round((cutoff - h) * 60))
        label = f"{h:02d}:{m:02d} ET"

        early = trades[trades["entry_hour_et"] < cutoff]
        after = trades[trades["entry_hour_et"] >= cutoff]

        rows.append({
            "cutoff": label,
            "trades_before_cutoff": len(early),
            "wins_before_cutoff": int((early["outcome"] == "WIN").sum()),
            "losses_before_cutoff": int((early["outcome"] == "LOSS").sum()),
            "pnl_before_cutoff": float(early["pnl"].sum()),
            "trades_kept_after_cutoff": len(after),
            "wins_kept": int((after["outcome"] == "WIN").sum()),
            "losses_kept": int((after["outcome"] == "LOSS").sum()),
            "win_rate_kept": float((after["outcome"] == "WIN").mean()) if len(after) else np.nan,
            "pnl_kept": float(after["pnl"].sum()),
            "pnl_delta_vs_baseline": float(after["pnl"].sum()) - baseline_pnl,
            "baseline_total": total,
            "baseline_wins": baseline_wins,
            "baseline_losses": baseline_losses,
            "baseline_pnl": baseline_pnl,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Entry-time-safe range / gap filter counterfactual
# ---------------------------------------------------------------------------

def _range_filter_counterfactuals(
    trades: pd.DataFrame,
    spx: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build entry-time-safe features and test whether filtering on them improves
    outcomes.

    Features (all safe to observe at or before entry):
      prior_day_range_pts:
        yesterday's High - Low; fully known before today's open.
      open_gap_pts:
        today's daily Open - yesterday's Close; known at 09:30 ET.
      entry_move_from_daily_open_pts:
        underlyingOpen (entry price) - today's daily Open.
        This measures "how far has SPX already moved from the 09:30 open by
        the time we enter?" — positive = price moved up since open, negative =
        price moved down. Knowable at entry because the daily open occurs
        before any trade in this sample (earliest entry 10:55 ET).

    IMPORTANT NOTE ON DAILY HIGH/LOW/CLOSE:
      The spx_daily.csv columns High, Low, and Close are end-of-day values.
      Using them in an entry filter would leak future intraday information.
      They are NOT used here. Only Open (09:30), and prior-day data, are used.

    CAUTION (small-sample): With 39 trades and 9 losses, splitting by a
    percentile-based binary filter leaves 4-20 trades per bucket. Results are
    directional indicators only.
    """
    trades = trades.copy()

    # Build prior-day features from the SPX daily file.
    # Sort by date to compute shifts correctly.
    spx_prior = spx[["trade_date", "Open", "High", "Low", "Close"]].drop_duplicates(
        "trade_date"
    ).sort_values("trade_date").reset_index(drop=True)
    spx_prior["prior_close"] = spx_prior["Close"].shift(1)
    spx_prior["prior_high"] = spx_prior["High"].shift(1)
    spx_prior["prior_low"] = spx_prior["Low"].shift(1)
    spx_prior["prior_day_range_pts"] = spx_prior["prior_high"] - spx_prior["prior_low"]
    spx_prior["open_gap_pts"] = spx_prior["Open"] - spx_prior["prior_close"]
    spx_prior["open_gap_abs_pts"] = spx_prior["open_gap_pts"].abs()
    # Rename daily Open to avoid collision with the "Open" column already in trades.
    spx_prior.rename(columns={"Open": "daily_Open_safe"}, inplace=True)

    merge_cols = [
        "trade_date", "daily_Open_safe", "prior_day_range_pts",
        "open_gap_pts", "open_gap_abs_pts",
    ]
    # Drop any previously merged columns with these names to avoid duplicates.
    for col in merge_cols[1:]:
        if col in trades.columns:
            trades = trades.drop(columns=[col])
    if "daily_Open_safe" in trades.columns:
        trades = trades.drop(columns=["daily_Open_safe"])

    trades = trades.merge(spx_prior[merge_cols], on="trade_date", how="left")

    trades["entry_move_from_daily_open_pts"] = (
        trades["underlyingOpen"] - trades["daily_Open_safe"]
    )
    trades["entry_move_from_daily_open_abs_pts"] = trades["entry_move_from_daily_open_pts"].abs()

    baseline_pnl = float(trades["pnl"].sum())
    baseline_wr = float((trades["outcome"] == "WIN").mean())
    total = len(trades)

    rows = []

    def _counterfactual(label: str, skip_mask: pd.Series) -> dict:
        skipped = trades[skip_mask]
        kept = trades[~skip_mask]
        return {
            "filter": label,
            "trades_skipped": len(skipped),
            "wins_skipped": int((skipped["outcome"] == "WIN").sum()),
            "losses_skipped": int((skipped["outcome"] == "LOSS").sum()),
            "trades_kept": len(kept),
            "wins_kept": int((kept["outcome"] == "WIN").sum()),
            "losses_kept": int((kept["outcome"] == "LOSS").sum()),
            "win_rate_kept": float((kept["outcome"] == "WIN").mean()) if len(kept) else np.nan,
            "pnl_kept": float(kept["pnl"].sum()),
            "pnl_delta_vs_baseline": float(kept["pnl"].sum()) - baseline_pnl,
            "baseline_total": total,
            "baseline_win_rate": baseline_wr,
            "baseline_pnl": baseline_pnl,
        }

    # --- Prior-day range filters (median split and 75th percentile) ---
    range_median = trades["prior_day_range_pts"].median()
    range_p75 = trades["prior_day_range_pts"].quantile(0.75)
    if pd.notna(range_median):
        rows.append(_counterfactual(
            f"Skip if prior_day_range > median ({range_median:.1f} pts) [entry-safe]",
            trades["prior_day_range_pts"] > range_median,
        ))
    if pd.notna(range_p75):
        rows.append(_counterfactual(
            f"Skip if prior_day_range > p75 ({range_p75:.1f} pts) [entry-safe]",
            trades["prior_day_range_pts"] > range_p75,
        ))

    # --- Open gap filters ---
    gap_median = trades["open_gap_abs_pts"].median()
    gap_p75 = trades["open_gap_abs_pts"].quantile(0.75)
    if pd.notna(gap_median):
        rows.append(_counterfactual(
            f"Skip if abs(open_gap) > median ({gap_median:.1f} pts) [entry-safe]",
            trades["open_gap_abs_pts"] > gap_median,
        ))
    if pd.notna(gap_p75):
        rows.append(_counterfactual(
            f"Skip if abs(open_gap) > p75 ({gap_p75:.1f} pts) [entry-safe]",
            trades["open_gap_abs_pts"] > gap_p75,
        ))

    # --- Entry-move-from-open filters ---
    entry_move_p50 = trades["entry_move_from_daily_open_abs_pts"].median()
    entry_move_p75 = trades["entry_move_from_daily_open_abs_pts"].quantile(0.75)
    if pd.notna(entry_move_p50):
        rows.append(_counterfactual(
            f"Skip if abs(entry_move_from_open) > median ({entry_move_p50:.1f} pts) [entry-safe]",
            trades["entry_move_from_daily_open_abs_pts"] > entry_move_p50,
        ))
    if pd.notna(entry_move_p75):
        rows.append(_counterfactual(
            f"Skip if abs(entry_move_from_open) > p75 ({entry_move_p75:.1f} pts) [entry-safe]",
            trades["entry_move_from_daily_open_abs_pts"] > entry_move_p75,
        ))

    return pd.DataFrame(rows), trades[
        [
            "openDate", "outcome", "pnl",
            "prior_day_range_pts", "open_gap_pts", "open_gap_abs_pts",
            "entry_move_from_daily_open_pts", "entry_move_from_daily_open_abs_pts",
        ]
    ].copy()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_entry_filter_analysis(
    trades: pd.DataFrame,
    spx: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Compute and write all entry-time-safe filter counterfactuals.

    Parameters
    ----------
    trades : enriched trades DataFrame (from VannaProxyValidator after
             engineer_features() — must have openDate, pnl, outcome,
             underlyingOpen, trade_date).
    spx    : enriched SPX daily DataFrame (must have trade_date, Open,
             High, Low, Close).
    output_dir : path to the outputs/ directory.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trades_work = trades.copy()
    spx_work = spx.copy()

    # Ensure trade_date is date type for merge.
    if "trade_date" not in trades_work.columns:
        trades_work["trade_date"] = pd.to_datetime(trades_work["openDate"]).dt.date
    if "trade_date" not in spx_work.columns:
        spx_work["trade_date"] = pd.to_datetime(spx_work["Date"]).dt.date

    # ---- 1. Time-of-day filter ----
    tod_df = _time_of_day_counterfactuals(trades_work, TIME_CUTOFFS_ET)

    # ---- 2. Entry-time-safe range / gap filter ----
    range_df, enriched_trades = _range_filter_counterfactuals(trades_work, spx_work)

    # ---- Write CSVs ----
    tod_df.to_csv(output_dir / "entry_filter_time_of_day.csv", index=False)
    range_df.to_csv(output_dir / "entry_filter_range_gap.csv", index=False)
    enriched_trades.to_csv(output_dir / "entry_filter_trade_features.csv", index=False)

    # ---- Write report ----
    _write_entry_filter_report(trades_work, tod_df, range_df, enriched_trades, output_dir)

    print(f"Entry filter analysis complete. Outputs written to {output_dir}")


def _write_entry_filter_report(
    trades: pd.DataFrame,
    tod_df: pd.DataFrame,
    range_df: pd.DataFrame,
    enriched_trades: pd.DataFrame,
    output_dir: Path,
) -> None:
    total = len(trades)
    wins = int((trades["outcome"] == "WIN").sum())
    losses = int((trades["outcome"] == "LOSS").sum())
    baseline_pnl = float(trades["pnl"].sum())
    baseline_wr = float((trades["outcome"] == "WIN").mean())

    lines = []
    lines.append("""
ENTRY FILTER REPORT — TRADABLE COUNTERFACTUALS ONLY
=====================================================

╔══════════════════════════════════════════════════════════════════════════════╗
║  TRADABLE COUNTERFACTUALS — ZERO LOOK-AHEAD BIAS                           ║
║                                                                              ║
║  Every filter in this report uses ONLY information available at or before   ║
║  trade entry. No exit price, intraday price, or day's High/Low/Close is     ║
║  used. These counterfactuals CAN be tested as live entry rules.             ║
║                                                                              ║
║  CONTRAST with MOVEMENT_PRESSURE_REPORT.txt which uses exit data and is     ║
║  explicitly labeled POST-HOC / BIASED.                                       ║
╚══════════════════════════════════════════════════════════════════════════════╝

⚠ SAMPLE SIZE CAUTION: 39 trades (9 losses, 30 wins) is a small sample.
  Results below are triage / hypothesis generation — NOT statistical proof.
  Validate any promising filter on the next 50+ forward trades before acting.

BASELINE (ALL 39 TRADES)
------------------------""")
    lines.append(f"Total trades : {total}")
    lines.append(f"Wins         : {wins}")
    lines.append(f"Losses       : {losses}")
    lines.append(f"Win rate     : {baseline_wr:.1%}")
    lines.append(f"Total P/L    : ${baseline_pnl:,.0f}")

    # ---- Time-of-day section ----
    lines.append("""
════════════════════════════════════════════════════════════════════════════════
SECTION 1: TIME-OF-DAY FILTER (TRADABLE — entry time known at entry)
════════════════════════════════════════════════════════════════════════════════

WHY THIS IS TRADABLE:
  You know what time it is before you submit an entry order. This filter asks:
  "What if we only entered trades after a given time cutoff?" It requires no
  future information whatsoever.

METHODOLOGY:
  For each cutoff (11:00, 11:30, 12:00 ET), trades entered before the cutoff
  are treated as "would have been skipped." Wins and losses removed by each
  cutoff are explicitly reported so you can see the full cost/benefit.

  All times are assumed ET (trades run 09:30–16:00 ET).
""")

    for _, row in tod_df.iterrows():
        wr_kept = row["win_rate_kept"]
        wr_str = f"{wr_kept:.1%}" if pd.notna(wr_kept) else "N/A"
        delta_sign = "+" if row["pnl_delta_vs_baseline"] >= 0 else ""
        lines.append(f"  Cutoff: {row['cutoff']}")
        lines.append(f"    Trades entered BEFORE cutoff (skipped)  : {row['trades_before_cutoff']} "
                     f"({row['wins_before_cutoff']} wins, {row['losses_before_cutoff']} losses, "
                     f"P/L ${row['pnl_before_cutoff']:,.0f})")
        lines.append(f"    Trades entered AT/AFTER cutoff (kept)   : {row['trades_kept_after_cutoff']} "
                     f"({row['wins_kept']} wins, {row['losses_kept']} losses)")
        lines.append(f"    Win rate (kept)                         : {wr_str}")
        lines.append(f"    P/L (kept)                              : ${row['pnl_kept']:,.0f}")
        lines.append(f"    P/L delta vs baseline                   : {delta_sign}${row['pnl_delta_vs_baseline']:,.0f}")
        lines.append("")

    lines.append("Full table:")
    lines.append(tod_df[[
        "cutoff", "trades_before_cutoff", "wins_before_cutoff", "losses_before_cutoff",
        "trades_kept_after_cutoff", "wins_kept", "losses_kept", "win_rate_kept",
        "pnl_kept", "pnl_delta_vs_baseline",
    ]].to_string(index=False))

    # ---- Range / gap section ----
    lines.append("""
════════════════════════════════════════════════════════════════════════════════
SECTION 2: ENTRY-TIME-SAFE RANGE / GAP FILTERS (TRADABLE)
════════════════════════════════════════════════════════════════════════════════

WHY THESE ARE TRADABLE:
  - prior_day_range_pts: yesterday's full High–Low range. Settled before today's
    open. A persistently wide prior day is a rough proxy for elevated-volatility
    regime entering today. Knowable from the previous session's close.
  - open_gap_abs_pts: abs(today's 09:30 open − yesterday's close). Knowable at
    market open, before entry.
  - entry_move_from_daily_open_abs_pts: abs(underlyingOpen at entry −
    today's 09:30 open price). Measures how far SPX has moved from the market
    open by the time we enter. Positive = up from open; negative = down.
    Knowable AT ENTRY because the daily open (09:30) precedes all trades in
    this sample (earliest entry 10:55).

WHAT IS NOT USED:
  daily High, daily Low, daily Close, underlyingClose — all are end-of-day or
  intraday future values relative to entry, and are NOT used here.

METHODOLOGY:
  Median-split and 75th-percentile-split binary filters for each feature.
  "Skip" means: exclude trades where the feature exceeds the threshold.
  Because these features are entry-time-safe, this IS a legitimate
  forward-looking counterfactual.

⚠ LIMITATION: Prior-day range and gap are weak predictors of intraday movement
  on a specific day. With 39 trades, the signal-to-noise ratio is low. These
  filters are a starting point, not a proven edge.
""")

    for _, row in range_df.iterrows():
        wr_kept = row["win_rate_kept"]
        wr_str = f"{wr_kept:.1%}" if pd.notna(wr_kept) else "N/A"
        delta_sign = "+" if row["pnl_delta_vs_baseline"] >= 0 else ""
        lines.append(f"  Filter: {row['filter']}")
        lines.append(f"    Trades skipped   : {row['trades_skipped']} "
                     f"({row['wins_skipped']} wins, {row['losses_skipped']} losses)")
        lines.append(f"    Trades kept      : {row['trades_kept']} "
                     f"({row['wins_kept']} wins, {row['losses_kept']} losses)")
        lines.append(f"    Win rate (kept)  : {wr_str}")
        lines.append(f"    P/L (kept)       : ${row['pnl_kept']:,.0f}")
        lines.append(f"    P/L delta        : {delta_sign}${row['pnl_delta_vs_baseline']:,.0f}")
        lines.append("")

    lines.append("Full table:")
    lines.append(range_df[[
        "filter", "trades_skipped", "wins_skipped", "losses_skipped",
        "trades_kept", "wins_kept", "losses_kept", "win_rate_kept",
        "pnl_kept", "pnl_delta_vs_baseline",
    ]].to_string(index=False))

    # ---- Entry-feature per-trade breakdown ----
    lines.append("""
════════════════════════════════════════════════════════════════════════════════
SECTION 3: ENTRY-TIME-SAFE FEATURES PER TRADE
════════════════════════════════════════════════════════════════════════════════
(entry_move_from_daily_open_pts positive = SPX moved UP from 09:30 open to entry)
""")
    display = enriched_trades[[
        "openDate", "outcome", "pnl",
        "prior_day_range_pts", "open_gap_pts",
        "entry_move_from_daily_open_pts", "entry_move_from_daily_open_abs_pts",
    ]].copy()
    display["openDate"] = pd.to_datetime(display["openDate"]).dt.strftime("%Y-%m-%d %H:%M")
    lines.append(display.to_string(index=False))

    lines.append(dedent("""

NEXT STEPS
----------
1. Paper-forward-test any promising filter on the next 50+ trades before
   trusting it. 39 trades is triage, not proof.
2. If time-of-day filter shows positive P/L delta with acceptable win-rate
   retention: add a configurable entry cutoff to your bot (e.g., skip entries
   before 11:30 ET).
3. If prior-day range filter shows positive delta: cross-reference with a
   volatility proxy (e.g., VIX open) for a more robust regime signal.
4. For intraday path-based risk management (first-breach stop), see
   INTRADAY_STOP_SIMULATION.txt.
"""))

    report = "\n".join(lines)
    (output_dir / "ENTRY_FILTER_REPORT.txt").write_text(report)
