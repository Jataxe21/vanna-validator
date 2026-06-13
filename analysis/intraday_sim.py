#!/usr/bin/env python3
"""
Intraday simulation for the live 0DTE SPX short iron butterfly bot.

REAL BOT BASELINE MODELED HERE
------------------------------
- Entry context (not re-simulated): body strike is centered at the max-absolute-GEX strike,
  entry window is 11:00 ET to 14:00 ET, and center_strike extraction from trade description
  is used as the position center.
- Profit target: close at 30% profit on credit received
    -> target close mark ~= 0.70 * entry credit.
- Stop loss: close when mark >= 1.5x entry credit.
- Hard loss cap: per-contract P/L floor of -$1,500.
- Timed exit: unconditional close at 15:15 ET (45 min before 16:00 ET expiry)
  if target/stop have not already triggered.

IMPORTANT APPROXIMATION (NO INTRADAY OPTION MARKS AVAILABLE)
-------------------------------------------------------------
Intraday option prices are unavailable in this repository, so the fly mark is inferred from
underlying distance to the body strike:

    d = abs(underlying - body)
    mark(d, t) ~= credit - decay_component(d, t) + distance_component(d)

where decay is strongest when d is small and time has elapsed, and distance_component grows
when d moves away from the body. This is a crude approximation and is explicitly treated as
"directional only". Stop-side behavior (distance-driven adverse movement) is more reliable
than exact target timing under this approximation.

If data/spx_intraday.csv is absent, this module writes a stub report and exits gracefully.
"""

from pathlib import Path
from textwrap import dedent

import numpy as np
import pandas as pd


PROFIT_TARGET_FRAC = 0.30
STOP_CREDIT_MULT = 1.5
HARD_CAP_LOSS_PER_CONTRACT = 1500
TIMED_EXIT_ET = "15:15"
ALT_STOP_DISTANCES = [8, 10, 12, 15]

# Reported live total from trades.csv context; used for explicit delta reporting.
ACTUAL_RECORDED_TOTAL_PNL = 10850.0

CONTRACT_MULTIPLIER = 100.0
# Mark approximation knobs (documented assumptions; directional only)
MARK_DISTANCE_CUSHION_PTS = 2.0
MARK_DISTANCE_SLOPE = 0.11
MARK_MAX_MULTIPLE = 3.0


INTRADAY_SCHEMA = dedent("""
    Expected schema for data/spx_intraday.csv
    ------------------------------------------
    Column      Type        Description
    timestamp   datetime    Bar timestamp (e.g., 2026-05-04 11:00:00)
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


def _timed_exit_timestamp(open_dt: pd.Timestamp) -> pd.Timestamp:
    """Return same-day 15:15 ET timestamp for a trade's open day."""
    hour, minute = [int(x) for x in TIMED_EXIT_ET.split(":")]
    return open_dt.normalize() + pd.Timedelta(hours=hour, minutes=minute)


def _estimate_mark(credit: float, distance_pts: float, elapsed_min: float, total_min: float) -> float:
    """Approximate iron-fly mark from distance to body and elapsed time."""
    total_min = max(total_min, 1.0)
    elapsed_frac = float(np.clip(elapsed_min / total_min, 0.0, 1.0))

    # Decay benefit is strongest when price is pinned near the body (small distance).
    pin_factor = float(np.clip(1.0 - distance_pts / MARK_DISTANCE_CUSHION_PTS, 0.0, 1.0))
    decay_component = credit * PROFIT_TARGET_FRAC * elapsed_frac * pin_factor

    # Distance expansion captures adverse intrinsic pressure when underlying moves away.
    distance_component = max(distance_pts - MARK_DISTANCE_CUSHION_PTS, 0.0) * MARK_DISTANCE_SLOPE

    raw_mark = credit - decay_component + distance_component
    max_mark = max(credit * MARK_MAX_MULTIPLE, credit + 40.0)
    return float(np.clip(raw_mark, 0.0, max_mark))


def _estimate_pnl(credit: float, est_mark: float) -> float:
    """Estimate per-contract P/L and apply hard loss cap."""
    pnl = (credit - est_mark) * CONTRACT_MULTIPLIER
    return float(max(pnl, -HARD_CAP_LOSS_PER_CONTRACT))


def _simulate_trade_path(
    trade: pd.Series,
    intraday: pd.DataFrame,
    alt_stop_dist: float | None = None,
) -> dict:
    """Simulate first-firing exit for baseline or alt-stop variant."""
    body = trade["center_strike"]
    open_dt = pd.to_datetime(trade["openDate"])
    close_dt = pd.to_datetime(trade["closeDate"])
    credit = float(trade["openPrice"])
    actual_pnl = float(trade["pnl"])
    outcome = trade.get("outcome", "WIN" if actual_pnl > 0 else "LOSS")

    variant_label = "baseline_real" if alt_stop_dist is None else f"alt_stop_{int(alt_stop_dist)}"

    if np.isnan(body):
        return {
            "variant": variant_label,
            "stop_distance_pts": alt_stop_dist,
            "first_exit_reason": "no_data",
            "exit_timestamp": pd.NaT,
            "est_mark": np.nan,
            "est_pnl": actual_pnl,
            "actual_pnl": actual_pnl,
            "outcome": outcome,
            "center_strike": body,
            "open_price": credit,
            "trade_openDate": open_dt,
            "trade_closeDate": close_dt,
            "note": "center_strike extraction failed",
        }

    if np.isnan(credit) or credit <= 0:
        return {
            "variant": variant_label,
            "stop_distance_pts": alt_stop_dist,
            "first_exit_reason": "no_data",
            "exit_timestamp": pd.NaT,
            "est_mark": np.nan,
            "est_pnl": actual_pnl,
            "actual_pnl": actual_pnl,
            "outcome": outcome,
            "center_strike": body,
            "open_price": credit,
            "trade_openDate": open_dt,
            "trade_closeDate": close_dt,
            "note": "invalid openPrice",
        }

    timed_exit_ts = _timed_exit_timestamp(open_dt)
    end_ts = min(close_dt, timed_exit_ts)
    total_minutes = max((timed_exit_ts - open_dt).total_seconds() / 60.0, 1.0)

    window = intraday[(intraday["timestamp"] >= open_dt) & (intraday["timestamp"] <= end_ts)].copy()

    if window.empty:
        return {
            "variant": variant_label,
            "stop_distance_pts": alt_stop_dist,
            "first_exit_reason": "no_data",
            "exit_timestamp": pd.NaT,
            "est_mark": np.nan,
            "est_pnl": actual_pnl,
            "actual_pnl": actual_pnl,
            "outcome": outcome,
            "center_strike": body,
            "open_price": credit,
            "trade_openDate": open_dt,
            "trade_closeDate": close_dt,
            "note": "no intraday bars in trade window",
        }

    target_mark = (1.0 - PROFIT_TARGET_FRAC) * credit
    stop_mark = STOP_CREDIT_MULT * credit

    for row in window.itertuples(index=False):
        ts = row.timestamp
        elapsed_minutes = max((ts - open_dt).total_seconds() / 60.0, 0.0)

        dist_close = abs(float(row.close) - body)
        dist_intrabar = max(abs(float(row.high) - body), abs(float(row.low) - body))

        mark_close = _estimate_mark(credit, dist_close, elapsed_minutes, total_minutes)
        mark_intrabar = _estimate_mark(credit, dist_intrabar, elapsed_minutes, total_minutes)

        # Real stop first: if breached intrabar, treat as fired at this bar close.
        if mark_intrabar >= stop_mark:
            est_mark = mark_intrabar
            return {
                "variant": variant_label,
                "stop_distance_pts": alt_stop_dist,
                "first_exit_reason": "stop_1.5x",
                "exit_timestamp": ts,
                "est_mark": est_mark,
                "est_pnl": _estimate_pnl(credit, est_mark),
                "actual_pnl": actual_pnl,
                "outcome": outcome,
                "center_strike": body,
                "open_price": credit,
                "trade_openDate": open_dt,
                "trade_closeDate": close_dt,
                "note": "",
            }

        # Alternative distance stop uses high/low breach; decision at bar close.
        if (
            alt_stop_dist is not None
            and ((float(row.high) >= body + alt_stop_dist) or (float(row.low) <= body - alt_stop_dist))
        ):
            est_mark = _estimate_mark(credit, float(alt_stop_dist), elapsed_minutes, total_minutes)
            return {
                "variant": variant_label,
                "stop_distance_pts": alt_stop_dist,
                "first_exit_reason": variant_label,
                "exit_timestamp": ts,
                "est_mark": est_mark,
                "est_pnl": _estimate_pnl(credit, est_mark),
                "actual_pnl": actual_pnl,
                "outcome": outcome,
                "center_strike": body,
                "open_price": credit,
                "trade_openDate": open_dt,
                "trade_closeDate": close_dt,
                "note": "",
            }

        # Profit target approximation evaluated at bar close.
        if mark_close <= target_mark:
            est_mark = mark_close
            return {
                "variant": variant_label,
                "stop_distance_pts": alt_stop_dist,
                "first_exit_reason": "target",
                "exit_timestamp": ts,
                "est_mark": est_mark,
                "est_pnl": _estimate_pnl(credit, est_mark),
                "actual_pnl": actual_pnl,
                "outcome": outcome,
                "center_strike": body,
                "open_price": credit,
                "trade_openDate": open_dt,
                "trade_closeDate": close_dt,
                "note": "",
            }

    # If nothing fired in-window, force timed exit at final available bar <= 15:15.
    last_row = window.iloc[-1]
    elapsed_minutes = max((last_row["timestamp"] - open_dt).total_seconds() / 60.0, 0.0)
    dist_close = abs(float(last_row["close"]) - body)
    est_mark = _estimate_mark(credit, dist_close, elapsed_minutes, total_minutes)

    return {
        "variant": variant_label,
        "stop_distance_pts": alt_stop_dist,
        "first_exit_reason": "timed_1515",
        "exit_timestamp": last_row["timestamp"],
        "est_mark": est_mark,
        "est_pnl": _estimate_pnl(credit, est_mark),
        "actual_pnl": actual_pnl,
        "outcome": outcome,
        "center_strike": body,
        "open_price": credit,
        "trade_openDate": open_dt,
        "trade_closeDate": close_dt,
        "note": "",
    }


def run_intraday_stop_simulation(
    trades: pd.DataFrame,
    intraday_file: Path,
    output_dir: Path,
) -> None:
    """Run (or gracefully skip) intraday baseline + alt-stop simulation."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not Path(intraday_file).exists():
        try:
            open_dates = pd.to_datetime(trades["openDate"])
            date_range = f"{open_dates.min().strftime('%Y-%m-%d')} to {open_dates.max().strftime('%Y-%m-%d')}"
        except Exception:
            date_range = ""
        _write_stub_report(output_dir, date_range=date_range)
        print(
            "[intraday_sim] data/spx_intraday.csv not found — "
            "intraday stop simulation skipped. "
            "A stub note has been written to outputs/INTRADAY_STOP_SIMULATION.txt"
        )
        return

    try:
        intraday = pd.read_csv(intraday_file)
        intraday["timestamp"] = pd.to_datetime(intraday["timestamp"])
        for col in ["open", "high", "low", "close"]:
            intraday[col] = pd.to_numeric(intraday[col], errors="coerce")
        intraday = intraday.dropna(subset=["timestamp", "open", "high", "low", "close"])
        intraday = intraday.sort_values("timestamp").reset_index(drop=True)
    except Exception as exc:
        print(f"[intraday_sim] Failed to load {intraday_file}: {exc}")
        _write_stub_report(output_dir, error=str(exc))
        return

    trades = trades.copy()
    required_cols = {"openDate", "closeDate", "openPrice", "pnl", "description"}
    if not required_cols.issubset(set(trades.columns)):
        missing = sorted(required_cols - set(trades.columns))
        _write_stub_report(output_dir, error=f"Missing required trade columns: {missing}")
        return

    trades["openDate"] = pd.to_datetime(trades["openDate"])
    trades["closeDate"] = pd.to_datetime(trades["closeDate"])
    trades["openPrice"] = pd.to_numeric(trades["openPrice"], errors="coerce")
    trades["pnl"] = pd.to_numeric(trades["pnl"], errors="coerce")

    if "center_strike" not in trades.columns:
        trades["center_strike"] = trades["description"].apply(_center_strike)
    if "outcome" not in trades.columns:
        trades["outcome"] = np.where(trades["pnl"] > 0, "WIN", "LOSS")

    baseline_rows = []
    variant_rows = []

    for _, trade in trades.iterrows():
        baseline_rows.append(_simulate_trade_path(trade, intraday, alt_stop_dist=None))
        for dist in ALT_STOP_DISTANCES:
            variant_rows.append(_simulate_trade_path(trade, intraday, alt_stop_dist=float(dist)))

    baseline_df = pd.DataFrame(baseline_rows)
    variants_df = pd.DataFrame(variant_rows)
    trades_detail_df = pd.concat([baseline_df, variants_df], ignore_index=True)

    simulated_baseline_total = float(baseline_df["est_pnl"].sum())
    actual_total_from_data = float(trades["pnl"].sum())

    baseline_valid = baseline_df[["est_pnl", "actual_pnl"]].dropna()
    if len(baseline_valid) > 1 and baseline_valid["est_pnl"].std() > 0 and baseline_valid["actual_pnl"].std() > 0:
        baseline_corr = float(baseline_valid["est_pnl"].corr(baseline_valid["actual_pnl"]))
    else:
        baseline_corr = np.nan
    baseline_mae = float((baseline_valid["est_pnl"] - baseline_valid["actual_pnl"]).abs().mean()) if len(baseline_valid) else np.nan

    summary_rows = []
    for dist in ALT_STOP_DISTANCES:
        variant_name = f"alt_stop_{dist}"
        sub = variants_df[variants_df["variant"] == variant_name]

        stopped_mask = sub["first_exit_reason"] == variant_name
        trades_stopped = int(stopped_mask.sum())
        stops_on_winners = int((stopped_mask & (sub["actual_pnl"] > 0)).sum())
        stops_on_losers = int((stopped_mask & (sub["actual_pnl"] <= 0)).sum())

        total_pnl_variant = float(sub["est_pnl"].sum())
        delta_vs_simulated_baseline = total_pnl_variant - simulated_baseline_total
        delta_vs_actual_recorded = total_pnl_variant - ACTUAL_RECORDED_TOTAL_PNL

        summary_rows.append(
            {
                "stop_distance_pts": dist,
                "trades_stopped": trades_stopped,
                "stops_on_winners": stops_on_winners,
                "stops_on_losers": stops_on_losers,
                "total_pnl_variant": total_pnl_variant,
                "delta_vs_simulated_baseline": delta_vs_simulated_baseline,
                "delta_vs_actual_recorded": delta_vs_actual_recorded,
                "whipsaw_warning": "YES — stop hurt overall" if delta_vs_simulated_baseline < 0 else "no",
            }
        )

    summary_df = pd.DataFrame(summary_rows)

    trades_detail_df.to_csv(output_dir / "intraday_stop_simulation_trades.csv", index=False)
    summary_df.to_csv(output_dir / "intraday_stop_simulation_summary.csv", index=False)

    _write_full_report(
        trades=trades,
        baseline_df=baseline_df,
        variants_df=variants_df,
        detail_df=trades_detail_df,
        summary_df=summary_df,
        simulated_baseline_total=simulated_baseline_total,
        actual_total_from_data=actual_total_from_data,
        baseline_mae=baseline_mae,
        baseline_corr=baseline_corr,
        output_dir=output_dir,
    )

    print(f"[intraday_sim] Real-rule baseline + alternative-stop simulation complete. Outputs written to {output_dir}")


def _write_stub_report(output_dir: Path, error: str = "", date_range: str = "") -> None:
    report = """
INTRADAY REAL-BASELINE STOP SIMULATION
======================================

STATUS: SKIPPED — data/spx_intraday.csv not found.

This module models the REAL bot exits as the baseline (30% target, 1.5x-credit
stop, -$1,500 hard cap, 15:15 ET timed exit), then compares alternative
underlying-distance stops against that baseline.

No intraday data file was detected, so path simulation was skipped. All other
analyses ran normally.

HOW TO ENABLE THIS SIMULATION
------------------------------
1. Obtain 1-minute SPX (or ES futures) OHLC data covering your trade range{date_range_note}.
2. Format as CSV columns: timestamp, open, high, low, close.

{schema}

3. Save as data/spx_intraday.csv in repository root.
4. Re-run: python analysis/vanna_analysis.py

Outputs generated when intraday file is present:
    outputs/INTRADAY_STOP_SIMULATION.txt
    outputs/intraday_stop_simulation_summary.csv
    outputs/intraday_stop_simulation_trades.csv
""".format(
        schema=INTRADAY_SCHEMA,
        date_range_note=f" ({date_range})" if date_range else " (see data/trades.csv)",
    )

    if error:
        report += f"\nERROR DETAIL: {error}\n"

    (output_dir / "INTRADAY_STOP_SIMULATION.txt").write_text(report)


def _write_full_report(
    trades: pd.DataFrame,
    baseline_df: pd.DataFrame,
    variants_df: pd.DataFrame,
    detail_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    simulated_baseline_total: float,
    actual_total_from_data: float,
    baseline_mae: float,
    baseline_corr: float,
    output_dir: Path,
) -> None:
    total = len(trades)
    wins = int((trades["pnl"] > 0).sum())
    losses = int((trades["pnl"] <= 0).sum())

    lines = []
    lines.append(
        dedent(
            f"""
INTRADAY REAL-BASELINE STOP SIMULATION
======================================

⚠ SAMPLE-SIZE CAUTION: {total} trades ({losses} losses). Directional only.
  Treat this as hypothesis generation, not statistical proof.

BASELINE MODELED (REAL BOT RULES)
---------------------------------
- Profit target: {PROFIT_TARGET_FRAC:.0%} of credit (mark <= {(1-PROFIT_TARGET_FRAC):.2f}x credit)
- Stop loss: mark >= {STOP_CREDIT_MULT:.1f}x credit
- Hard cap: per-contract P/L floor at -${HARD_CAP_LOSS_PER_CONTRACT:,.0f}
- Timed exit: {TIMED_EXIT_ET} ET unconditional close
- Entry context: 11:00-14:00 ET (not re-simulated), center strike from trade description

MARK APPROXIMATION (EXPLICITLY CRUDE)
-------------------------------------
No intraday option marks are available, so mark is inferred from underlying distance
from body strike plus elapsed-time decay near the body.
Stop-side behavior is more reliable than precise target timing under this approximation.

BASELINE RECONCILIATION CREDIBILITY CHECK
-----------------------------------------
Simulated baseline total P/L : ${simulated_baseline_total:,.0f}
Actual recorded total (data) : ${actual_total_from_data:,.0f}
Actual recorded total (fixed): ${ACTUAL_RECORDED_TOTAL_PNL:,.0f}
Mean absolute error (trade)  : ${baseline_mae:,.1f}
Correlation (sim vs actual)  : {baseline_corr:.3f}

ALTERNATIVE DISTANCE STOPS VS REAL BASELINE
-------------------------------------------
(Each variant keeps real target + timed exit, and adds first-breach abs(underlying-body)>=N)
"""
        ).strip()
    )

    lines.append(summary_df.to_string(index=False))

    for _, row in summary_df.iterrows():
        if row["delta_vs_simulated_baseline"] < 0:
            lines.append(
                f"\n⚠ WHIPSAW WARNING N={int(row['stop_distance_pts'])}: "
                f"delta_vs_simulated_baseline = ${row['delta_vs_simulated_baseline']:,.0f}"
            )

    lines.append(
        dedent(
            """

PER-TRADE X PER-VARIANT DETAIL
------------------------------
Columns include first_exit_reason (target / stop_1.5x / timed_1515 / alt_stop_N),
exit_timestamp, est_pnl, and actual_pnl.
"""
        ).rstrip()
    )

    detail_display = detail_df[
        [
            "trade_openDate",
            "variant",
            "first_exit_reason",
            "exit_timestamp",
            "est_pnl",
            "actual_pnl",
            "outcome",
            "stop_distance_pts",
            "note",
        ]
    ].copy()
    detail_display["trade_openDate"] = pd.to_datetime(detail_display["trade_openDate"]).dt.strftime("%Y-%m-%d %H:%M")
    detail_display["exit_timestamp"] = pd.to_datetime(detail_display["exit_timestamp"]).dt.strftime("%Y-%m-%d %H:%M").fillna("—")
    lines.append(detail_display.to_string(index=False))

    lines.append("\nINTRADAY DATA SCHEMA USED\n--------------------------")
    lines.append(INTRADAY_SCHEMA)

    report = "\n".join(lines)
    (output_dir / "INTRADAY_STOP_SIMULATION.txt").write_text(report)
