#!/usr/bin/env python3
"""
0DTE SPX Short Iron Butterfly — Loss Attribution & Tradable Filter Analysis

Uses the data currently available in this repo:
- completed 0DTE SPX iron butterfly trades (data/trades.csv)
- daily SPX OHLCV history (data/spx_daily.csv)
- optional intraday SPX bars (data/spx_intraday.csv) — enables first-breach simulation

ESTABLISHED FINDING:
  Losses are NOT explained by a "proxy-vanna"/volatility regime.
  They ARE explained by adverse underlying MOVEMENT and DISTANCE of price from
  the body strike AT EXIT.

BIAS WARNING:
  Exit-derived metrics (underlying_move_abs_pts, abs_exit_distance_from_center)
  are OUTCOMES, not entry inputs. Any "skip threshold" derived from them is
  POST-HOC and subject to look-ahead bias — it re-describes the losers after
  the fact and is NOT a tradable edge.

  Tradable counterfactuals (entry time filters, intraday first-breach stop) are
  clearly separated in ENTRY_FILTER_REPORT.txt and INTRADAY_STOP_SIMULATION.txt.
"""

from pathlib import Path
from textwrap import dedent
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
TRADES_FILE = DATA_DIR / "trades.csv"
SPX_FILE = DATA_DIR / "spx_daily.csv"


class VannaProxyValidator:
    def __init__(self):
        self.trades = None
        self.spx = None
        self.summary = {}
        self.movement_thresholds = None
        self.center_distance_summary = None

    def load_data(self):
        self.trades = pd.read_csv(TRADES_FILE)
        self.spx = pd.read_csv(SPX_FILE)

        self.trades["openDate"] = pd.to_datetime(self.trades["openDate"])
        self.trades["closeDate"] = pd.to_datetime(self.trades["closeDate"])
        self.trades["trade_date"] = self.trades["openDate"].dt.date
        self.trades["pnl"] = pd.to_numeric(self.trades["pnl"], errors="coerce")
        self.trades["underlyingOpen"] = pd.to_numeric(self.trades["underlyingOpen"], errors="coerce")
        self.trades["underlyingClose"] = pd.to_numeric(self.trades["underlyingClose"], errors="coerce")
        self.trades["openPrice"] = pd.to_numeric(self.trades.get("openPrice"), errors="coerce")
        self.trades["closePrice"] = pd.to_numeric(self.trades.get("closePrice"), errors="coerce")
        self.trades["outcome"] = np.where(self.trades["pnl"] > 0, "WIN", "LOSS")

        self.spx["Date"] = pd.to_datetime(self.spx["Date"])
        self.spx["trade_date"] = self.spx["Date"].dt.date
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col in self.spx.columns:
                self.spx[col] = pd.to_numeric(self.spx[col], errors="coerce")
        self.spx = self.spx.sort_values("Date").reset_index(drop=True)

    @staticmethod
    def center_strike(description):
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

    def engineer_features(self):
        spx = self.spx.copy()
        trades = self.trades.copy()

        spx["daily_return"] = spx["Close"].pct_change()
        spx["day_move_pct"] = (spx["Close"] - spx["Open"]) / spx["Open"]
        spx["daily_range_pct"] = (spx["High"] - spx["Low"]) / spx["Open"]
        spx["gap_abs_pct"] = (spx["Open"] - spx["Close"].shift(1)).abs() / spx["Close"].shift(1)
        spx["rv_5d"] = spx["daily_return"].rolling(5, min_periods=3).std() * np.sqrt(252)
        spx["range_5d"] = spx["daily_range_pct"].rolling(5, min_periods=3).mean()

        # Percentile ranks are more stable than fixed thresholds in a short sample.
        for col in ["daily_range_pct", "rv_5d", "range_5d", "gap_abs_pct"]:
            spx[f"{col}_rank"] = spx[col].rank(pct=True)

        spx["proxy_vanna_score"] = (
            45 * spx["daily_range_pct_rank"].fillna(0.5)
            + 35 * spx["rv_5d_rank"].fillna(0.5)
            + 20 * spx["gap_abs_pct_rank"].fillna(0.5)
        ).clip(0, 100)
        spx["proxy_vanna_env"] = pd.qcut(
            spx["proxy_vanna_score"].rank(method="first"),
            q=3,
            labels=["LOW", "MEDIUM", "HIGH"],
        )

        market_cols = [
            "trade_date", "Open", "High", "Low", "Close", "day_move_pct",
            "daily_range_pct", "rv_5d", "range_5d", "proxy_vanna_score", "proxy_vanna_env",
        ]
        trades = trades.merge(spx[market_cols], on="trade_date", how="left")

        trades["center_strike"] = trades["description"].apply(self.center_strike)
        trades["underlying_move_pts"] = trades["underlyingClose"] - trades["underlyingOpen"]
        trades["underlying_move_abs_pts"] = trades["underlying_move_pts"].abs()
        trades["underlying_move_pct"] = trades["underlying_move_pts"] / trades["underlyingOpen"]
        trades["entry_distance_from_center"] = trades["underlyingOpen"] - trades["center_strike"]
        trades["exit_distance_from_center"] = trades["underlyingClose"] - trades["center_strike"]
        trades["abs_entry_distance_from_center"] = trades["entry_distance_from_center"].abs()
        trades["abs_exit_distance_from_center"] = trades["exit_distance_from_center"].abs()
        trades["distance_expansion_pts"] = trades["abs_exit_distance_from_center"] - trades["abs_entry_distance_from_center"]
        trades["trade_duration_min"] = (trades["closeDate"] - trades["openDate"]).dt.total_seconds() / 60
        trades["pnl_per_minute"] = trades["pnl"] / trades["trade_duration_min"].replace(0, np.nan)

        large_move_cutoff = trades["underlying_move_abs_pts"].quantile(0.67)
        trades["loss_attribution"] = "Winner"
        trades.loc[(trades["outcome"] == "LOSS") & (trades["proxy_vanna_env"] == "HIGH"), "loss_attribution"] = "High proxy-vanna / volatility regime"
        trades.loc[(trades["outcome"] == "LOSS") & (trades["loss_attribution"] == "Winner") & (trades["underlying_move_abs_pts"] >= large_move_cutoff), "loss_attribution"] = "Directional move / gamma pressure"
        trades.loc[(trades["outcome"] == "LOSS") & (trades["loss_attribution"] == "Winner"), "loss_attribution"] = "Unclassified / execution / noise"

        self.trades = trades
        self.spx = spx

    def summarize(self):
        trades = self.trades
        wins = trades[trades["outcome"] == "WIN"]
        losses = trades[trades["outcome"] == "LOSS"]

        env_summary = trades.dropna(subset=["proxy_vanna_env"]).groupby("proxy_vanna_env", observed=False).agg(
            trades=("pnl", "size"),
            wins=("outcome", lambda s: int((s == "WIN").sum())),
            losses=("outcome", lambda s: int((s == "LOSS").sum())),
            total_pnl=("pnl", "sum"),
            avg_pnl=("pnl", "mean"),
            avg_abs_move_pts=("underlying_move_abs_pts", "mean"),
            avg_range_pct=("daily_range_pct", "mean"),
        ).reset_index()
        env_summary["win_rate"] = env_summary["wins"] / env_summary["trades"]

        high_losses = int((losses["proxy_vanna_env"] == "HIGH").sum())
        baseline_win_rate = float((trades["outcome"] == "WIN").mean())
        baseline_pnl = float(trades["pnl"].sum())
        skip_high = trades[trades["proxy_vanna_env"] != "HIGH"]

        loss_scores = losses["proxy_vanna_score"].dropna()
        win_scores = wins["proxy_vanna_score"].dropna()
        t_p = float(stats.ttest_ind(loss_scores, win_scores, equal_var=False).pvalue) if len(loss_scores) > 1 and len(win_scores) > 1 else np.nan
        mw_p = float(stats.mannwhitneyu(loss_scores, win_scores, alternative="two-sided").pvalue) if len(loss_scores) > 1 and len(win_scores) > 1 else np.nan

        self.summary = {
            "total_trades": int(len(trades)),
            "wins": int(len(wins)),
            "losses": int(len(losses)),
            "classified_env_trades": int(trades["proxy_vanna_env"].notna().sum()),
            "unclassified_env_trades": int(trades["proxy_vanna_env"].isna().sum()),
            "baseline_win_rate": baseline_win_rate,
            "baseline_pnl": baseline_pnl,
            "high_proxy_vanna_losses": high_losses,
            "high_proxy_vanna_loss_share": high_losses / len(losses) if len(losses) else np.nan,
            "skip_high_trades": int(len(skip_high)),
            "skip_high_win_rate": float((skip_high["outcome"] == "WIN").mean()) if len(skip_high) else np.nan,
            "skip_high_pnl": float(skip_high["pnl"].sum()) if len(skip_high) else np.nan,
            "avg_loss_proxy_vanna_score": float(losses["proxy_vanna_score"].mean()),
            "avg_win_proxy_vanna_score": float(wins["proxy_vanna_score"].mean()),
            "ttest_p_value": t_p,
            "mannwhitney_p_value": mw_p,
            "corr_pnl_proxy_vanna_score": float(trades[["pnl", "proxy_vanna_score"]].corr().iloc[0, 1]),
            "corr_pnl_abs_underlying_move": float(trades[["pnl", "underlying_move_abs_pts"]].corr().iloc[0, 1]),
            "corr_pnl_abs_exit_distance": float(trades[["pnl", "abs_exit_distance_from_center"]].corr().iloc[0, 1]),
            "loss_attribution": losses["loss_attribution"].value_counts().to_dict(),
            "env_summary": env_summary,
        }

    def analyze_movement_thresholds(self):
        """Post-hoc diagnostic: how do losses cluster by exit-measured movement?

        POST-HOC / LOOK-AHEAD BIAS WARNING
        -----------------------------------
        underlying_move_abs_pts = abs(underlyingClose - underlyingOpen) — this is measured at
        the ACTUAL EXIT. It is an outcome, not an entry- or mid-trade-decision input.
        Flagging trades by "underlying_move >= threshold" is a RE-DESCRIPTION of the losers,
        not predictive skill. The "post_hoc_separation_score" and
        "pnl_impact_if_skipped_POST_HOC_BIASED" columns therefore CANNOT be acted on at entry.
        They are retained here as a diagnostic to show WHERE losses cluster, nothing more.

        For genuinely tradable counterfactuals, see ENTRY_FILTER_REPORT.txt and
        INTRADAY_STOP_SIMULATION.txt.
        """
        trades = self.trades.copy()
        total_losses = int((trades["outcome"] == "LOSS").sum())
        total_wins = int((trades["outcome"] == "WIN").sum())
        baseline_pnl = float(trades["pnl"].sum())

        thresholds = [2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20, 22.5, 25]
        rows = []
        for threshold in thresholds:
            flagged = trades[trades["underlying_move_abs_pts"] >= threshold]
            kept = trades[trades["underlying_move_abs_pts"] < threshold]
            flagged_losses = int((flagged["outcome"] == "LOSS").sum())
            flagged_wins = int((flagged["outcome"] == "WIN").sum())
            kept_losses = int((kept["outcome"] == "LOSS").sum())
            kept_wins = int((kept["outcome"] == "WIN").sum())
            rows.append({
                "threshold_pts": threshold,
                "flagged_trades": int(len(flagged)),
                "flagged_wins": flagged_wins,
                "flagged_losses": flagged_losses,
                "kept_trades": int(len(kept)),
                "kept_wins": kept_wins,
                "kept_losses": kept_losses,
                "loss_capture_rate": flagged_losses / total_losses if total_losses else np.nan,
                "winner_penalty_rate": flagged_wins / total_wins if total_wins else np.nan,
                "flagged_total_pnl": float(flagged["pnl"].sum()),
                "kept_total_pnl": float(kept["pnl"].sum()),
                "kept_win_rate": kept_wins / len(kept) if len(kept) else np.nan,
                "pnl_removed_if_flagged_bucket": float(flagged["pnl"].sum()),
                # BIASED: uses exit-measured movement — cannot be acted on at entry
                "pnl_impact_if_skipped_POST_HOC_BIASED": float(kept["pnl"].sum() - baseline_pnl),
                "losses_avoided_per_winner_sacrificed": flagged_losses / flagged_wins if flagged_wins else np.inf,
            })

        df = pd.DataFrame(rows)
        # Post-hoc separation quality score — diagnostic only, biased (uses exit outcomes).
        # Renamed from "kill_switch_score" to prevent misuse as a strategy signal.
        df["post_hoc_separation_score (diagnostic only, biased)"] = (
            2.0 * df["loss_capture_rate"].fillna(0)
            - 1.0 * df["winner_penalty_rate"].fillna(0)
            + 0.0001 * (-df["pnl_removed_if_flagged_bucket"].fillna(0))
        )
        self.movement_thresholds = df

        # Center distance diagnostics.
        distance_bins = [0, 5, 10, 15, 20, 30, 50, np.inf]
        trades["exit_distance_bucket"] = pd.cut(
            trades["abs_exit_distance_from_center"],
            bins=distance_bins,
            labels=["0-5", "5-10", "10-15", "15-20", "20-30", "30-50", "50+"],
            include_lowest=True,
        )
        center_summary = trades.groupby("exit_distance_bucket", observed=False).agg(
            trades=("pnl", "size"),
            wins=("outcome", lambda s: int((s == "WIN").sum())),
            losses=("outcome", lambda s: int((s == "LOSS").sum())),
            total_pnl=("pnl", "sum"),
            avg_pnl=("pnl", "mean"),
            avg_underlying_move=("underlying_move_abs_pts", "mean"),
        ).reset_index()
        center_summary["win_rate"] = center_summary["wins"] / center_summary["trades"].replace(0, np.nan)
        self.center_distance_summary = center_summary

    def write_outputs(self):
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.trades.to_csv(OUTPUT_DIR / "trade_proxy_analysis.csv", index=False)
        self.spx.to_csv(OUTPUT_DIR / "spx_proxy_regimes.csv", index=False)
        self.summary["env_summary"].to_csv(OUTPUT_DIR / "environment_summary.csv", index=False)
        self.movement_thresholds.to_csv(OUTPUT_DIR / "movement_threshold_analysis.csv", index=False)
        self.center_distance_summary.to_csv(OUTPUT_DIR / "center_strike_distance_analysis.csv", index=False)
        self.write_report()
        self.write_movement_report()
        self.write_charts()
        self.write_movement_charts()

    def write_report(self):
        s = self.summary
        verdict = "YES - justify deeper true vanna/GEX testing" if s["high_proxy_vanna_losses"] >= 5 else "NO - not enough loss clustering in high proxy-vanna regimes"
        report = f"""
VANNA PROXY VALIDATOR REPORT
============================

DATA LIMITATION
---------------
This uses completed trades plus daily SPX OHLCV only. It is not true strike-level
vanna/GEX because historical option-chain Greeks, IV, OI, volume, and bid/ask are
not present. Read 'proxy vanna' as a volatility/range/stress-regime proxy.

SAMPLE
------
Total trades: {s['total_trades']}
Wins: {s['wins']}
Losses: {s['losses']}
Baseline win rate: {s['baseline_win_rate']:.1%}
Baseline total P/L: ${s['baseline_pnl']:,.0f}
Proxy-regime classified trades: {s['classified_env_trades']}
Proxy-regime unclassified trades: {s['unclassified_env_trades']}  (usually caused by missing SPX daily row)

KEY QUESTION
------------
Do 5+ losses cluster in HIGH proxy-vanna / high volatility regimes?
Answer: {verdict}
High proxy-vanna losses: {s['high_proxy_vanna_losses']} of {s['losses']} ({s['high_proxy_vanna_loss_share']:.1%})

WIN VS LOSS COMPARISON
----------------------
Average proxy-vanna score on wins: {s['avg_win_proxy_vanna_score']:.1f}
Average proxy-vanna score on losses: {s['avg_loss_proxy_vanna_score']:.1f}
Welch t-test p-value: {s['ttest_p_value']:.4f}
Mann-Whitney p-value: {s['mannwhitney_p_value']:.4f}

CORRELATIONS
------------
P/L vs proxy-vanna score: {s['corr_pnl_proxy_vanna_score']:.3f}
P/L vs absolute underlying move: {s['corr_pnl_abs_underlying_move']:.3f}
P/L vs absolute exit distance from center strike: {s['corr_pnl_abs_exit_distance']:.3f}

COUNTERFACTUAL: SKIP HIGH PROXY-VANNA DAYS
------------------------------------------
Trades retained: {s['skip_high_trades']} of {s['total_trades']}
Win rate if skipped: {s['skip_high_win_rate']:.1%}
Total P/L if skipped: ${s['skip_high_pnl']:,.0f}
P/L delta vs baseline: ${s['skip_high_pnl'] - s['baseline_pnl']:,.0f}

LOSS ATTRIBUTION PROXY
----------------------
"""
        for key, value in s["loss_attribution"].items():
            report += f"{key}: {value}\n"
        report += "\nENVIRONMENT SUMMARY\n-------------------\n"
        report += s["env_summary"].to_string(index=False)
        report += dedent("""

NEXT STEPS
----------
1. Treat this as a screening test, not proof.
2. If high proxy-vanna days explain many losses, paper-test a no-trade filter.
3. If movement/gamma pressure is stronger, use MOVEMENT_PRESSURE_REPORT.txt first.
4. For true vanna/GEX, collect historical SPX chains with IV, delta, gamma, vega, OI, and volume.
5. Keep sample-size discipline: 39 trades is useful for triage, not final proof.
""")
        (OUTPUT_DIR / "VANNA_PROXY_REPORT.txt").write_text(report)

    def write_movement_report(self):
        trades = self.trades
        losses = trades[trades["outcome"] == "LOSS"]
        wins = trades[trades["outcome"] == "WIN"]
        thresholds = self.movement_thresholds.copy()

        # Pick the threshold with best post-hoc separation, but only to illustrate the
        # diagnostic — NOT as a recommended entry rule.
        score_col = "post_hoc_separation_score (diagnostic only, biased)"
        candidate_pool = thresholds[(thresholds["flagged_trades"] >= 3) & (thresholds["kept_trades"] >= 10)].copy()
        if candidate_pool.empty:
            candidate_pool = thresholds.copy()
        best = candidate_pool.sort_values(score_col, ascending=False).iloc[0]

        loss_move_q = losses["underlying_move_abs_pts"].describe(percentiles=[0.25, 0.5, 0.75]).to_string()
        win_move_q = wins["underlying_move_abs_pts"].describe(percentiles=[0.25, 0.5, 0.75]).to_string()

        report = f"""
MOVEMENT / GAMMA PRESSURE DIAGNOSTIC REPORT
============================================

╔══════════════════════════════════════════════════════════════════════════════╗
║  POST-HOC DIAGNOSTIC — NOT A TRADABLE STRATEGY                             ║
║                                                                              ║
║  ALL movement / exit-distance metrics in this report are computed from      ║
║  EXIT data (underlyingClose at trade exit). They are OUTCOMES, not inputs   ║
║  you can observe at entry or mid-trade. Thresholds derived from exit        ║
║  movement RE-DESCRIBE the losers after the fact; they do not predict them.  ║
║                                                                              ║
║  For TRADABLE counterfactuals, see:                                          ║
║    • outputs/ENTRY_FILTER_REPORT.txt  (entry-time & prior-day range filters)║
║    • outputs/INTRADAY_STOP_SIMULATION.txt  (first-breach stop, if intraday  ║
║      data is present)                                                        ║
╚══════════════════════════════════════════════════════════════════════════════╝

WHAT THIS DIAGNOSTIC TESTS
---------------------------
It answers: "After the fact, on trades that lost — how large was the underlying
move and how far was price from the body strike at exit?" This is useful for
understanding the MECHANISM of losses, not for predicting or filtering them.

DATA LIMITATION
---------------
This uses entry and exit underlying prices only, NOT second-by-second or
minute-by-minute SPX path data. Every metric below (underlying_move_abs_pts,
abs_exit_distance_from_center) is measured AT THE ACTUAL EXIT and is therefore
unknowable at entry. Do NOT attempt to use these numbers as live rules.

BASELINE
--------
Total trades: {len(trades)}
Wins: {len(wins)}
Losses: {len(losses)}
Total P/L: ${trades['pnl'].sum():,.0f}
Baseline win rate: {(trades['outcome'].eq('WIN').mean()):.1%}

CORE FINDINGS (DIAGNOSTIC ONLY)
---------------------------------
Correlation: P/L vs absolute SPX move during trade (EXIT metric): {self.summary['corr_pnl_abs_underlying_move']:.3f}
Correlation: P/L vs absolute exit distance from center strike (EXIT metric): {self.summary['corr_pnl_abs_exit_distance']:.3f}

Loss movement distribution (EXIT-MEASURED — post-hoc):
{loss_move_q}

Win movement distribution (EXIT-MEASURED — post-hoc):
{win_move_q}

BEST POST-HOC SEPARATION THRESHOLD (DIAGNOSTIC ONLY — BIASED, NOT TRADABLE)
-----------------------------------------------------------------------------
⚠ WARNING: The number below is derived from EXIT-measured movement. Because a
  large loss BY DEFINITION has a large exit move, "flag trades that moved >=X pts"
  is a circular re-description of the losers. It is NOT a forward-looking signal
  you can act on at trade entry. The P/L delta shown is similarly BIASED.

Threshold: {best['threshold_pts']} SPX points (exit-measured, post-hoc)
Flagged trades: {int(best['flagged_trades'])}
Flagged losses: {int(best['flagged_losses'])} of {len(losses)} ({best['loss_capture_rate']:.1%})
Flagged winners: {int(best['flagged_wins'])} of {len(wins)} ({best['winner_penalty_rate']:.1%})
P/L of flagged trades: ${best['flagged_total_pnl']:,.0f}
If flagged trades had been skipped, kept P/L would be: ${best['kept_total_pnl']:,.0f}
P/L impact if skipped (POST-HOC BIASED — uses exit-measured movement,
  cannot be acted on at entry): ${best['pnl_impact_if_skipped_POST_HOC_BIASED']:,.0f}

IMPORTANT INTERPRETATION
------------------------
A profitable flagged bucket does NOT automatically mean the threshold is bad.
It may mean winners also move, but losses become catastrophic when movement
continues. With minute-level data, the correct implementation may be:
- reduce size after X points,
- take profit earlier when centered,
- exit only if move is away from center plus trade price deteriorates,
- avoid entries when price is already unstable.

POST-HOC THRESHOLD TABLE (EXIT DATA — DIAGNOSTIC ONLY, BIASED)
---------------------------------------------------------------
NOTE: "pnl_impact_if_skipped_POST_HOC_BIASED" uses exit-measured movement and
cannot be acted on at entry. It is labeled BIASED to prevent misuse.
"""
        display_cols = [
            "threshold_pts", "flagged_trades", "flagged_wins", "flagged_losses",
            "loss_capture_rate", "winner_penalty_rate", "flagged_total_pnl",
            "kept_total_pnl", "pnl_impact_if_skipped_POST_HOC_BIASED",
            "post_hoc_separation_score (diagnostic only, biased)",
        ]
        report += thresholds[display_cols].to_string(index=False)
        report += "\n\nCENTER STRIKE EXIT-DISTANCE SUMMARY (POST-HOC — EXIT DATA)\n"
        report += "------------------------------------------------------------\n"
        report += "abs_exit_distance_from_center is measured at exit — NOT known at entry.\n\n"
        report += self.center_distance_summary.to_string(index=False)
        report += dedent("""

PRACTICAL NEXT STEP
-------------------
Do not automate a hard stop from this report alone. Every input here is
post-hoc exit data; using it as a live signal would produce look-ahead bias.

The genuinely actionable next steps are:
1. See ENTRY_FILTER_REPORT.txt for time-of-day and prior-day range filters
   that ARE knowable at entry.
2. See INTRADAY_STOP_SIMULATION.txt for a first-breach stop simulation using
   intraday SPX path data — the tradable counterfactual.
3. Collect 50+ more trades before drawing firm conclusions (39-trade sample).
""")
        (OUTPUT_DIR / "MOVEMENT_PRESSURE_REPORT.txt").write_text(report)

    def write_charts(self):
        trades = self.trades
        wins = trades[trades["outcome"] == "WIN"]
        losses = trades[trades["outcome"] == "LOSS"]
        env_summary = self.summary["env_summary"]

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("0DTE SPX SIB - Proxy Vanna / Volatility Regime Analysis", fontsize=15, fontweight="bold")

        ax = axes[0, 0]
        ax.scatter(wins["proxy_vanna_score"], wins["pnl"], color="green", alpha=0.7, label="Wins")
        ax.scatter(losses["proxy_vanna_score"], losses["pnl"], color="red", alpha=0.8, label="Losses")
        ax.axhline(0, color="black", linestyle="--")
        ax.set_title("P/L vs Proxy-Vanna Score")
        ax.set_xlabel("Proxy-vanna score")
        ax.set_ylabel("P/L ($)")
        ax.legend()
        ax.grid(alpha=0.25)

        ax = axes[0, 1]
        ax.bar(env_summary["proxy_vanna_env"].astype(str), env_summary["win_rate"], color=["green", "orange", "red"], alpha=0.75)
        ax.axhline(self.summary["baseline_win_rate"], color="blue", linestyle="--", label="Baseline")
        ax.set_ylim(0, 1)
        ax.set_title("Win Rate by Proxy-Vanna Environment")
        ax.set_ylabel("Win rate")
        ax.legend()
        ax.grid(axis="y", alpha=0.25)

        ax = axes[1, 0]
        losses["loss_attribution"].value_counts().plot(kind="bar", ax=ax, color="tomato")
        ax.set_title("Loss Attribution Proxy")
        ax.set_ylabel("Number of losses")
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.25)

        ax = axes[1, 1]
        ax.scatter(trades["underlying_move_abs_pts"], trades["pnl"], c=np.where(trades["outcome"] == "WIN", "green", "red"), alpha=0.75)
        ax.axhline(0, color="black", linestyle="--")
        ax.set_title("P/L vs Absolute Underlying Move During Trade")
        ax.set_xlabel("Absolute SPX move during trade")
        ax.set_ylabel("P/L ($)")
        ax.grid(alpha=0.25)

        plt.tight_layout()
        fig.savefig(OUTPUT_DIR / "vanna_proxy_summary.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    def write_movement_charts(self):
        trades = self.trades
        wins = trades[trades["outcome"] == "WIN"]
        losses = trades[trades["outcome"] == "LOSS"]
        thresholds = self.movement_thresholds

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("0DTE SPX SIB - Movement / Gamma Pressure Diagnostics", fontsize=15, fontweight="bold")

        ax = axes[0, 0]
        ax.scatter(wins["underlying_move_abs_pts"], wins["pnl"], color="green", alpha=0.7, label="Wins")
        ax.scatter(losses["underlying_move_abs_pts"], losses["pnl"], color="red", alpha=0.8, label="Losses")
        ax.axhline(0, color="black", linestyle="--")
        ax.set_title("P/L vs Absolute SPX Move")
        ax.set_xlabel("Absolute SPX move during trade")
        ax.set_ylabel("P/L ($)")
        ax.legend()
        ax.grid(alpha=0.25)

        ax = axes[0, 1]
        ax.plot(thresholds["threshold_pts"], thresholds["loss_capture_rate"], marker="o", label="Loss capture rate", color="red")
        ax.plot(thresholds["threshold_pts"], thresholds["winner_penalty_rate"], marker="o", label="Winner penalty rate", color="green")
        ax.set_title("Threshold Tradeoff: Losses Captured vs Winners Sacrificed")
        ax.set_xlabel("Movement threshold (SPX points)")
        ax.set_ylabel("Rate")
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.grid(alpha=0.25)

        ax = axes[1, 0]
        ax.plot(thresholds["threshold_pts"], thresholds["flagged_total_pnl"], marker="o", label="P/L of flagged bucket")
        ax.plot(thresholds["threshold_pts"], thresholds["kept_total_pnl"], marker="o", label="P/L if kept under threshold")
        ax.axhline(0, color="black", linestyle="--")
        ax.set_title("P/L by Movement Threshold Bucket")
        ax.set_xlabel("Movement threshold (SPX points)")
        ax.set_ylabel("P/L ($)")
        ax.legend()
        ax.grid(alpha=0.25)

        ax = axes[1, 1]
        ax.scatter(wins["abs_exit_distance_from_center"], wins["pnl"], color="green", alpha=0.7, label="Wins")
        ax.scatter(losses["abs_exit_distance_from_center"], losses["pnl"], color="red", alpha=0.8, label="Losses")
        ax.axhline(0, color="black", linestyle="--")
        ax.set_title("P/L vs Exit Distance from Center Strike")
        ax.set_xlabel("Absolute exit distance from body strike")
        ax.set_ylabel("P/L ($)")
        ax.legend()
        ax.grid(alpha=0.25)

        plt.tight_layout()
        fig.savefig(OUTPUT_DIR / "movement_pressure_summary.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    def run(self):
        self.load_data()
        self.engineer_features()
        self.summarize()
        self.analyze_movement_thresholds()
        self.write_outputs()
        print(f"Analysis complete. Outputs written to {OUTPUT_DIR}")

        # --- Tradable entry filters (entry-time-safe, zero look-ahead) ---
        try:
            from entry_filters import run_entry_filter_analysis
        except ImportError:
            import sys
            sys.path.insert(0, str(Path(__file__).parent))
            from entry_filters import run_entry_filter_analysis
        run_entry_filter_analysis(self.trades, self.spx, OUTPUT_DIR)

        # --- Intraday first-breach stop simulation (only if data file is present) ---
        try:
            from intraday_sim import run_intraday_stop_simulation
        except ImportError:
            import sys
            sys.path.insert(0, str(Path(__file__).parent))
            from intraday_sim import run_intraday_stop_simulation
        run_intraday_stop_simulation(self.trades, DATA_DIR / "spx_intraday.csv", OUTPUT_DIR)


if __name__ == "__main__":
    VannaProxyValidator().run()
