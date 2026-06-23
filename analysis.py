"""
LPDDR5 SI Analysis - Data Analysis & Visualization
7 publication-quality plots saved to plots/ at 150 dpi
"""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

try:
    from PyLTSpice import RawRead as _RawRead
except ImportError:
    _RawRead = None  # plot7 gracefully skipped

# Constants
VDDQ: float = 0.50
VREF: float = 0.25

DATA_DIR = Path("data")
PLOT_DIR = Path("plots")
RAW_DIR  = Path("raw_outputs")

plt.style.use("dark_background")
COLORS = plt.cm.tab10.colors


# ─────────────────── data loader ────────────────────

def load_results() -> pd.DataFrame:
    """Load simulation_results.csv; raise if missing."""
    path = DATA_DIR / "simulation_results.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Results not found: {path}. Run extract_metrics.py first."
        )
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} simulation records")
    return df


# ─────────────────── individual plots ───────────────

def plot1_overshoot_vs_odt(df: pd.DataFrame) -> None:
    """Overshoot vs ODT - 5 lines, one per Ron value."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, ron in enumerate([28, 30, 34, 38, 40]):
        grp = (
            df[df["Ron_ohm"] == ron]
            .groupby("ODT_ohm")["overshoot_pct"]
            .mean()
            .reset_index()
        )
        ax.plot(
            grp["ODT_ohm"], grp["overshoot_pct"],
            marker="o", color=COLORS[i], linewidth=2,
            label=f"Ron={ron}ohm", markersize=8,
        )

    ax.axhline(15, color="red", linestyle="--", alpha=0.85, label="Analysis limit 15%")
    ax.set_xlabel("ODT Resistance (ohm)", fontsize=12)
    ax.set_ylabel("Overshoot (%)", fontsize=12)
    ax.set_title("LPDDR5 6400 Mbps Overshoot vs ODT per Ron", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "plot1_overshoot_vs_odt.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[OK] Plot 1: Overshoot vs ODT")


def plot2_eye_height_heatmap(df: pd.DataFrame) -> None:
    """Eye height heatmap - rows=ODT, cols=trace_length."""
    fig, ax = plt.subplots(figsize=(10, 7))
    pivot = df.pivot_table(
        values="eye_height_mV",
        index="ODT_ohm",
        columns="trace_length_mm",
        aggfunc="mean",
    )
    sns.heatmap(
        pivot, annot=True, fmt=".0f", cmap="RdYlGn",
        linewidths=0.5, ax=ax,
        cbar_kws={"label": "Eye Height (mV)"},
    )
    ax.set_title(
        "Eye Height Heatmap: ODT x Trace Length\n(mean across Ron & C_decap)",
        fontsize=14,
    )
    ax.set_xlabel("Trace Length (mm)", fontsize=12)
    ax.set_ylabel("ODT Resistance (ohm)", fontsize=12)
    ax.text(
        0.01, 0.01, "Analysis pass min: 100 mV",
        transform=ax.transAxes, color="white", fontsize=9, alpha=0.7,
    )
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "plot2_eye_height_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[OK] Plot 2: Eye Height Heatmap")


def plot3_ssn_vs_cdecap(df: pd.DataFrame) -> None:
    """SSN vs C_decap - shows ESL resonance effect across three regions."""
    fig, ax = plt.subplots(figsize=(11, 6))

    for i, ron in enumerate([28, 34, 40]):
        grp = (
            df[df["Ron_ohm"] == ron]
            .groupby("C_decap_nF")["SSN_mV"]
            .mean()
        )
        ax.semilogx(
            grp.index, grp.values,
            marker="D", color=COLORS[i], linewidth=2,
            label=f"Ron={ron}ohm", markersize=8,
        )

    # ESL effect regions
    ax.axvspan(47,   470,  alpha=0.08, color="green",  label="(1) Decreasing (cap dominant)")
    ax.axvspan(470,  1000, alpha=0.08, color="yellow", label="(2) Saturation")
    ax.axvspan(1000, 4700, alpha=0.08, color="red",    label="(3) Increasing (ESL dominant)")

    ax.axhline(100.0, color="red", linestyle="--", alpha=0.85, label="Analysis pass limit 100 mV")
    ax.set_xlabel("C_decap (nF) - log scale", fontsize=12)
    ax.set_ylabel("SSN peak-to-peak (mV)", fontsize=12)
    ax.set_title(
        "SSN vs C_decap - ESL Resonance Effect (ESL=0.4nH)\n"
        "(1) Decreasing  (2) Saturation  (3) Increasing",
        fontsize=14,
    )
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "plot3_ssn_vs_cdecap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[OK] Plot 3: SSN vs C_decap (ESL effect)")


def plot4_setup_margin_vs_tracelen(df: pd.DataFrame) -> None:
    """Setup margin vs trace length - 5 lines, one per ODT value."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, odt in enumerate(sorted(df["ODT_ohm"].unique())):
        grp = (
            df[df["ODT_ohm"] == odt]
            .groupby("trace_length_mm")["setup_margin_ps"]
            .mean()
            .reset_index()
        )
        ax.plot(
            grp["trace_length_mm"], grp["setup_margin_ps"],
            marker="s", color=COLORS[i], linewidth=2,
            label=f"ODT={int(odt)}ohm", markersize=8,
        )

    ax.axhline(0, color="red", linestyle="--", alpha=0.85, label="Analysis limit 0 ps")
    ax.set_xlabel("Trace Length (mm)", fontsize=12)
    ax.set_ylabel("Setup Margin (ps)", fontsize=12)
    ax.set_title(
        "Timing Margin vs Trace Length per ODT\n(analysis budget = UI/2 = 78.125 ps)",
        fontsize=14,
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "plot4_setup_margin_vs_tracelen.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[OK] Plot 4: Setup Margin vs Trace Length")


def plot5_jedec_compliance(df: pd.DataFrame) -> None:
    """Analysis pass rate by each of the 4 design parameters (2x2 subplots)."""
    overall = df["analysis_pass"].mean() * 100
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    params = [
        ("Ron_ohm",         "Ron (ohm)"),
        ("ODT_ohm",         "ODT (ohm)"),
        ("trace_length_mm", "Trace Length (mm)"),
        ("C_decap_nF",      "C_decap (nF)"),
    ]

    for ax, (col, label) in zip(axes, params):
        compliance = df.groupby(col)["analysis_pass"].mean() * 100
        bars = ax.bar(
            range(len(compliance)), compliance.values,
            color=[COLORS[i % 10] for i in range(len(compliance))],
            edgecolor="white", linewidth=0.5,
        )
        ax.set_xticks(range(len(compliance)))
        ax.set_xticklabels([str(v) for v in compliance.index], fontsize=9)
        ax.set_xlabel(label, fontsize=11)
        ax.set_ylabel("Analysis Pass Rate (%)", fontsize=11)
        ax.set_title(f"Pass Rate by {label}", fontsize=12)
        ax.set_ylim(0, 110)
        ax.axhline(100, color="lime", linestyle="--", alpha=0.4)
        ax.grid(True, alpha=0.2, axis="y")
        for bar, val in zip(bars, compliance.values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{val:.0f}%", ha="center", va="bottom", fontsize=8,
            )

    fig.suptitle(
        f"Analysis Pass Rate by Parameter\n"
        f"(Overall: {overall:.1f}%)",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "plot5_jedec_compliance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[OK] Plot 5: Analysis compliance by Parameter")


def plot6_correlation_matrix(df: pd.DataFrame) -> None:
    """Pearson correlation heatmap - all inputs + SI outputs."""
    cols = [
        "Ron_ohm", "ODT_ohm", "trace_length_mm", "C_decap_nF",
        "overshoot_pct", "undershoot_pct", "prop_delay_ps",
        "eye_height_mV", "SSN_mV", "transition_pp_mV",
        "setup_margin_ps", "rise_time_ps",
        "analysis_pass",
    ]
    corr = df[cols].corr()
    mask = np.eye(len(cols), dtype=bool)

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="coolwarm",
        center=0, square=True, linewidths=0.5,
        mask=mask, ax=ax,
        cbar_kws={"label": "Pearson r"},
    )
    ax.set_title("SI Metrics Correlation Matrix (Inputs & Outputs)", fontsize=14)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "plot6_correlation_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[OK] Plot 6: Correlation Matrix")


def plot7_best_worst_waveform(df: pd.DataFrame) -> None:
    """Best (max eye) vs worst (min eye) V(net_rx) overlay from real .raw data."""
    if _RawRead is None:
        print("[WARN] PyLTSpice unavailable - skipping Plot 7 (waveform overlay)")
        return

    pass_df = df[df["analysis_pass"] == 1]
    fail_df = df[df["analysis_pass"] == 0]

    if pass_df.empty or fail_df.empty:
        print("[WARN] Need both passing and failing cases for waveform overlay - skipping")
        return

    best_row  = pass_df.loc[pass_df["eye_height_mV"].idxmax()]
    worst_row = fail_df.loc[fail_df["eye_height_mV"].idxmin()]

    def raw_name(row: pd.Series) -> str:
        return (
            f"sim_Ron{int(row['Ron_ohm'])}_ODT{int(row['ODT_ohm'])}_"
            f"TL{int(row['trace_length_mm'])}_C{int(row['C_decap_nF'])}.raw"
        )

    fig, ax = plt.subplots(figsize=(12, 7))

    for row, label, color, ls in [
        (best_row,  "Best - Analysis pass", "lime", "-"),
        (worst_row, "Worst - Analysis FAIL", "red",  "--"),
    ]:
        raw_path = RAW_DIR / raw_name(row)
        if not raw_path.exists():
            print(f"  [WARN] Not found: {raw_path}")
            continue
        try:
            raw  = _RawRead(str(raw_path))
            time = np.array(raw.get_trace("time").get_wave(0)) * 1e12    # -> ps
            v_rx = np.array(raw.get_trace("V(net_rx)").get_wave(0)) * 1e3  # -> mV
            del raw
            info = (
                f"Ron={int(row['Ron_ohm'])}ohm  ODT={int(row['ODT_ohm'])}ohm  "
                f"TL={int(row['trace_length_mm'])}mm  C={int(row['C_decap_nF'])}nF\n"
                f"Eye={row['eye_height_mV']:.0f}mV  SSN={row['SSN_mV']:.1f}mV"
            )
            ax.plot(time, v_rx, color=color, linestyle=ls, linewidth=2,
                    label=f"{label}\n{info}")
        except Exception as exc:
            print(f"  [WARN] Could not load {raw_path.name}: {exc}")

    ax.axhline(VDDQ * 1000, color="cyan",   linestyle=":", alpha=0.6,
               label=f"VDDQ ({VDDQ}V)")
    ax.axhline(VREF * 1000, color="yellow", linestyle=":", alpha=0.6,
               label=f"VREF ({VREF}V)")
    ax.axhline(0,           color="white",  linestyle=":", alpha=0.3)
    ax.set_xlabel("Time (ps)", fontsize=12)
    ax.set_ylabel("V(net_rx) (mV)", fontsize=12)
    ax.set_title(
        "Best vs Worst Case: V(net_rx) Waveform\n(Real LTspice Simulation Data)",
        fontsize=14,
    )
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 4000])
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "plot7_best_worst_waveform.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[OK] Plot 7: Best vs Worst Waveform")


# ─────────────────── main pipeline ──────────────────

def run_analysis() -> pd.DataFrame:
    """Generate all 7 plots and print summary statistics."""
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_results()
    df = df.dropna(subset=["prop_delay_ps", "setup_margin_ps", "rise_time_ps"])
    print(f"Clean rows after NaN drop: {len(df)}\n")

    plot1_overshoot_vs_odt(df)
    plot2_eye_height_heatmap(df)
    plot3_ssn_vs_cdecap(df)
    plot4_setup_margin_vs_tracelen(df)
    plot5_jedec_compliance(df)
    plot6_correlation_matrix(df)
    plot7_best_worst_waveform(df)

    # ── summary ───────────────────────────────────────
    pass_n    = int(df["analysis_pass"].sum())
    pass_rate = pass_n / len(df) * 100

    pass_df = df[df["analysis_pass"] == 1]
    if pass_df.empty:
        best = df.nlargest(1, "eye_height_mV").iloc[0]
        best_label = "Best fail"
    else:
        best = pass_df.nlargest(1, "eye_height_mV").iloc[0]
        best_label = "Best pass"
    worst = df.nsmallest(1, "eye_height_mV").iloc[0]

    print(f"\n{'='*60}")
    print("ANALYSIS SUMMARY")
    print(f"{'='*60}")
    print(f"  Total records : {len(df)}")
    print(f"  Analysis pass    : {pass_n}/{len(df)} ({pass_rate:.1f}%)")
    print(f"\n  {best_label:<9}| Ron={best['Ron_ohm']:.0f}ohm  ODT={best['ODT_ohm']:.0f}ohm  "
          f"TL={best['trace_length_mm']:.0f}mm  C={best['C_decap_nF']:.0f}nF")
    print(f"         Eye={best['eye_height_mV']:.0f}mV  "
          f"SSN={best['SSN_mV']:.1f}mV  Over={best['overshoot_pct']:.1f}%")
    print(f"\n  Worst | Ron={worst['Ron_ohm']:.0f}ohm  ODT={worst['ODT_ohm']:.0f}ohm  "
          f"TL={worst['trace_length_mm']:.0f}mm  C={worst['C_decap_nF']:.0f}nF")
    print(f"         Eye={worst['eye_height_mV']:.0f}mV  "
          f"SSN={worst['SSN_mV']:.1f}mV  Over={worst['overshoot_pct']:.1f}%")
    print(f"{'='*60}")

    return df


if __name__ == "__main__":
    run_analysis()



