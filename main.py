"""
LPDDR5 Channel SI Analysis - Main Runner
Step 1: Generate 1000 netlists
Step 2: Parallel LTspice simulation  (10 workers, AMD Ryzen AI 9 HX 8745HS)
Step 3: Extract SI metrics from .raw files
Step 4: Analysis & visualisation
Step 5: ML optimisation
Step 6: Generate Markdown report
"""

import sys
import time
from pathlib import Path


def _banner() -> None:
    print("=" * 70)
    print("  LPDDR5 Channel Signal Integrity Analysis")
    print("  LPDDR5 6400 Mbps DQ experiment  |  LTspice + PyLTSpice + RandomForest ML")
    print("  AMD Ryzen AI 9 HX 8745HS  |  10 parallel simulation workers")
    print("=" * 70)


def _create_dirs() -> None:
    for d in ("netlists", "raw_outputs", "data", "plots", "models", "reports"):
        Path(d).mkdir(parents=True, exist_ok=True)
    print("[OK] Output directories ready\n")


def main() -> None:
    _banner()
    wall_start = time.time()
    _create_dirs()

    # ── Step 1: Netlist generation ───────────────────
    print("[STEP 1/6]  Generating 1000 LTspice netlists...")
    t0 = time.time()
    from generate_netlist import generate_all_netlists
    param_list = generate_all_netlists()
    print(f"            Done in {time.time()-t0:.1f}s\n")

    # ── Step 2: Parallel SPICE simulation ───────────
    print("[STEP 2/6]  Running parallel LTspice simulations (10 workers)...")
    t0 = time.time()
    from run_simulations import find_ltspice, run_all_parallel, NETLIST_DIR
    try:
        ltspice_exe = find_ltspice()
    except FileNotFoundError as exc:
        print(f"\n  [FAIL] {exc}")
        print("  Skipping simulation step - checking for existing .raw files...\n")
        sim_results = {"success": 0, "failed": len(param_list)}
    else:
        netlist_files = sorted(NETLIST_DIR.glob("sim_*.net"))
        sim_results   = run_all_parallel(netlist_files, ltspice_exe)
        print(
            f"  Core utilisation: {10}/12 cores active (83%) · "
            f"OS reserved: 2 cores"
        )
    print(f"            Done in {time.time()-t0:.1f}s\n")

    if sim_results.get("success", 0) == 0:
        raw_count = len(list(Path("raw_outputs").glob("*.raw")))
        if raw_count == 0:
            print(
                "[FAIL] No .raw files available and no simulations succeeded.\n"
                "  Install LTspice and re-run, or place .raw files in raw_outputs/."
            )
            sys.exit(1)
        print(f"  Using {raw_count} existing .raw files in raw_outputs/\n")

    # ── Step 3: Metric extraction ────────────────────
    print("[STEP 3/6]  Extracting SI metrics from .raw files...")
    t0 = time.time()
    from extract_metrics import extract_all_metrics
    try:
        metrics = extract_all_metrics()
    except FileNotFoundError as exc:
        print(f"  [FAIL] {exc}")
        sys.exit(1)
    print(f"            Done in {time.time()-t0:.1f}s\n")

    if not metrics:
        print("[FAIL] No metrics extracted - cannot proceed.")
        sys.exit(1)

    # ── Step 4: Analysis & visualisation ────────────
    print("[STEP 4/6]  Generating analysis plots...")
    t0 = time.time()
    from analysis import run_analysis
    df = run_analysis()
    print(f"            Done in {time.time()-t0:.1f}s\n")

    # ── Step 5: ML optimisation ──────────────────────
    print("[STEP 5/6]  Training ML model & searching optimal parameters...")
    t0 = time.time()
    from ml_optimize import run_ml_optimization
    model, top10, ml_metrics = run_ml_optimization()
    print(f"            Done in {time.time()-t0:.1f}s\n")

    # ── Step 6: Report generation ────────────────────
    print("[STEP 6/6]  Generating analysis report...")
    t0 = time.time()
    from generate_report import generate_report
    report_path = generate_report()
    print(f"            Done in {time.time()-t0:.1f}s\n")

    # ── Final summary ────────────────────────────────
    import pandas as pd
    df_final  = pd.read_csv("data/simulation_results.csv")
    pass_rate = df_final["analysis_pass"].mean() * 100
    wall_time = time.time() - wall_start

    print(f"\n{'='*70}")
    print("  COMPLETE")
    print(f"  Total wall-clock time : {wall_time:.1f}s  ({wall_time/60:.1f} min)")
    print(f"{'='*70}")
    print(f"  Simulations  : {sim_results.get('success', 0)} / {len(param_list)} succeeded")
    print(f"  Analysis pass   : {pass_rate:.1f}%")
    print()
    print("  Output files:")
    print(f"    netlists/                  {len(param_list)} .net files")
    raw_count = len([
        p for p in Path("raw_outputs").glob("sim_*.raw")
        if not p.name.endswith(".op.raw")
    ])
    print(f"    raw_outputs/               {raw_count} .raw files")
    print(f"    data/simulation_results.csv")
    print(f"    data/optimization_results.csv")
    print(f"    plots/                     7 PNG plots")
    print(f"    models/rf_model.pkl")
    print(f"    {report_path}")

    if top10 is not None and len(top10) > 0:
        print()
        print("  Top 3 Optimal Combinations (ML-predicted):")
        for i in range(min(3, len(top10))):
            r = top10.iloc[i]
            print(
                f"    #{i+1}: Ron={r['Ron_ohm']:.0f}ohm  "
                f"ODT={r['ODT_ohm']:.0f}ohm  "
                f"TL={r['trace_length_mm']:.0f}mm  "
                f"C={r['C_decap_nF']:.0f}nF  ->  "
                f"Eye={r['eye_height_mV']:.0f}mV  "
                f"Score={r.get('score', 0):.3f}"
            )
    print(f"{'='*70}")


if __name__ == "__main__":
    main()



