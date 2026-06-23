"""
LPDDR5 SI Analysis - Parallel Batch Simulation Runner
Optimised for AMD Ryzen AI 9 HX 8745HS (12 cores / 24 threads)
MAX_WORKERS = 10  -> leaves 2 cores free for OS + file I/O
"""

import csv
import os
import shutil
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

MAX_WORKERS: int = 10   # tuned for 12-core 8745HS
TIMEOUT:     int = 30   # seconds per simulation

NETLIST_DIR    = Path("netlists")
RAW_OUTPUT_DIR = Path("raw_outputs")
DATA_DIR       = Path("data")


def find_ltspice() -> str:
    """
    Auto-detect LTspice installation path on Windows.

    Returns:
        Absolute path string to LTspice.exe

    Raises:
        FileNotFoundError: when LTspice cannot be located
    """
    username = os.environ.get("USERNAME", "")
    ltspice_paths = [
        rf"C:\Users\{username}\AppData\Local\Programs\ADI\LTspice\LTspice.exe",
        r"C:\Program Files\ADI\LTspice\LTspice.exe",
        r"C:\Program Files (x86)\ADI\LTspice\LTspice.exe",
        r"C:\Program Files (x86)\LTC\LTspiceXVII\XVIIx64.exe",
    ]
    ltspice_exe = next((p for p in ltspice_paths if os.path.exists(p)), None)
    if not ltspice_exe:
        raise FileNotFoundError(
            "LTspice not found.\n"
            "Install from: https://www.analog.com/en/resources/design-tools-and-calculators/"
            "ltspice-simulator.html\n"
            "Or update ltspice_paths list in run_simulations.py"
        )
    return ltspice_exe


def run_single_sim(args: tuple) -> dict:
    """
    Execute one LTspice simulation in batch mode.

    Args:
        args: (netlist_path_str, ltspice_exe_str)

    Returns:
        Dict with keys: file, success, [returncode | error]
    """
    netlist_path, ltspice_exe = args
    try:
        result = subprocess.run(
            [ltspice_exe, "-b", "-Run", netlist_path],
            capture_output=True,
            timeout=TIMEOUT,
        )
        return {
            "file":       netlist_path,
            "success":    result.returncode == 0,
            "returncode": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"file": netlist_path, "success": False, "error": "timeout"}
    except Exception as exc:
        return {"file": netlist_path, "success": False, "error": str(exc)}


def move_raw_files() -> int:
    """
    Move generated .raw files from netlists/ to raw_outputs/.

    Returns:
        Number of files moved
    """
    RAW_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    moved = 0
    for raw_file in NETLIST_DIR.glob("sim_*.raw"):
        if raw_file.name.endswith(".op.raw"):
            continue
        shutil.move(str(raw_file), str(RAW_OUTPUT_DIR / raw_file.name))
        moved += 1
    # Also move .log files
    for log_file in NETLIST_DIR.glob("*.log"):
        shutil.move(str(log_file), str(RAW_OUTPUT_DIR / log_file.name))
    return moved


def run_all_parallel(netlist_files: list, ltspice_exe: str) -> dict:
    """
    Execute all simulations concurrently with ProcessPoolExecutor.

    Args:
        netlist_files: Sorted list of Path objects for .net files
        ltspice_exe:   Path to LTspice executable

    Returns:
        Summary dict: {success, failed, errors, elapsed_seconds}
    """
    RAW_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    args = [(str(f), ltspice_exe) for f in netlist_files]
    results: dict = {"success": 0, "failed": 0, "errors": []}
    total = len(args)
    completed = 0
    start_time = time.time()

    print(f"\nStarting {total} simulations - {MAX_WORKERS} parallel workers")
    print(f"LTspice: {ltspice_exe}\n")

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(run_single_sim, a): a for a in args}

        for future in as_completed(futures):
            completed += 1
            res = future.result()

            if res["success"]:
                results["success"] += 1
            else:
                results["failed"] += 1
                results["errors"].append(res)

            if completed % 50 == 0 or completed == total:
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                eta  = (total - completed) / rate if rate > 0 else 0
                print(
                    f"  [{completed:4d}/{total}] "
                    f"OK={results['success']:4d}  "
                    f"FAIL={results['failed']:3d}  "
                    f"{rate:.1f} sim/s  "
                    f"ETA={eta:.0f}s"
                )

    # Save failed list
    if results["errors"]:
        failed_path = DATA_DIR / "failed_sims.csv"
        with open(failed_path, "w", newline="") as fh:
            writer = csv.DictWriter(
                fh, fieldnames=["file", "success", "error", "returncode"]
            )
            writer.writeheader()
            for err in results["errors"]:
                writer.writerow({
                    "file":       err.get("file", ""),
                    "success":    False,
                    "error":      err.get("error", err.get("returncode", "unknown")),
                    "returncode": err.get("returncode", ""),
                })
        print(f"\n  [WARN] Failed list -> {failed_path}")

    moved = move_raw_files()
    print(f"  [OK] Moved {moved} .raw files -> {RAW_OUTPUT_DIR}/")

    results["elapsed_seconds"] = time.time() - start_time
    return results


def main() -> dict:
    """Entry point: find LTspice, collect netlists, run parallel batch."""
    ltspice_exe = find_ltspice()
    print(f"LTspice detected: {ltspice_exe}")

    netlist_files = sorted(NETLIST_DIR.glob("sim_*.net"))
    if not netlist_files:
        raise FileNotFoundError(
            f"No .net files in {NETLIST_DIR}/. Run generate_netlist.py first."
        )

    results = run_all_parallel(netlist_files, ltspice_exe)

    elapsed = results["elapsed_seconds"]
    print(f"\n{'='*60}")
    print("SIMULATION COMPLETE")
    print(f"{'='*60}")
    print(f"  Total:   {results['success'] + results['failed']}")
    print(f"  Success: {results['success']}")
    print(f"  Failed:  {results['failed']}")
    print(f"  Time:    {elapsed:.1f}s  ({elapsed/60:.1f} min)")
    print(f"  Workers: {MAX_WORKERS}/12 cores  (OS reserved: 2)")
    print(f"{'='*60}")
    return results


if __name__ == "__main__":
    main()

