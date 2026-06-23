"""
LPDDR5 SI Analysis - Raw File Parser & SI Metrics Extractor
Processes .raw files ONE AT A TIME (del after use) to stay under ~50MB RAM peak.
1000 files x ~500KB = ~500MB total - never loaded simultaneously.
"""

import csv
import re
from pathlib import Path
from typing import Optional

import numpy as np

try:
    from PyLTSpice import RawRead
except ImportError as exc:
    raise ImportError(
        "PyLTSpice not installed. Run: pip install PyLTSpice"
    ) from exc

# LPDDR5 DQ-interface experiment constants
VDDQ: float = 0.50             # V
VREF: float = 0.25             # V
BIT_PERIOD: float = 156.25e-12 # s, 6400 Mbps UI
TDS:  float = 0.5 * BIT_PERIOD # s, analysis timing budget for this simplified model
TR_TF: float = 10e-12
TON: float = 78.125e-12

# Analysis pass limits. These are not a JEDEC conformance test.
ANALYSIS_OVERSHOOT_MAX:    float = 15.0  # %
ANALYSIS_EYE_HEIGHT_MIN:   float = 100.0 # mV
ANALYSIS_SSN_MAX:          float = 100.0 # mV
ANALYSIS_SETUP_MARGIN_MIN: float = 0.0   # ps
NOISE_WINDOW: float = 10e-12             # s, centered on bit sample points

RAW_DIR  = Path("raw_outputs")
DATA_DIR = Path("data")


# ─────────────────────────── helpers ───────────────────────────

def find_crossing(
    time_arr: np.ndarray,
    volt_arr: np.ndarray,
    threshold: float,
    rising: bool = True,
    start_idx: int = 0,
) -> Optional[float]:
    """
    Linear-interpolation threshold crossing search.

    Args:
        time_arr:  Time axis (s)
        volt_arr:  Voltage waveform (V)
        threshold: Crossing level (V)
        rising:    True -> rising edge, False -> falling
        start_idx: First index to consider

    Returns:
        Crossing time in seconds, or None if not found
    """
    t = time_arr[start_idx:]
    v = volt_arr[start_idx:]
    for i in range(len(v) - 1):
        if rising:
            if v[i] < threshold <= v[i + 1]:
                frac = (threshold - v[i]) / (v[i + 1] - v[i])
                return float(t[i] + frac * (t[i + 1] - t[i]))
        else:
            if v[i] > threshold >= v[i + 1]:
                frac = (v[i] - threshold) / (v[i] - v[i + 1])
                return float(t[i] + frac * (t[i + 1] - t[i]))
    return None


def _sample_at(time_arr: np.ndarray, volt_arr: np.ndarray, sample_time: float) -> float:
    """Linear interpolation sample helper."""
    return float(np.interp(sample_time, time_arr, volt_arr))


# ─────────────────────── core extractor ────────────────────────

def extract_metrics_from_raw(raw_path: str, params: dict) -> Optional[dict]:
    """
    Open one .raw file, extract SI metrics, then free memory.

    Args:
        raw_path: Absolute path to .raw file
        params:   Dict with Ron_ohm, ODT_ohm, trace_length_mm, C_decap_nF

    Returns:
        Metrics dict, or None on parse/waveform error
    """
    try:
        raw   = RawRead(raw_path)
        time  = np.array(raw.get_trace("time").get_wave(0),        dtype=np.float64)
        v_rx  = np.array(raw.get_trace("V(net_rx)").get_wave(0),   dtype=np.float64)
        v_drv = np.array(raw.get_trace("V(net_driver)").get_wave(0), dtype=np.float64)
        del raw  # free ~500KB immediately

        # ── a) Overshoot ──────────────────────────────────────
        v_max    = float(np.max(v_rx))
        overshoot = max(0.0, (v_max - VDDQ) / VDDQ * 100.0)

        # ── b) Undershoot (t > 100ps) ─────────────────────────
        late_mask = time > 100e-12
        v_rx_late = v_rx[late_mask]
        v_min     = float(np.min(v_rx_late)) if len(v_rx_late) else 0.0
        undershoot = max(0.0, abs(min(v_min, 0.0)) / VDDQ * 100.0)

        # ── c) Propagation delay ──────────────────────────────
        # Use 50% of each waveform's own peak swing (robust when signal
        # does not reach VREF due to heavy capacitive loading)
        drv_swing = v_drv.max() - v_drv.min()
        rx_swing  = v_rx.max()  - v_rx.min()
        v50_drv = v_drv.min() + 0.5 * drv_swing if drv_swing > 0.01 else VREF
        v50_rx  = v_rx.min()  + 0.5 * rx_swing  if rx_swing  > 0.005 else VREF

        t50_drv = find_crossing(time, v_drv, v50_drv, rising=True)
        t50_rx  = find_crossing(time, v_rx,  v50_rx,  rising=True)
        prop_delay = (
            (t50_rx - t50_drv) * 1e12
            if (t50_drv is not None and t50_rx is not None)
            else float("nan")
        )
        sample_shift = (t50_rx - t50_drv) if (t50_drv is not None and t50_rx is not None) else 0.0

        # ── d) Eye height - sample at PULSE bit-centre times ──
        PULSE_HIGH_CTR = TR_TF + 0.5 * TON
        PULSE_LOW_CTR = TR_TF + TON + TR_TF + 0.5 * (
            BIT_PERIOD - (2 * TR_TF + TON)
        )

        eye_ones:  list[float] = []
        eye_zeros: list[float] = []
        settled_pp: list[float] = []
        for n in range(int((time[-1] - 100e-12) / BIT_PERIOD) + 2):
            t_high = n * BIT_PERIOD + PULSE_HIGH_CTR + sample_shift
            t_low  = n * BIT_PERIOD + PULSE_LOW_CTR + sample_shift
            if 100e-12 <= t_high < time[-1]:
                eye_ones.append(_sample_at(time, v_rx, t_high))
                w = v_rx[(time >= t_high - NOISE_WINDOW) & (time <= t_high + NOISE_WINDOW)]
                if len(w) > 1:
                    settled_pp.append(float(np.ptp(w)))
            if 100e-12 <= t_low < time[-1]:
                eye_zeros.append(_sample_at(time, v_rx, t_low))
                w = v_rx[(time >= t_low - NOISE_WINDOW) & (time <= t_low + NOISE_WINDOW)]
                if len(w) > 1:
                    settled_pp.append(float(np.ptp(w)))

        mean_ones  = float(np.mean(eye_ones))  if eye_ones  else float(np.max(v_rx))
        mean_zeros = float(np.mean(eye_zeros)) if eye_zeros else float(np.min(v_rx))
        eye_height = (mean_ones - mean_zeros) * 1000.0  # mV

        # ── e) Settled sample noise and transition peak-to-peak ─
        ssn = max(settled_pp) * 1000.0 if settled_pp else 0.0
        transition_pp = 0.0
        t_first_rx = t50_rx if t50_rx is not None else (
            find_crossing(time, v_rx, v_rx.min() + 0.1 * rx_swing, rising=True)
        )
        if t_first_rx is not None:
            w_mask   = (time >= t_first_rx - 100e-12) & (time <= t_first_rx + 100e-12)
            v_window = v_rx[w_mask]
            if len(v_window) > 0:
                transition_pp = float(np.ptp(v_window)) * 1000.0  # mV

        # ── f) Rise time 10% -> 90% at V(net_rx) ─────────────
        # The receiver is ODT-biased and may not swing rail-to-rail, so use
        # the measured Rx swing instead of VDDQ absolute thresholds.
        v10 = v_rx.min() + 0.10 * rx_swing
        v90 = v_rx.min() + 0.90 * rx_swing
        start_idx = int(np.searchsorted(time, 100e-12))
        t10 = find_crossing(time, v_rx, v10, rising=True, start_idx=start_idx)
        t90 = (
            find_crossing(time, v_rx, v90, rising=True, start_idx=int(np.searchsorted(time, t10)))
            if t10 is not None
            else None
        )
        rise_time = (
            (t90 - t10) * 1e12
            if (t10 is not None and t90 is not None)
            else float("nan")
        )

        # ── g) Setup-like aperture margin ─────────────────────
        setup_margin = (
            BIT_PERIOD * 1e12 * 0.5 - rise_time * 0.5
            if not np.isnan(rise_time)
            else float("nan")
        )

        # ── h) Analysis pass/fail ─────────────────────────────
        analysis_pass = bool(
            overshoot  < ANALYSIS_OVERSHOOT_MAX
            and eye_height > ANALYSIS_EYE_HEIGHT_MIN
            and ssn        < ANALYSIS_SSN_MAX
            and not np.isnan(setup_margin)
            and setup_margin > ANALYSIS_SETUP_MARGIN_MIN
        )

        def _r(val: float, n: int = 4) -> Optional[float]:
            return round(val, n) if not np.isnan(val) else None

        return {
            "Ron_ohm":          params["Ron_ohm"],
            "ODT_ohm":          params["ODT_ohm"],
            "trace_length_mm":  params["trace_length_mm"],
            "C_decap_nF":       params["C_decap_nF"],
            "overshoot_pct":    round(overshoot,  4),
            "undershoot_pct":   round(undershoot, 4),
            "prop_delay_ps":    _r(prop_delay),
            "eye_height_mV":    round(eye_height, 4),
            "SSN_mV":           round(ssn,         4),
            "transition_pp_mV":  round(transition_pp, 4),
            "setup_margin_ps":  _r(setup_margin),
            "rise_time_ps":     _r(rise_time),
            "analysis_pass":       int(analysis_pass),
        }

    except Exception as exc:
        print(f"    [FAIL] {Path(raw_path).name}: {exc}")
        return None


def parse_filename_params(raw_filename: str) -> Optional[dict]:
    """
    Extract sweep parameters from filename.
    Expected format: sim_Ron{r}_ODT{o}_TL{t}_C{c}.raw
    """
    try:
        match = re.fullmatch(
            r"sim_Ron(?P<ron>\d+(?:\.\d+)?)_ODT(?P<odt>\d+(?:\.\d+)?)_"
            r"TL(?P<tl>\d+(?:\.\d+)?)_C(?P<cdecap>\d+(?:\.\d+)?)\.raw",
            raw_filename,
        )
        if match is None:
            raise ValueError("not a sweep transient raw file")
        return {
            "Ron_ohm":         float(match.group("ron")),
            "ODT_ohm":         float(match.group("odt")),
            "trace_length_mm": float(match.group("tl")),
            "C_decap_nF":      float(match.group("cdecap")),
        }
    except Exception as exc:
        print(f"    [FAIL] Cannot parse filename '{raw_filename}': {exc}")
        return None


# ─────────────────────── main pipeline ─────────────────────────

def extract_all_metrics() -> list[dict]:
    """
    Process every .raw file sequentially; save results to CSV.

    Returns:
        List of metric dicts for all successfully parsed files
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    raw_files = sorted(
        p for p in RAW_DIR.glob("sim_*.raw")
        if not p.name.endswith(".op.raw")
    )
    if not raw_files:
        raise FileNotFoundError(
            f"No .raw files in {RAW_DIR}/. Run run_simulations.py first."
        )

    total = len(raw_files)
    print(f"\nExtracting metrics from {total} .raw files (one at a time)...\n")

    all_metrics: list[dict] = []
    failed = 0

    for idx, raw_path in enumerate(raw_files):
        params = parse_filename_params(raw_path.name)
        if params is None:
            failed += 1
            continue

        metrics = extract_metrics_from_raw(str(raw_path), params)
        if metrics is not None:
            all_metrics.append(metrics)
        else:
            failed += 1

        if (idx + 1) % 100 == 0 or (idx + 1) == total:
            print(
                f"  [{idx+1:4d}/{total}]  "
                f"extracted={len(all_metrics)}  failed={failed}"
            )

    if not all_metrics:
        print("[FAIL] No metrics extracted.")
        return []

    # Save to CSV
    out_path = DATA_DIR / "simulation_results.csv"
    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(all_metrics[0].keys()))
        writer.writeheader()
        writer.writerows(all_metrics)

    pass_n    = sum(1 for m in all_metrics if m.get("analysis_pass") == 1)
    pass_rate = pass_n / len(all_metrics) * 100

    print(f"\n[OK] Saved {len(all_metrics)} records -> {out_path}")
    print(f"[OK] Analysis pass rate: {pass_n}/{len(all_metrics)} ({pass_rate:.1f}%)")
    return all_metrics


if __name__ == "__main__":
    extract_all_metrics()

