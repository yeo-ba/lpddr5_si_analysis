"""
LPDDR5 SI Analysis - LTspice Netlist Generator
LPDDR5 6400 Mbps DQ-interface memory-down topology
Parametric sweep: 5x5x5x8 = 1000 combinations
"""

import csv
import itertools
from pathlib import Path

# LPDDR5 DQ-interface experiment constants
VDDQ: float = 0.50          # V, LPDDR5 DQ I/O supply nominal
VREF: float = 0.25          # V, nominal mid-supply reference
BIT_PERIOD_PS: float = 156.25
TR_TF_PS: float = 10.0
TON_PS: float = 78.125
Z0: float = 42.0            # Ohm (FR4, εr=4.3)
PROP_VELOCITY: float = 1.8e8  # m/s

# Parametric sweep ranges
RON_VALUES:     list[int] = [28, 30, 34, 38, 40]                       # Ohm
ODT_VALUES:     list[int] = [40, 48, 60, 80, 120]                      # Ohm
TRACE_LENGTHS:  list[int] = [5, 10, 15, 20, 25]                        # mm
C_DECAP_VALUES: list[int] = [47, 100, 220, 470, 680, 1000, 2200, 4700] # nF

OUTPUT_DIR = Path("netlists")
DATA_DIR   = Path("data")


def generate_netlist(ron: float, odt: float,
                     trace_length_mm: float, c_decap_nf: float) -> str:
    """
    Generate LTspice netlist string for LPDDR5 memory-down channel.

    Args:
        ron: Driver output impedance (Ohm)
        odt: On-die termination resistance (Ohm)
        trace_length_mm: PCB trace length (mm)
        c_decap_nf: Decoupling capacitance (nF)

    Returns:
        Netlist text in LTspice format
    """
    td = (trace_length_mm * 1e-3) / PROP_VELOCITY  # propagation delay (s)
    c_decap_f = c_decap_nf * 1e-9                  # nF -> F
    odt_leg = 2.0 * odt

    return f"""; LPDDR5 Channel SI Model - 6400 Mbps DQ experiment
; Ron={ron}ohm  ODT={odt}ohm  TL={trace_length_mm}mm  Cdecap={c_decap_nf}nF
; ============================================================
* Memory-down topology, VDDQ={VDDQ}V, 6400Mbps (UI={BIT_PERIOD_PS}ps)

* Driver: PULSE source + series Ron
V1 net_source 0 PULSE(0 {VDDQ} 0 {TR_TF_PS}p {TR_TF_PS}p {TON_PS}p {BIT_PERIOD_PS}p)
R_drv net_source net_driver {ron}

* PCB Trace: lossless transmission line (Td={td:.4e}s, Z0={Z0}ohm)
T1 net_driver 0 net_via_in 0 Td={td:.4e} Z0={Z0}

* Via parasitics: pi-model
L_via  net_via_in net_pkg_in 0.4n
C_via1 net_via_in 0 0.075p
C_via2 net_pkg_in 0 0.075p

* Package bump (L + R + C)
L_pkg net_pkg_in net_rx_pre 0.4n
R_pkg net_rx_pre net_rx 0.07
C_pkg net_rx 0 0.15p

* ODT split termination: DC bias = VDDQ/2 = {VREF}V
* Each leg is 2x ODT so the Thevenin equivalent equals the requested ODT.
R_odt_up   net_vddq net_rx {odt_leg}
V_vddq     net_vddq 0      {VDDQ}
R_odt_down net_rx   0      {odt_leg}

* Decoupling capacitor with ESL + ESR
C_decap net_rx   net_esl {c_decap_f:.4e}
L_esl   net_esl  net_esr 0.4n
R_esr   net_esr  0 0.02

* Receiver input capacitance
C_rx net_rx 0 1p

* Save node voltages for PyLTSpice extraction
.save V(net_source) V(net_driver) V(net_rx)

* Simulation command (1ps max step, 2ns window)
.tran 0 2n 0 1p

* Measurement directives
.measure TRAN v_max MAX V(net_rx)
.measure TRAN v_min MIN V(net_rx) FROM=100p TO=2n
.measure TRAN t_rise TRIG V(net_source) VAL={0.1 * VDDQ:.3f} RISE=1 TARG V(net_rx) VAL={0.5 * VDDQ:.3f} RISE=1

.backanno
.end
"""


def generate_all_netlists() -> list[dict]:
    """
    Generate all 1000 netlist files and save parameter index CSV.

    Returns:
        List of parameter dicts (index, filename, Ron, ODT, TL, C_decap)
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    combinations = list(itertools.product(
        RON_VALUES, ODT_VALUES, TRACE_LENGTHS, C_DECAP_VALUES
    ))
    print(f"Generating {len(combinations)} netlists -> {OUTPUT_DIR}/")

    param_list: list[dict] = []

    for idx, (ron, odt, tl, cdecap) in enumerate(combinations):
        filename = f"sim_Ron{ron}_ODT{odt}_TL{tl}_C{cdecap}.net"
        filepath = OUTPUT_DIR / filename

        with open(filepath, "w", encoding="ascii") as fh:
            fh.write(generate_netlist(ron, odt, tl, cdecap))

        param_list.append({
            "index":           idx,
            "filename":        filename,
            "Ron_ohm":         ron,
            "ODT_ohm":         odt,
            "trace_length_mm": tl,
            "C_decap_nF":      cdecap,
        })

        if (idx + 1) % 200 == 0:
            print(f"  [{idx+1:4d}/{len(combinations)}] netlists written")

    # Save parameter index
    index_path = DATA_DIR / "param_index.csv"
    with open(index_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(param_list[0].keys()))
        writer.writeheader()
        writer.writerows(param_list)

    print(f"[OK] {len(param_list)} netlists saved -> {OUTPUT_DIR}/")
    print(f"[OK] Parameter index     -> {index_path}")
    return param_list


if __name__ == "__main__":
    generate_all_netlists()

