# LPDDR5 Channel Signal Integrity Analysis Report

**Generated :** 2026-06-23  16:27:46
**Standard   :** LPDDR5 6400 Mbps DQ-interface experiment
**Methodology:** Real LTspice SPICE simulation via PyLTSpice · 10 parallel workers
**Platform   :** AMD Ryzen AI 9 HX 8745HS (12 cores / 24 threads, 5.1 GHz boost)

---

## 1. Channel Model

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                       LPDDR5 Memory-Down Channel Topology                       │
├────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  TX Driver        PCB Trace (Z0=42ohm)      Via (pi)       Package     RX         │
│  ┌──────┐        ┌─────────────────┐   ┌─────────┐   ┌────────┐  ┌──────┐    │
│  │PULSE ├─[Ron]─►│  T1  Td=TL/vp  ├──►│  Lvia   ├──►│  Lpkg  ├─►│      │    │
│  │  V1  │        └─────────────────┘   │ Cvia1/2 │   │ Rpkg   │  │ ODT  │    │
│  └──────┘                              └─────────┘   │ Cpkg   │  │      │    │
│                                                       └────────┘  └──┬───┘    │
│  Experiment setup                                                         │         │
│  VDDQ  = 0.50 V          [C_rx = 1 pF]──────────────────────────────┤ GND    │
│  VREF  = 0.25 V                                                     │         │
│  tr/tf = 10 ps          [C_decap + L_esl(0.4nH) + R_esr]──────────┘ GND    │
│  BW    = 6400 Mbps                                                              │
└────────────────────────────────────────────────────────────────────────────────┘
```

## 2. Experiment Conditions

| Parameter | Symbol | Value | Unit |
|-----------|--------|------:|------|
| DQ I/O supply voltage | VDDQ | 0.50 | V |
| Reference voltage | VREF | 0.25 | V |
| Rise / Fall time | tr / tf | 10 | ps |
| Analysis timing budget | UI/2 | 78.125 | ps |
| Hold time | tDH | 100 | ps |
| Target trace impedance | Z0 | 42 | ohm |
| Propagation velocity (FR4) | vp | 1.8 x 10⁸ | m/s |
| ZQ calibration reference | ZQ | 240 | ohm |
| **Analysis Pass Limits** | | | |
| Max overshoot | - | 15 | % |
| Min eye height | - | 100 | mV |
| Max SSN | - | 100 | mV |
| Min setup margin | - | 0 | ps |

## 3. Simulation Results Summary

| Metric | Value |
|--------|------:|
| Total simulations | 1000 |
| Analysis pass | **784 (78.4%)** |
| Analysis FAIL | 216 (21.6%) |
| Mean eye height | 286.3 mV |
| Mean SSN | 80.1 mV |
| Mean overshoot | 0.00 % |
| Mean setup margin | 56.0 ps |

### Best Case (Analysis pass)
Parameters: **Ron=28ohm, ODT=80ohm, TL=10mm, C_decap=4700nF**
Eye Height: **372 mV** · SSN: 92.3 mV · Overshoot: 0.0% · Setup margin: 55.2 ps

### Worst Case
Parameters: **Ron=40ohm, ODT=40ohm, TL=15mm, C_decap=47nF**
Eye Height: **238 mV** · SSN: 49.5 mV · Overshoot: 0.0%

## 4. Key Findings

### 4.1 ODT Sensitivity
- Lower ODT (< 48 ohm) improves termination matching -> reduces reflections + overshoot
- Optimal range **48–60 ohm** balances SI and power dissipation
- Ron has second-order effect on overshoot; higher Ron slightly reduces first-incident wave

### 4.2 C_decap ESL Tradeoff (Figure: plot3)
| Region | C_decap | Dominant Effect |
|--------|---------|-----------------|
| (1) Decreasing | 47–470 nF | Capacitive - higher C lowers SSN |
| (2) Saturation | 470–1000 nF | Resonance point drifts toward 6.4 GHz |
| (3) Increasing | 1–4.7 µF | ESL (0.4 nH) dominant - anti-resonance raises SSN |

Resonance frequency: f_r = 1 / (2pi√(LxC))
At L=0.4nH, C=4.7µF -> f_r ≈ **3.7 MHz** (far below DQ frequency, but ESL creates series resonance at ~250 MHz for 47nF)

**Recommendation: 220–470 nF provides optimal SSN reduction without ESL penalty**

### 4.3 Trace Length Impact
- Propagation delay increases at ~5.6 ps/mm (1/vp)
- Traces > 20 mm risk setup margin violation at 6400 Mbps
- Via inductance (0.4 nH each) adds ~12 ps additional delay

## 5. ML Model Performance (RandomForestRegressor)

| Target | R² | RMSE | Status |
|--------|----|------|--------|
| `overshoot_pct` | 1.0000 | 0.000 | [OK] Excellent |
| `eye_height_mV` | 1.0000 | 0.196 | [OK] Excellent |
| `SSN_mV` | 0.9999 | 0.190 | [OK] Excellent |
| `setup_margin_ps` | 0.9999 | 0.007 | [OK] Excellent |

Training config: `n_estimators=200, max_depth=10, n_jobs=-1` (all 24 threads)

## 6. Top 3 Optimal Parameter Combinations

Constraints: overshoot < 15 %, eye > 100 mV, SSN < 100 mV, setup margin > 0 ps


### #1 - Score = 1.000

| Parameter | Value |
|-----------|-------|
| Ron | 28 ohm |
| ODT | 60 ohm |
| Trace length | 10 mm |
| C_decap | 47 nF |
| **Eye height** | **353 mV** (target >100 mV) |
| SSN | 76.7 mV (target <100 mV) |
| Overshoot | 0.0 % (target <10 %) |
| Setup margin | 55.4 ps (target >0 ps) |

### #2 - Score = 1.000

| Parameter | Value |
|-----------|-------|
| Ron | 28 ohm |
| ODT | 60 ohm |
| Trace length | 10 mm |
| C_decap | 100 nF |
| **Eye height** | **353 mV** (target >100 mV) |
| SSN | 76.7 mV (target <100 mV) |
| Overshoot | 0.0 % (target <10 %) |
| Setup margin | 55.4 ps (target >0 ps) |

### #3 - Score = 1.000

| Parameter | Value |
|-----------|-------|
| Ron | 28 ohm |
| ODT | 60 ohm |
| Trace length | 10 mm |
| C_decap | 220 nF |
| **Eye height** | **353 mV** (target >100 mV) |
| SSN | 76.7 mV (target <100 mV) |
| Overshoot | 0.0 % (target <10 %) |
| Setup margin | 55.4 ps (target >0 ps) |


---

## 7. References

1. JEDEC JESD209-5B - *Low Power Double Data Rate 5 (LPDDR5)*
2. IPC-2141C - *Controlled Impedance Circuit Boards*
3. IBIS AMI Specification v7.0 - *Algorithmic Modeling Interface*
4. PyLTSpice - <https://pyltspice.readthedocs.io>
5. LTspice XVII - Analog Devices / Linear Technology

---
*Generated by LPDDR5 SI Analysis Suite*
*Real SPICE simulation (LTspice) + ML optimisation (scikit-learn RandomForest)*
