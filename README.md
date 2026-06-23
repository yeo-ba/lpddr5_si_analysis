# LPDDR5 6400 Mbps Channel SI Analysis

End-to-end LTspice and Python workflow for a simplified LPDDR5 DQ-interface
memory-down channel experiment.

This is not a JEDEC conformance test. It uses LPDDR5-like operating conditions
and project-defined analysis pass limits.

## Experiment Conditions

| Item | Value |
|------|------:|
| VDDQ | 0.50 V |
| VREF | 0.25 V |
| Data rate | 6400 Mbps |
| Unit interval | 156.25 ps |
| PULSE rise/fall | 10 ps |
| PULSE high time | 78.125 ps |
| Timing budget | UI/2 = 78.125 ps |
| Trace impedance | 42 ohm |

## Sweep

| Parameter | Values |
|-----------|--------|
| Ron | 28, 30, 34, 38, 40 ohm |
| ODT effective | 40, 48, 60, 80, 120 ohm |
| Trace length | 5, 10, 15, 20, 25 mm |
| C_decap | 47, 100, 220, 470, 680, 1000, 2200, 4700 nF |

The ODT netlist uses split termination with each leg set to `2 * ODT`, so the
Thevenin equivalent resistance equals the requested ODT value.

## Analysis Pass Limits

| Metric | Limit |
|--------|------:|
| Overshoot | < 15 % |
| Eye height | > 100 mV |
| SSN | < 100 mV |
| Setup margin | > 0 ps |

## ML Optimization Constraints

| Metric | Constraint |
|--------|-----------:|
| Overshoot | < 15 % |
| Eye height | > 100 mV |
| SSN | < 100 mV |
| Setup margin | > 0 ps |

The composite score is a normalized desirability score where higher is better.

## Run

```bash
python main.py
```

Individual steps:

```bash
python generate_netlist.py
python run_simulations.py
python extract_metrics.py
python analysis.py
python ml_optimize.py
python generate_report.py
```

## Outputs

```text
netlists/
raw_outputs/
data/simulation_results.csv
data/optimization_results.csv
plots/
models/rf_model.pkl
reports/LPDDR5_SI_Report.md
```
