"""
LPDDR5 SI Analysis - ML Optimization Pipeline
RandomForestRegressor (multi-output) + constraint-based optimisation
n_jobs=-1 -> all 24 threads on AMD Ryzen AI 9 HX 8745HS
"""

import itertools
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

DATA_DIR  = Path("data")
PLOT_DIR  = Path("plots")
MODEL_DIR = Path("models")

FEATURES: list[str] = ["Ron_ohm", "ODT_ohm", "trace_length_mm", "C_decap_nF"]
TARGETS:  list[str] = ["overshoot_pct", "eye_height_mV", "SSN_mV", "setup_margin_ps"]

# Stricter than the analysis pass limits for optimisation search
CONSTRAINTS: dict[str, tuple[str, float]] = {
    "overshoot_pct":   ("max", 10.0),   # < 10 %
    "eye_height_mV":   ("min", 100.0),  # > 100 mV
    "SSN_mV":          ("max", 100.0),  # < 100 mV
    "setup_margin_ps": ("min", 0.0),    # > 0 ps
}

plt.style.use("dark_background")


# ───────────────────── data ─────────────────────────

def load_data() -> pd.DataFrame:
    """Load and clean simulation_results.csv for ML training."""
    path = DATA_DIR / "simulation_results.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Results not found: {path}. Run extract_metrics.py first."
        )
    df = pd.read_csv(path).dropna(subset=FEATURES + TARGETS)
    print(f"ML dataset: {len(df)} clean records")
    return df


# ─────────────────── training ───────────────────────

def train_model(df: pd.DataFrame) -> tuple:
    """
    Train multi-output RandomForest; print per-target metrics.

    Args:
        df: Clean simulation dataframe

    Returns:
        (model, X_test, y_test, y_pred, metrics_dict)
    """
    X = df[FEATURES].values
    y = df[TARGETS].values

    # Stratified split preserves analysis_pass ratio
    try:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.20,
            stratify=df["analysis_pass"].values,
            random_state=42,
        )
    except ValueError:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.20, random_state=42
        )

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1,   # all 24 threads on Ryzen AI 9 HX 8745HS
    )

    print(
        f"\nTraining RandomForestRegressor  "
        f"(n_estimators=200, max_depth=10, n_jobs=-1 -> 24 threads)..."
    )
    model.fit(X_tr, y_tr)
    print("[OK] Training complete\n")

    y_pred   = model.predict(X_te)
    metrics: dict[str, dict] = {}

    print(f"{'Target':25s}  {'R²':>8}  {'RMSE':>10}  {'MAE':>10}")
    print("-" * 60)
    for i, target in enumerate(TARGETS):
        r2   = r2_score(y_te[:, i], y_pred[:, i])
        rmse = float(np.sqrt(mean_squared_error(y_te[:, i], y_pred[:, i])))
        mae  = float(mean_absolute_error(y_te[:, i], y_pred[:, i]))
        metrics[target] = {"R2": r2, "RMSE": rmse, "MAE": mae}
        print(f"  {target:23s}  {r2:8.4f}  {rmse:10.4f}  {mae:10.4f}")

    return model, X_te, y_te, y_pred, metrics


# ─────────────── feature importance plot ────────────

def plot_feature_importance(model: RandomForestRegressor) -> None:
    """Bar chart of RandomForest feature importances (averaged over all targets)."""
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    importances = model.feature_importances_
    std = np.std(
        [tree.feature_importances_ for tree in model.estimators_], axis=0
    )
    order = np.argsort(importances)[::-1]

    colors = plt.cm.viridis(np.linspace(0.25, 0.85, len(FEATURES)))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(
        range(len(FEATURES)),
        importances[order],
        yerr=std[order],
        color=colors, edgecolor="white", linewidth=0.5, capsize=6,
    )
    ax.set_xticks(range(len(FEATURES)))
    ax.set_xticklabels([FEATURES[i] for i in order], fontsize=11)
    ax.set_ylabel("Mean Decrease in Impurity", fontsize=11)
    ax.set_title(
        "RandomForest Feature Importance\n(averaged across all SI targets)",
        fontsize=13,
    )
    ax.grid(True, alpha=0.2, axis="y")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "ml_feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[OK] Feature importance plot saved")


# ──────────────── optimisation ───────────────────────

def compute_score(row: pd.Series) -> float:
    """
    Composite optimisation score in a 0..1-ish desirability scale.
    Higher is better; each term is normalised against the lowered pass limits.
    """
    eye_score = min(max(row["eye_height_mV"] / 350.0, 0.0), 1.0)
    ssn_score = min(max((350.0 - row["SSN_mV"]) / 200.0, 0.0), 1.0)
    overshoot_score = min(max((15.0 - row["overshoot_pct"]) / 15.0, 0.0), 1.0)
    setup_score = min(max((row["setup_margin_ps"] + 80.0) / 120.0, 0.0), 1.0)

    return (
        0.35 * eye_score
        + 0.25 * ssn_score
        + 0.20 * setup_score
        + 0.20 * overshoot_score
    )


def find_optimal_combinations(
    model: RandomForestRegressor,
) -> pd.DataFrame:
    """
    Predict SI metrics for the full 1000-point design grid and rank by score.

    Args:
        model: Trained RandomForestRegressor

    Returns:
        Top-10 feasible combinations as DataFrame
    """
    ron_vals  = [28, 30, 34, 38, 40]
    odt_vals  = [40, 48, 60, 80, 120]
    tl_vals   = [5, 10, 15, 20, 25]
    c_vals    = [47, 100, 220, 470, 680, 1000, 2200, 4700]

    grid = np.array(list(itertools.product(ron_vals, odt_vals, tl_vals, c_vals)))
    y_pred = model.predict(grid)

    pred_df = pd.DataFrame(grid, columns=FEATURES)
    for i, t in enumerate(TARGETS):
        pred_df[t] = y_pred[:, i]

    # Apply constraints
    mask = (
        (pred_df["overshoot_pct"]   < CONSTRAINTS["overshoot_pct"][1])
        & (pred_df["eye_height_mV"]  > CONSTRAINTS["eye_height_mV"][1])
        & (pred_df["SSN_mV"]         < CONSTRAINTS["SSN_mV"][1])
        & (pred_df["setup_margin_ps"] > CONSTRAINTS["setup_margin_ps"][1])
    )

    feasible = pred_df[mask].copy()
    if feasible.empty:
        print("[WARN] No combinations satisfy all constraints - relaxing to analysis pass limits")
        mask2 = (
            (pred_df["overshoot_pct"]   < 15.0)
            & (pred_df["eye_height_mV"]  > 100.0)
            & (pred_df["SSN_mV"]         < 100.0)
            & (pred_df["setup_margin_ps"] > 0.0)
        )
        feasible = pred_df[mask2].copy()
        if feasible.empty:
            print("[WARN] No combinations satisfy analysis pass limits - ranking all simulated conditions")
            feasible = pred_df.copy()

    feasible["score"] = feasible.apply(compute_score, axis=1)
    return feasible.nlargest(10, "score").reset_index(drop=True)


# ─────────────────── main pipeline ──────────────────

def run_ml_optimization() -> tuple:
    """Full ML pipeline: load -> train -> feature importance -> optimise -> save."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    model, X_te, y_te, y_pred, metrics = train_model(df)
    plot_feature_importance(model)

    print("\nSearching optimal combinations across full design grid...")
    top10 = find_optimal_combinations(model)

    print(f"\n{'='*90}")
    print("TOP 10 OPTIMAL PARAMETER COMBINATIONS (predicted)")
    print(f"{'='*90}")
    hdr = (f"{'#':>3}  {'Ron':>5}  {'ODT':>5}  {'TL':>5}  {'Cdecap':>8}  "
           f"{'Eye(mV)':>9}  {'SSN(mV)':>8}  {'Over%':>7}  {'Setup(ps)':>10}  {'Score':>7}")
    print(hdr)
    print("-" * 90)

    for i, row in top10.iterrows():
        print(
            f"  {i+1:2d}. "
            f"Ron={row['Ron_ohm']:3.0f}ohm  "
            f"ODT={row['ODT_ohm']:3.0f}ohm  "
            f"TL={row['trace_length_mm']:4.0f}mm  "
            f"C={row['C_decap_nF']:5.0f}nF  ->  "
            f"Eye={row['eye_height_mV']:6.0f}mV  "
            f"SSN={row['SSN_mV']:5.1f}mV  "
            f"Over={row['overshoot_pct']:4.1f}%  "
            f"Setup={row['setup_margin_ps']:6.1f}ps  "
            f"Score={row['score']:5.3f}"
        )

    # Persist model
    model_path = MODEL_DIR / "rf_model.pkl"
    with open(model_path, "wb") as fh:
        pickle.dump(model, fh)
    print(f"\n[OK] Model saved      -> {model_path}")

    # Persist optimisation table
    opt_path = DATA_DIR / "optimization_results.csv"
    top10.to_csv(opt_path, index=False)
    print(f"[OK] Optimal params   -> {opt_path}")

    return model, top10, metrics


if __name__ == "__main__":
    run_ml_optimization()


