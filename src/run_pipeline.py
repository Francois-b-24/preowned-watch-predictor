from __future__ import annotations
from pathlib import Path
from sklearn.model_selection import train_test_split
import numpy as np
import yaml

from src.data_utils import load_data, get_feature_matrix
from src.training import evaluate_models, select_and_train_best
from src.interpret import get_personas, log_and_save_personas, analyze_price_segments
from src.inference import fit_conformal, predict_with_interval

BASE_DIR = Path(__file__).resolve().parent.parent
CFG = yaml.safe_load(open(BASE_DIR / "config" / "params.yaml", "r", encoding="utf-8"))


def main():
    df = load_data()
    log_and_save_personas(get_personas())
    analyze_price_segments(df)

    X, y = get_feature_matrix(df)
    Xtr_full, Xte, ytr_full, yte = train_test_split(
        X, y,
        test_size=CFG["training"]["test_size"],
        random_state=CFG["training"]["random_state"],
    )
    Xtr, Xcal, ytr, ycal = train_test_split(
        Xtr_full, ytr_full,
        test_size=0.2,
        random_state=CFG["training"]["random_state"],
    )

    df_results, models = evaluate_models(Xtr, ytr)
    print("\n=== Résultats CV (train) ===")
    print(df_results.to_string(index=False))

    summary = select_and_train_best(df_results, models, Xtr, ytr, Xte, yte)
    print(f"\nBest: {summary['best_model']} | R2_test={summary['R2_test']:.3f} RMSE={summary['RMSE_test']:.3f} MAE={summary['MAE_test']:.3f}")
    print(f"Saved → {summary['model_path']}")
    try:
        ycal_eur = np.exp(ycal) - 1.0   # calibration en euros (pas en log)
        q = fit_conformal(Xcal, ycal_eur, alpha=0.1)
        print(f"Conformal calibration done (alpha=0.1). q = {q:.2f} €")
    except Exception as e:
        print(f"[WARN] Conformal calibration failed: {e}")
