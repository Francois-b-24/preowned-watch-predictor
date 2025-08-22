from __future__ import annotations
from pathlib import Path
from sklearn.model_selection import train_test_split
import pandas as pd
import yaml

from src.data_utils import load_data, get_feature_matrix
from src.training import evaluate_models, select_and_train_best
from src.interpret import get_personas, log_and_save_personas, analyze_price_segments, shap_top_features, learning_curve_plot

BASE_DIR = Path(__file__).resolve().parent.parent
CFG = yaml.safe_load(open(BASE_DIR / "config" / "params.yaml", "r", encoding="utf-8"))


def main():
    df = load_data()
    personas = get_personas()
    log_and_save_personas(personas)
    analyze_price_segments(df)

    X, y = get_feature_matrix(df)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=CFG["training"]["test_size"], random_state=CFG["training"]["random_state"])

    df_results, models = evaluate_models(Xtr, ytr)
    print("=== Résultats CV (train) ===")
    print(df_results.to_string(index=False))

    summary = select_and_train_best(df_results, models, Xtr, ytr, Xte, yte)
    print(f"Best: {summary['best_model']} | R2_test={summary['R2_test']:.3f} RMSE={summary['RMSE_test']:.3f} MAE={summary['MAE_test']:.3f}")
    print(f"Saved → {summary['model_path']}")

    # Interprétation
    best_name = df_results.iloc[0]["Model"]
    best_pipe = models[best_name]
    shap_top_features(best_pipe, Xtr.sample(min(1000, len(Xtr)), random_state=CFG['training']['random_state']))
    learning_curve_plot(best_pipe, Xtr, ytr, tag=best_name)


if __name__ == "__main__":
    main()
