from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, cross_validate, GridSearchCV, train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import optuna
import joblib

try:
    import mlflow
    _HAS_MLFLOW = True
except Exception:
    _HAS_MLFLOW = False

from .preprocess import build_preprocessor
from .data_utils import load_data, get_feature_matrix

BASE_DIR = Path(__file__).resolve().parent.parent
CFG = yaml.safe_load(open(BASE_DIR / "config" / "params.yaml", "r", encoding="utf-8"))
REPORTS_DIR = BASE_DIR / "reports"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = CFG["training"]["random_state"]
CV_SPLITS = CFG["training"]["cv_splits"]
OPTUNA_TRIALS = CFG["training"]["optuna_trials"]
MODEL_PATH = MODELS_DIR / CFG["model"]["filename"]

MLFLOW_ENABLED = False
MLFLOW_TRACKING_URI = None
MLFLOW_EXPERIMENT = None
if _HAS_MLFLOW:
    MLFLOW_ENABLED = bool(CFG.get("mlflow", {}).get("enabled", False))
    MLFLOW_TRACKING_URI = CFG.get("mlflow", {}).get("tracking_uri", None)
    MLFLOW_EXPERIMENT = CFG.get("mlflow", {}).get("experiment", "watch-price")
    if MLFLOW_ENABLED and MLFLOW_TRACKING_URI:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT)


def _evaluate_with_cv(model: Pipeline, X, y, cv) -> dict:
    scores = cross_validate(
        model, X, y, cv=cv,
        scoring=["r2", "neg_root_mean_squared_error", "neg_mean_absolute_error"],
        n_jobs=-1,
    )
    return {
        "R2_mean": scores["test_r2"].mean(),
        "R2_std": scores["test_r2"].std(),
        "RMSE_mean": -scores["test_neg_root_mean_squared_error"].mean(),
        "MAE_mean": -scores["test_neg_mean_absolute_error"].mean(),
    }


def evaluate_models(X_train, y_train) -> tuple[pd.DataFrame, dict]:
    pre = build_preprocessor()
    cv = KFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    results, models = [], {}

    # Linear Regression
    lr = Pipeline([("pre", pre), ("lr", LinearRegression())])
    res = _evaluate_with_cv(lr, X_train, y_train, cv)
    results.append({"Model": "LinearRegression", **res}); models["LinearRegression"] = lr

    # Ridge (grid)
    ridge = Pipeline([("pre", pre), ("ridge", Ridge())])
    grid = GridSearchCV(ridge, {"ridge__alpha": [0.01, 0.1, 1, 10, 100]}, cv=cv, scoring="r2", n_jobs=-1)
    grid.fit(X_train, y_train)
    ridge_best = grid.best_estimator_
    res = _evaluate_with_cv(ridge_best, X_train, y_train, cv)
    results.append({"Model": "Ridge", **res}); models["Ridge"] = ridge_best

    # HGB (Optuna)
    def objective(trial: optuna.Trial) -> float:
        params = {
            "max_iter": trial.suggest_int("max_iter", 200, 1000),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 100),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 31, 255),
            "random_state": RANDOM_STATE,
        }
        pipe = Pipeline([("pre", pre), ("hgb", HistGradientBoostingRegressor(**params))])
        sc = cross_validate(pipe, X_train, y_train, cv=cv, scoring="r2", n_jobs=-1)
        return sc["test_score"].mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=OPTUNA_TRIALS)
    hgb = Pipeline([("pre", pre), ("hgb", HistGradientBoostingRegressor(random_state=RANDOM_STATE, **study.best_params))])
    res = _evaluate_with_cv(hgb, X_train, y_train, cv)
    results.append({"Model": "HGB", **res}); models["HGB"] = hgb

    df_results = pd.DataFrame(results).sort_values("R2_mean", ascending=False).reset_index(drop=True)

    out_csv = REPORTS_DIR / "model_comparison.csv"
    try:
        df_results.to_csv(out_csv, index=False)
    except Exception:
        pass

    return df_results, models


def select_and_train_best(df_results: pd.DataFrame, models: dict, X_train, y_train, X_test, y_test) -> dict:
    best_name = df_results.iloc[0]["Model"]
    best_model = models[best_name]
    best_model.fit(X_train, y_train)

    # Test metrics
    y_pred = best_model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = float(mean_absolute_error(y_test, y_pred))

    joblib.dump(best_model, MODEL_PATH)

    # Optional MLflow logging
    if _HAS_MLFLOW and MLFLOW_ENABLED:
        with mlflow.start_run():
            mlflow.log_metric("R2_test", float(r2))
            mlflow.log_metric("RMSE_test", float(rmse))
            mlflow.log_metric("MAE_test", float(mae))
            mlflow.log_param("best_model", best_name)
            # Log key hyperparameters if present
            try:
                if hasattr(best_model, "named_steps"):
                    if "ridge" in best_model.named_steps:
                        mlflow.log_param("ridge_alpha", best_model.named_steps["ridge"].alpha)
                    if "hgb" in best_model.named_steps:
                        hgb = best_model.named_steps["hgb"]
                        for p in ["max_iter","learning_rate","max_depth","min_samples_leaf","max_leaf_nodes"]:
                            mlflow.log_param(p, getattr(hgb, p, None))
            except Exception:
                pass
            # Log artifacts if available
            try:
                cmp = REPORTS_DIR / "model_comparison.csv"
                if cmp.exists():
                    mlflow.log_artifact(str(cmp), artifact_path="reports")
            except Exception:
                pass

    return {"best_model": best_name, "R2_test": float(r2), "RMSE_test": rmse, "MAE_test": mae, "model_path": str(MODEL_PATH)}
