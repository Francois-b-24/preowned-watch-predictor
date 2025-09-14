from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, cross_validate, GridSearchCV, train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
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

    # Lasso (grid)
    lasso = Pipeline([("pre", pre), ("lasso", Lasso(max_iter=10000, random_state=RANDOM_STATE))])
    grid = GridSearchCV(lasso, {"lasso__alpha": [1e-3, 1e-2, 1e-1, 1, 10]}, cv=cv, scoring="r2", n_jobs=-1)
    grid.fit(X_train, y_train)
    lasso_best = grid.best_estimator_
    res = _evaluate_with_cv(lasso_best, X_train, y_train, cv)
    results.append({"Model": "Lasso", **res}); models["Lasso"] = lasso_best

    # ElasticNet (grid)
    enet = Pipeline([("pre", pre), ("enet", ElasticNet(max_iter=10000, random_state=RANDOM_STATE))])
    grid = GridSearchCV(enet, {"enet__alpha": [1e-3, 1e-2, 1e-1, 1], "enet__l1_ratio": [0.2, 0.5, 0.8]}, cv=cv, scoring="r2", n_jobs=-1)
    grid.fit(X_train, y_train)
    enet_best = grid.best_estimator_
    res = _evaluate_with_cv(enet_best, X_train, y_train, cv)
    results.append({"Model": "ElasticNet", **res}); models["ElasticNet"] = enet_best

    # RandomForest (grid léger)
    rf = Pipeline([("pre", pre), ("rf", RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1))])
    grid = GridSearchCV(rf, {"rf__n_estimators": [200, 400], "rf__max_depth": [None, 10, 20], "rf__min_samples_leaf": [1, 3, 5]}, cv=cv, scoring="r2", n_jobs=-1)
    grid.fit(X_train, y_train)
    rf_best = grid.best_estimator_
    res = _evaluate_with_cv(rf_best, X_train, y_train, cv)
    results.append({"Model": "RandomForest", **res}); models["RandomForest"] = rf_best

    # ExtraTrees (grid léger)
    et = Pipeline([("pre", pre), ("et", ExtraTreesRegressor(random_state=RANDOM_STATE, n_jobs=-1))])
    grid = GridSearchCV(et, {"et__n_estimators": [300, 600], "et__max_depth": [None, 15, 30], "et__min_samples_leaf": [1, 2, 4]}, cv=cv, scoring="r2", n_jobs=-1)
    grid.fit(X_train, y_train)
    et_best = grid.best_estimator_
    res = _evaluate_with_cv(et_best, X_train, y_train, cv)
    results.append({"Model": "ExtraTrees", **res}); models["ExtraTrees"] = et_best

    # GradientBoosting (grid)
    gbr = Pipeline([("pre", pre), ("gbr", GradientBoostingRegressor(random_state=RANDOM_STATE))])
    grid = GridSearchCV(gbr, {"gbr__n_estimators": [200, 400], "gbr__learning_rate": [0.03, 0.1, 0.2], "gbr__max_depth": [2, 3, 4]}, cv=cv, scoring="r2", n_jobs=-1)
    grid.fit(X_train, y_train)
    gbr_best = grid.best_estimator_
    res = _evaluate_with_cv(gbr_best, X_train, y_train, cv)
    results.append({"Model": "GradientBoosting", **res}); models["GradientBoosting"] = gbr_best

    # KNN Regressor (grid)
    knn = Pipeline([("pre", pre), ("knn", KNeighborsRegressor())])
    grid = GridSearchCV(knn, {"knn__n_neighbors": [3, 5, 11, 21], "knn__weights": ["uniform", "distance"]}, cv=cv, scoring="r2", n_jobs=-1)
    grid.fit(X_train, y_train)
    knn_best = grid.best_estimator_
    res = _evaluate_with_cv(knn_best, X_train, y_train, cv)
    results.append({"Model": "KNeighbors", **res}); models["KNeighbors"] = knn_best

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
