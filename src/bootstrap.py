"""
Bootstrap de l'architecture modulaire + compatibilité ascendante.
Ce script crée les nouveaux modules (data_utils, preprocess, training, interpret),
un fichier `config/params.yaml`, un `run_pipeline.py` et un `Makefile` s'ils n'existent pas.
Ensuite, il vous invite à exécuter `python src/run_pipeline.py`.
"""

from __future__ import annotations
import os
from pathlib import Path
import textwrap
import sys

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
CONFIG_DIR = BASE_DIR / "config"
REPORTS_DIR = BASE_DIR / "reports"
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

# --- Contenus des fichiers à créer ---
DATA_UTILS = """\
from __future__ import annotations
from pathlib import Path
import os
import pandas as pd
import numpy as np
import yaml

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "config" / "params.yaml"

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

RANDOM_STATE = CFG["training"]["random_state"]


def segment_prix(prix: float) -> str:
    if prix < 5000:
        return "Entrée de gamme"
    if prix <= 20000:
        return "Moyen de gamme"
    return "Haut de gamme"


def load_data() -> pd.DataFrame:
    path = BASE_DIR / CFG["data"]["processed_path"]
    df = pd.read_excel(path)
    df = df.dropna(subset=["prix", "marque", "modele"])  # colonnes indispensables
    df["segment_prix"] = df["prix"].apply(segment_prix)
    df["log_prix"] = np.log(df["prix"] + 1)
    return df


def get_feature_matrix(df: pd.DataFrame):
    feats = CFG["features"]["columns"]
    X = df[feats]
    y = df["log_prix"]
    return X, y
"""

PREPROCESS = """\
from __future__ import annotations
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from category_encoders import TargetEncoder
import yaml
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
CFG = yaml.safe_load(open(BASE_DIR / "config" / "params.yaml", "r", encoding="utf-8"))


def build_preprocessor() -> ColumnTransformer:
    material_cats = CFG["preprocess"]["material_categories"]
    return ColumnTransformer([
        ("brand", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), ["marque"]),
        ("model_te", Pipeline([("te", TargetEncoder(min_samples_leaf=50, smoothing=10))]), ["modele"]),
        ("seg", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), ["segment_prix"]),
        ("mat", Pipeline([
            ("imp", SimpleImputer(strategy="constant", fill_value="")),
            ("ord", OrdinalEncoder(categories=material_cats, handle_unknown="use_encoded_value", unknown_value=-1))
        ]), ["matiere_boitier", "matiere_bracelet"]),
        ("tech", Pipeline([
            ("imp", SimpleImputer(strategy="mean")),
            ("sc", StandardScaler())
        ]), ["reserve_de_marche", "diametre", "etencheite", "comptage_fonctions"])], remainder="drop")
"""

TRAINING = """\
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
"""

INTERPRET = """\
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import learning_curve

from .preprocess import build_preprocessor

BASE_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def get_personas() -> dict:
    return {
        "Revendeur pro": {
            "Objectif": "Estimer la valeur d'achat au plus juste (±5%)",
            "Usage": "Quotidien",
            "Sortie attendue": "Prix attendu + intervalle de confiance",
            "Canal": "API / Web",
        },
        "Particulier": {
            "Objectif": "Vérifier si une annonce est surpayée (±10%)",
            "Usage": "Ponctuel",
            "Sortie attendue": "Score de fair‑price + conseils",
            "Canal": "Mobile / Web",
        },
        "Plateforme": {
            "Objectif": "Détecter les annonces outliers (±20%)",
            "Usage": "Automatique",
            "Sortie attendue": "Flag outlier + explications",
            "Canal": "Dashboard",
        },
    }


def log_and_save_personas(personas: dict):
    print("\n=== Personas ===")
    for name, attrs in personas.items():
        print(f"\n{name}:")
        for k, v in attrs.items():
            print(f"  - {k}: {v}")
    md = ["# Personas\n"]
    for n, attrs in personas.items():
        md.append(f"\n## {n}\n")
        for k, v in attrs.items():
            md.append(f"- **{k}**: {v}")
    (REPORTS_DIR / "personas.md").write_text("\n".join(md), encoding="utf-8")


def analyze_price_segments(df: pd.DataFrame):
    print("\n=== Segments de prix (counts) ===")
    counts = df["segment_prix"].value_counts()
    print(counts.to_string())

    print("\n=== Statistiques prix par segment ===")
    stats = df.groupby("segment_prix")["prix"].describe().round(2)
    print(stats.to_string())

    counts.to_csv(REPORTS_DIR / "segment_counts.csv", header=["count"])
    stats.to_csv(REPORTS_DIR / "segment_price_stats.csv")

    fig, ax = plt.subplots()
    df.boxplot(column="prix", by="segment_prix", ax=ax)
    ax.set_title("Distribution du prix par segment")
    ax.set_ylabel("Prix affiché")
    fig.suptitle("")
    plt.tight_layout()
    fig.savefig(REPORTS_DIR / "segment_price_boxplot.png")
    plt.close(fig)


def shap_top_features(best_pipe, X_sample, topn: int = 10):
    model = best_pipe.named_steps.get("hgb")
    if model is None:
        print("[SHAP] Modèle non arborescent — skip")
        return
    pre = best_pipe.named_steps["pre"]
    Xs = pre.transform(X_sample)
    expl = shap.TreeExplainer(model)
    sv = expl.shap_values(Xs)
    fn = pre.get_feature_names_out()
    imp = (
        pd.DataFrame({"feature": fn, "importance": np.abs(sv).mean(axis=0)})
        .sort_values("importance", ascending=False)
    )
    top = imp.head(topn)[::-1]
    fig, ax = plt.subplots()
    ax.barh(top["feature"], top["importance"])  # couleurs par défaut
    ax.set_xlabel("SHAP importance")
    ax.set_title(f"Top {topn} SHAP feature importances")
    plt.tight_layout()
    fig.savefig(REPORTS_DIR / "shap_top10.png")
    plt.close(fig)


def learning_curve_plot(best_pipe, X, y, tag: str = "best"):
    sizes, tr_s, val_s = learning_curve(best_pipe, X, y, cv=5, train_sizes=np.linspace(0.1, 1, 5), scoring="r2", n_jobs=-1)
    import pandas as pd
    df_lc = pd.DataFrame({"train_size": sizes, "train_r2": tr_s.mean(axis=1), "val_r2": val_s.mean(axis=1)})
    fig, ax = plt.subplots()
    ax.plot(df_lc["train_size"], df_lc["train_r2"], marker="o", label="Train R2")
    ax.plot(df_lc["train_size"], df_lc["val_r2"], marker="o", label="Validation R2")
    ax.set_xlabel("Train set size")
    ax.set_ylabel("R2 score")
    ax.set_title(f"Learning Curves — {tag}")
    ax.legend()
    plt.tight_layout()
    fig.savefig(REPORTS_DIR / "learning_curve.png")
    plt.close(fig)
"""

RUN_PIPELINE = """\
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
    print("\n=== Résultats CV (train) ===")
    print(df_results.to_string(index=False))

    summary = select_and_train_best(df_results, models, Xtr, ytr, Xte, yte)
    print(f"\nBest: {summary['best_model']} | R2_test={summary['R2_test']:.3f} RMSE={summary['RMSE_test']:.3f} MAE={summary['MAE_test']:.3f}")
    print(f"Saved → {summary['model_path']}")

    # Interprétation
    best_name = df_results.iloc[0]["Model"]
    best_pipe = models[best_name]
    shap_top_features(best_pipe, Xtr.sample(min(1000, len(Xtr)), random_state=CFG['training']['random_state']))
    learning_curve_plot(best_pipe, Xtr, ytr, tag=best_name)


if __name__ == "__main__":
    main()
"""

PARAMS_YAML = """\
data:
  processed_path: "data/processed/propre.xlsx"
model:
  filename: "final_pipeline.joblib"
features:
  columns:
    - marque
    - modele
    - segment_prix
    - matiere_boitier
    - matiere_bracelet
    - reserve_de_marche
    - diametre
    - etencheite
    - comptage_fonctions
preprocess:
  material_categories:
    - ["acier", "or/acier", "or massif"]
    - ["acier", "or/acier", "or massif"]
training:
  random_state: 42
  test_size: 0.2
  cv_splits: 5
  optuna_trials: 30
mlflow:
  enabled: false        # passer à true pour activer MLflow
  tracking_uri: null    # ex: "file:./mlruns" ou http(s)://...
  experiment: "watch-price"
"""

MAKEFILE = """\
.PHONY: train clean test mlflow-ui app

train:
	python src/run_pipeline.py

clean:
	rm -f models/*.joblib
	rm -f reports/*.png reports/*.csv reports/*.md

# Run unit tests quietly
test:
	pytest -q

# Launch local MLflow UI (uses MLFLOW_BACKEND or defaults to ./mlruns)
mlflow-ui:
	mlflow ui --backend-store-uri $${MLFLOW_BACKEND:-mlruns}

# Launch the Streamlit app
app:
	streamlit run src/streamlit_app.py
"""

PYPROJECT_TOML = """\
[project]
name = "preowned-watch-predictor"
version = "0.1.0"
requires-python = ">=3.10"
description = "Price prediction for pre-owned luxury watches"
authors = [{name = "Your Name"}]
readme = "README.md"
dependencies = [
  "pandas",
  "numpy",
  "scikit-learn",
  "category-encoders",
  "optuna",
  "shap",
  "matplotlib",
  "pyyaml",
  "mlflow",
  "streamlit",
  "openpyxl"
]

[tool.pytest.ini_options]
addopts = "-q"
pythonpath = ["src"]

[tool.black]
line-length = 100

[tool.ruff]
line-length = 100
select = ["E","F","I"]
"""

TEST_DATA_UTILS = """\
import os
from pathlib import Path
import pytest

from src.data_utils import load_data, get_feature_matrix

DATA_FILE = Path("data/processed/propre.xlsx")

@pytest.mark.skipif(not DATA_FILE.exists(), reason="Missing data file: data/processed/propre.xlsx")
def test_load_and_features():
    df = load_data()
    assert not df.empty
    X, y = get_feature_matrix(df)
    assert len(X) == len(y)
    assert X.shape[0] > 0
"""

TEST_TRAINING = """\
import os
from pathlib import Path
import pytest
import pandas as pd

from src.data_utils import load_data, get_feature_matrix
from src.training import evaluate_models

DATA_FILE = Path("data/processed/propre.xlsx")

@pytest.mark.skipif(not DATA_FILE.exists(), reason="Missing data file: data/processed/propre.xlsx")
def test_evaluate_models_runs():
    df = load_data()
    X, y = get_feature_matrix(df)
    # Petit sous-échantillon pour accélérer ce test si dataset massif
    if len(df) > 5000:
        df = df.sample(2000, random_state=42)
        X, y = get_feature_matrix(df)
    results, models = evaluate_models(X, y)
    assert not results.empty
    assert isinstance(models, dict)
"""

# --- Inference helpers ---
INFERENCE = """\
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent
CFG = yaml.safe_load(open(BASE_DIR / "config" / "params.yaml", "r", encoding="utf-8"))
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / CFG["model"]["filename"]

_FEATURES = CFG["features"]["columns"]

_model_cache = None

def load_model():
    global _model_cache
    if _model_cache is None:
        _model_cache = joblib.load(MODEL_PATH)
    return _model_cache


def ensure_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Add any missing feature columns as NA/empty defaults to avoid KeyErrors
    for col in _FEATURES:
        if col not in df.columns:
            df[col] = np.nan if col not in ("marque", "modele", "segment_prix", "matiere_boitier", "matiere_bracelet") else ""
    return df[_FEATURES]


def predict_prices(df_feats: pd.DataFrame) -> pd.DataFrame:
    #Takes a DataFrame with feature columns and returns a DataFrame with predictions in euros.
    model = load_model()
    X = ensure_feature_columns(df_feats)
    y_log = model.predict(X)
    price = np.exp(y_log) - 1.0
    out = df_feats.copy()
    out["predicted_price"] = price
    return out
"""

STREAMLIT_APP = """\
from __future__ import annotations
from pathlib import Path
import io
import numpy as np
import pandas as pd
import yaml
import streamlit as st

from .inference import predict_prices, ensure_feature_columns

BASE_DIR = Path(__file__).resolve().parent.parent
CFG = yaml.safe_load(open(BASE_DIR / "config" / "params.yaml", "r", encoding="utf-8"))
FEATURES = CFG["features"]["columns"]

st.set_page_config(page_title="Watch Price Predictor", page_icon="⌚", layout="centered")
st.title("⌚ Prévision de prix de montres d'occasion")
st.caption("Modèle entraîné hors-ligne — fournissez des caractéristiques pour estimer le prix.")

mode = st.sidebar.radio("Mode", ["Formulaire", "Fichier (CSV/Excel)"])

SEG_OPTS = ["Entrée de gamme", "Moyen de gamme", "Haut de gamme"]

if mode == "Fichier (CSV/Excel)":
    up = st.file_uploader("Déposez un CSV ou un Excel contenant les colonnes features", type=["csv", "xlsx"])
    if up is not None:
        try:
            if up.name.lower().endswith(".csv"):
                df = pd.read_csv(up)
            else:
                df = pd.read_excel(up)
        except Exception as e:
            st.error(f"Erreur de lecture du fichier: {e}")
            df = None
        if df is not None:
            missing = [c for c in FEATURES if c not in df.columns]
            if missing:
                st.warning("Colonnes manquantes ajoutées par défaut : " + ", ".join(missing))
            preds = predict_prices(df)
            st.success(f"Prédictions effectuées sur {len(preds)} ligne(s).")
            st.dataframe(preds)
            csv = preds.to_csv(index=False).encode("utf-8")
            st.download_button("Télécharger les prédictions (CSV)", data=csv, file_name="watch_predictions.csv", mime="text/csv")
else:
    with st.form("price_form"):
        col1, col2 = st.columns(2)
        with col1:
            marque = st.text_input("Marque", "Rolex")
            modele = st.text_input("Modèle", "Submariner")
            segment_prix = st.selectbox("Segment de prix (estimé)", SEG_OPTS)
            matiere_boitier = st.text_input("Matière boîtier", "acier")
            matiere_bracelet = st.text_input("Matière bracelet", "acier")
        with col2:
            reserve_de_marche = st.number_input("Réserve de marche (h)", min_value=0.0, value=48.0, step=1.0)
            diametre = st.number_input("Diamètre (mm)", min_value=0.0, value=40.0, step=0.5)
            etencheite = st.number_input("Étanchéité (m)", min_value=0.0, value=100.0, step=10.0)
            comptage_fonctions = st.number_input("# Complications/fonctions", min_value=0, value=1, step=1)
        submitted = st.form_submit_button("Estimer le prix")
    if submitted:
        row = {
            "marque": marque,
            "modele": modele,
            "segment_prix": segment_prix,
            "matiere_boitier": matiere_boitier,
            "matiere_bracelet": matiere_bracelet,
            "reserve_de_marche": reserve_de_marche,
            "diametre": diametre,
            "etencheite": etencheite,
            "comptage_fonctions": comptage_fonctions,
        }
        df = pd.DataFrame([row])
        pred = predict_prices(df)
        price = float(pred.loc[0, "predicted_price"])
        st.subheader("Prix estimé")
        st.metric("€", f"{price:,.0f}".replace(",", " "))
        st.caption("Note : estimation ponctuelle. Ajouter un intervalle d'incertitude au besoin.")
"""


# --- Bootstrap ---
def write_file(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(content, encoding="utf-8")
        print(f"[CREATED] {path}")
    else:
        print(f"[SKIPPED] {path} (exists)")


def main():
    print("Bootstrapping l'architecture modulaire…")
    write_file(SRC_DIR / "data_utils.py", DATA_UTILS)
    write_file(SRC_DIR / "preprocess.py", PREPROCESS)
    write_file(SRC_DIR / "training.py", TRAINING)
    write_file(SRC_DIR / "interpret.py", INTERPRET)
    write_file(SRC_DIR / "run_pipeline.py", RUN_PIPELINE)
    write_file(CONFIG_DIR / "params.yaml", PARAMS_YAML)
    write_file(BASE_DIR / "Makefile", MAKEFILE)
    write_file(BASE_DIR / "pyproject.toml", PYPROJECT_TOML)
    write_file(BASE_DIR / "tests" / "test_data_utils.py", TEST_DATA_UTILS)
    write_file(BASE_DIR / "tests" / "test_training.py", TEST_TRAINING)
    write_file(SRC_DIR / "inference.py", INFERENCE)
    write_file(SRC_DIR / "streamlit_app.py", STREAMLIT_APP)

    print("\n✅ Bootstrap terminé. Prochaines étapes :")
    print("   1) Vérifier/éditer config/params.yaml si besoin")
    print("   2) Lancer l'entraînement :  make train   (ou  python src/run_pipeline.py)")
    print("   3) Lancer l'application :  make app     (ou  streamlit run src/streamlit_app.py)")


if __name__ == "__main__":
    main()