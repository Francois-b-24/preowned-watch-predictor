"""
Bootstrap de l'architecture modulaire + compatibilité ascendante.
Ce script crée les nouveaux modules (data_utils, preprocess, feature_selection, training, interpret),
un fichier `config/params.yaml`, un `run_pipeline.py` et un `Makefile` s'ils n'existent pas.
Ensuite, il vous invite à exécuter `python src/run_pipeline.py`.
"""

from __future__ import annotations
from pathlib import Path

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
    df["segment_prix"] = df["prix"].apply(segment_prix)  # analyse/rapports uniquement
    df["log_prix"] = np.log(df["prix"] + 1)
    return df


def get_feature_matrix(df: pd.DataFrame):
    feats = CFG["features"]["columns"]  # 'segment_prix' exclu des features (anti-fuite)
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
        # 'segment_prix' exclu pour éviter la fuite de cible
        ("mat", Pipeline([
            ("imp", SimpleImputer(strategy="constant", fill_value="")),
            ("ord", OrdinalEncoder(categories=material_cats, handle_unknown="use_encoded_value", unknown_value=-1))
        ]), ["matiere_boitier", "matiere_bracelet"]),
        ("tech", Pipeline([
            ("imp", SimpleImputer(strategy="mean")),
            ("sc", StandardScaler())
        ]), ["reserve_de_marche", "diametre", "etencheite", "comptage_fonctions"])], remainder="drop")
"""

FEATURE_SELECTION = """\
from __future__ import annotations
from pathlib import Path
import yaml

from sklearn.feature_selection import SelectKBest, f_regression, SelectFromModel
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor

BASE_DIR = Path(__file__).resolve().parent.parent
CFG = yaml.safe_load(open(BASE_DIR / "config" / "params.yaml", "r", encoding="utf-8"))


def build_feature_selector():
    fs_cfg = CFG.get("feature_selection", {})
    if not fs_cfg.get("enabled", False):
        return "passthrough"

    method = str(fs_cfg.get("method", "kbest")).lower()

    if method == "kbest":
        k = int(fs_cfg.get("k", 20))
        return SelectKBest(score_func=f_regression, k=k)

    if method == "lasso":
        alpha = float(fs_cfg.get("lasso_alpha", 0.001))
        return SelectFromModel(Lasso(alpha=alpha, max_iter=10000, random_state=CFG["training"]["random_state"]))

    if method == "rf_importance":
        n = int(fs_cfg.get("rf_n_estimators", 300))
        thresh = fs_cfg.get("threshold", "median")
        est = RandomForestRegressor(n_estimators=n, random_state=CFG["training"]["random_state"], n_jobs=-1)
        return SelectFromModel(estimator=est, threshold=thresh)

    return "passthrough"
"""

TRAINING = """\
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, cross_validate, GridSearchCV
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
from .feature_selection import build_feature_selector

BASE_DIR = Path(__file__).resolve().parent.parent
CFG = yaml.safe_load(open(BASE_DIR / "config" / "params.yaml", "r", encoding="utf-8"))
REPORTS_DIR = BASE_DIR / "reports"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = CFG["training"]["random_state"]
CV_SPLITS = CFG["training"]["cv_splits"]
OPTUNA_TRIALS = CFG["training"].get("optuna_trials", 30)
MODEL_PATH = MODELS_DIR / CFG["model"]["filename"]

MLFLOW_ENABLED = False
if _HAS_MLFLOW:
    MLFLOW_ENABLED = bool(CFG.get("mlflow", {}).get("enabled", False))
    if MLFLOW_ENABLED and CFG.get("mlflow", {}).get("tracking_uri", None):
        mlflow.set_tracking_uri(CFG["mlflow"]["tracking_uri"])
        mlflow.set_experiment(CFG.get("mlflow", {}).get("experiment", "watch-price"))


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
    lr = Pipeline([("pre", pre), ("fs", build_feature_selector()), ("lr", LinearRegression())])
    res = _evaluate_with_cv(lr, X_train, y_train, cv)
    results.append({"Model": "LinearRegression", **res}); models["LinearRegression"] = lr

    # Ridge (grid)
    ridge = Pipeline([("pre", pre), ("fs", build_feature_selector()), ("ridge", Ridge())])
    grid = GridSearchCV(ridge, {"ridge__alpha": [0.01, 0.1, 1, 10, 100]}, cv=cv, scoring="r2", n_jobs=-1)
    grid.fit(X_train, y_train)
    ridge_best = grid.best_estimator_
    res = _evaluate_with_cv(ridge_best, X_train, y_train, cv)
    results.append({"Model": "Ridge", **res}); models["Ridge"] = ridge_best

    # Lasso (grid)
    lasso = Pipeline([("pre", pre), ("fs", build_feature_selector()), ("lasso", Lasso(max_iter=10000, random_state=RANDOM_STATE))])
    grid = GridSearchCV(lasso, {"lasso__alpha": [1e-3, 1e-2, 1e-1, 1, 10]}, cv=cv, scoring="r2", n_jobs=-1)
    grid.fit(X_train, y_train)
    lasso_best = grid.best_estimator_
    res = _evaluate_with_cv(lasso_best, X_train, y_train, cv)
    results.append({"Model": "Lasso", **res}); models["Lasso"] = lasso_best

    # ElasticNet (grid)
    enet = Pipeline([("pre", pre), ("fs", build_feature_selector()), ("enet", ElasticNet(max_iter=10000, random_state=RANDOM_STATE))])
    grid = GridSearchCV(enet, {"enet__alpha": [1e-3, 1e-2, 1e-1, 1], "enet__l1_ratio": [0.2, 0.5, 0.8]}, cv=cv, scoring="r2", n_jobs=-1)
    grid.fit(X_train, y_train)
    enet_best = grid.best_estimator_
    res = _evaluate_with_cv(enet_best, X_train, y_train, cv)
    results.append({"Model": "ElasticNet", **res}); models["ElasticNet"] = enet_best

    # RandomForest (grid léger)
    rf = Pipeline([("pre", pre), ("fs", build_feature_selector()), ("rf", RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1))])
    grid = GridSearchCV(rf, {"rf__n_estimators": [200, 400], "rf__max_depth": [None, 10, 20], "rf__min_samples_leaf": [1, 3, 5]}, cv=cv, scoring="r2", n_jobs=-1)
    grid.fit(X_train, y_train)
    rf_best = grid.best_estimator_
    res = _evaluate_with_cv(rf_best, X_train, y_train, cv)
    results.append({"Model": "RandomForest", **res}); models["RandomForest"] = rf_best

    # ExtraTrees (grid léger)
    et = Pipeline([("pre", pre), ("fs", build_feature_selector()), ("et", ExtraTreesRegressor(random_state=RANDOM_STATE, n_jobs=-1))])
    grid = GridSearchCV(et, {"et__n_estimators": [300, 600], "et__max_depth": [None, 15, 30], "et__min_samples_leaf": [1, 2, 4]}, cv=cv, scoring="r2", n_jobs=-1)
    grid.fit(X_train, y_train)
    et_best = grid.best_estimator_
    res = _evaluate_with_cv(et_best, X_train, y_train, cv)
    results.append({"Model": "ExtraTrees", **res}); models["ExtraTrees"] = et_best

    # GradientBoosting (grid)
    gbr = Pipeline([("pre", pre), ("fs", build_feature_selector()), ("gbr", GradientBoostingRegressor(random_state=RANDOM_STATE))])
    grid = GridSearchCV(gbr, {"gbr__n_estimators": [200, 400], "gbr__learning_rate": [0.03, 0.1, 0.2], "gbr__max_depth": [2, 3, 4]}, cv=cv, scoring="r2", n_jobs=-1)
    grid.fit(X_train, y_train)
    gbr_best = grid.best_estimator_
    res = _evaluate_with_cv(gbr_best, X_train, y_train, cv)
    results.append({"Model": "GradientBoosting", **res}); models["GradientBoosting"] = gbr_best

    # KNN Regressor (grid)
    knn = Pipeline([("pre", pre), ("fs", build_feature_selector()), ("knn", KNeighborsRegressor())])
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
        pipe = Pipeline([("pre", pre), ("fs", build_feature_selector()), ("hgb", HistGradientBoostingRegressor(**params))])
        sc = cross_validate(pipe, X_train, y_train, cv=cv, scoring="r2", n_jobs=-1)
        return sc["test_score"].mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=OPTUNA_TRIALS)
    hgb = Pipeline([("pre", pre), ("fs", build_feature_selector()), ("hgb", HistGradientBoostingRegressor(random_state=RANDOM_STATE, **study.best_params))])
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

    y_pred = best_model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = float(mean_absolute_error(y_test, y_pred))

    joblib.dump(best_model, MODEL_PATH)

    if _HAS_MLFLOW and MLFLOW_ENABLED:
        with mlflow.start_run():
            mlflow.log_metric("R2_test", float(r2))
            mlflow.log_metric("RMSE_test", float(rmse))
            mlflow.log_metric("MAE_test", float(mae))
            mlflow.log_param("best_model", best_name)
    return {"best_model": best_name, "R2_test": float(r2), "RMSE_test": rmse, "MAE_test": mae, "model_path": str(MODEL_PATH)}
"""

INTERPRET = """\
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def get_personas() -> dict:
    return {
        "Revendeur pro": {"Objectif": "Estimer la valeur d'achat au plus juste (±5%)"},
        "Particulier": {"Objectif": "Vérifier si une annonce est surpayée (±10%)"},
        "Plateforme": {"Objectif": "Détecter les annonces outliers (±20%)"},
    }


def log_and_save_personas(personas: dict):
    md = ["# Personas\\n"]
    for n, attrs in personas.items():
        md.append(f"\\n## {n}\\n")
        for k, v in attrs.items():
            md.append(f"- **{k}**: {v}")
    (REPORTS_DIR / "personas.md").write_text("\\n".join(md), encoding="utf-8")


def analyze_price_segments(df: pd.DataFrame):
    counts = df["segment_prix"].value_counts()
    stats = df.groupby("segment_prix")["prix"].describe().round(2)
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
"""

RUN_PIPELINE = """\
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
    print("\\n=== Résultats CV (train) ===")
    print(df_results.to_string(index=False))

    summary = select_and_train_best(df_results, models, Xtr, ytr, Xte, yte)
    print(f\"\\nBest: {summary['best_model']} | R2_test={summary['R2_test']:.3f} RMSE={summary['RMSE_test']:.3f} MAE={summary['MAE_test']:.3f}\")
    print(f\"Saved → {summary['model_path']}\")
    try:
        ycal_eur = np.exp(ycal) - 1.0   # calibration en euros (pas en log)
        q = fit_conformal(Xcal, ycal_eur, alpha=0.1)
        print(f\"Conformal calibration done (alpha=0.1). q = {q:.2f} €\")
    except Exception as e:
        print(f\"[WARN] Conformal calibration failed: {e}\")
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
    # segment_prix exclu des features pour éviter la fuite de cible
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
feature_selection:
  enabled: true
  method: "lasso"
  k: 20
  lasso_alpha: 0.001
  rf_n_estimators: 300
  threshold: "median"
training:
  random_state: 42
  test_size: 0.2
  cv_splits: 5
  optuna_trials: 30
mlflow:
  enabled: false
  tracking_uri: null
  experiment: "watch-price"

# === Choix (categoriels) & Plages (numériques) pour l'UI Streamlit ===
choices:
  marque: ["Rolex", "Omega", "Patek Philippe", "Audemars Piguet", "Cartier", "TAG Heuer", "Breitling", "IWC"]
  modele: ["Submariner", "Daytona", "Datejust", "Speedmaster", "Seamaster", "Royal Oak", "Santos", "Navitimer", "Carrera", "Portugieser"]
  matiere_boitier: ["acier", "or/acier", "or massif"]
  matiere_bracelet: ["acier", "or/acier", "or massif"]
ranges:
  reserve_de_marche: {min: 0, max: 120, step: 1}
  diametre: {min: 30, max: 45, step: 1}
  etencheite: {min: 0, max: 300, step: 10}
  comptage_fonctions: {min: 0, max: 10, step: 1}
"""

MAKEFILE = """\
.PHONY: train clean app
train:
\tpython src/run_pipeline.py
clean:
\trm -f models/*.joblib
\trm -f reports/*.png reports/*.csv reports/*.md
app:
\tstreamlit run src/streamlit_app.py
"""


PYPROJECT_TOML = """\
[build-system]
requires = ["poetry-core>=1.9.0"]
build-backend = "poetry.core.masonry.api"

[tool.poetry]
name = "preowned-watch-predictor"
version = "0.1.0"
description = "Price prediction for pre-owned luxury watches"
readme = "README.md"
authors = ["BOUSSENGUI François <you@example.com>"]
license = "MIT"
keywords = ["ml", "regression", "watches", "price-prediction"]
# Package the 'src' directory as a package named 'src' (current code imports from 'src.*')
packages = [{ include = "src" }]

[tool.poetry.dependencies]
python = ">=3.10"
pandas = ">=2.0"
numpy = ">=1.24"
scikit-learn = ">=1.2,<2.0"
category-encoders = ">=2.6"
optuna = ">=3"
shap = ">=0.44"
matplotlib = ">=3.7"
pyyaml = ">=6.0"
mlflow = ">=2.5"
streamlit = ">=1.25"
openpyxl = ">=3.1"

[tool.poetry.group.dev.dependencies]
pytest = ">=7.4"
pytest-cov = ">=4.1"
black = ">=23.7"
ruff = ">=0.3.0"
mypy = ">=1.5"
ipykernel = ">=6.25"

[tool.pytest.ini_options]
addopts = "-q"
testpaths = ["tests"]
pythonpath = ["src"]

[tool.black]
line-length = 100
target-version = ["py310"]

[tool.ruff]
line-length = 100
select = ["E", "F", "I", "UP", "B"]
ignore = ["E203"]

[tool.ruff.lint.isort]
known-first-party = ["src"]
profile = "black"

[tool.poetry.scripts]
run-pipeline = "src.run_pipeline:main"

[tool.mypy]
python_version = "3.10"
ignore_missing_imports = true
warn_unused_ignores = true
warn_redundant_casts = true
warn_unused_configs = true
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
    # segment_prix présent pour l'analyse, mais exclu des features du modèle (anti-fuite)
    assert "segment_prix" in df.columns
    assert "segment_prix" not in X.columns
"""

TEST_TRAINING = """\
import os
from pathlib import Path
import pytest

from src.data_utils import load_data, get_feature_matrix
from src.training import evaluate_models

DATA_FILE = Path("data/processed/propre.xlsx")

@pytest.mark.skipif(not DATA_FILE.exists(), reason="Missing data file: data/processed/propre.xlsx")
def test_evaluate_models_runs():
    df = load_data()
    X, y = get_feature_matrix(df)
    # Sous-échantillon éventuel pour accélérer si dataset massif
    if len(df) > 5000:
        df = df.sample(2000, random_state=42)
        X, y = get_feature_matrix(df)
    results, models = evaluate_models(X, y)
    assert not results.empty
    assert isinstance(models, dict)
"""

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
    # Add any missing feature columns as NA/empty defaults
    for col in _FEATURES:
        if col not in df.columns:
            df[col] = np.nan if col not in ("marque", "modele", "matiere_boitier", "matiere_bracelet") else ""
    return df[_FEATURES]


def predict_with_interval(df_feats: pd.DataFrame, alpha: float = 0.1) -> pd.DataFrame:
    model = load_model()
    X = ensure_feature_columns(df_feats)
    y_log = model.predict(X)
    point = np.exp(y_log) - 1.0
    out = df_feats.copy()
    out["predicted_price"] = point
    # Try to load conformal q if present
    try:
        q = float(np.load(MODEL_PATH.with_name("conformal_q.npy")).ravel()[0])
        out["pi_low"] = np.maximum(0.0, point - q)
        out["pi_high"] = point + q
        out["pi_width"] = out["pi_high"] - out["pi_low"]
    except Exception:
        pass
    return out


def fit_conformal(calib_X: pd.DataFrame, calib_y_eur: pd.Series, alpha: float = 0.1) -> float:
    # calib_y_eur doit être en EUROS (pas en log)
    model = load_model()
    Xc = ensure_feature_columns(calib_X)
    y_log = model.predict(Xc)
    point = np.exp(y_log) - 1.0
    abs_res = np.abs(point - calib_y_eur.values)
    n = len(abs_res)
    if n == 0:
        return float("nan")
    idx = int(np.ceil((n + 1) * (1 - alpha))) - 1
    idx = int(np.clip(idx, 0, n - 1))
    q = float(np.sort(abs_res)[idx])
    try:
        import numpy as _np
        _np.save(MODEL_PATH.with_name("conformal_q.npy"), _np.array([q], dtype=float))
    except Exception:
        pass
    return q
"""

STREAMLIT_APP = """\
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
import streamlit as st

from .inference import predict_with_interval

BASE_DIR = Path(__file__).resolve().parent.parent
CFG = yaml.safe_load(open(BASE_DIR / "config" / "params.yaml", "r", encoding="utf-8"))
FEATURES = CFG["features"]["columns"]
CHOICES = CFG.get("choices", {})
RANGES = CFG.get("ranges", {})

st.set_page_config(page_title="Watch Price Predictor", page_icon="⌚", layout="centered")
st.title("⌚ Prévision de prix de montres d'occasion")
st.caption("Modèle entraîné hors-ligne — choisissez des caractéristiques pour estimer le prix.")

mode = st.sidebar.radio("Mode", ["Formulaire", "Fichier (CSV/Excel)"])

if mode == "Fichier (CSV/Excel)":
    up = st.file_uploader("Déposez un CSV ou un Excel contenant les colonnes features", type=["csv", "xlsx"])
    if up is not None:
        try:
            df = pd.read_csv(up) if up.name.lower().endswith(".csv") else pd.read_excel(up)
        except Exception as e:
            st.error(f"Erreur de lecture du fichier: {e}")
            df = None
        if df is not None:
            missing = [c for c in FEATURES if c not in df.columns]
            if missing:
                st.warning("Colonnes manquantes ajoutées par défaut : " + ", ".join(missing))
            preds = predict_with_interval(df, alpha=0.1)
            st.success(f"Prédictions effectuées sur {len(preds)} ligne(s).")
            st.dataframe(preds)
            st.download_button("Télécharger les prédictions (CSV)", data=preds.to_csv(index=False).encode("utf-8"), file_name="watch_predictions.csv", mime="text/csv")
else:
    with st.form("price_form"):
        col1, col2 = st.columns(2)

        def _build_range_list(key, default_min, default_max, default_step):
            r = RANGES.get(key, {"min": default_min, "max": default_max, "step": default_step})
            start = int(r.get("min", default_min)); stop = int(r.get("max", default_max)); step = int(r.get("step", default_step))
            return list(range(start, stop + step, step))

        with col1:
            marque = st.selectbox("Marque", CHOICES.get("marque", []))
            modele = st.selectbox("Modèle", CHOICES.get("modele", []))
            matiere_boitier = st.selectbox("Matière boîtier", CHOICES.get("matiere_boitier", []))
            matiere_bracelet = st.selectbox("Matière bracelet", CHOICES.get("matiere_bracelet", []))

        with col2:
            reserve_de_marche = st.select_slider("Réserve de marche (h)", options=_build_range_list("reserve_de_marche", 0, 120, 1), value=48)
            diametre = st.select_slider("Diamètre (mm)", options=_build_range_list("diametre", 30, 45, 1), value=40)
            etencheite = st.select_slider("Étanchéité (m)", options=_build_range_list("etencheite", 0, 300, 10), value=100)
            comptage_fonctions = st.select_slider("# Complications/fonctions", options=_build_range_list("comptage_fonctions", 0, 10, 1), value=1)

        submitted = st.form_submit_button("Estimer le prix")
    if submitted:
        row = {
            "marque": marque,
            "modele": modele,
            "matiere_boitier": matiere_boitier,
            "matiere_bracelet": matiere_bracelet,
            "reserve_de_marche": reserve_de_marche,
            "diametre": diametre,
            "etencheite": etencheite,
            "comptage_fonctions": comptage_fonctions,
        }
        pred = predict_with_interval(pd.DataFrame([row]), alpha=0.1)
        price = float(pred.loc[0, "predicted_price"])
        st.subheader("Prix estimé (intervalle 90%)")
        if "pi_low" in pred.columns:
            lo = float(pred.loc[0, "pi_low"]); hi = float(pred.loc[0, "pi_high"])
            st.metric("€", f"{price:,.0f}".replace(",", " "), help=f"Intervalle: {lo:,.0f} – {hi:,.0f} €".replace(",", " "))
        else:
            st.metric("€", f"{price:,.0f}".replace(",", " "))
        st.caption("Confiance empirique basée sur calibration split-conformal (alpha=0.1).")
"""

def write_file(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(content, encoding="utf-8")
        print(f"[CREATED] {path}")
    else:
        print(f"[SKIPPED] {path} (exists)")

def main():
    print("Bootstrapping l'architecture modulaire…")
    (BASE_DIR / "reports").mkdir(exist_ok=True)
    (BASE_DIR / "models").mkdir(exist_ok=True)
    (BASE_DIR / "data" / "processed").mkdir(parents=True, exist_ok=True)
    (SRC_DIR / "__init__.py").write_text('"""src package for preowned-watch-predictor."""\n', encoding="utf-8")

    # Écriture des fichiers
    (SRC_DIR / "data_utils.py").write_text(DATA_UTILS, encoding="utf-8")
    (SRC_DIR / "preprocess.py").write_text(PREPROCESS, encoding="utf-8")
    (SRC_DIR / "feature_selection.py").write_text(FEATURE_SELECTION, encoding="utf-8")
    (SRC_DIR / "training.py").write_text(TRAINING, encoding="utf-8")
    (SRC_DIR / "interpret.py").write_text(INTERPRET, encoding="utf-8")
    (SRC_DIR / "run_pipeline.py").write_text(RUN_PIPELINE, encoding="utf-8")
    (CONFIG_DIR / "params.yaml").write_text(PARAMS_YAML, encoding="utf-8")
    (SRC_DIR / "inference.py").write_text(INFERENCE, encoding="utf-8")
    (SRC_DIR / "streamlit_app.py").write_text(STREAMLIT_APP, encoding="utf-8")

    # Fichiers racine
    (BASE_DIR / "Makefile").write_text(MAKEFILE, encoding="utf-8")
    (BASE_DIR / "pyproject.toml").write_text(PYPROJECT_TOML, encoding="utf-8")

    # Tests
    (BASE_DIR / "tests").mkdir(parents=True, exist_ok=True)
    (BASE_DIR / "tests" / "test_data_utils.py").write_text(TEST_DATA_UTILS, encoding="utf-8")
    (BASE_DIR / "tests" / "test_training.py").write_text(TEST_TRAINING, encoding="utf-8")

    print("\n✅ Bootstrap terminé. Prochaines étapes :")
    print("   1) Vérifier/éditer config/params.yaml (choices/ranges)")
    print("   2) Entraîner :  python src/run_pipeline.py")
    print("   3) Lancer l'app : streamlit run src/streamlit_app.py")

if __name__ == "__main__":
    main()