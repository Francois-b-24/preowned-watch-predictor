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
