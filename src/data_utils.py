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
