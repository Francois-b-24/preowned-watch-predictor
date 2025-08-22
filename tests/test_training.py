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
