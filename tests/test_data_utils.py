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
