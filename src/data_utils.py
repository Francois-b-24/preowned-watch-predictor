import pandas as pd

def load_data(path: str) -> pd.DataFrame:
    """
    Charge les données depuis un fichier Excel ou CSV.
    """
    if path.lower().endswith('.xlsx'):
        return pd.read_excel(path)
    else:
        return pd.read_csv(path)

def segment_prix(val: float) -> str:
    """
    Classement d'une montre en segment métier selon le prix.
    """
    if val < 5000:
        return 'Entrée de gamme'
    elif val <= 20000:
        return 'Moyen de gamme'
    else:
        return 'Haut de gamme'
