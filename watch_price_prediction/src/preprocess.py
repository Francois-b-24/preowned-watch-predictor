import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from category_encoders import TargetEncoder

def build_preprocessor() -> ColumnTransformer:
    """
    Construit un ColumnTransformer pour prétraiter les données:
      - OneHotEncoder pour marque et segment_prix
      - TargetEncoder pour modele
      - OrdinalEncoder pour matières
      - StandardScaler pour variables techniques
    """
    material_cats = ['acier', 'or/acier', 'or massif']
    return ColumnTransformer(transformers=[
        ('brand', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), ['marque']),
        ('model_te', Pipeline([
            ('te', TargetEncoder(min_samples_leaf=50, smoothing=10))
        ]), ['modele']),
        ('seg', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), ['segment_prix']),
        ('mat', Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='')),
            ('ordinal', OrdinalEncoder(
                categories=[material_cats, material_cats],
                handle_unknown='use_encoded_value',
                unknown_value=-1
            ))
        ]), ['matiere_boitier', 'matiere_bracelet']),
        ('tech', Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ]), ['reserve_de_marche', 'diametre', 'etencheite', 'comptage_fonctions'])
    ], remainder='drop')
