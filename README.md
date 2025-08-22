# luxury-watch-price-predictor

**Prédiction de prix de montres de luxe d'occasion**

## 🚀 Objectif
Ce projet fournit un pipeline complet de Machine Learning pour estimer le **prix affiché** de montres de luxe de seconde main, en s’appuyant sur les données de Chrono24. Il propose :
- Extraction, nettoyage et préparation des données
- Modélisation (régression linéaire, Ridge, HGB, tuning Optuna)
- Interprétabilité (SHAP)
- Validation croisée et learning curves
- Packaging du modèle final

## 📁 Structure du projet
```
preowned-watch-predictor/
├── config/
│   └── params.yaml                 # Chemins, features, CV, Optuna, MLflow (désactivé par défaut)
├── data/
│   ├── raw/                        # Données brutes (NE PAS versionner)
│   └── processed/                  # Données prêtes (ex: propre.xlsx — NE PAS versionner)
├── models/
│   └── final_pipeline.joblib       # Modèle entraîné (NE PAS versionner)
├── reports/
│   ├── personas.md                 # Personas (généré)
│   ├── segment_counts.csv          # Comptes par segment (généré)
│   ├── segment_price_stats.csv     # Stats par segment (généré)
│   ├── segment_price_boxplot.png   # Boxplot prix par segment (généré)
│   ├── model_comparison.csv        # Comparaison R² CV des modèles (généré)
│   ├── shap_top10.png              # Importances SHAP si HGB (généré)
│   └── learning_curve.png          # Courbe d’apprentissage (généré)
├── src/
│   ├── bootstrap.py                # Script qui génère les fichiers manquants (déjà en place)
│   ├── run_pipeline.py             # Point d’entrée : analyse + comparaison modèles + sauvegarde
│   ├── data_utils.py               # load_data(), segment_prix(), get_feature_matrix()
│   ├── preprocess.py               # build_preprocessor()
│   ├── training.py                 # CV, tuning (Grid/Optuna), sélection & sauvegarde du meilleur
│   └── interpret.py                # Personas, segments, SHAP, learning curves
├── tests/
│   ├── test_data_utils.py          # Test de fumée sur chargement/features
│   └── test_training.py            # Test de fumée sur l’évaluation des modèles
├── Makefile                        # Raccourcis: make train / test / mlflow-ui / clean
├── pyproject.toml                  # Dépendances + config pytest/black/ruff
├── .gitignore                      # À créer/compléter (ci-dessous)
└── README.md                       # Présentation, usage, résultats (à compléter)
```

## 🛠️ Installation
1. **Cloner le dépôt**
   ```bash
   git clone https://github.com/<utilisateur>/luxury-watch-price-predictor.git
   cd luxury-watch-price-predictor
   ```

2. **Créer un environnement virtuel et installer**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Préparer les données**
   - Placer `propre.xlsx` dans `data/processed/`


## ▶️ Exécution du pipeline
Pour lancer toutes les étapes (1→12) et générer : personas, analyses, modèles, rapports et sauvegarde finale :
```bash
python src/run_pipeline.py
```

## 🔍 Résultats
- **Modèle final** enregistré dans `models/final_pipeline.joblib`
- **Graphiques** : `reports/shap_top10.png`, `reports/learning_curve.png`

## 📦 Déploiement API (FastAPI)
1. Positionner-vous à la racine du projet
2. Construire l’image Docker :
3. Lancer le conteneur :
4. Tester l’endpoint :
   

## 🤝 Contribuer
1. Forker le dépôt
2. Créer une branche `feature/ma-fonction`
3. Commit & Push
4. Ouvrir un Pull Request

## 📄 Licence
Ce projet est sous licence MIT. Voir [LICENSE](LICENSE) pour plus de détails.




