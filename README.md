# luxury-watch-price-predictor

**Prédiction de prix de montres de luxe d'occasion**

Ce projet fournit **un pipeline complet de Machine Learning** pour estimer le prix affiché de montres de seconde main (données Chrono24) **et une application Streamlit** (formulaire uniquement) pour réaliser des prédictions interactives.

---

## 🚀 Fonctionnalités
- Extraction, nettoyage et préparation des données
- Modélisation : Régression Linéaire, Ridge, Lasso, ElasticNet, RandomForest, ExtraTrees, GradientBoosting, HistGradientBoosting (Optuna)
- Interprétabilité (SHAP)
- Validation croisée, courbe d’apprentissage
- Sauvegarde du pipeline final (`models/final_pipeline.joblib`)
- **Application Streamlit** (sans upload de fichier, **sans segment de prix**, **sans sélection du modèle** de montre)

---

## 📁 Structure du projet
```
├── config/
│   └── params.yaml                 # Chemins, features, CV, Optuna, etc.
├── data/
│   ├── raw/                        # Données brutes (NE PAS versionner)
│   └── processed/                  # Données prêtes (ex: propre.xlsx — NE PAS versionner)
├── models/
│   ├── final_pipeline.joblib       # Modèle entraîné (NE PAS versionner)
│   └── estimator.py                # Variante de l’app Streamlit (équivalente à src/streamlit_app.py)
├── reports/
│   ├── personas.md                 # Généré
│   ├── segment_counts.csv          # Généré
│   ├── segment_price_stats.csv     # Généré
│   ├── segment_price_boxplot.png   # Généré
│   ├── model_comparison.csv        # Généré
│   ├── shap_top10.png              # Généré
│   └── learning_curve.png          # Généré
├── src/
│   ├── bootstrap.py                # Génère les fichiers manquants
│   ├── run_pipeline.py             # Exécution : analyse + comparaison modèles + sauvegarde
│   ├── data_utils.py               # load_data(), get_feature_matrix(), etc.
│   ├── preprocess.py               # build_preprocessor()
│   ├── training.py                 # CV, tuning (Grid/Optuna), sélection & sauvegarde du meilleur
│   ├── interpret.py                # Personas, segments, SHAP, learning curves
│   ├── inference.py                # Fonctions d’inférence (chargement & prédiction)
│   └── streamlit_app.py            # Application Streamlit (formulaire)
├── tests/
│   ├── test_data_utils.py          # Smoke tests
│   └── test_training.py            # Smoke tests
├── Makefile                        # make train / test / mlflow-ui / clean / app
├── pyproject.toml                  # Dépendances + config pytest/black/ruff
├── .gitignore                      # À compléter
└── README.md
```

---
## 🛠️ Installation
1. **Cloner le dépôt**
   ```bash
   git clone https://github.com/<utilisateur>/luxury-watch-price-predictor.git
   cd luxury-watch-price-predictor

2. **Créer un environnement virtuel et installer**
   ```bash
  python3 -m venv venv
  source venv/bin/activate
  pip install -r requirements.txt
   ```

3. **Préparer les données**
   - Placer `propre.xlsx` dans `data/processed/`
   - Le fichier doit contenir les features définies dans config/params.yaml


## ▶️ Exécuter le pipeline ML

Recommandé (depuis la racine) :
```bash
python -m src.run_pipeline
# ou
make train
```
Cela entraîne les modèles, compare les performances (CV), et sauvegarde le meilleur pipeline dans models/final_pipeline.joblib.

## 🔧 Dépannage rapide
	•	ModuleNotFoundError: No module named 'src' : lancez depuis la racine du projet, ou utilisez python -m src.run_pipeline.
	•	ufunc 'isnan' not supported ... : assurez-vous de saisir des nombres pour les champs numériques. L’app nettoie/convertit automatiquement, mais si vous avez modifié le schéma des features, alignez config/params.yaml puis ré-entraînez (make train).

⸻

## 🔍 Résultats
	•	Modèle final : models/final_pipeline.joblib
	•	Graphiques : reports/shap_top10.png, reports/learning_curve.png

⸻

## ▶️ Lancer l’application Streamlit

Depuis la racine du projet :
```bash
streamlit run src/streamlit_app.py
 ou
make app
```

Fonctionnement : choisissez une marque puis renseignez les autres caractéristiques (matières, réserve de marche, diamètre, étanchéité, nombre de fonctions). L’application retourne le prix estimé.

Remarques : l’app n’utilise pas de fichier CSV/Excel; elle ne demande ni segment de prix ni modèle de montre. Si ces features existent encore dans params.yaml, elles sont neutralisées côté inférence.

⸻

## 📦 Déploiement API (optionnel)

Alternative à Streamlit si vous souhaitez un endpoint. Voir un éventuel src/api.py. Déploiement possible via Docker/Cloud Run/Render, etc.

⸻

## 🤝 Contribuer
	1.	Forker le dépôt
	2.	Créer une branche feature/ma-fonction
	3.	Commit & Push
	4.	Ouvrir un Pull Request


## 📄 Licence
Ce projet est sous licence MIT. Voir [LICENSE](LICENSE) pour plus de détails.




