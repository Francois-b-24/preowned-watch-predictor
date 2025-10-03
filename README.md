# luxury-watch-price-predictor

# preowned-watch-predictor

**Prédiction de prix de montres de luxe d'occasion**

Ce projet fournit **un pipeline complet de Machine Learning** pour estimer le prix affiché de montres de seconde main (données Chrono24) **et une application Streamlit** pour réaliser des prédictions interactives.

---

## 🚀 Fonctionnalités
- Extraction, nettoyage et préparation des données (`data_utils.py`)
- Modélisation : LinearRegression, Ridge, Lasso, ElasticNet, RandomForest, ExtraTrees, GradientBoosting, HistGradientBoosting, KNN (Optuna)
- Sélection de variables optionnelle (kbest, lasso, random forest importance)
- Calibration conformale (intervalle de prédiction en euros)
- Interprétation simple (segments prix, boxplots)
- Sauvegarde du pipeline final (`models/final_pipeline.joblib`)
- **Application Streamlit** avec listes déroulantes et curseurs (valeurs basées sur la base ou forcées par YAML)
- Tests de base (`tests/`)

---

## 📁 Structure du projet

---

## 📁 Structure du projet
```
├── config/
│   └── params.yaml                 # Configuration : features, preprocess, choix UI, entraînement
├── data/
│   └── processed/propre.xlsx       # Données prêtes (non versionnées)
├── models/
│   ├── final_pipeline.joblib       # Modèle entraîné
│   └── conformal_q.npy             # Quantile d’erreur pour intervalle de confiance
├── reports/                        # Rapports (comparaison modèles, boxplots…)
├── src/
│   ├── init.py
│   ├── bootstrap.py                # Génère/écrit les fichiers du projet
│   ├── run_pipeline.py             # Pipeline complet d’entraînement et calibration
│   ├── data_utils.py               # Chargement & préparation (ajout log_prix, segment_prix)
│   ├── preprocess.py               # build_preprocessor()
│   ├── feature_selection.py        # Sélecteurs de variables
│   ├── training.py                 # CV, tuning Optuna, sauvegarde du meilleur modèle
│   ├── interpret.py                # Segments prix et graphiques
│   ├── inference.py                # Fonctions d’inférence + intervalle conformal
│   └── streamlit_app.py            # Application Streamlit
├── tests/
│   ├── test_data_utils.py
│   └── test_training.py
├── Makefile                        # make train / clean / app
├── pyproject.toml                  # Dépendances (Poetry), scripts, config black/ruff/mypy/pytest
└── README.md
```

⸻

## 🛠️ Installation avec Poetry
1. **Cloner le dépôt**
```bash
	git clone https://github.com/<utilisateur>/preowned-watch-predictor.git
	cd preowned-watch-predictor
```

2.	Installer les dépendances avec Poetry
```bash
	poetry install
```

⚠️ Python 3.10–3.12 recommandé.
Si vous êtes en Python 3.13, créez un venv compatible :

```bash
brew install python@3.12
poetry env use /opt/homebrew/bin/python3.12
poetry install
```

3.	Préparer les données
	•	Placer propre.xlsx dans data/processed/
	•	Le fichier doit contenir au minimum : prix, marque, modele, ainsi que les features définies dans config/params.yaml.

⸻

## ▶️ Exécuter le pipeline ML
Depuis la racine :
```bash
poetry run python -m src.run_pipeline
```
ou via le script Poetry (déclaré dans pyproject.toml) :

```bash
poetry run run-pipeline
```

ou via le Makefile :

```bash
poetry run make train
```

Le pipeline :
	•	compare plusieurs modèles avec validation croisée,
	•	sélectionne le meilleur,
	•	entraîne et sauvegarde le pipeline (models/final_pipeline.joblib),
	•	calibre un intervalle de confiance (split-conformal).

⸻

## ▶️ Lancer l’application Streamlit

Depuis la racine :

```bash
poetry run streamlit run src/streamlit_app.py
```
ou 

```bash
poetry run make app
```
⸻

Fonctionnement :
	•	Choisissez “Basées sur la base de données” pour que les listes (marques, modèles, matières) et les curseurs soient construits automatiquement (top-K valeurs, quantiles 1–99 %).
	•	Ou choisissez “Forcer les valeurs du YAML” pour utiliser les valeurs définies dans config/params.yaml.
	•	La liste des modèles se filtre dynamiquement selon la marque choisie.
	•	Bloc 🔎 Diagnostic des données intégré pour afficher lignes/colonnes détectées.

⸻


## 🧪 Tests

```bash
poetry run pytest
# ou avec couverture
poetry run pytest --cov
```

⸻


## 🔧 Dépannage rapide
	•	ModuleNotFoundError: No module named ‘src’ : lancez toujours avec poetry run ... depuis la racine.
	•	Python 3.13 non supporté : créez un venv Python 3.12 via poetry env use.
	•	UI vide en mode “Basées sur la base de données” : vérifiez que data/processed/propre.xlsx existe et contient les colonnes attendues (voir diagnostic dans l’app).
	•	Pas d’intervalle affiché : relancez run_pipeline pour recalculer models/conformal_q.npy.

⸻

## 🤝 Contribuer
	1.	Forker le dépôt
	2.	Créer une branche feature/ma-fonction
	3.	Commit & Push
	4.	Ouvrir un Pull Request

⸻

## 📄 Licence
Ce projet est sous licence MIT. Voir [LICENSE](LICENSE) pour plus de détails.




