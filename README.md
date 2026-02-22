# AirParadis — Analyse de sentiment (MLOps)

## Objectif
Prédire si un tweet exprime un sentiment négatif afin d’anticiper un bad buzz.

## Contenu du projet (livrables)
- **Préparation des données** : `00_prepare_data.ipynb` → `data/processed/{train,val,test}.csv`
- **Modèle classique** : `01_modele_classique_tfidf.ipynb` (TF-IDF + LogisticRegression) + artefacts dans `artifacts/`
- **Modèle sur mesure avancé** : `02_modele_sur_mesure_avance_keras.ipynb` (Keras) + `exported_model/`
- **API FastAPI** : `api/main.py` (endpoints `/health`, `/predict`, `/feedback`)
- **Tests unitaires** : `tests/` (pytest)
- **Interface de test** : `interface/app_streamlit.py` (Streamlit)
- **Tracking MLflow** : `mlruns/`

## Arborescence
- `api/` : API FastAPI
- `interface/` : interface Streamlit
- `tests/` : tests unitaires
- `data/processed/` : datasets train/val/test
- `artifacts/` : tokenizer + metrics + baseline
- `exported_model/` : modèle Keras exporté (SavedModel)
- `mlruns/` : runs MLflow (tracking local)

## Installation (local)
Activer l’environnement conda `sentiment`, puis :

```bash
pip install -r api/requirements.txt
pip install -r interface/requirements.txt
pip install pytest httpx