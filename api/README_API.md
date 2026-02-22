# API — Prédiction de sentiment

## Lancer en local
Depuis la racine du projet :

```bash
pip install -r api/requirements.txt
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload