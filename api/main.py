import json
from pathlib import Path

import tensorflow as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

MODEL_DIR = "exported_model"
TOKENIZER_PATH = "artifacts/tokenizer.json"
CONFIG_PATH = "artifacts/preprocess_config.json"

app = FastAPI(title="AirParadis Sentiment API")

model = None
tokenizer = None
seq_len = None


class PredictRequest(BaseModel):
    text: str


def load_tokenizer():
    with open(TOKENIZER_PATH, "r", encoding="utf-8") as f:
        tok_json = f.read()
    return tf.keras.preprocessing.text.tokenizer_from_json(tok_json)


def load_seq_len():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return int(cfg["seq_len"])


def ensure_ready():
    if model is None or tokenizer is None or seq_len is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Check exported_model/ and artifacts/ files."
        )


@app.on_event("startup")
def startup():
    global model, tokenizer, seq_len

    if not Path(MODEL_DIR).exists():
        raise RuntimeError(f"Missing folder: {MODEL_DIR}")
    if not Path(TOKENIZER_PATH).exists():
        raise RuntimeError(f"Missing file: {TOKENIZER_PATH}")
    if not Path(CONFIG_PATH).exists():
        raise RuntimeError(f"Missing file: {CONFIG_PATH}")

    model = tf.keras.models.load_model(MODEL_DIR)
    tokenizer = load_tokenizer()
    seq_len = load_seq_len()


def preprocess(text: str):
    ensure_ready()
    seq = tokenizer.texts_to_sequences([text])
    return tf.keras.preprocessing.sequence.pad_sequences(
        seq, maxlen=seq_len, padding="post", truncating="post"
    )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest):
    ensure_ready()
    text = req.text.strip()
    if not text:
        return {"label": "unknown", "score": 0.0}

    X = preprocess(text)
    proba = float(model.predict(X, verbose=0).ravel()[0])
    label = "negative" if proba >= 0.5 else "not_negative"
    return {"label": label, "score": proba}

    from pydantic import BaseModel

class FeedbackRequest(BaseModel):
    text: str
    predicted_label: str
    predicted_score: float
    is_correct: bool

@app.post("/feedback")
def feedback(req: FeedbackRequest):
    return {"status": "ok"}