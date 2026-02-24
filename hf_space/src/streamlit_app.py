from pathlib import Path
import json
import streamlit as st
import tensorflow as tf
import keras
from tensorflow.keras.preprocessing.text import tokenizer_from_json

# ------------------------------------------------------------
# Chemins
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "exported_model"
TOKENIZER_PATH = BASE_DIR / "tokenizer.json"

# ------------------------------------------------------------
# Interface
# ------------------------------------------------------------

st.title("Air Paradis — Détection de tweets négatifs")

# ------------------------------------------------------------
# Chargement modèle + tokenizer
# ------------------------------------------------------------

@st.cache_resource
def load_assets():
    # Chargement SavedModel via TFSMLayer (Keras 3)
    model = keras.layers.TFSMLayer(
        str(MODEL_DIR),
        call_endpoint="serving_default"
    )

    # Chargement tokenizer
    tokenizer_data = json.loads(TOKENIZER_PATH.read_text(encoding="utf-8"))
    tokenizer = tokenizer_from_json(json.dumps(tokenizer_data))

    return model, tokenizer


model, tokenizer = load_assets()

# ------------------------------------------------------------
# Zone utilisateur
# ------------------------------------------------------------

text = st.text_area("Entrez un tweet")

if st.button("Prédire"):
    if text.strip():

        # -------- Préprocessing --------
        seq = tokenizer.texts_to_sequences([text])

        padded = tf.keras.preprocessing.sequence.pad_sequences(
            seq,
            maxlen=100
        )

        # 🔥 Correction : cast en float32 (signature du SavedModel)
        padded = tf.convert_to_tensor(padded, dtype=tf.float32)

        # -------- Inference --------
        pred = model(padded)

        # Certains SavedModel renvoient un dict
        if isinstance(pred, dict):
            pred = list(pred.values())[0]

        score = float(pred.numpy()[0][0])
        label = "negative" if score > 0.5 else "not_negative"

        st.write("Label :", label)
        st.write("Score :", score)