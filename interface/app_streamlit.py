import streamlit as st
import requests

st.set_page_config(page_title="AirParadis - Test API", layout="centered")

st.title("Interface de test — API Sentiment")

api_url = st.text_input("URL de l'API", "http://127.0.0.1:8000")

st.subheader("1) Prédiction")
tweet = st.text_area("Texte du tweet", height=120)

if st.button("Prédire"):
    if not tweet.strip():
        st.warning("Saisis un tweet.")
    else:
        try:
            r = requests.post(f"{api_url}/predict", json={"text": tweet}, timeout=30)
            st.write("Status:", r.status_code)
            if r.headers.get("content-type", "").startswith("application/json"):
                pred = r.json()
                st.json(pred)

                st.session_state["last_tweet"] = tweet
                st.session_state["last_pred"] = pred
            else:
                st.code(r.text)
        except Exception as e:
            st.error(str(e))

st.divider()

st.subheader("2) Feedback")
if "last_pred" not in st.session_state:
    st.info("Fais d’abord une prédiction pour activer le feedback.")
else:
    pred = st.session_state["last_pred"]
    last_tweet = st.session_state["last_tweet"]

    st.write("Dernière prédiction :")
    st.json(pred)

    is_correct = st.radio("La prédiction est-elle correcte ?", [True, False], index=0)

    if st.button("Envoyer feedback"):
        payload = {
            "text": last_tweet,
            "predicted_label": pred.get("label", ""),
            "predicted_score": float(pred.get("score", 0.0)),
            "is_correct": is_correct,
        }
        try:
            r = requests.post(f"{api_url}/feedback", json=payload, timeout=30)
            st.write("Status:", r.status_code)
            if r.headers.get("content-type", "").startswith("application/json"):
                st.json(r.json())
            else:
                st.code(r.text)
        except Exception as e:
            st.error(str(e))