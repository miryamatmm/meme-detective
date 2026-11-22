import os

from openai import OpenAI
from PIL import Image
import requests
import streamlit as st
from dotenv import load_dotenv

API_URL = "http://serving-api:8080/predict"
N8N_WEBHOOK_URL = "http://n8n:5678/webhook/human-validation"
load_dotenv()

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

SHARED_DATA_DIR = "/data/temp_uploads"
os.makedirs(SHARED_DATA_DIR, exist_ok=True)


def generate_explanation(label):
    try:
        prompt = f"Tu es un expert en culture internet. Explique de façon drôle et concise (max 2 phrases) l'origine ou la signification du meme : '{label}'."

        response = client.chat.completions.create(
            model="gpt-3.5-turbo", # Ou gpt-3.5-turbo
            messages=[
                {"role": "system", "content": "Tu es un assistant expert en memes."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Oups, l'IA est timide aujourd'hui. (Erreur: {str(e)})"


st.set_page_config(page_title="Meme Detective", layout="centered")
st.title("Meme Classification (MLOps)")

if "prediction" not in st.session_state:
    st.session_state.prediction = None
if "uploaded_file_name" not in st.session_state:
    st.session_state.uploaded_file_name = None

uploaded_file = st.file_uploader("Charge ton mème ici", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    file_path = os.path.join(SHARED_DATA_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Image à analyser", width=300)

    if st.button("Predire le Template"):
        with st.spinner("Interrogation du modèle..."):
            try:
                with open(file_path, "rb") as f:
                    files = {"file": (uploaded_file.name, f, uploaded_file.type)}
                    response = requests.post(API_URL, files=files)
                
                if response.status_code == 200:
                    result = response.json()
                    st.session_state.prediction = result
                    st.session_state.uploaded_file_name = uploaded_file.name
                else:
                    st.error(f"Erreur API : {response.status_code}")
            except Exception as e:
                st.error(f"Erreur connexion : {e}")

if st.session_state.prediction:
    res = st.session_state.prediction
    pred_label = res.get("prediction")
    conf = res.get("confidence_score", 0)
    
    st.divider()
    st.markdown(f"### Resultat : **{pred_label}**")
    st.progress(int(conf))
    st.caption(f"Confiance : {conf}%")

    with st.spinner("L'IA analyse le contexte du meme..."):
        explanation = generate_explanation(pred_label)


    st.info(f"**L'instant Culture :** {explanation}")

    st.markdown("---")
    st.write("Le résultat semble incorrect ?")

    if st.button("Notifier l'utilisateur (Correction Humaine)"):
        with st.spinner("Contact de l'agent n8n..."):
            try:
                payload = {
                    "filename": st.session_state.uploaded_file_name,
                    "prediction": pred_label,
                    "confidence": conf
                }
                requests.post(N8N_WEBHOOK_URL, json=payload)
                st.success("Agent notifie ! Verifie Discord pour valider.")
            except Exception as e:
                st.error(f"Erreur n8n : {e}")