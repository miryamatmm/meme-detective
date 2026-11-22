# ==================================== IMPORTS ====================================
import streamlit as st
from PIL import Image
import tempfile
import os
import requests
import uuid

# ==================================== CONFIG API ====================================
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8080")
CONFIDENCE_THRESHOLD = 50.0


# ==================================== N8N CALL ====================================
def call_n8n_llm(prediction, confidence, image_url):
    payload = {
        "event_id": str(uuid.uuid4()),
        "prediction": prediction,
        "score": confidence,
        "image_url": image_url
    }

    try:
        response = requests.post(
            "http://localhost:5678/webhook/predicition",
            json=payload,
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}


# ==================================== STREAMLIT STATE ====================================
if "feedback_given" not in st.session_state:
    st.session_state.feedback_given = False

if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None

if "messages" not in st.session_state:
    st.session_state.messages = []


# ==================================== CONFIG ====================================
st.set_page_config(
    page_title="Meme Template Chatbot",
    page_icon="",
    layout="centered"
)

st.title("Meme Template Chatbot")
st.caption(
    "Projet pédagogique – Master Informatique Data Science – Université Claude Bernard Lyon 1"
)
st.caption(
    "Encadré par : Haytham Elghazel"
)
st.caption(
    "Équipe : Wyatt Leaux, AbdelKarim Bencheikh, Imane Gara, Niama Chibani, Miryam Atamna, Olivia Chen"
)

st.text("Upload un mème → prédiction ML → explication LLM")

# ==================================== IMAGE UPLOAD ====================================
uploaded_file = st.file_uploader(
    "Upload un mème",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    # Charger l'image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Mème reçu", width=300)

    # Message utilisateur
    user_message = "Voici mon mème"
    with st.chat_message("user"):
        st.markdown(user_message)

    st.session_state.messages.append({
        "role": "user",
        "content": user_message
    })

    # ===================== SAUVEGARDE TEMPORAIRE =====================
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image_path = tmp.name
        image.save(image_path)

    # ===================== PREDICTION ML (API) =====================
    with st.spinner("Analyse du mème en cours..."):
        try:
            with open(image_path, "rb") as f:
                files = {"file": (os.path.basename(image_path), f, "image/jpeg")}
                resp = requests.post(f"{API_BASE_URL}/predict", files=files, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            label = data.get("prediction")
            confidence = float(data.get("confidence_score", 0.0))
        except Exception as e:
            label = None
            confidence = 0.0
            st.error(f"Erreur appel API: {e}")

        st.session_state.prediction_result = {
            "label": label,
            "confidence": confidence
        }
        st.session_state.feedback_given = False

    os.remove(image_path)

    # ===================== RÉPONSE ASSISTANT =====================
    if label is None:
        assistant_msg = "Impossible d'analyser l'image."

    elif confidence < CONFIDENCE_THRESHOLD:
        assistant_msg = (
            f"**Résultat incertain**\n\n"
            f"Template le plus probable : **{label}**\n"
            f"Confiance : **{confidence:.1f}%**\n\n"
            "Le modèle n'est pas assez sûr de lui."
        )

    else:
        # 🔗 APPEL N8N + LLM
        llm_result = call_n8n_llm(
            prediction=label,
            confidence=confidence,
            image_url="uploaded_via_streamlit"
        )

        if "explanation" in llm_result:
            assistant_msg = llm_result["explanation"]
        else:
            assistant_msg = (
                f"**Template détecté : {label}**\n\n"
                f"Confiance : **{confidence:.2f}%**\n\n"
                "Ce template est bien reconnu par le modèle."
            )

    # Message assistant
    with st.chat_message("assistant"):
        st.markdown(assistant_msg)

    st.session_state.messages.append({
        "role": "assistant",
        "content": assistant_msg
    })

    # ===================== FEEDBACK UTILISATEUR =====================
    if st.session_state.prediction_result and not st.session_state.feedback_given:
        st.markdown("### La prédiction est-elle correcte ?")
        col1, col2 = st.columns(2)

        with col1:
            correct = st.button("✅ Oui")
        with col2:
            incorrect = st.button("❌ Non")

        if correct:
            st.success("Merci pour ton retour !")
            st.session_state.feedback_given = True

        if incorrect:
            st.session_state.show_correction = True

    # ===================== CORRECTION =====================
    if st.session_state.get("show_correction", False):
        st.markdown("### La prédiction est incorrecte")
        correct_label = st.text_input(
            "Quel est le bon template ?",
            placeholder="ex: Drake, Two Buttons..."
        )

        if st.button("Envoyer la correction"):
            st.success("Merci ! Ta correction a été enregistrée.")
            st.session_state.feedback_given = True
            st.session_state.show_correction = False


# ==================================== EVIDENTLY REPORT ====================================
report_path = "evidently_report.html"

if os.path.exists(report_path):
    with open(report_path, "r", encoding="utf-8") as f:
        report_html = f.read()

    st.markdown("### 📊 Rapport Evidently")
    st.components.v1.html(report_html, height=1000, scrolling=True)
else:
    st.warning("⚠️ Le rapport Evidently n'a pas encore été généré.")
