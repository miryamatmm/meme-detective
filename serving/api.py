import pickle
import os
from io import BytesIO

import clip
import numpy as np
import pandas as pd
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image
from sklearn.preprocessing import LabelEncoder
import torch
from xgboost import XGBClassifier

app = FastAPI(title="Meme Classifier API (CLIP + PCA + Auto-Train)")

current_file_dir = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.normpath(os.path.join(current_file_dir, "..", "artifacts"))

if os.path.exists("/data"):
    DATA_DIR = "/data"
else:
    DATA_DIR = os.path.normpath(os.path.join(current_file_dir, "..", "data"))

PROD_DATA_FILE = os.path.join(DATA_DIR, "prod_data.csv")
REF_DATA_FILE = os.path.join(DATA_DIR, "ref_data.csv")

MODEL = None
SCALER = None
PCA = None
ENCODER = None
CLIP_MODEL = None
CLIP_PREPROCESS = None
DEVICE = "cpu"

@app.on_event("startup")
def load_artifacts():
    global MODEL, SCALER, PCA, ENCODER, CLIP_MODEL, CLIP_PREPROCESS
    print("Chargement du pipeline complet...")
    try:
        with open(os.path.join(ARTIFACTS_DIR, "model.pickle"), "rb") as f:
            MODEL = pickle.load(f)
        with open(os.path.join(ARTIFACTS_DIR, "scaler.pickle"), "rb") as f:
            SCALER = pickle.load(f)
        with open(os.path.join(ARTIFACTS_DIR, "pca.pickle"), "rb") as f:
            PCA = pickle.load(f)
        with open(os.path.join(ARTIFACTS_DIR, "label_encoder.pickle"), "rb") as f:
            ENCODER = pickle.load(f)

        CLIP_MODEL, CLIP_PREPROCESS = clip.load("ViT-B/32", device=DEVICE)

        print("API prete : Modele, CLIP et Encoder charges.")
    except Exception as e:
        print(f"ERREUR CRITIQUE au chargement : {e}")

def get_image_vector(image_bytes):
    if not CLIP_MODEL:
        raise ValueError("Le modèle CLIP n'est pas chargé.")

    image = Image.open(BytesIO(image_bytes)).convert('RGB')
    image_input = CLIP_PREPROCESS(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        features_raw = CLIP_MODEL.encode_image(image_input)
        features_raw = features_raw.cpu().numpy()

    features_scaled = SCALER.transform(features_raw)
    features_pca = PCA.transform(features_scaled)

    return features_pca

def retrain_model():
    global MODEL, ENCODER
    print("[Auto-Train] Demarrage du reentrainement...")
    
    try:
        if not os.path.exists(REF_DATA_FILE):
            print("Pas de ref_data.csv, impossible de reentrainer.")
            return

        df_ref = pd.read_csv(REF_DATA_FILE)
        if len(df_ref) > 2000:
            SAMPLES_PER_CLASS = 30
            df_ref = df_ref.groupby('target', group_keys=False).apply(
                lambda x: x.sample(min(len(x), SAMPLES_PER_CLASS))
            )
            df_ref = df_ref.sample(frac=1).reset_index(drop=True)

        if os.path.exists(PROD_DATA_FILE) and os.path.getsize(PROD_DATA_FILE) > 0:
            df_prod = pd.read_csv(PROD_DATA_FILE)
            common_cols = [c for c in df_ref.columns if c in df_prod.columns]
            df_prod = df_prod[common_cols]
            df_final = pd.concat([df_ref, df_prod], ignore_index=True)
        else:
            print("Pas de donnees de prod valides, annulation.")
            return

        print(f"Entrainement sur {len(df_final)} images...")

        X = df_final.iloc[:, :-1].values 
        y_strings = df_final.iloc[:, -1].values

        new_encoder = LabelEncoder()
        y_encoded = new_encoder.fit_transform(y_strings)

        old_params = MODEL.get_params()
        if "use_label_encoder" in old_params:
            del old_params["use_label_encoder"]
        old_params["n_estimators"] = 50
        old_params["max_depth"] = 6
        old_params["tree_method"] = "hist"
        old_params["n_jobs"] = -1

        new_model = XGBClassifier(**old_params)
        new_model.fit(X, y_encoded)

        MODEL = new_model
        ENCODER = new_encoder
        print("Reentrainement termine")
        with open(os.path.join(ARTIFACTS_DIR, "model.pickle"), "wb") as f:
            pickle.dump(MODEL, f)

        with open(os.path.join(ARTIFACTS_DIR, "label_encoder.pickle"), "wb") as f:
            pickle.dump(ENCODER, f)

        print(f"SUCCES : Modele et Encoder mis a jour ({len(df_final)} entrees).")

    except Exception as e:
        print(f"CRASH Reentrainement : {e}")

@app.get("/")
def health_check():
    return {"status": "ok", "engine": "CLIP + XGBoost"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not MODEL:
        raise HTTPException(500, "Modèle non chargé")
    try:
        image_bytes = await file.read()
        features = get_image_vector(image_bytes)

        pred_idx = MODEL.predict(features)[0]
        probas = MODEL.predict_proba(features)[0]
        confidence = float(np.max(probas) * 100)

        try:
            label = ENCODER.inverse_transform([pred_idx])[0]
        except ValueError:
            label = f"Unknown_Class_{pred_idx}"
            print(f"Warning: Classe {pred_idx} inconnue de l'encodeur.")

        return {
            "prediction": label,
            "confidence_score": round(confidence, 2),
            "filename": file.filename
        }
    except Exception as e:
        print(f"Erreur Predict: {e}")
        raise HTTPException(500, str(e))

@app.post("/feedback")
async def feedback(
    background_tasks: BackgroundTasks, 
    file: UploadFile = File(...), 
    correct_label: str = Form(...)
):
    try:
        image_bytes = await file.read()
        features = get_image_vector(image_bytes)

        label_clean = correct_label.strip().replace(" ", "_")
        if not label_clean or label_clean == "undefined":
            return {"status": "error", "message": "Label invalide reçu"}

        target_dir = os.path.join(DATA_DIR, label_clean)
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        file_location = os.path.join(target_dir, file.filename)
        with open(file_location, "wb") as f:
            f.write(image_bytes)

        vector_list = features[0].tolist()
        cols = [f"PCA_{i}" for i in range(len(vector_list))] + ['target']
        df_new = pd.DataFrame([vector_list + [label_clean]], columns=cols)

        header = not os.path.exists(PROD_DATA_FILE)
        df_new.to_csv(PROD_DATA_FILE, mode='a', header=header, index=False)

        K_TRIGGER = 5
        try:
            if os.path.exists(PROD_DATA_FILE):
                with open(PROD_DATA_FILE, "r") as f:
                    row_count = sum(1 for line in f) - 1

                if row_count > 0 and row_count % K_TRIGGER == 0:
                    print(f"Trigger atteint ({row_count} feedbacks). Lancement Background Task...")
                    background_tasks.add_task(retrain_model)
        except Exception as e:
            print(f"Erreur lecture trigger : {e}")

        return {
            "status": "success", 
            "message": f"Feedback enregistré pour {label_clean}", 
            "filename": file.filename
        }
    except Exception as e:
        print(f"Erreur Feedback: {e}")
        raise HTTPException(500, str(e))

@app.get("/force-retrain")
def force_retrain_endpoint(background_tasks: BackgroundTasks):
    background_tasks.add_task(retrain_model)
    return {"message": "Réentraînement forcé lancé en arrière-plan."}