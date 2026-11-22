import pickle
import torch
import clip 
from PIL import Image
import numpy as np
import os


IMAGE_PATH = "mind_different.webp"  
ARTIFACTS_DIR = "artifacts"
DEVICE = "cpu"

print("Chargement des artifacts")

try:
    with open(os.path.join(ARTIFACTS_DIR, "model.pickle"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(ARTIFACTS_DIR, "scaler.pickle"), "rb") as f:
        scaler = pickle.load(f)
    with open(os.path.join(ARTIFACTS_DIR, "pca.pickle"), "rb") as f:
        pca = pickle.load(f)
    with open(os.path.join(ARTIFACTS_DIR, "label_encoder.pickle"), "rb") as f:
        encoder = pickle.load(f)

    clip_model, preprocess = clip.load("ViT-B/32", device=DEVICE)
    
    print("Tout est chargé (CLIP + XGBoost + PCA).")

except FileNotFoundError as e:
    print(f"Fichier manquant : {e}")
    exit()
except Exception as e:
    print(f"Erreur de chargement : {e}")
    exit()

def predict_meme(image_path):
    if not os.path.exists(image_path):
        print(f"L'image {image_path} n'existe pas.")
        return None, 0

    try:
        img = Image.open(image_path).convert('RGB')
        img_input = preprocess(img).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            features_raw = clip_model.encode_image(img_input)
            features_raw = features_raw.cpu().numpy()
        
        features_scaled = scaler.transform(features_raw) 
        features_pca = pca.transform(features_scaled)    
        
        pred_index = model.predict(features_pca)[0]
        probas = model.predict_proba(features_pca)[0]
        confidence = np.max(probas) * 100
        
        pred_label = encoder.inverse_transform([pred_index])[0]
        
        return pred_label, confidence

    except Exception as e:
        print(f"Erreur pendant la prédiction : {e}")
        return None, 0

print(f"\nAnalyse de : {IMAGE_PATH}")
result, conf = predict_meme(IMAGE_PATH)

print("-" * 30)
if result:
    print(f"RÉSULTAT : {result}")
    print(f"Confiance : {conf:.2f}%")
else:
    print("Échec de la prédiction.")
print("-" * 30)