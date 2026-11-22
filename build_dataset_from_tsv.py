import pandas as pd
import requests
import os
from PIL import Image
from io import BytesIO

TSV_FILE = "data/memes_data.tsv"
OUTPUT_DIR = "data/raw_images"
MIN_IMAGES = 20   
MAX_IMAGES = 50  

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/91.0.4472.124 Safari/537.36"
}


try:
    df = pd.read_csv(TSV_FILE, sep='\t', on_bad_lines='skip')
except:
    df = pd.read_csv(TSV_FILE, sep='\t', error_bad_lines=False)

counts = df['MemeLabel'].value_counts()
valid_classes = counts[counts >= MIN_IMAGES].index.tolist()
print(f"{len(valid_classes)} classes trouvées (Top: {valid_classes[:3]})...")


for label in valid_classes:
    safe_label = "".join([c if c.isalnum() else "_" for c in str(label)]).strip()
    class_dir = os.path.join(OUTPUT_DIR, safe_label)
    os.makedirs(class_dir, exist_ok=True)
    
    urls = df[df['MemeLabel'] == label]['ImageURL'].tolist()
    
    count = 0
    for url in urls:
        if count >= MAX_IMAGES:
            break
        
        url = str(url).strip()
        if url.startswith("//"):
            url = "https:" + url  

        try:
            response = requests.get(url, headers=HEADERS, timeout=3)
            if response.status_code == 200:
                try:
                    img = Image.open(BytesIO(response.content)).convert('RGB')
                    img.save(os.path.join(class_dir, f"{count}.jpg"))
                    count += 1
                except:
                    pass # Fichier corrompu
        except:
            pass # Lien mort

    print(f"📁 {label} : {count} images téléchargées.")
print("="*50)
print("\n Téléchargement Terminé")