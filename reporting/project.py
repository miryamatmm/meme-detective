import pandas as pd
import os

# --- CORRECTION : Imports basés sur ta capture d'écran ---
from evidently import Report
from evidently.presets import DataDriftPreset, ClassificationPreset # Comme sur l'image
from evidently.ui.workspace import Workspace

def create_dashboard():
    print("🚀 Démarrage du reporting...")

    # 1. Chargement des données
    ref_path = "/data/ref_data.csv"
    prod_path = "/data/prod_data.csv"

    # Sécurité si les fichiers n'existent pas encore
    if not os.path.exists(ref_path):
        print(f"❌ {ref_path} introuvable.")
        return

    reference_data = pd.read_csv(ref_path)
    
    # Gestion du fichier de prod vide ou inexistant
    if os.path.exists(prod_path) and os.path.getsize(prod_path) > 0:
        current_data = pd.read_csv(prod_path)
    else:
        print("⚠️ Prod vide : utilisation des données de référence pour test.")
        current_data = reference_data.iloc[:10]

    # 2. Workspace (Obligatoire pour 'evidently ui')
    ws = Workspace("workspace")
    project = ws.create_project("Meme Detective Project")
    project.save()

    # 3. Création du Rapport
    # Utilisation des Presets importés depuis 'evidently.presets'
    report = Report(metrics=[
        DataDriftPreset(),
        ClassificationPreset()
    ])

    report.run(reference_data=reference_data, current_data=current_data)

    # 4. Ajout au Workspace
    ws.add_report(project.id, report)
    print("✅ Rapport généré avec succès.")

if __name__ == "__main__":
    create_dashboard()