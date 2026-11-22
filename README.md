# Meme Detective (projet MLOps)

**Meme Detective** classe automatiquement des mèmes Internet parmi 81 catégories, et s'améliore tout seul grâce aux corrections de ses utilisateurs.

Le projet couvre toute la chaîne MLOps : constitution du dataset par web scraping, modèle de vision (CLIP + XGBoost), API de prédiction, interface utilisateur, réentraînement automatique, monitoring et agent IA.

> Projet réalisé dans le cadre de l'UE Machine Learning du Master Data Science de l'Université Lyon 1 (2025)

---

## Fonctionnalités

* **Classification d'images** : chaque image est transformée en vecteur par CLIP (ViT-B/32), normalisée, réduite par PCA (512 → 330 dimensions), puis classée par XGBoost.
* **Réentraînement automatique (humain dans la boucle)** : l'utilisateur corrige une prédiction dans l'interface ; après 5 corrections, l'API réentraîne le modèle en arrière-plan et le recharge sans interruption du service.
* **Architecture micro-services** : API, interface, orchestrateur et monitoring déployés dans des conteneurs Docker.
* **Monitoring** : suivi de la dérive des données de production avec Evidently AI.
* **Agent IA** : workflows n8n qui utilisent l'API OpenAI pour commenter les prédictions et notifier l'utilisateur.

## Résultats

* Dataset : 81 catégories de mèmes, enrichi par augmentation d'images (rotations, recadrages, variations de couleurs).
* Accuracy de 0,97 (F1 macro de 0,97) sur le jeu de test.

**Limite identifiée :** les augmentations ont été générées avant la séparation train/test, donc des variantes d'une même image peuvent se retrouver dans les deux jeux. Ce score est probablement optimiste ; une évaluation plus fiable séparerait les images d'origine avant de les augmenter.

---

## Installation et lancement

### 1. Prérequis
* Docker et Docker Compose
* Une clé API OpenAI (pour l'agent IA)

### 2. Clé API
Créez un fichier `.env` dans le dossier `webapp/` :

```env
OPENAI_API_KEY=sk-proj-votre-cle-api-ici
```

### 3. Lancement
Le modèle entraîné est fourni dans `artifacts/`, il n'y a rien à entraîner. Depuis la racine :

```bash
docker-compose -f serving/docker-compose.yml up -d --build
docker-compose -f webapp/docker-compose.yml up -d --build
```

### 4. Accès aux services

| Service | URL | Description |
| :--- | :--- | :--- |
| **Interface (Streamlit)** | [http://localhost:8081](http://localhost:8081) | Tester le modèle et corriger ses prédictions |
| **API (FastAPI)** | [http://localhost:8080/docs](http://localhost:8080/docs) | Documentation Swagger de l'API |
| **Orchestrateur (n8n)** | [http://localhost:5678](http://localhost:5678) | Workflows de l'agent IA |

### 5. Activer l'agent IA (n8n)
1. Ouvrez n8n sur [http://localhost:5678](http://localhost:5678) et créez un compte local.
2. Importez les fichiers `.json` du dossier `workflows/` (menu en haut à droite → *Import from File*).
3. Dans *Credentials*, ajoutez un credential **OpenAI API** avec votre clé, et sélectionnez-le dans le nœud OpenAI.
4. Activez le workflow (bouton *Active*).

---

## Données

Les images d'entraînement proviennent d'un scraping automatisé de l'API d'un site public de mèmes, avec le script `build_dataset_from_tsv.py`. Il n'est pas nécessaire de le relancer : les modèles entraînés sont fournis.

## Structure du projet

```text
meme-detective/
├── artifacts/                 # Modèle, scaler, PCA et encodeur entraînés (.pickle)
├── serving/                   # API de prédiction et de réentraînement (FastAPI, CLIP, XGBoost)
├── webapp/                    # Interface utilisateur (Streamlit)
├── reporting/                 # Monitoring de la dérive (Evidently AI)
├── workflows/                 # Workflows n8n de l'agent IA
├── data/                      # Données de production collectées
├── training.ipynb             # Entraînement initial du modèle
├── build_dataset_from_tsv.py  # Constitution du dataset par web scraping
├── predict_test.py            # Test d'une prédiction en local, sans Docker
└── README.md
```
