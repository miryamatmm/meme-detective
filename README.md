# Meme Detective (MLOps Project)

**Meme Detective** est une solution complète de classification de mèmes Internet utilisant une approche MLOps moderne.

Le projet combine un modèle de Vision par Ordinateur (CLIP), une interface utilisateur interactive, un système de réentraînement automatique (Continuous Training) et un **Agent IA** orchestré par n8n pour générer des réponses contextuelles.

---

##  Fonctionnalités Clés

* **Modèle Pré-entraîné** : Le projet inclut les artefacts (`model.pickle`, `pca.pickle`, etc.) et est prêt à l'emploi.
* **Architecture Micro-services** : Déploiement conteneurisé via Docker (API, Webapp, Orchestrateur, Monitoring).
* **Continuous Training** : Le modèle se réentraîne automatiquement en arrière-plan grâce au feedback utilisateur (Human-in-the-loop).
* **Data Pipeline** : Les données d'entraînement initiales ont été constituées via un script de **webscraping** automatisé.
* **Agent IA** : Intégration de ChatGPT via n8n pour analyser et commenter les prédictions.

## Installation et Déploiement

### 1. Pré-requis
* Docker & Docker Compose installés.
* Une clé API OpenAI (pour l'Agent IA).

### 2. Configuration de la Clé API
L'Agent IA nécessite une clé OpenAI. Créez un fichier `.env` dans le dossier webapp :

```bash
# Créez le fichier .env
touch .env
```

Ouvrez ce fichier et ajoutez votre clé :
```env
OPENAI_API_KEY=sk-proj-votre-cle-api-ici...
```

### 3. Lancement (Docker)
Le modèle étant déjà entraîné et présent dans le dépôt Git, vous n'avez pas besoin de lancer de script Python. Déployez simplement les conteneurs (depuis la racine):

```bash
docker-compose -f serving/docker-compose.yml up -d --build
docker-compose -f webapp/docker-compose.yml up -d --build
```

### 4. Accès aux Services

| Service | URL | Description |
| :--- | :--- | :--- |
| **Webapp (Streamlit)** | [http://localhost:8081](http://localhost:8081) | Interface pour tester et corriger le modèle. |
| **API (FastAPI)** | [http://localhost:8080/docs](http://localhost:8080/docs) | Swagger de l'API de prédiction. |
| **Orchestrateur (n8n)** | [http://localhost:5678](http://localhost:5678) | Gestion des workflows IA. |
---

## Configuration de l'Agent IA (n8n)

Les workflows n8n ne sont pas activés par défaut. Vous devez les importer pour activer la boucle de feedback intelligente.

1.  Accédez à **n8n** sur [http://localhost:5678](http://localhost:5678).
2.  Créez un compte local (admin / password).
3.  **Importer les Workflows :**
    * Allez dans le menu (haut droite) → `Import from File`.
    * Sélectionnez les fichiers `.json` situés dans le dossier `/workflows` de ce projet.
4.  **Configurer les Credentials :**
    * Allez dans l'onglet `Credentials` de n8n.
    * Ajoutez un credential **OpenAI API** avec votre clé.
    * Assurez-vous que le nœud OpenAI dans le workflow utilise ce credential.
5.  **Activer** le workflow (bouton "Active" en haut à droite).

---

## Origine des Données & Entraînement

### Dataset (Webscraping)
Les images utilisées pour entraîner le modèle initial proviennent d'un scraping automatisé via l'API d'un site public de mèmes.
Le script utilisé est disponible ici : \`scripts/build_dataset_from_tsv.py\`.

*Note : Il n'est pas nécessaire d'exécuter ce script pour lancer le projet, car les artefacts entraînés sont fournis.*

### Réentraînement Automatique
Le système apprend de ses erreurs :
1.  L'utilisateur corrige une prédiction sur l'interface Webapp.
2.  L'image et le label correct sont envoyés à l'API.
3.  L'Agent IA (n8n) notifie l'utilisateur.
4.  Une fois **5 nouveaux feedbacks** reçus, l'API déclenche un réentraînement en arrière-plan (Hot Reload).

---

##  Structure du Projet

```text
ML_Project/
├── artifacts/        # Modèles entraînés (.pickle) inclus dans Git
├── serving/          # API d'inférence (FastAPI, CLIP, XGBoost)
├── webapp/           # Interface utilisateur (Streamlit)
├── workflows/        # Workflows n8n (JSON) pour l'Agent IA
├── reporting/        # Monitoring (Evidently AI)
├── scripts/          # Scripts utilitaires (Scraping, Training manuel)
├── docker-compose.yml
└── README.md
```
