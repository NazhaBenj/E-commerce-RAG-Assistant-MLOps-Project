# 🛒 E-commerce RAG Assistant – MLOps Project

## 1. Présentation du projet

**E-commerce RAG Assistant** est un assistant conversationnel intelligent dédié à la documentation de **Jumia Maroc**.

Le projet repose sur une architecture Retrieval-Augmented Generation (RAG) permettant de répondre aux questions des utilisateurs à partir d'un corpus documentaire constitué de documents relatifs aux services de Jumia Maroc.

L'objectif est de développer une solution RAG fiable, évaluable et reproductible, tout en intégrant des pratiques MLOps pour le versionnement des données, le suivi des expériences, les tests automatisés et l'intégration continue.

## 2. Objectifs

- Collecter et préparer des documents provenant de différentes sources documentaires.
- Construire un corpus documentaire structuré et enrichi de métadonnées.
- Implémenter un système de recherche hybride combinant recherche vectorielle et BM25.
- Générer des réponses contextualisées à l'aide d'un LLM.
- Fournir des références documentaires pour vérifier les réponses.
- Comparer différentes configurations de chunking.
- Évaluer la qualité du retrieval et de la génération.
- Assurer le suivi des expériences et la reproductibilité du pipeline.

## 3. Architecture du projet

Le système suit les principales étapes suivantes :

1. **Ingestion documentaire** : chargement et nettoyage des documents.
2. **Préparation du corpus** : extraction du texte et structuration des métadonnées.
3. **Chunking** : segmentation des documents en passages exploitables.
4. **Embeddings** : transformation des passages en représentations vectorielles.
5. **Indexation** : stockage des vecteurs dans ChromaDB.
6. **Retrieval hybride** : combinaison de la recherche vectorielle et de BM25 avec Reciprocal Rank Fusion (RRF).
7. **Génération** : production d'une réponse à partir du contexte récupéré avec Llama 3.2.
8. **Évaluation et monitoring** : mesure des performances et suivi des expériences.

## 4. Technologies utilisées

| Domaine | Technologies |
|---|---|
| Langage | Python 3.11 |
| Ingestion | PyPDF, BeautifulSoup, lxml |
| Traitement des données | Pandas, NumPy |
| Embeddings | Sentence Transformers |
| Modèle d'embeddings | all-MiniLM-L6-v2 |
| Base vectorielle | ChromaDB |
| Recherche hybride | BM25, RRF |
| LLM | Llama 3.2 |
| Exécution du LLM | Ollama |
| Interface | Streamlit |
| Évaluation | Scikit-learn |
| Tracking | MLflow |
| Versionnement des données | DVC |
| Tests | Pytest |
| Qualité et sécurité du code | Ruff, Bandit |
| CI/CD | GitHub Actions |

## 5. Structure du dépôt

```text
E-commerce-RAG-Assistant-MLOps-Project/
│
├── app/
│   └── streamlit_app.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── evaluation/
│
├── docs/
│
├── indexes/
│   ├── chroma_ecommerce/
│   └── experiments/
│
├── prompts/
│
├── reports/
│   └── evaluation/
│
├── scripts/
│
├── src/
│   ├── ingestion/
│   ├── rag/
│   ├── evaluation/
│   ├── experiments/
│   ├── monitoring/
│   └── utils/
│
├── tests/
│
├── .github/
│   └── workflows/
│
├── dvc.yaml
├── dvc.lock
├── params.yaml
├── requirements.txt
└── README.md
```

## 6. Installation et configuration

### Prérequis

- Python 3.11
- Git
- Ollama
- GitHub pour accéder au dépôt

### Cloner le dépôt

```bash
git clone https://github.com/NazhaBenj/E-commerce-RAG-Assistant-MLOps-Project.git

cd E-commerce-RAG-Assistant-MLOps-Project
```

### Créer un environnement virtuel

Sous Windows :

```powershell
python -m venv .venv

.venv\Scripts\activate
```

### Installer les dépendances

```bash
python -m pip install --upgrade pip

pip install -r requirements.txt
```

### Préparer Ollama

Installer Ollama, puis télécharger le modèle :

```bash
ollama pull llama3.2:latest
```

Vérifier que le service Ollama est accessible avant de lancer l'application.

## 7. Configuration du projet

Les paramètres principaux sont centralisés dans `params.yaml`.

Configuration de référence :

```yaml
chunking:
  chunk_size: 300
  chunk_overlap: 50

embeddings:
  provider: sentence-transformers
  model_name: sentence-transformers/all-MiniLM-L6-v2

vector_store:
  provider: chromadb
  persist_directory: indexes/chroma_ecommerce
  collection_name: jumia_documents

retrieval:
  top_k: 3

llm:
  provider: ollama
  model_name: llama3.2:latest
  temperature: 0.1
  max_tokens: 512
```

La configuration de chunking retenue pour le système est **300 tokens avec un overlap de 50**.

## 8. Exécution de l'application

Après avoir préparé les données, construit l'index vectoriel et vérifié la disponibilité du modèle Ollama, lancer :

```bash
streamlit run app/streamlit_app.py
```

L'interface permet à l'utilisateur de :
<img width="1545" height="884" alt="image" src="https://github.com/user-attachments/assets/581e009c-e887-4dac-9a4f-e75c4d713d40" />


- Poser des questions sur les services de Jumia Maroc.
- Consulter les réponses générées.
- Accéder aux sources documentaires utilisées.
- Démarrer une nouvelle conversation.

## 9. Expérimentations

Deux configurations de chunking ont été comparées :

- Configuration A : chunk size = 300, overlap = 50.
- Configuration B : chunk size = 500, overlap = 100.

Les expériences sont suivies avec MLflow.
<img width="1711" height="397" alt="image" src="https://github.com/user-attachments/assets/011e5201-67ed-4159-8492-bb1b2f7c004d" />


### Résultats de l'évaluation Retrieval

| Métrique | 300/50 | 500/100 |
|---|---:|---:|
| Nombre de chunks | 339 | 213 |
| Recall@5 | 1.000 | 1.000 |
| Precision@5 | 0.267 | 0.200 |
| MRR | 0.778 | 0.667 |

La configuration 300/50 obtient de meilleurs résultats en Precision@5 et MRR, avec un Recall@5 identique.

Elle est donc retenue comme configuration de référence pour le projet.

Les résultats doivent être interprétés en tenant compte de la taille limitée du jeu de questions d'évaluation.

## 10. Évaluation et Quality Gates

Le projet intègre deux niveaux d'évaluation :

**Retrieval**
- Recall@K
- Precision@K
- Mean Reciprocal Rank (MRR)

**Generation**
- Exactitude de classification du statut de réponse
- Faithfulness proxy
- Answer relevance proxy
- Citation rate

Les Quality Gates vérifient notamment les seuils définis dans `params.yaml` :

- Recall@5 minimum : 1.00
- Precision@5 minimum : 0.25
- MRR minimum : 0.75
- Nombre minimal de questions évaluées : 3
- Présence d'un cas hors périmètre.

La configuration 300/50 a obtenu la décision `ACCEPTED`.

## 11. Reproductibilité avec DVC

DVC permet de versionner les sorties du pipeline et de suivre les dépendances entre les différentes étapes.

Les étapes déclarées dans `dvc.yaml` comprennent :

- `prepare_corpus`
- `chunk`
- `build_index`
- `build_index_300_50`
- `build_index_500_100`
- `evaluate_retrieval`

Pour vérifier l'état du pipeline :

```bash
dvc status
```

Pour exécuter le pipeline :

```bash
dvc repro
```

Le fichier `dvc.lock` conserve les références des paramètres, dépendances et sorties des étapes exécutées.

## 12. Suivi des expériences avec MLflow

MLflow est utilisé pour suivre les expériences et conserver les paramètres, métriques et artefacts associés.

Deux expériences principales sont disponibles :

- `Ecommerce-RAG-Chunking-Experiments`
- `Ecommerce-RAG-Generation-Evaluation`

Elles permettent de comparer les configurations de chunking et d'analyser les résultats de génération.

## 13. Tests et CI/CD

Les tests automatisés sont développés avec Pytest.

Pour les exécuter :

```bash
pytest -v
```

Le workflow GitHub Actions comprend trois étapes :

1. **Quality** : vérification syntaxique, tests automatisés et Quality Gates.
2. **Staging** : validation de l'artefact candidat.
3. **Production** : promotion et publication de l'artefact validé.

Le workflow est déclenché manuellement depuis GitHub Actions. La publication de l'artefact validé ne correspond pas à un déploiement sur un serveur de production.

## 14. Limites et perspectives

Les principales limites du projet concernent notamment :

- La taille limitée du jeu de questions utilisé pour l'évaluation.
- La dépendance à la qualité et à l'actualité des documents sources.
- Les limites du modèle de langage et le risque de réponses incorrectes.
- La nécessité d'améliorer progressivement les métriques de génération.

Des améliorations futures pourraient porter sur l'élargissement du corpus, l'enrichissement du jeu d'évaluation et l'amélioration de la robustesse des réponses.

