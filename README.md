# AI Customer Insights Engine

**AI Customer Insights Engine** est une application d'intelligence artificielle permettant d'interroger en langage naturel une base de plusieurs dizaines de milliers d'avis clients.

Le projet repose sur une architecture **Retrieval-Augmented Generation (RAG)** : les avis les plus pertinents sont d'abord recherchés dans une base vectorielle, puis transmis à un modèle de langage afin de générer une réponse fondée sur les données disponibles.

L'objectif est de permettre une **exploration simple et rapide des retours clients**, tout en limitant les réponses non fondées sur les données.

Ce projet s'inscrit dans la continuité d'un précédent projet NLP, **[Classification Bank Customer Reviews](https://github.com/florian-borius/classification_bank_customer_reviews)**, visant à évaluer la satisfaction client et à identifier les signaux d'insatisfaction potentiellement associés au *churn*.

🚀 **[Accéder à l'application](https://ai-customer-insights-engine.streamlit.app/)**

## 🎯 Objectifs

Ce projet vise à :

* exploiter un large volume d'avis clients ;
* permettre leur interrogation en langage naturel ;
* identifier les informations les plus pertinentes pour chaque question ;
* générer des réponses synthétiques à partir des avis récupérés ;
* comparer différentes stratégies de recherche en termes de qualité, de temps d'exécution et de coût.

## 📊 Données

Le système s'appuie sur un jeu de données de **48 000+ avis clients issus du secteur bancaire**, collectés sur Trustpilot auprès de quatre banques en ligne dans le cadre d'un projet personnel à vocation pédagogique. Les données sont utilisées pour expérimenter et évaluer le système RAG.

Les avis sont traités comme un **corpus commun**, sans distinction entre les banques lors de la recherche sémantique. L'origine de chaque avis est néanmoins conservée dans les métadonnées des chunks, permettant notamment d'envisager ultérieurement un filtrage ou une analyse par banque.

## 🏗️ Architecture

Le projet repose sur deux pipelines distincts : un **pipeline d'indexation**, exécuté en amont pour construire la base vectorielle, et un **pipeline de recherche et génération**, exécuté à chaque question utilisateur.

### Pipeline d'indexation

```text
Avis clients
     ↓
Prétraitement
     ↓
Découpage en chunks
     ↓
Génération des embeddings
     ↓
Base vectorielle Chroma
```

Les avis clients sont prétraités, puis découpés en *chunks* lorsque leur longueur le nécessite. Ces derniers sont ensuite transformés en représentations vectorielles (*embeddings*) et stockés dans une base de données Chroma afin de permettre leur recherche par similarité sémantique.

Le découpage en *chunks* est réalisé avec les paramètres suivants :

* **Taille maximale d'un chunk :** 700 caractères
* **Chevauchement entre chunks :** 100 caractères

### Pipeline de recherche et génération

```text
Question utilisateur
     ↓
Embedding de la question
     ↓
Recherche sémantique
     ↓
Contextes pertinents
     ↓
Modèle de langage
     ↓
Réponse générée
```

À chaque question, celle-ci est transformée en représentation vectorielle (*embedding*) afin de rechercher dans Chroma les chunks les plus similaires. Les contextes récupérés sont ensuite transmis au modèle de langage, qui génère une réponse fondée sur les informations fournies.

Deux configurations de recherche sémantique ont été évaluées (Cf. section "Évaluation et résultats").

## 🛠️ Composants techniques

| Composant         | Technologie                             |
| ----------------- | --------------------------------------- |
| Langage           | Python                                  |
| Framework RAG     | LangChain                               |
| Embeddings        | `paraphrase-multilingual-MiniLM-L12-v2` |
| Base vectorielle  | Chroma                                  |
| Recherche         | Similarity Search                       |
| Reranker          | `BAAI/bge-reranker-v2-m3` *(évalué)*    |
| Modèle de langage | GPT-4.1-mini                            |
| Évaluation RAG.   | Ragas                                   |
| Interface         | Streamlit                               |

Le projet est conçu de manière modulaire afin de séparer les différentes étapes du pipeline d'indexation et du pipeline de recherche et génération.

## 📏 Évaluation et résultats

L'objectif de l'évaluation a été de **comparer les performances de deux configurations de recherche** :

* **Retriever seul** : recherche des 5 chunks les plus similaires ;
* **Retriever + Reranker** : recherche initiale de 20 chunks, puis sélection des 5 chunks les plus pertinents par le Cross-Encoder `BAAI/bge-reranker-v2-m3`.

Les deux configurations ont été comparées selon trois dimensions :

* **Qualité des contextes et des réponses**
* **Temps d'exécution** de l'appel au pipeline
* **Coût** associé à l'utilisation du modèle de langage OpenAI

La qualité a été évaluée à l'aide de trois métriques :

* **Context Relevance** : mesure la pertinence des contextes récupérés par rapport à la question posée.
* **Faithfulness** : mesure dans quelle mesure la réponse générée est fidèle aux informations contenues dans les contextes récupérés.
* **Answer Relevancy** : mesure la pertinence de la réponse générée par rapport à la question posée.

Les métriques **Context Precision** et **Context Recall** n'ont pas été retenues, car leur utilisation nécessite des données de référence permettant de déterminer les contextes pertinents pour chaque question.

L'évaluation a reposé sur une approche **LLM-as-a-Judge**, mise en œuvre avec **Ragas**, permettant d'évaluer automatiquement les résultats selon les trois métriques retenues. Elle a été réalisée sur un jeu de **20 questions**, générées à partir du corpus d'avis clients puis vérifiées avant leur utilisation.

### Résultats

| Métrique | Retriever seul | Retriever + Reranker | Écart |
|---|---:|---:|---:|
|  |  |  |  |
| **Context Relevance** | **0,94** | **0,97** | **+0,04** |
| Temps moyen | 0,32 s | 9,25 s | +8,93 s |
|  |  |  |  |
| **Faithfulness** | **0,75** | **0,87** | **+0,12** |
| Temps moyen | 3,44 s | 10,63 s | +7,20 s |
| Coût moyen | 0,000205 $ | 0,000247 $ | +0,000042 $ |
|  |  |  |  |
| **Answer Relevancy** | **0,90** | **0,97** | **+0,07** |
| Temps moyen | 2,65 s | 19,82 s | +17,17 s |
| Coût moyen | 0,000201 $ | 0,000247 $ | +0,000046 $ |

### Interprétation

L'ajout du reranker améliore les trois métriques de qualité évaluées : **Context Relevance**, **Faithfulness** et **Answer Relevancy**. L'amélioration est particulièrement marquée pour la **Faithfulness**, qui progresse de **0,75 à 0,87**.

En contrepartie, le reranking entraîne une **augmentation importante du temps d'exécution**, tandis que l'augmentation du coût par requête reste **faible en valeur absolue**.

Le reranking permet donc d'obtenir de meilleurs résultats en termes de qualité, mais au prix d'une latence sensiblement plus élevée et d'un léger surcoût.

Pour l'application déployée, la configuration **Retriever seul** a finalement été retenue, offrant le meilleur compromis entre **qualité, temps d'exécution et coût**.

## 💬 Exemples d'utilisation

L'application permet d'interroger la base d'avis clients en langage naturel.

Quelques exemples de questions :

* *Quels types de problèmes rencontrent les clients avec le service client ?*
* *Comment les clients décrivent-ils leur expérience lors de l'ouverture d'un compte ?*
* *Quels sont les éléments de satisfaction des clients concernant les frais bancaires ?*

Pour chaque question, le système recherche les avis les plus pertinents, puis utilise ces informations pour générer une réponse contextualisée.

---

## 📁 Structure du projet

```text
AI-Customer-Insights-Engine/
│
├── config/
│   └── config.py
│
├── data/
│   ├── chroma_db/
│   └── evaluation/
│
├── notebooks/
│   ├── 01_data/
│   ├── 02_validation/
│   └── 03_evaluation/
│
├── src/
│   ├── data/
│   │   └── preprocess_dataset.py
│   │
│   ├── rag/
│   │   ├── build_vector_store.py
│   │   ├── document_processing.py
│   │   ├── rag_chain.py
│   │   ├── reranker.py
│   │   └── retriever.py
│   │
│   └── security/
│       └── request_guard.py
│
├── streamlit_app/
│   └── app.py
│
├── tests/
│   ├── test_build_vector_store.py
│   ├── test_document_processing.py
│   ├── test_preprocess_dataset.py
│   ├── test_reranker.py
│   └── test_retriever.py
│
├── .gitattributes 
├── .gitignore
├── pyproject.toml
└── README.md
```

* **`config/`** : paramètres et constantes du projet.
* **`data/`** : base vectorielle Chroma et données utilisées pour l'évaluation.
* **`notebooks/`** : collecte et exploration des données, validation des composants et évaluation du système RAG.
* **`src/data/`** : prétraitement des données.
* **`src/rag/`** : composants des pipelines d'indexation et de recherche et génération.
* **`src/security/`** : protection de l'application contre les requêtes excessives.
* **`streamlit_app/`** : interface utilisateur Streamlit.
* **`tests/`** : tests unitaires des principaux composants.

Le projet sépare les différentes étapes du pipeline afin de faciliter sa compréhension, sa maintenance et son évolution.

---

## ⚙️ Installation et lancement

### Prérequis

* Python 3.11+
* Une clé API OpenAI

### Installation

Cloner le dépôt puis installer les dépendances :

```bash
git clone <repository-url>
cd AI-Customer-Insights-Engine
pip install .
```

Configurer ensuite la clé API OpenAI dans une variable d'environnement :

```bash
export OPENAI_API_KEY="your-api-key"
```

### Lancement de l'application

L'application peut être lancée avec Streamlit :

```bash
streamlit run streamlit_app/app.py
```

L'application permet alors d'interroger la base d'avis clients directement depuis une interface web.

---

## 🔮 Axes d'amélioration

Cette version constitue une première itération fonctionnelle du système.
Plusieurs axes d'amélioration pourraient être explorés pour le faire évoluer, tant sur la qualité des résultats que sur ses performances, son évaluation et ses fonctionnalités.

### Données

* Améliorer le nettoyage et l'anonymisation des avis, notamment afin de limiter la présence d'informations personnelles.
* Évaluer la qualité des avis et filtrer les contenus trop bruités, peu informatifs ou difficilement exploitables.

### Indexation

* Expérimenter différentes stratégies de *chunking* : taille des chunks, taille du chevauchement et choix des séparateurs.
* Tester différents modèles d'embeddings pour l'indexation / le retrieval.

### Retrieval

* Expérimenter différentes stratégies de recherche et différents paramètres du retriever, notamment le nombre de contextes récupérés.
* Tester différents modèles de *Cross-Encoder* pour le reranking.
* Classifier le sentiment de la question afin d'orienter la recherche vers des avis positifs, négatifs ou neutres.
* Expérimenter la reformulation des questions utilisateur afin d'améliorer la recherche.
* Exploiter les métadonnées des chunks (note, date de publication, etc.) pour adapter ou filtrer la recherche.
* Améliorer la diversité des contextes récupérés afin de limiter la redondance des informations.
* Expérimenter une recherche hybride combinant recherche sémantique et recherche lexicale (par exemple BM25).
* Étudier l'impact d'une réduction de la fenêtre temporelle du corpus.

### Génération

* Tester différents modèles de langage pour la génération des réponses.

### Évaluation

* Étendre et diversifier le jeu de données d'évaluation, avec des questions représentatives des usages réels et variés.
* Tester différents modèles de langage utilisés comme LLM-as-a-Judge pour l'évaluation.
* Tracer systématiquement les expérimentations afin de comparer différentes configurations et leurs performances, notamment avec MLflow.
* Compléter l'évaluation LLM-as-a-Judge par une validation humaine sur un échantillon de questions afin de confronter les scores automatiques à une évaluation manuelle.

### Production

* Automatiser la mise à jour régulière de la base vectorielle afin d'intégrer de nouveaux avis clients.
* Prévoir des mécanismes de *fallback* en cas d'indisponibilité temporaire du modèle de langage.
* Implémenter un système de cache pour les questions similaires.
* Réduire la latence du pipeline en optimisant l’infrastructure d’exécution, notamment via des machines plus performantes ou des services d’inférence plus rapides.
* Mettre en place un système de monitoring permettant de suivre les questions posées, les réponses générées, les performances, les temps d'exécution, les coûts et l'empreinte carbone.

### Fonctionnalités

* Permettre de préciser, en complément de la question, une période temporelle afin de cibler les avis correspondants.
* Développer un Agent IA doté de différents outils permettant notamment d'effectuer des classements, des comparaisons ou des analyses statistiques sur le corpus d'avis.

---

## 👤 Auteur

**Florian BORIUS**

Projet réalisé dans le cadre de mon portfolio Data & AI, avec pour objectif d'explorer la conception, l'évaluation et le déploiement d'une application RAG de bout en bout.