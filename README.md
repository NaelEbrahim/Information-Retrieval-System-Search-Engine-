# 🔍 Information Retrieval System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-Web%20Framework-orange)](https://flask.palletsprojects.com/)
[![Gensim](https://img.shields.io/badge/Gensim-Word2Vec-purple)](https://radimrehurek.com/gensim/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-TFIDF-green)](https://scikit-learn.org/)
[![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-red)](https://faiss.ai/)
[![NLTK](https://img.shields.io/badge/NLTK-NLP-yellow)](https://www.nltk.org/)

**Keywords:** Python, Flask, TF-IDF, Word2Vec, FAISS, Information Retrieval, Search Engine, NLP, Query Expansion, Evaluation Metrics  

A sophisticated Information Retrieval system with a web-based UI, multiple retrieval models, heuristic query expansion, and evaluation tools. Ideal for learning IR concepts and practical implementation.

---

## 📑 Table of Contents
- [Overview](#-overview)
- [Features](#-features)
- [System Architecture](#-system-architecture)
- [Folder Structure](#-folder-structure)
- [Getting Started](#-getting-started)
- [Building Models & Indices](#-building-required-models-and-indices)
- [Usage](#usage)
- [Technologies](#-technologies-used)
- [Future Work](#-future-work)
- [Author](#-author)

---

## 📖 Overview

This project is a practical and educational Information Retrieval (IR) system. It implements a search engine with multiple retrieval models, query expansion, and evaluation tools, all accessible via a web-based UI. Built with Python and Flask, it emphasizes modularity and extensibility.

---

## ✨ Features

* **Web-Based UI:** Simple and intuitive interface for searching and viewing documents.  
* **Multiple Retrieval Models:**
  * **TF-IDF:** Classical vector space model for information retrieval.  
  * **Word2Vec:** Neural network-based model capturing semantic relationships between words.  
  * **Hybrid Model:** Combines TF-IDF and Word2Vec scores for improved ranking.  
  * **FAISS-based Search:** Efficient similarity search using dense vectors from Word2Vec.  
* **Query Suggestion:** Autocomplete user queries from the dataset vocabulary.  
* **Query Expansion:** Semantic query expansion using Word2Vec.  
* **Evaluation Services:** Evaluate retrieval performance using TREC and ANTIQUE datasets.  
* **Modular Architecture:** Clear separation of services for indexing, retrieval, NLP, and evaluation.

---

## 🏛️ System Architecture

Key components:

* **`app.py`** – Main Flask app handling requests, templates, and orchestrating searches.  
* **`search_engine.py`** – Core search logic delegating to retrieval models.  
* **Retrieval Models:**
  * `tfidf_service.py` – TF-IDF model with vectorization and scoring.  
  * `word2vec_service.py` – Word2Vec model with document vectorization.  
  * `hybrid_search_service.py` – Combines TF-IDF and Word2Vec results.  
  * `vector_store_service.py` – FAISS index management for vector search.  
* **`inverted_index_service.py`** – Manages the inverted index.  
* **`document_service.py`** – Loads and accesses documents.  
* **`preprocessor.py`** – Tokenization, stemming, and stopword removal.  
* **Evaluation Services:**
  * `trec_evaluation_service.py` – Evaluate on TREC datasets.  
  * `antique_evaluation_service.py` – Evaluate on ANTIQUE dataset.  
  * `metrics_service.py` – Computes Precision, Recall, MAP.

---

## Folder Structure

```
.
├── README.md
├── app.py
├── database
│   ├── index_files
│   │   ├── antique
│   │   │   ├── doc_id_to_index.joblib
│   │   │   ├── doc_ids.joblib
│   │   │   ├── faiss.index
│   │   │   ├── inverted_index.joblib
│   │   │   └── train
│   │   └── trec
│   │       ├── doc_id_to_index.joblib
│   │       ├── doc_ids.joblib
│   │       ├── faiss.index
│   │       └── inverted_index.joblib
│   ├── tfidf_files
│   │   ├── antique
│   │   │   ├── tfidf_matrix.joblib
│   │   │   └── tfidf_vectorizer.joblib
│   │   └── trec
│   │       ├── tfidf_matrix.joblib
│   │       └── tfidf_vectorizer.joblib
│   └── word2vec_files
│       ├── antique
│       │   ├── doc_vectors.joblib
│       │   └── word2vec.model
│       └── trec
│           ├── doc_vectors.joblib
│           ├── word2vec.model
│           ├── word2vec.model.syn1neg.npy
│           └── word2vec.model.wv.vectors.npy
├── model_building_documentation.txt
├── requirements.txt
├── scripts
│   ├── __init__.py
│   ├── build_index.py
│   └── load_datasets.py
├── services
│   ├── __init__.py
│   ├── evaluation
│   │   ├── antique_evaluation_service.py
│   │   ├── metrics_service.py
│   │   └── trec_evaluation_service.py
│   ├── helpers
│   │   ├── query_expander_service.py
│   │   └── query_suggestion_service.py
│   ├── indexing
│   │   └── inverted_index_service.py
│   ├── modeling
│   │   ├── tfidf_service.py
│   │   └── word2vec_service.py
│   ├── nlp
│   │   ├── preprocessor.py
│   │   └── spell_corrector.py
│   ├── retrieval
│   │   ├── document_service.py
│   │   ├── hybrid_search_service.py
│   │   ├── tfidf_service.py
│   │   ├── vector_store_service.py
│   │   └── word2vec_service.py
│   └── search
│       └── search_engine.py
├── static
│   └── css
│       └── style.css
├── structure.md
└── templates
    ├── base.html
    ├── document.html
    ├── index.html
    ├── not_found.html
    └── results.html
```

---

## 🚀 Getting Started

### Prerequisites

* Python 3.8+  
* Pip package manager

### Installation & Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-directory>
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Build models and indices (see below).

4. Run the app:
```bash
python app.py
```
Visit `http://127.0.0.1:5000` in a browser.

---

## 🛠️ Building Required Models and Indices

**Run these commands in order from the project root:**

1. Preprocess NLTK data:
```bash
python -m services.nlp.preprocessor
```

2. Load datasets:
```bash
python -m scripts.load_datasets
```

3. Train TF-IDF models:
```bash
python -m services.modeling.tfidf_service
```

4. Train Word2Vec models:
```bash
python -m services.modeling.word2vec_service
```

5. Build inverted index:
```bash
python -m scripts.build_index
```

6. Build FAISS vector stores:
```bash
python -m services.retrieval.vector_store_service
```

> **Note:** Scripts must contain `if __name__ == "__main__":` for module execution.

---

## Usage

### Searching

1. Open `http://127.0.0.1:5000`.  
2. Enter a query.  
3. Select dataset and retrieval model.  
4. Click **Search** to view results.

### Evaluation

Run evaluation scripts:
```bash
python -m services.evaluation.antique_evaluation_service
python -m services.evaluation.trec_evaluation_service
```

---

## 🛠️ Technologies Used

* Python – core language  
* Flask – web framework  
* Gensim – Word2Vec model  
* Scikit-learn – TF-IDF and similarity calculations  
* NLTK – NLP preprocessing  
* FAISS – vector similarity search  
* NumPy – numerical computations

---

## 🚧 Future Work

* Integrate advanced retrieval models (e.g., BERT).  
* Add user feedback for relevance refinement.  
* Support distributed indexing and search for large datasets.  
* More evaluation metrics and visualization.

---

## 👨‍💻 Author

**Nael Ebrahim**  
Software Engineer
