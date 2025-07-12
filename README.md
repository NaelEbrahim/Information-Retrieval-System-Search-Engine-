
# Information Retrieval System

## 📖 Overview

This project is a sophisticated Information Retrieval (IR) system designed to serve as a practical and educational tool for students and instructors in software engineering and computer science. It provides a hands-on implementation of a search engine, complete with a web-based user interface, multiple retrieval models, and evaluation components. The system is built using Python and Flask, with a focus on modularity and extensibility.

## ✨ Features

*   **Web-Based UI:** A simple and intuitive web interface built with Flask for searching and viewing documents.
*   **Multiple Retrieval Models:**
    *   **TF-IDF:** A classical vector space model for information retrieval.
    *   **Word2Vec:** A neural network-based model for capturing semantic relationships between words.
    *   **Hybrid Model:** A combination of TF-IDF and Word2Vec scores for improved ranking.
    *   **FAISS-based Search:** A highly efficient similarity search for dense vectors, integrated with the Word2Vec model.
*   **Query Suggestion:** Autocompletes user queries based on the dataset's vocabulary.
*   **Query Expansion:** "Smart" query expansion using Word2Vec to improve search results.
*   **Evaluation Services:** Built-in support for evaluating retrieval performance using standard IR datasets like TREC and ANTIQUE.
*   **Modular Architecture:** The project is structured into services for different functionalities, making it easy to understand, maintain, and extend.

## 🏛️ System Architecture

The system is composed of several key components:

*   **`app.py`:** The main Flask application that handles web requests, renders templates, and orchestrates the search process.
*   **`search_engine.py`:** The core of the search functionality, which delegates search requests to the appropriate retrieval model.
*   **Retrieval Models:**
    *   **`tf_idf_singleton_service.py`:** Implements the TF-IDF model, including vectorization and scoring.
    *   **`word2vec_singleton_service.py`:** Implements the Word2Vec model, including document vectorization and scoring.
    *   **`hybrid_search_service.py`:** Combines the results of the TF-IDF and Word2Vec models.
    *   **`vector_store_singleton_service.py`:** Manages the FAISS index for efficient vector similarity search.
*   **`inverted_index_singleton_service.py`:** Manages the inverted index, a core data structure for efficient retrieval.
*   **`document_service_singleton.py`:** Handles loading and accessing document content.
*   **`preprocessor.py`:** Responsible for text preprocessing tasks such as tokenization, stemming, and stopword removal.
*   **Evaluation Services:**
    *   **`TREC_Evaluation_service.py`:** Provides tools for evaluating the system on TREC datasets.
    *   **`ANTIQUE_Evaluation_service.py`:** Provides tools for evaluating the system on the ANTIQUE dataset.
    *   **`Metrics_service.py`:** Calculates standard IR metrics such as Precision, Recall, and Mean Average Precision (MAP).

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
│   │   └── inverted_index_singleton_service.py
│   ├── modeling
│   │   ├── tfidf_service.py
│   │   └── word2vec_service.py
│   ├── nlp
│   │   ├── preprocessor.py
│   │   └── spell_corrector.py
│   ├── retrieval
│   │   ├── document_service_singleton.py
│   │   ├── hybrid_search_service.py
│   │   ├── tf_idf_singleton_service.py
│   │   ├── vector_store_singleton_service.py
│   │   └── word2vec_singleton_service.py
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

## 🚀 Getting Started

### Prerequisites

*   Python 3.8+
*   Pip for package management

### Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Build Models and Indices:**
    Before running the application, you must build the necessary models. Please follow the instructions in the **"Building Required Models and Indices"** section below.

4.  **Run the Application:**
    Once the setup is complete, you can run the Flask application:
    ```bash
    python app.py
    ```
    The application will be available at `http://127.0.0.1:5000`.

## 🛠️ Building Required Models and Indices

This is a mandatory one-time setup process. Before running the application for the first time, you must build the data models. This involves training the TF-IDF and Word2Vec models and then creating the inverted index.

**Run the following commands from the project's root directory in the exact order shown:**

1.  **Train TF-IDF Models:**
    ```bash
    python -m services.modeling.tfidf_service
    ```

2.  **Train Word2Vec Models:**
    ```bash
    python -m services.modeling.word2vec_service
    ```

3.  **Build the Inverted Index:**
    ```bash
    python -m scripts.build_index
    ```

> **Note:** For a detailed explanation of the model building and loading architecture, please see the `model_building_documentation.txt` file in this repository.

## Usage

### Searching

1.  Open your web browser and navigate to `http://127.0.0.1:5000`.
2.  Enter your search query in the search box.
3.  Select the dataset and retrieval model you want to use.
4.  Click the "Search" button to view the results.

### Evaluation

The evaluation services can be used to measure the performance of the retrieval models. You can run the evaluation scripts from the command line:

```bash
python -m services.evaluation.antique_evaluation_service
python -m services.evaluation.trec_evaluation_service
```

## 🛠️ Technologies Used

*   **Python:** The core programming language.
*   **Flask:** A lightweight web framework for the user interface.
*   **Gensim:** For Word2Vec model training and implementation.
*   **Scikit-learn:** For TF-IDF vectorization and cosine similarity calculations.
*   **NLTK:** For natural language processing tasks like tokenization and stopword removal.
*   **FAISS:** A library for efficient similarity search and clustering of dense vectors.
*   **NumPy:** For numerical operations.

## 퓨 Future Work

*   **Integration of more advanced retrieval models:** Such as BERT or other transformer-based models.
*   **User feedback and relevance feedback:** Allow users to provide feedback on search results to improve future rankings.
*   **Distributed indexing and search:** To support larger datasets and higher query loads.
*   **More comprehensive evaluation metrics:** And visualization of evaluation results.
