from rich import print
import os
import joblib
import numpy as np
from gensim.models import Word2Vec
from services.retrieval.document_service_singleton import DocumentService
from services.nlp.preprocessor import Preprocessor

class Word2VecService:
    def __init__(self, model_path='word2vec.model', vectors_path='doc_vectors.joblib'):
        """Initializes the Word2Vec Service."""
        self.preprocessor = Preprocessor()
        self.model_path = model_path
        self.vectors_path = vectors_path
        self.model = None
        self.doc_vectors = None

    def _document_vector(self, doc_tokens, model):
        """Calculates the vector representation of a document."""
        vectors = [model.wv[token] for token in doc_tokens if token in model.wv]
        if not vectors:
            return np.zeros(model.vector_size)
        return np.mean(vectors, axis=0)

    def train_and_save_model(self, ds):
        """Trains the Word2Vec model and saves it to disk."""
        doc_service = DocumentService()
        print(f"Fetching documents for Word2Vec training from {ds}...")
        documents = doc_service.get_docs_store(ds)
        print(f"[green]Documents fetched successfully.[/green]")
        print(f"Found {documents.count()} documents in the database.")

        if not documents:
            print(f"[red]No documents found. Aborting training.[/red]")
            return

        print(f"Preprocessing and tokenizing {documents.count()} documents...")
        tokenized_docs = [self.preprocessor.process(doc.text) for doc in documents]

        print("Training Word2Vec model...")
        self.model = Word2Vec(sentences=tokenized_docs, vector_size=100, window=5, min_count=2, workers=4)
        print("Word2Vec model trained successfully.")

        print("Creating document vectors...")
        self.doc_vectors = np.array([self._document_vector(tokens, self.model) for tokens in tokenized_docs])

        print("Saving Word2Vec model and document vectors to disk...")
        model_dir = os.path.join('database', ds)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        self.model.save(os.path.join(model_dir, self.model_path))
        joblib.dump(self.doc_vectors, os.path.join(model_dir, self.vectors_path))
        print("[green]Word2Vec data saved successfully.[/green]")

    def load_model(self, ds):
        """Loads the Word2Vec model and document vectors from disk."""
        print(f"Loading Word2Vec model and vectors for {ds}...")
        model_dir = os.path.join('database' , 'word2vec_files' , ds)
        model_file = os.path.join(model_dir, self.model_path)
        vectors_file = os.path.join(model_dir, self.vectors_path)
        
        try:
            self.model = Word2Vec.load(model_file)
            self.doc_vectors = joblib.load(vectors_file)
            print("[green]Word2Vec data loaded successfully.[/green]")
            return True
        except Exception as e:
            print(e)
            #print(f"[red]Word2Vec model or vectors not found for {ds}. Please train the model first.[/red]")
            return False

    def get_model(self):
        """Returns the loaded Word2Vec model."""
        return self.model

    def get_doc_vectors(self):
        """Returns the loaded document vectors."""
        return self.doc_vectors

    def get_query_vector(self, tokens, model):
        return self._document_vector(tokens, model)


if __name__ == '__main__':
    w2v_service = Word2VecService()
    for dataset in DocumentService.available_datasets:
        w2v_service.train_and_save_model(dataset)
