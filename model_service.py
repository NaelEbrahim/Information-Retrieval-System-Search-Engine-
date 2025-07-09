from rich import print
import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from services.document_service_singleton import DocumentService
from preprocessor import Preprocessor

class TFIDFService:
    def __init__(self, vectorizer_path='tfidf_vectorizer.joblib', matrix_path='tfidf_matrix.joblib'):
        """Initializes the Model Service."""
        self.preprocessor = Preprocessor()
        self.vectorizer_path = vectorizer_path
        self.matrix_path = matrix_path
        self.vectorizer = None
        self.tfidf_matrix = None

    def train_and_save_model(self, ds):
        """Trains the TF-IDF model and saves it to disk."""
        doc_service = DocumentService()

        # First, get a count of documents to verify.
        print(f"Fetching documents for training from '{ds}'...")
        documents = doc_service.get_docs_store(ds)
        print(f"Documents fetched successfully for dataset: '{ds}'")
        print(f"Found {documents.count()} documents in the database.")

        if not documents:
            print(f"No documents found in the database for dataset: '{ds}'. Aborting training.")
            return

        print(f"Training model on {documents.count()} documents...")
        corpus = [doc.text for doc in documents]            

        # Use the preprocessor's process method as the analyzer
        self.vectorizer = TfidfVectorizer(analyzer=self.preprocessor.process, max_features=5000)
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)

        print(f"Saving TF-IDF model to disk for dataset: '{ds}'...")
        if not os.path.exists(os.path.join('database', ds)):
            os.makedirs(os.path.join('database', ds))
        joblib.dump(self.vectorizer, os.path.join('database', f"{ds}/" + self.vectorizer_path))
        joblib.dump(self.tfidf_matrix, os.path.join('database', f"{ds}/" + self.matrix_path))
        print(f"[green]TF-IDF model saved successfully for dataset: '{ds}'[/green]")

    def load_model(self, ds):
        """Loads the TF-IDF model from disk."""
        print(f"Loading TF-IDF model from disk for dataset: '{ds}'")
        try:
            self.vectorizer = joblib.load(os.path.join('database/tfidf_files', f"{ds}/" + self.vectorizer_path))
            self.tfidf_matrix = joblib.load(os.path.join('database/tfidf_files', f"{ds}/" + self.matrix_path))
            print(f"[green]TF-IDF model loaded successfully for dataset: '{ds}'[/green]")
        except FileNotFoundError:
            print(f"[red]TF-Idf files not found for {ds}. Please train the model first.[/red]")
            return False
        return True

    def get_tfidf_matrix(self):
        return self.tfidf_matrix

    def get_vectorizer(self):
        return self.vectorizer

if __name__ == '__main__':
    model_service = TFIDFService()
    for dataset in DocumentService.available_datasets:
        model_service.train_and_save_model(dataset)
