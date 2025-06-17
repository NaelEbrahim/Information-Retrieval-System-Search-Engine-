import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from ir_project.document_service import DocumentService
from ir_project.preprocessor import Preprocessor

class ModelService:
    def __init__(self, vectorizer_path='tfidf_vectorizer.pkl', matrix_path='tfidf_matrix.pkl'):
        """Initializes the Model Service."""
        self.preprocessor = Preprocessor()
        self.vectorizer_path = vectorizer_path
        self.matrix_path = matrix_path
        self.vectorizer = None
        self.tfidf_matrix = None

    def train_and_save_model(self):
        """Trains the TF-IDF model and saves it to disk."""
        doc_service = DocumentService()

        # First, get a count of documents to verify.
        doc_service.cursor.execute("SELECT COUNT(*) as count FROM documents")
        count = doc_service.cursor.fetchone()['count']
        print(f"Found {count} documents in the database.")

        print("Fetching documents for training...")
        documents = doc_service.get_all_documents()
        doc_service.close_connection()

        if not documents:
            print("No documents found in the database. Aborting training.")
            return

        print(f"Training model on {len(documents)} documents...")
        corpus = [doc['text'] for doc in documents]
        
        # Use the preprocessor's process method as the analyzer
        self.vectorizer = TfidfVectorizer(analyzer=self.preprocessor.process)
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)

        print("Saving model to disk...")
        with open(self.vectorizer_path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
        with open(self.matrix_path, 'wb') as f:
            pickle.dump(self.tfidf_matrix, f)
        print("Model saved successfully.")

    def load_model(self):
        """Loads the TF-IDF model from disk."""
        print("Loading model from disk...")
        try:
            with open(self.vectorizer_path, 'rb') as f:
                self.vectorizer = pickle.load(f)
            with open(self.matrix_path, 'rb') as f:
                self.tfidf_matrix = pickle.load(f)
            print("Model loaded successfully.")
        except FileNotFoundError:
            print("Model files not found. Please train the model first.")
            return False
        return True

    def get_tfidf_matrix(self):
        return self.tfidf_matrix

    def get_vectorizer(self):
        return self.vectorizer

if __name__ == '__main__':
    model_service = ModelService()
    model_service.train_and_save_model()
