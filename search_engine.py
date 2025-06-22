from sklearn.metrics.pairwise import cosine_similarity
from model_service import ModelService
from preprocessor import Preprocessor
from document_service import DocumentService

class SearchEngine:
    def __init__(self):
        """Initializes the Search Engine."""
        self.preprocessor = Preprocessor()
        self.model_service = ModelService()
        self.model_loaded = self.model_service.load_model()
        if self.model_loaded:
            self.vectorizer = self.model_service.get_vectorizer()
            self.tfidf_matrix = self.model_service.get_tfidf_matrix()
        
        # Fetch all documents to map indices to doc_ids
        self.doc_service = DocumentService()
        self.documents = self.doc_service.get_all_documents()
        self.doc_id_map = {i: doc.doc_id for i, doc in enumerate(self.documents)}

        # Verify that the documents are not empty
        if not self.documents:
            raise RuntimeError("No documents found in the database. Aborting search engine initialization.")

    def search(self, query, top_n=10):
        """
        Performs a search for a given query and returns the top N results.
        """
        if not self.model_loaded:
            print("Model is not loaded. Cannot perform search.")
            return []

        # Preprocess the query
        processed_query = ' '.join(self.preprocessor.process(query))

        # Transform the query into a TF-IDF vector
        query_vector = self.vectorizer.transform([processed_query])

        # Calculate cosine similarity
        cosine_similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()

        # Get the top N document indices
        # We use argpartition for efficiency, as it's faster than sorting the whole array
        top_doc_indices = cosine_similarities.argsort()[-top_n:][::-1]

        # get the documents with non-zero cosine similarity
        # non_zero_indices = [i for i in top_doc_indices if cosine_similarities[i] > 0]

        # Get the results
        results = []
        for i in top_doc_indices:
            doc_id = self.doc_id_map[i]
            score = cosine_similarities[i]
            # Fetch the full document details from our initial list
            doc_details = next((doc for doc in self.documents if doc.doc_id == doc_id), None)
            if doc_details:
                results.append({'doc_id': doc_id, 'score': score, 'text': doc_details.text})
        
        return results

if __name__ == '__main__':
    search_engine = SearchEngine()
    if search_engine.model_loaded:
        print("Search engine initialized.")
        # Example search
        search_query = "who is Juan que reía"
        search_results = search_engine.search(search_query)

        print(f"\nSearch results for: '{search_query}'")
        for result in search_results:
            print(f"  Score: {result['score']:.4f}, Doc ID: {result['doc_id']}")
            print(f"    Text: {result['text'][:150]}...") # Print snippet
