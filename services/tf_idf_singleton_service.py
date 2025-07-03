from rich import print
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from model_service import TFIDFService
from services.document_service_singleton import DocumentService
from preprocessor import Preprocessor
from sklearn.metrics.pairwise import cosine_similarity
from inverted_index_singleton_service import InvertedIndexSingletonService

class TFIDFSingletonService:
    _initialized = False
    _instance = None
    index_service = None
    tf_idf_services = {}
    available_datasets = DocumentService.available_datasets
    preprocessor = Preprocessor()

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(TFIDFSingletonService, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        """Initializes the TF-IDF Singleton Service."""
        if self._initialized:
            return
        for dataset_name in self.available_datasets:
            self.tf_idf_services[dataset_name] = TFIDFService()
            self.tf_idf_services[dataset_name].load_model(dataset_name)
        
        self.index_service = InvertedIndexSingletonService()
        self._initialized = True

    def get_tfidf_service(self, dataset_name):
        if dataset_name not in self.available_datasets:
            raise ValueError(f"Invalid dataset name: {dataset_name}")
        return self.tf_idf_services[dataset_name]

    def search(self, query, dataset_name, top_n=10):
        processed_tokens = self.preprocessor.process(query)
        candidate_indices = set()
        for token in processed_tokens:
            if token in self.index_service.inverted_indices[dataset_name]:
                candidate_indices.update(self.index_service.inverted_indices[dataset_name][token])
        
        candidate_indices = list(candidate_indices)
        
        vectorizer = self.get_tfidf_service(dataset_name).get_vectorizer()
        tfidf_matrix = self.get_tfidf_service(dataset_name).get_tfidf_matrix()

        query_vector = vectorizer.transform([" ".join(processed_tokens)])
        if query_vector.count_nonzero() == 0:
            print("Warning: Query vector is all zeros after processing")
            return []

        query_vector = query_vector.reshape(1, -1)

        if len(candidate_indices) == 0:
            print("No matching documents found for the query")
            return []
        
        candidate_doc_vectors = tfidf_matrix[candidate_indices]
        
        try:
            cosine_similarities = cosine_similarity(query_vector, candidate_doc_vectors).flatten()
            return self._get_top_results(cosine_similarities, top_n, dataset_name, candidate_indices)
        except Exception as e:
            print(f"Error calculating cosine similarity: {str(e)}")
            return 0, []

    def _get_top_results(self, cosine_similarities, top_n, dataset_name, candidate_indices):
        top_doc_indecies = np.argsort(cosine_similarities)[::-1]
        results = []
        for i in top_doc_indecies:
            original_index = candidate_indices[i]
            if cosine_similarities[i] > 0:
                doc_id = self.index_service.index_to_doc_ids[dataset_name].get(original_index)
                if doc_id:
                    results.append({"doc_id": doc_id, "score": cosine_similarities[i] })
        
        return len(results), sorted(results, key=lambda x: x["score"], reverse=True)[:top_n]

if __name__ == "__main__":
    tf_idf_service = TFIDFSingletonService()
    query = "who is Saddam Hussein"
    dataset_name = "antique/train"
    top_n = 10

    print('Searching for query: ', query)
    print('In dataset: ', dataset_name)

    print(tf_idf_service.search(query, dataset_name, top_n))
    