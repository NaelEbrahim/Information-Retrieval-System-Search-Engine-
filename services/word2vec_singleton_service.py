
from rich import print
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from word2vec_service import Word2VecService
from services.document_service_singleton import DocumentService
from preprocessor import Preprocessor
from sklearn.metrics.pairwise import cosine_similarity
from inverted_index_singleton_service import InvertedIndexSingletonService

class Word2VecSingletonService:
    _initialized = False
    _instance = None
    index_service = None
    word2vec_services = {}
    available_datasets = DocumentService.available_datasets
    preprocessor = Preprocessor()

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(Word2VecSingletonService, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        """Initializes the Word2Vec Singleton Service."""
        if self._initialized:
            return
        for dataset_name in self.available_datasets:
            self.word2vec_services[dataset_name] = Word2VecService()
            self.word2vec_services[dataset_name].load_model(dataset_name)
        
        self.index_service = InvertedIndexSingletonService()
        self._initialized = True

    def get_word2vec_service(self, dataset_name):
        if dataset_name not in self.available_datasets:
            raise ValueError(f"Invalid dataset name: {dataset_name}")
        return self.word2vec_services[dataset_name]

    def search(self, query, dataset_name, top_n=10):
        processed_tokens = self.preprocessor.process(query)
        candidate_indices = set()
        for token in processed_tokens:
            if token in self.index_service.inverted_indices[dataset_name]:
                candidate_indices.update(self.index_service.inverted_indices[dataset_name][token])
        
        candidate_indices = list(candidate_indices)
        
        w2v_service = self.get_word2vec_service(dataset_name)
        model = w2v_service.get_model()
        doc_vectors = w2v_service.get_doc_vectors()

        if model is None or doc_vectors is None:
            print(f"Word2Vec model for {dataset_name} is not loaded.")
            return 0, []

        query_vector = self._document_vector(processed_tokens, model)
        if np.count_nonzero(query_vector) == 0:
            print("Warning: Query vector is all zeros after processing")
            return 0, []
            
        query_vector = query_vector.reshape(1, -1)
        
        if len(candidate_indices) == 0:
            print("No matching documents found for the query")
            return 0, []
            
        candidate_doc_vectors = doc_vectors[candidate_indices]
        
        try:
            cosine_similarities = cosine_similarity(query_vector, candidate_doc_vectors).flatten()
            return self._get_top_results(cosine_similarities, top_n, dataset_name, candidate_indices)
        except Exception as e:
            print(f"Error calculating cosine similarity: {str(e)}")
            return 0, []

    def _document_vector(self, doc_tokens, model):
        """Calculates the vector representation of a document."""
        vectors = [model.wv[token] for token in doc_tokens if token in model.wv]
        if not vectors:
            return np.zeros(model.vector_size)
        return np.mean(vectors, axis=0)

    def _get_top_results(self, cosine_similarities, top_n, dataset_name, candidate_indices):
        top_doc_indecies = np.argsort(cosine_similarities)[::-1]
        results = []
        for i in top_doc_indecies:
            original_index = candidate_indices[i]
            if cosine_similarities[i] > 0:
                doc_id = self.index_service.index_to_doc_ids[dataset_name].get(original_index)
                if doc_id:
                    results.append({"doc_id": doc_id, "score": cosine_similarities[i]})
        if (top_n == -1):
            return len(results), sorted(results, key=lambda x: x["score"], reverse=True)
        return len(results), sorted(results, key=lambda x: x["score"], reverse=True)[:top_n]



    def get_query_vector(self, tokens, model):
        return self._document_vector(tokens, model)



if __name__ == "__main__":
    w2v_singleton_service = Word2VecSingletonService()
    query = "who is Saddam Hussein"
    dataset_name = "antique/train"
    top_n = 10

    print('Searching for query: ', query)
    print('In dataset: ', dataset_name)

    print(w2v_singleton_service.search(query, dataset_name, top_n))
