from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from model_service import ModelService
from preprocessor import Preprocessor
from document_service import DocumentService

class SearchEngine:
    def __init__(self):
        """Initializes the Search Engine."""
        self.preprocessor = Preprocessor()
        self.model_service = ModelService()
        self.model_loaded_antique = self.model_service.load_model('antique/train')
        if self.model_loaded_antique:
            self.vectorizer_antique = self.model_service.get_vectorizer()
            self.tfidf_matrix_antique = self.model_service.get_tfidf_matrix()
            print("Antique model loaded successfully.")

        self.model_loaded_trec = self.model_service.load_model('trec-tot/2023/train')
        if self.model_loaded_trec:
            self.vectorizer_trec = self.model_service.get_vectorizer()
            self.tfidf_matrix_trec = self.model_service.get_tfidf_matrix()
            print("TREC model loaded successfully.")
        
        # Fetch all documents to map indices to doc_ids
        self.doc_service = DocumentService()
        self.documents_trec = self.doc_service.get_docs_store('trec-tot/2023/train')
        self.doc_id_map_trec = {i: doc.doc_id for i, doc in enumerate(self.documents_trec)}

        self.documents_antique = self.doc_service.get_docs_store('antique/train')
        self.doc_id_map_antique = {i: doc.doc_id for i, doc in enumerate(self.documents_antique)}

        # Verify that the documents are not empty
        if not self.documents_trec:
            raise RuntimeError("No documents found in the database. Aborting search engine initialization.")

    def search(self, query, top_n=10, dataset_name='trec-tot/2023/train'):
        """
        Performs a search for a given query and returns the top N results.
        """
        if not self.model_loaded_trec and not self.model_loaded_antique:
            print("Model is not loaded. Cannot perform search.")
            return []

        # Preprocess the query
        print("Preprocessing query...")
        processed_query = ' '.join(self.preprocessor.process(query))
        print("Preprocessed query:", processed_query)

        # Transform the query into a TF-IDF vector
        if dataset_name == 'trec-tot/2023/train':
            query_vector = self.vectorizer_trec.transform([processed_query])
            print("Query vector:", query_vector)
        elif dataset_name == 'antique/train':
            query_vector = self.vectorizer_antique.transform([processed_query])
            print("Query vector:", query_vector)

        # Calculate cosine similarity
        if dataset_name == 'trec-tot/2023/train':
            cosine_similarities = cosine_similarity(query_vector, self.tfidf_matrix_trec).flatten()
        elif dataset_name == 'antique/train':
            cosine_similarities = cosine_similarity(query_vector, self.tfidf_matrix_antique).flatten()
        print("Cosine similarities:", cosine_similarities)

        # Get the top N document indices
        # We use argpartition for efficiency, as it's faster than sorting the whole array
        top_doc_indices = np.argpartition(cosine_similarities, -top_n)[-top_n:][::-1]
        
        print("Top document indices:", top_doc_indices)

        # get the documents with non-zero cosine similarity
        non_zero_indices = [i for i in top_doc_indices if cosine_similarities[i] > 0]

        # Get the results
        results = []
        for i in non_zero_indices:
            doc_id = self.doc_id_map_trec[i] if dataset_name == 'trec-tot/2023/train' else self.doc_id_map_antique[i]
            score = cosine_similarities[i]
            # Fetch the full document details from our initial list
            doc_details = self.doc_service.get_document(doc_id, dataset_name)
            if doc_details:
                results.append({'doc_id': doc_id, 'score': score, 'text': doc_details.text})
        
        return results

if __name__ == '__main__':
    search_engine = SearchEngine()
    if search_engine.model_loaded_trec or search_engine.model_loaded_antique:
        print("Search engine initialized.")
        # Example search

        for query in ['who is Juan que reía', 'who is Saddam Hussein', 'who is Barack Obama', 'who is Donald Trump', 'who is Bill Clinton', 'who is Bill Gates', 'who is Bill Clinton', 'who is Bill Clinton', 'who is Bill Clinton', 'who is Bill Clinton']:
            print("Searching for: ", query)
            search_results = search_engine.search(query, dataset_name='antique/train')
            # search_results = search_engine.search(search_query, dataset_name='trec-tot/2023/train')

            print(f"\nSearch results for: '{query}'")
            for result in search_results:
                print(f"  Score: {result['score']:.4f}, Doc ID: {result['doc_id']}")
                print(f"    Text: {result['text'][:150]}...") # Print snippet

            print("\n")
            print("--------------------------------")
