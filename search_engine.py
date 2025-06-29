from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from model_service import ModelService
from preprocessor import Preprocessor
from document_service import DocumentService
from word2vec_service import Word2VecService

class SearchEngine:
    def __init__(self):
        """Initializes the Search Engine."""
        self.preprocessor = Preprocessor()
        self.model_service = ModelService()
        self.w2v_service = Word2VecService()

        # Load TF-IDF models
        self.model_loaded_antique = self.model_service.load_model('antique/train')
        if self.model_loaded_antique:
            self.vectorizer_antique = self.model_service.get_vectorizer()
            self.tfidf_matrix_antique = self.model_service.get_tfidf_matrix()
            print("Antique TF-IDF model loaded successfully.")

        self.model_loaded_trec = self.model_service.load_model('trec-tot/2023/train')
        if self.model_loaded_trec:
            self.vectorizer_trec = self.model_service.get_vectorizer()
            self.tfidf_matrix_trec = self.model_service.get_tfidf_matrix()
            print("TREC TF-IDF model loaded successfully.")

        # Load Word2Vec models
        self.w2v_model_loaded_antique = self.w2v_service.load_model('antique/train')
        if self.w2v_model_loaded_antique:
            self.w2v_model_antique = self.w2v_service.get_model()
            self.doc_vectors_antique = self.w2v_service.get_doc_vectors()
            print("Antique Word2Vec model loaded successfully.")

        self.w2v_model_loaded_trec = self.w2v_service.load_model('trec-tot/2023/train')
        if self.w2v_model_loaded_trec:
            self.w2v_model_trec = self.w2v_service.get_model()
            self.doc_vectors_trec = self.w2v_service.get_doc_vectors()
            print("TREC Word2Vec model loaded successfully.")
        
        # Fetch all documents to map indices to doc_ids
        self.doc_service = DocumentService()
        self.documents_trec = self.doc_service.get_docs_store('trec-tot/2023/train')
        self.doc_id_map_trec = {i: doc.doc_id for i, doc in enumerate(self.documents_trec)}

        self.documents_antique = self.doc_service.get_docs_store('antique/train')
        self.doc_id_map_antique = {i: doc.doc_id for i, doc in enumerate(self.documents_antique)}

        # Verify that the documents are not empty
        if not self.documents_trec:
            raise RuntimeError("No documents found in the database. Aborting search engine initialization.")

    def search(self, query, top_n=10, dataset_name='trec-tot/2023/train', model_type='tfidf'):
        """
        Performs a search for a given query and returns the top N results.
        """
        if model_type == 'tfidf':
            return self.search_tfidf(query, top_n, dataset_name)
        elif model_type == 'word2vec':
            return self.search_word2vec(query, top_n, dataset_name)
        else:
            raise ValueError("Invalid model_type specified. Choose 'tfidf' or 'word2vec'.")

    def search_tfidf(self, query, top_n=10, dataset_name='trec-tot/2023/train'):
        """
        Performs a search using the TF-IDF model.
        """
        if not self.model_loaded_trec and not self.model_loaded_antique:
            print("TF-IDF model is not loaded. Cannot perform search.")
            return []

        # Preprocess the query
        print("Preprocessing query...")
        processed_query = ' '.join(self.preprocessor.process(query))
        print("Preprocessed query:", processed_query)

        # Transform the query into a TF-IDF vector
        if dataset_name == 'trec-tot/2023/train':
            query_vector = self.vectorizer_trec.transform([processed_query])
        elif dataset_name == 'antique/train':
            query_vector = self.vectorizer_antique.transform([processed_query])

        # Calculate cosine similarity
        if dataset_name == 'trec-tot/2023/train':
            cosine_similarities = cosine_similarity(query_vector, self.tfidf_matrix_trec).flatten()
        elif dataset_name == 'antique/train':
            cosine_similarities = cosine_similarity(query_vector, self.tfidf_matrix_antique).flatten()

        # Get the top N document indices
        top_doc_indices = np.argpartition(cosine_similarities, -top_n)[-top_n:][::-1]
        
        non_zero_indices = [i for i in top_doc_indices if cosine_similarities[i] > 0]

        # Get the results
        results = []
        for i in non_zero_indices:
            doc_id = self.doc_id_map_trec[i] if dataset_name == 'trec-tot/2023/train' else self.doc_id_map_antique[i]
            score = cosine_similarities[i]
            doc_details = self.doc_service.get_document(doc_id, dataset_name)
            if doc_details:
                results.append({'doc_id': doc_id, 'score': score, 'text': doc_details.text})
        
        return results

    def search_word2vec(self, query, top_n=10, dataset_name='trec-tot/2023/train'):
        """
        Performs a search using the Word2Vec model.
        """
        if not self.w2v_model_loaded_trec and not self.w2v_model_loaded_antique:
            print("Word2Vec model is not loaded. Cannot perform search.")
            return []

        # Preprocess the query
        print("Preprocessing query...")
        processed_tokens = self.preprocessor.process(query)
        print("Preprocessed query tokens:", processed_tokens)

        # Get the appropriate model and document vectors
        if dataset_name == 'trec-tot/2023/train':
            model = self.w2v_model_trec
            doc_vectors = self.doc_vectors_trec
            doc_id_map = self.doc_id_map_trec
        elif dataset_name == 'antique/train':
            model = self.w2v_model_antique
            doc_vectors = self.doc_vectors_antique
            doc_id_map = self.doc_id_map_antique
        else:
            return []

        # Calculate query vector
        query_vector = self.w2v_service._document_vector(processed_tokens, model)
        query_vector = query_vector.reshape(1, -1)

        # Calculate cosine similarity
        cosine_similarities = cosine_similarity(query_vector, doc_vectors).flatten()

        # Get the top N document indices
        top_doc_indices = np.argpartition(cosine_similarities, -top_n)[-top_n:][::-1]
        
        non_zero_indices = [i for i in top_doc_indices if cosine_similarities[i] > 0]

        # Get the results
        results = []
        for i in non_zero_indices:
            doc_id = doc_id_map[i]
            score = cosine_similarities[i]
            doc_details = self.doc_service.get_document(doc_id, dataset_name)
            if doc_details:
                results.append({'doc_id': doc_id, 'score': score, 'text': doc_details.text})
        
        return results

if __name__ == '__main__':
    search_engine = SearchEngine()
    if search_engine.model_loaded_trec or search_engine.model_loaded_antique:
        print("Search engine initialized.")
        # Example search

        for query in ['who is Juan que reía', 'who is Saddam Hussein', 'who is Barack Obama', 'who is Donald Trump', 'who is Bill Clinton', 'who is Bill Gates']:
            print("Searching (TF-IDF) for: ", query)
            search_results_tfidf = search_engine.search(query, dataset_name='antique/train', model_type='tfidf')
            print(f"\nTF-IDF Search results for: '{query}'")
            for result in search_results_tfidf:
                print(f"  Score: {result['score']:.4f}, Doc ID: {result['doc_id']}")
                print(f"    Text: {result['text'][:150]}...")
            print("\n" + "="*50 + "\n")

            print("Searching (Word2Vec) for: ", query)
            search_results_w2v = search_engine.search(query, dataset_name='antique/train', model_type='word2vec')
            print(f"\nWord2Vec Search results for: '{query}'")
            for result in search_results_w2v:
                print(f"  Score: {result['score']:.4f}, Doc ID: {result['doc_id']}")
                print(f"    Text: {result['text'][:150]}...")
            print("\n" + "="*50 + "\n")
