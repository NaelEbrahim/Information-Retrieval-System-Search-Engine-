from rich import print
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from inverted_index_singleton_service import InvertedIndexSingletonService
from preprocessor import Preprocessor
from services.document_service_singleton import DocumentService
from services.tf_idf_singleton_service import TFIDFSingletonService
from services.word2vec_singleton_service import Word2VecSingletonService
from services.vector_store_singleton_service import VectorStoreSingletonService

class HybridSearchService:
    def __init__(self):
        """Initializes the Hybrid Search Service."""
        self.preprocessor = Preprocessor()
        self.doc_service = DocumentService()
        self.index_service = InvertedIndexSingletonService()
        self.tfidf_service = TFIDFSingletonService()
        self.w2v_service = Word2VecSingletonService()
        self.vector_store_service = VectorStoreSingletonService()

    def search(self, query, dataset_name, top_n=10, alpha=0.3, beta=0.7):
        """Performs a hybrid search for a given query on a specific dataset."""
        print(f"\nSearching for: '{query}' in '{dataset_name}'")

        tfidf_service = self.tfidf_service.get_tfidf_service(dataset_name)
        w2v_service = self.w2v_service.get_word2vec_service(dataset_name)

        vectorizer = tfidf_service.get_vectorizer()
        tfidf_matrix = tfidf_service.get_tfidf_matrix()
        w2v_model = w2v_service.get_model()
        doc_vectors = w2v_service.get_doc_vectors()

        if (
            not all(
                [
                    vectorizer,
                    tfidf_matrix is not None,
                    w2v_model,
                    doc_vectors is not None,
                ]
            )
            or dataset_name not in self.index_service.inverted_indices
        ):
            print(
                f"Models not loaded for dataset '{dataset_name}', cannot perform search."
            )
            return []

        processed_tokens = self.preprocessor.process(query)
        processed_tokens = expand_query_terms_smart(processed_tokens, w2v_model)
        print(f"Processed query tokens: {processed_tokens}")

        candidate_indices = set()
        for token in processed_tokens:
            if token in self.index_service.inverted_indices[dataset_name]:
                candidate_indices.update(self.index_service.inverted_indices[dataset_name][token])
        
        candidate_indices = list(candidate_indices)

        if not candidate_indices:
            print("🔍 Using FAISS index as fallback (no candidates from inverted index)")
            query_vector_w2v = w2v_service.get_query_vector(processed_tokens, w2v_model)
            if np.all(query_vector_w2v == 0):
                print(" Skipping query: no valid Word2Vec representation.")
                return 0, []
            faiss_results = self.vector_store_service.search(query_vector_w2v, dataset_name, top_n)
            return len(faiss_results), [
                {"doc_id": doc_id, "score": score} for doc_id, score in faiss_results
            ]

        if not candidate_indices:
                return 0,[]

        try:
            # TF-IDF scores
            processed_query_str = " ".join(processed_tokens)
            query_vector_tfidf = vectorizer.transform([processed_query_str])
            candidate_tfidf_matrix = tfidf_matrix[candidate_indices]
            tfidf_scores = cosine_similarity(
                query_vector_tfidf, candidate_tfidf_matrix
            ).flatten()

            # Word2Vec scores
            query_vector_w2v = w2v_service._document_vector(
                processed_tokens, w2v_model
            ).reshape(1, -1)
            if np.all(query_vector_w2v == 0):
                print("Skipping query: no valid Word2Vec representation.")
                return 0, []
            candidate_doc_vectors = doc_vectors[candidate_indices]
            w2v_scores = cosine_similarity(
                query_vector_w2v, candidate_doc_vectors
            ).flatten()

            # Scale scores to [0, 1] range
            scaler = MinMaxScaler()
            norm_tfidf_scores = scaler.fit_transform(tfidf_scores[:, np.newaxis]).ravel()
            norm_w2v_scores = scaler.fit_transform(w2v_scores[:, np.newaxis]).ravel()

            # Combine scores
            combined_scores = alpha * norm_tfidf_scores + beta * norm_w2v_scores

            # Get top results from candidates
            top_candidate_indices = np.argsort(combined_scores)[::-1]

            results = []
            index_to_doc_id = self.index_service.index_to_doc_ids[dataset_name]
            for i in top_candidate_indices:
                original_index = candidate_indices[i]
                if combined_scores[i] > 0:
                    doc_id = index_to_doc_id.get(original_index)
                    if doc_id:
                        score = combined_scores[i]
                        results.append({"doc_id": doc_id, "score": score})

            if (top_n == -1):
                return len(results), sorted(results, key=lambda x: x["score"], reverse=True)
            return len(results), sorted(results, key=lambda x: x["score"], reverse=True)[:top_n]
        except Exception as e:
            print(f"Error calculating hybrid search: {str(e)}")
            return 0, []


    def search_faiss_index(self, query, dataset_name, top_n=10):
        """Performs FAISS-based semantic search using Word2Vec only."""
        print(f"\n Searching with FAISS for: '{query}' in '{dataset_name}'")

        w2v_service = self.w2v_service.get_word2vec_service(dataset_name)
        faiss_service = self.vector_store_service
        index_service = self.index_service

        w2v_model = w2v_service.get_model()
        if not w2v_model:
            print(f"Word2Vec model not available for '{dataset_name}'")
            return 0, []

        processed_tokens = self.preprocessor.process(query)
        # processed_tokens = expand_query_terms_smart(processed_tokens, w2v_model)
        # print(f"Processed query tokens: {processed_tokens}")

        query_vector = w2v_service.get_query_vector(processed_tokens, w2v_model)
        if np.all(query_vector == 0):
            print(" Skipping query: no valid Word2Vec representation.")
            return 0, []

        faiss_results = faiss_service.search(query_vector, dataset_name, top_n if top_n != -1 else 10000)

        index_to_doc_id = index_service.index_to_doc_ids.get(dataset_name, {})
        results = []
        for faiss_idx_str, score in faiss_results:
            faiss_idx = int(faiss_idx_str)
            doc_id = index_to_doc_id.get(faiss_idx)
            if doc_id:
                results.append({"doc_id": doc_id, "score": score})

        results = sorted(results, key=lambda x: x["score"], reverse=True)
        if top_n == -1:
            return len(results), results
        return len(results), results[:top_n]



if __name__ == "__main__":
    hybrid_search = HybridSearchService()

    # Test with the 'antique' dataset
    print("Testing with 'antique' dataset...")
    print("Hybrid search service initialized for 'antique/train'.")
    query_antique = "who is Saddam Hussein?"

    results_antique = hybrid_search.search(query_antique, "antique/train")
    for result in results_antique:
        print(f"  Score: {result['score']:.4f}, Doc ID: {result['doc_id']}")

    # Test with the 'trec-tot/2023/train' dataset
    print("\nTesting with 'trec-tot/2023/train' dataset...")
    print("Hybrid search service initialized for 'trec-tot/2023/train'.")
    query_trec = "who is juan que reía"

    results_trec = hybrid_search.search(query_trec, "trec-tot/2023/train")
    for result in results_trec:
        print(f"  Score: {result['score']:.4f}, Doc ID: {result['doc_id']}")


def is_valid_term(term):
    """
    Filter out invalid query expansion terms.
    Returns True only if term is alphabetic and not a stopword.
    """
    # Basic filtering
    if len(term) < 3 or not term.isalpha():
        return False

    # Optional: remove stopwords
    from nltk.corpus import stopwords
    STOPWORDS = set(stopwords.words('english'))

    if term.lower() in STOPWORDS:
        return False

    return True


def expand_query_terms_smart(query_terms, w2v_model, max_expansions=1, similarity_threshold=0.7):
    valid_query_terms = [term for term in query_terms if term in w2v_model.wv]

    if not valid_query_terms:
        return query_terms

    try:
        query_vector = np.mean([w2v_model.wv[term] for term in valid_query_terms], axis=0)
    except Exception as e:
        print(f"Error computing query vector: {e}")
        return query_terms

    expanded_terms = list(set(query_terms))  # Avoid duplicates

    for term in valid_query_terms:
        similar_words = w2v_model.wv.most_similar(term, topn=10)  # Get more candidates

        added = 0
        for word, score in similar_words:
            if added >= max_expansions:
                break
            if word in expanded_terms:
                continue
            if not is_valid_term(word):
                continue

            # Compute similarity with query vector
            try:
                sim = cosine_similarity([query_vector], [w2v_model.wv[word]])[0][0]
                if sim >= similarity_threshold:
                    expanded_terms.append(word)
                    added += 1
            except:
                continue

    return expanded_terms

