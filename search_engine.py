from services.tf_idf_singleton_service import TFIDFSingletonService
from services.document_service_singleton import DocumentService
from services.word2vec_singleton_service import Word2VecSingletonService
from services.hybrid_search_service import HybridSearchService
from services.vector_store_singleton_service import VectorStoreSingletonService

class SearchEngine:
    def __init__(self):
        """Initializes the Search Engine."""
        self.doc_service = DocumentService()
        self.tfidf_service = TFIDFSingletonService()
        self.w2v_service = Word2VecSingletonService()
        self.hybrid_service = HybridSearchService()



def search(
            self,
            query,
            top_n=10,
            dataset_name="trec",
            model_type="tfidf",
            alpha=0.3,
            beta=0.7,
        ):
        """
        Performs a search for a given query and returns the top N results.
        """
        results = []
        if model_type == "tfidf":
            num_results, results = self.tfidf_service.search(query, dataset_name, top_n)
        elif model_type == "word2vec":
            num_results, results = self.w2v_service.search(query, dataset_name, top_n)
        elif model_type == "hybrid":
            num_results, results = self.hybrid_service.search(query, dataset_name, top_n, alpha, beta)
        elif model_type == "search_with_vector_store":
            num_results, results = self.hybrid_service.search_faiss_index(query, dataset_name,top_n)
        else:
            raise ValueError(
                "Invalid model_type specified. Choose 'tfidf', 'word2vec', or 'hybrid'."
            )

        for result in results:
            doc_details = self.doc_service.get_document(result["doc_id"], dataset_name)
            if doc_details is None:
                result["text"] = "(Document not found)"
            elif isinstance(doc_details, dict):
                result["text"] = doc_details["text"]
            else:
                result["text"] = doc_details.text

        return num_results, results




if __name__ == "__main__":
    search_engine = SearchEngine()
    print("Search engine initialized.")
    query = "who is Juan que reía"

    print(f"\nSearching (Hybrid) for: '{query}' on antique dataset")
    num_results, results = search_engine.search(
        query, dataset_name="trec", model_type="hybrid"
    )
    for result in results:
        print(f"  Score: {result['score']:.4f}, Doc ID: {result['doc_id']}")
        if result.get("text"):
            print(f"    Text: {result.text[:150]}...")

    print("\n" + "=" * 50 + "\n")

    print("Searching (TF-IDF) for: ", query)
    search_results_tfidf = search_engine.search(
        query, dataset_name="trec", model_type="tfidf"
    )
    print(f"\nTF-IDF Search results for: '{query}'")
    for result in search_results_tfidf:
        print(f"  Score: {result['score']:.4f}, Doc ID: {result['doc_id']}")
        if result.get("text"):
            print(f"    Text: {result.text[:150]}...")
    print("\n" + "=" * 50 + "\n")

    print("Searching (Word2Vec) for: ", query)
    search_results_w2v = search_engine.search(
        query, dataset_name="trec", model_type="word2vec"
    )
    print(f"\nWord2Vec Search results for: '{query}'")
    for result in search_results_w2v:
        print(f"  Score: {result['score']:.4f}, Doc ID: {result['doc_id']}")
        if result.get("text"):
            print(f"    Text: {result.text[:150]}...")

    print("\n" + "=" * 50 + "\n")
