import json
import time
from services.tf_idf_singleton_service import TFIDFSingletonService
from services.word2vec_singleton_service import Word2VecSingletonService
from services.hybrid_search_service import HybridSearchService
from services.document_service_singleton import DocumentService
from Metrics_service import RetrievalEvaluator


class AntiqueEvaluationPipeline:
    def __init__(self):
        self.DATASET = 'antique'
        self.QUERIES_PATH = r"C:\Users\NAEL PC\.ir_datasets\antique\train\queries.txt"
        self.QRELS_PATH = r"C:\Users\NAEL PC\.ir_datasets\antique\train\qrels.txt"
        self.RESULTS_OUTPUT = "evaluation_results_antique.json"
        self.doc_service = DocumentService()
        self.tfidf_service = TFIDFSingletonService()
        self.w2v_service = Word2VecSingletonService()
        self.hybrid_service = HybridSearchService()

    def run(self):
        # Load queries
        queries = []
        with open(self.QUERIES_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                parts = line.strip().split()
                query_id = parts[0]
                query_text = ' '.join(parts[1:])
                queries.append({"id": query_id, "text": query_text})

        # Load qrels
        qrels = {}
        with open(self.QRELS_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                # Format: query_id Q0 doc_id relevance
                if len(parts) >= 4:
                    query_id = parts[0]
                    doc_id = parts[2]
                    qrels.setdefault(query_id, set()).add(doc_id)

        # Search and evaluate
        predictions = {}
        successful_queries = []

        for q in queries:
            query_id = str(q["id"])
            query_text = q["text"]
            print(f"\n🔍 Searching for Query ID {query_id}...")

            model_type = 'word2vec'
            top_n = 10

            if model_type == "tfidf":
                _, results = self.tfidf_service.search(query_text, self.DATASET, top_n)
            elif model_type == "word2vec":
                _, results = self.w2v_service.search(query_text, self.DATASET, top_n)
            elif model_type == "hybrid":
                _, results = self.hybrid_service.search(query_text, self.DATASET, top_n, 0.3, 0.7)
            elif model_type == "search_with_vector_store":
                _, results = self.hybrid_service.search_faiss_index(query_text, self.DATASET,top_n)
            else:
                continue

            ranked_doc_ids = [res['doc_id'] for res in results]
            predictions[query_id] = ranked_doc_ids

            retrieved_doc_ids = set(ranked_doc_ids)
            expected_doc_ids = qrels.get(query_id, set())

            if expected_doc_ids & retrieved_doc_ids:
                successful_queries.append(query_id)

        # Evaluate
        all_y_true = [list(qrels[qid]) for qid in predictions if qid in qrels]
        all_y_pred = [predictions[qid] for qid in predictions if qid in qrels]

        evaluator = RetrievalEvaluator(len(successful_queries) , len(queries),k=10)
        metrics = evaluator.evaluate(all_y_true, all_y_pred)

        print("\n📊 Evaluation Metrics:")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")

        with open(self.RESULTS_OUTPUT, "w") as f_out:
            json.dump(metrics, f_out, indent=2)

        with open("successful_queries_antique.txt", "w") as f:
            f.write("\n".join(successful_queries))

        print(f"\n✅ Done! Metrics saved to {self.RESULTS_OUTPUT}")


if __name__ == "__main__":
    start_time = time.time()

    AntiqueEvaluationPipeline().run()

    end_time = time.time()
    elapsed = end_time - start_time
    print(f"\n⏱️ Total execution time: {elapsed:.2f} seconds")