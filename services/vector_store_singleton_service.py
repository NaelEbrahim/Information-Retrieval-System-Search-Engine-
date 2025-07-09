import os
import faiss
import joblib
import numpy as np
from services.word2vec_singleton_service import Word2VecSingletonService
from services.document_service_singleton import DocumentService


class VectorStoreSingletonService:
    _instance = None
    vector_stores = {}
    doc_ids_map = {}
    index_paths = {}
    doc_ids_paths = {}
    available_datasets = DocumentService.available_datasets

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(VectorStoreSingletonService, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, "initialized"):
            self.initialized = True
            for dataset in self.available_datasets:
                self._init_dataset(dataset)

    def _init_dataset(self, dataset_name):
        index_path = f'database/index_files/{dataset_name}/faiss.index'
        doc_ids_path = f'database/index_files/{dataset_name}/doc_ids.joblib'

        self.index_paths[dataset_name] = index_path
        self.doc_ids_paths[dataset_name] = doc_ids_path

        if os.path.exists(index_path) and os.path.exists(doc_ids_path):
            index = faiss.read_index(index_path)
            doc_ids = joblib.load(doc_ids_path)
            self.vector_stores[dataset_name] = index
            self.doc_ids_map[dataset_name] = doc_ids
            print(f"[green]FAISS index loaded successfully for {dataset_name}[/green]")
        else:
            print(f"[yellow]FAISS index not found for {dataset_name}, building it...[/yellow]")
            self._build_faiss_index(dataset_name)

    def _build_faiss_index(self, dataset_name):
        w2v_service = Word2VecSingletonService().get_word2vec_service(dataset_name)
        matrix = w2v_service.get_doc_vectors()

        doc_ids = [str(i) for i in range(matrix.shape[0])]
        d = matrix.shape[1]

        index = faiss.IndexFlatL2(d)
        index.add(matrix)

        # Save both
        os.makedirs(os.path.dirname(self.index_paths[dataset_name]), exist_ok=True)
        faiss.write_index(index, self.index_paths[dataset_name])
        joblib.dump(doc_ids, self.doc_ids_paths[dataset_name])

        self.vector_stores[dataset_name] = index
        self.doc_ids_map[dataset_name] = doc_ids
        print(f"[blue]FAISS index built and saved for {dataset_name}[/blue]")

    def search(self, query_vector, dataset_name, top_n=10):
        if dataset_name not in self.vector_stores:
            self._init_dataset(dataset_name)

        index = self.vector_stores[dataset_name]
        doc_ids = self.doc_ids_map[dataset_name]

        D, I = index.search(np.array([query_vector]), top_n)
        return [(doc_ids[idx], float(D[0][i])) for i, idx in enumerate(I[0])]
