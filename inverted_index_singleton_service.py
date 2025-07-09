from rich import print
import os
import joblib
from collections import defaultdict
from tqdm import tqdm
from services.document_service_singleton import DocumentService


class InvertedIndexSingletonService:
    _instance = None
    available_datasets = DocumentService.available_datasets
    inverted_indices = {}
    doc_id_to_indices = {}
    index_to_doc_ids = {}
    folder_paths = {}

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(InvertedIndexSingletonService, cls).__new__(cls)
        return cls._instance

    def __init__(self, subfolder="index_files"):
        """Initializes the Inverted Index Service."""
        if not hasattr(self, "initialized"):
            for dataset_name in self.available_datasets:
                self.subfolder = f"index_files/{dataset_name}"
                self.folder_paths[dataset_name] = os.path.join(
                    "database", self.subfolder
                )
                os.makedirs(self.folder_paths[dataset_name], exist_ok=True)
                # Load Inverted Index
                self.inverted_indices[dataset_name] = self.load_inverted_index(
                    dataset_name
                )
                if self.inverted_indices[dataset_name] is not None:
                    print(
                        f"[green]Inverted index loaded successfully for {dataset_name}[/green]",
                    )
                else:
                    print(
                        f"[red]Inverted index not loaded successfully for {dataset_name}[/red]",
                    )
                self.doc_id_to_indices[dataset_name] = self.load_doc_id_to_index(
                    dataset_name
                )
                if self.doc_id_to_indices[dataset_name] is not None:
                    print(
                        f"[green]Doc ID to index mapping loaded successfully for {dataset_name}[/green]",
                    )
                else:
                    print(
                        f"[red]Doc ID to index mapping not loaded successfully for {dataset_name}[/red]",
                    )
                self.index_to_doc_ids[dataset_name] = (
                    {v: k for k, v in self.doc_id_to_indices[dataset_name].items()}
                    if self.doc_id_to_indices[dataset_name]
                    else {}
                )

            self.initialized = True

    def build_inverted_index(self, preprocessed_docs, doc_ids, dataset_name):
        """Builds and saves the inverted index for a given dataset."""
        print(f"\nBuilding Inverted Index for {dataset_name}...")
        inverted_index = defaultdict(set)

        for index_position, doc in tqdm(
            enumerate(preprocessed_docs), total=len(doc_ids)
        ):
            for token in set(doc):
                inverted_index[token].add(index_position)

        inverted_index = {
            term: list(doc_list) for term, doc_list in inverted_index.items()
        }
        print("\nInverted Index built successfully!")

        joblib.dump(
            inverted_index,
            os.path.join(self.folder_paths[dataset_name], "inverted_index.joblib"),
        )
        print(
            f"\n[green]Inverted Index for {dataset_name} saved successfully in {self.folder_paths[dataset_name]}/inverted_index.joblib![/green]"
        )
        return inverted_index

    def build_doc_id_to_index(self, doc_ids, dataset_name):
        """Builds and saves the doc_id to index mapping for a given dataset."""
        print(f"\nBuilding doc_id to index mapping for {dataset_name}...")
        doc_id_to_index = {doc_id: idx for idx, doc_id in enumerate(doc_ids)}

        joblib.dump(
            doc_id_to_index,
            os.path.join(self.folder_paths[dataset_name], "doc_id_to_index.joblib"),
        )
        print(
            f"\n[green]Doc ID to index mapping for {dataset_name} saved successfully in {self.folder_paths[dataset_name]}/doc_id_to_index.joblib![/green]"
        )
        return doc_id_to_index

    def load_inverted_index(self, dataset_name):
        """Loads the inverted index from disk for a given dataset."""
        try:
            print(
                f"Loading inverted index from {self.folder_paths[dataset_name]}/inverted_index.joblib"
            )
            return joblib.load(
                os.path.join(self.folder_paths[dataset_name], "inverted_index.joblib")
            )
        except FileNotFoundError:
            print(
                f"Inverted index not found at {self.folder_paths[dataset_name]}/{dataset_name}"
            )
            return None

    def load_doc_id_to_index(self, dataset_name):
        """Loads the doc_id to index mapping from disk for a given dataset."""
        try:
            print(
                f"Loading doc_id to index mapping from {self.folder_paths[dataset_name]}/doc_id_to_index.joblib"
            )
            return joblib.load(
                os.path.join(self.folder_paths[dataset_name], "doc_id_to_index.joblib")
            )
        except FileNotFoundError:
            print(
                f"Doc ID to index mapping not found at {self.folder_paths[dataset_name]}/{dataset_name}"
            )
            return None

    def get_top_docs(self, query_tokens, dataset_name, top_k=10):
        """Retrieves the top documents for a given query from a specific dataset."""
        if dataset_name not in self.available_datasets:
            print(f"Dataset '{dataset_name}' not found!")
            return []

        inverted_index = self.inverted_indices.get(dataset_name)
        index_to_doc_id = self.index_to_doc_ids.get(dataset_name)

        if not inverted_index or not index_to_doc_id:
            print(
                f"Inverted index or index to doc IDs not found for dataset '{dataset_name}'!"
            )
            return []

        doc_scores = defaultdict(int)
        for token in query_tokens:
            if token in inverted_index:
                for doc_index in inverted_index[token]:
                    doc_scores[doc_index] += 1

        sorted_docs = sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)

        top_doc_ids = [
            index_to_doc_id[doc_index] for doc_index, score in sorted_docs[:top_k]
        ]
        return top_doc_ids


if __name__ == "__main__":
    # Example usage
    service1 = InvertedIndexSingletonService()
    service2 = InvertedIndexSingletonService()

    print(f"service1 is service2: {service1 is service2}")

    # Example query for 'antique/train'
    query_antique = "example query antique"
    query_tokens_antique = query_antique.split()
    top_docs_antique = service1.get_top_docs(query_tokens_antique, "antique/train")
    print(
        f"Top documents for query '{query_antique}' in 'antique/train': {top_docs_antique}"
    )

    # Example query for 'trec-tot/2023/train'
    query_trec = "example query trec"
    query_tokens_trec = query_trec.split()
    top_docs_trec = service1.get_top_docs(query_tokens_trec, "trec-tot/2023/train")
    print(
        f"Top documents for query '{query_trec}' in 'trec-tot/2023/train': {top_docs_trec}"
    )
