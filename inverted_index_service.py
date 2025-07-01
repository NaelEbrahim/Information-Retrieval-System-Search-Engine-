import os
import joblib
from collections import defaultdict
from tqdm import tqdm


class InvertedIndexService:
    def __init__(
        self,
        subfolder="index_files",
        index_name="inverted_index.joblib",
        doc_id_map_name="doc_id_to_index.joblib",
    ):
        """Initializes the Inverted Index Service."""
        self.subfolder = subfolder
        self.index_name = index_name
        self.doc_id_map_name = doc_id_map_name
        self.folder_path = os.path.join("database", self.subfolder)
        os.makedirs(self.folder_path, exist_ok=True)

    def build_inverted_index(self, preprocessed_docs, doc_ids):
        """Builds and saves the inverted index."""
        print("\nBuilding Inverted Index...")
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

        joblib.dump(inverted_index, os.path.join(self.folder_path, self.index_name))
        print(f"\nInverted Index saved successfully in {self.folder_path}!")
        return inverted_index

    def build_doc_id_to_index(self, doc_ids):
        """Builds and saves the doc_id to index mapping."""
        print("\nBuilding doc_id to index mapping...")
        doc_id_to_index = {doc_id: idx for idx, doc_id in enumerate(doc_ids)}

        joblib.dump(
            doc_id_to_index, os.path.join(self.folder_path, self.doc_id_map_name)
        )
        print(f"\nMapping saved successfully in {self.folder_path}!")
        return doc_id_to_index

    def load_inverted_index(self):
        """Loads the inverted index from disk."""
        try:
            print(f"Loading inverted index from {self.folder_path}/{self.index_name}")
            return joblib.load(os.path.join(self.folder_path, self.index_name))
        except FileNotFoundError:
            print(f"Inverted index not found at {self.folder_path}/{self.index_name}")
            return None

    def load_doc_id_to_index(self):
        """Loads the doc_id to index mapping from disk."""
        try:
            print(
                f"Loading doc_id to index mapping from {self.folder_path}/{self.doc_id_map_name}"
            )
            return joblib.load(os.path.join(self.folder_path, self.doc_id_map_name))
        except FileNotFoundError:
            print(
                f"Doc ID to index mapping not found at {self.folder_path}/{self.doc_id_map_name}"
            )
            return None
