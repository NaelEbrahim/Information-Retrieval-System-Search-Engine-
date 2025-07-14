from rich import print
from ir_datasets.indices import Docstore
import scripts.load_datasets as load_datasets


class DocumentService:
    _instance = None
    available_datasets = ('antique', 'trec')  # ('antique/train', 'trec-tot/2023/train')

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(DocumentService, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        """Initializes the Document Service with a specific dataset."""
        if not hasattr(self, 'initialized'):
            try:
                self.datasets = load_datasets.load_datasets()
                for dataset in self.datasets:
                    print(f"loaded dataset: '{dataset}'")
                self.initialized = True
            except Exception as e:
                raise RuntimeError(f"Failed to load datasets: {e}")

    # def get_document(self, doc_id, dataset_name='trec'):
    #     """Retrieves a document by its ID."""
    #     if dataset_name not in self.available_datasets:
    #         raise ValueError(f"Invalid dataset name: {dataset_name}")
    #     try:
    #         dataset_name = normalize_dataset_name(dataset_name)
    #         return self.datasets[dataset_name].docs_store().get(doc_id)
    #     except KeyError:
    #         print(f"Document {doc_id} not found in dataset '{dataset_name}'")
    #         return None

    def get_document(self, doc_id, dataset_name='trec'):
        """Retrieves a document by its ID."""
        if dataset_name not in self.available_datasets:
            raise ValueError(f"Invalid dataset name: {dataset_name}")

        dataset = self.datasets[normalize_dataset_name(dataset_name)]
        if isinstance(dataset, list):
            for doc in dataset:
                if doc['doc_id'] == doc_id:
                    return doc
            print(f"Document {doc_id} not found in dataset '{dataset_name}'")
            return None
        else:
            try:
                return dataset.docs_store().get(doc_id)
            except KeyError:
                print(f"Document {doc_id} not found in dataset '{dataset_name}'")
                return None

    # returns Docstore object
    def get_docs_store(self, dataset_name='trec') -> Docstore:
        """Retrieves all documents from the dataset."""
        if dataset_name not in self.available_datasets:
            raise ValueError(f"Invalid dataset name: {dataset_name}")
        return self.datasets[normalize_dataset_name(dataset_name)].docs_store()


if __name__ == '__main__':
    # Example usage
    doc_service1 = DocumentService()
    doc_service2 = DocumentService()

    print(f"doc_service1 is doc_service2: {doc_service1 is doc_service2}")

    # print first 10 documents
    print("\nFirst 10 documents:")
    for i, doc in enumerate(doc_service1.get_docs_store('trec')):
        if i >= 10:
            break
        print(f"Document {i}: {doc.doc_id}: {doc.text}")

    # Retrieve the document
    # doc_id = '1'
    # print(f"Retrieving document {doc_id} from dataset trec-tot/2023/train...")
    # doc = doc_service1.get_document(doc_id, 'trec-tot/2023/train')
    # print(f"Retrieved: {doc}")

    print("\nDocument service test complete.")


def normalize_dataset_name(dataset_name):
    if dataset_name == "trec":
        return "trec-tot/2023/train"
    elif dataset_name == "antique":
        return "antique/train"
    else:
        return dataset_name
