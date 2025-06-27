from ir_datasets.indices import Docstore
import scripts.load_datasets as load_datasets


class DocumentService:
    available_datasets = ('antique/train', 'trec-tot/2023/train')

    def __init__(self, dataset_name='trec-tot/2023/train'):
        """Initializes the Document Service with a specific dataset."""
        try:
            self.datasets = load_datasets.load_datasets()
            for dataset in self.datasets:
                print('loaded dataset: ', dataset)
            
        except Exception as e:
            raise RuntimeError(f"Failed to load dataset '{dataset_name}': {e}")

    def get_document(self, doc_id, dataset_name = 'trec-tot/2023/train'):
        """Retrieves a document by its ID."""
        if dataset_name not in self.available_datasets:
            raise ValueError(f"Invalid dataset name: {dataset_name}")
        try:
            return self.datasets[dataset_name].docs_store().get(doc_id)
        except KeyError:
            print(f"Document {doc_id} not found in dataset {dataset_name}")
            return None

    # returns Docstore object
    def get_docs_store(self, dataset_name = 'trec-tot/2023/train') -> Docstore:
        """Retrieves all documents from the dataset."""
        if dataset_name not in self.available_datasets:
            raise ValueError(f"Invalid dataset name: {dataset_name}")
        return self.datasets[dataset_name].docs_store()

if __name__ == '__main__':
    # Example usage
    doc_service = DocumentService()

    # print first 10 documents
    print("\nFirst 10 documents:")
    for i, doc in enumerate(doc_service.get_docs_store('trec-tot/2023/train')):
        if i >= 10:
            break
        print(f"Document {i}: {doc.doc_id}: {doc.text}")

    # Retrieve the document
    # doc_id = '1'
    # print(f"Retrieving document {doc_id} from dataset trec-tot/2023/train...")
    # doc = doc_service.get_document(doc_id, 'trec-tot/2023/train')
    print(f"Retrieved: {doc}")

    print("\nDocument service test complete.")
