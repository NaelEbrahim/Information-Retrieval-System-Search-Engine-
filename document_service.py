import scripts.load_datasets as load_datasets

class DocumentService:
    def __init__(self, dataset_name='trec-tot/2023/train'):
        """Initializes the Document Service with a specific dataset."""
        try:
            self.datasets = load_datasets.load_datasets()
            self.docs_store_trec = self.datasets['trec-tot/2023/train'].docs_store()
            self.docs_store_antique = self.datasets['antique/train'].docs_store()
            for dataset in self.datasets:
                print('loaded dataset: ', dataset)
            
        except Exception as e:
            raise RuntimeError(f"Failed to load dataset '{dataset_name}': {e}")

    def get_document(self, doc_id, dataset_name = 'trec-tot/2023/train'):
        """Retrieves a document by its ID."""
        if dataset_name == 'trec-tot/2023/train':
            return self.docs_store_trec.get(doc_id)
        elif dataset_name == 'antique/train':
            return self.docs_store_antique.get(doc_id)

    def get_all_documents(self, dataset_name = 'trec-tot/2023/train'):
        """Retrieves all documents from the dataset."""
        if dataset_name == 'trec-tot/2023/train':
            return self.docs_store_trec
        elif dataset_name == 'antique/train':
            return self.docs_store_antique

if __name__ == '__main__':
    # Example usage
    doc_service = DocumentService()

    # print first 10 documents
    print("\nFirst 10 documents:")
    for i, doc in enumerate(doc_service.get_all_documents('trec-tot/2023/train')):
        if i >= 10:
            break
        print(f"Document {i}: {doc.doc_id}: {doc.text}")

    # Retrieve the document
    # doc_id = '1'
    # print(f"Retrieving document {doc_id} from dataset trec-tot/2023/train...")
    # doc = doc_service.get_document(doc_id, 'trec-tot/2023/train')
    print(f"Retrieved: {doc}")

    print("\nDocument service test complete.")
