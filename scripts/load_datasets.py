import ir_datasets
from ir_project.document_service import DocumentService

def load_datasets_to_db():
    """
    Loads the specified datasets and stores them in the database.
    """
    doc_service = DocumentService()
    datasets_to_load = [
        "trec-tot/2023/train",
        "antique/train",
    ]

    for dataset_name in datasets_to_load:
        print(f"Loading and storing {dataset_name}...")
        try:
            dataset = ir_datasets.load(dataset_name)
            count = 0
            for doc in dataset.docs_iter():
                doc_service.add_document(doc.doc_id, doc.text, {'dataset': dataset_name})
                count += 1
                if count % 1000 == 0:
                    print(f"  ...stored {count} documents")
            print(f"Finished storing {count} documents from {dataset_name}.")
            print("-"*20)
        except Exception as e:
            print(f"Could not process dataset {dataset_name}. Error: {e}")
    
    doc_service.close_connection()

if __name__ == "__main__":
    load_datasets_to_db()
