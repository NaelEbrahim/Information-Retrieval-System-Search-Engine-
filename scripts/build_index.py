import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tqdm import tqdm
from services.document_service_singleton import DocumentService
from preprocessor import Preprocessor
from inverted_index_singleton_service import InvertedIndexSingletonService


def main():
    # Initialize the preprocessor
    preprocessor = Preprocessor()

    # Initialize the document service
    doc_service = DocumentService()

    # Process Trec dataset
    print("\nProcessing Trec dataset...")
    documents_trec = doc_service.get_docs_store('trec-tot/2023/train')
    preprocessed_docs_trec = [preprocessor.process(doc.text) for doc in tqdm(documents_trec)]
    doc_ids_trec = [doc.doc_id for doc in documents_trec]
    
    trec_index_service = InvertedIndexSingletonService()
    trec_index_service.build_inverted_index(preprocessed_docs_trec, doc_ids_trec, 'trec-tot/2023/train')
    trec_index_service.build_doc_id_to_index(doc_ids_trec, 'trec-tot/2023/train')

    # Process Antique dataset
    print("\nProcessing Antique dataset...")
    documents_antique = doc_service.get_docs_store('antique/train')
    preprocessed_docs_antique = [preprocessor.process(doc.text) for doc in tqdm(documents_antique)]
    doc_ids_antique = [doc.doc_id for doc in documents_antique]

    antique_index_service = InvertedIndexSingletonService()
    antique_index_service.build_inverted_index(preprocessed_docs_antique, doc_ids_antique, 'antique/train')
    antique_index_service.build_doc_id_to_index(doc_ids_antique, 'antique/train')


if __name__ == '__main__':
    main()
