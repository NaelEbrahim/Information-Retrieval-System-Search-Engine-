import mysql.connector
import json
from ir_project.config import DB_CONFIG

class DocumentService:
    def __init__(self):
        """Initializes the Document Service and connects to the database."""
        self.db_connection = mysql.connector.connect(**DB_CONFIG)
        self.cursor = self.db_connection.cursor(dictionary=True)

    def add_document(self, doc_id, text, metadata=None):
        """Adds a document to the database."""
        if self.document_exists(doc_id):
            # print(f"Document {doc_id} already exists.")
            return

        sql = "INSERT INTO documents (doc_id, text, metadata) VALUES (%s, %s, %s)"
        # Convert metadata dict to a JSON string
        metadata_json = json.dumps(metadata) if metadata else None
        self.cursor.execute(sql, (doc_id, text, metadata_json))
        self.db_connection.commit()

    def get_document(self, doc_id):
        """Retrieves a document by its ID."""
        self.cursor.execute("SELECT * FROM documents WHERE doc_id = %s", (doc_id,))
        return self.cursor.fetchone()

    def document_exists(self, doc_id):
        """Checks if a document exists in the database."""
        self.cursor.execute("SELECT id FROM documents WHERE doc_id = %s", (doc_id,))
        return self.cursor.fetchone() is not None

    def get_all_documents(self):
        """Retrieves all documents from the database."""
        self.cursor.execute("SELECT * FROM documents")
        return self.cursor.fetchall()

    def close_connection(self):
        """Closes the database connection."""
        self.db_connection.close()

if __name__ == '__main__':
    # Example usage
    doc_service = DocumentService()

    # Add a sample document
    print("Adding a sample document...")
    sample_doc_id = 'sample_123'
    sample_text = 'This is the text for the sample document.'
    sample_metadata = {'source': 'test'}
    doc_service.add_document(sample_doc_id, sample_text, sample_metadata)

    # Retrieve the document
    print(f"Retrieving document {sample_doc_id}...")
    doc = doc_service.get_document(sample_doc_id)
    print(f"Retrieved: {doc}")

    # Check for existence
    print(f"Does document {sample_doc_id} exist? {doc_service.document_exists(sample_doc_id)}")

    doc_service.close_connection()
    print("\nDocument service test complete.")
