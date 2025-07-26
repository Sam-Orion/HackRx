# This script handles the document ingestion process, including loading, splitting, embedding, and storing the documents.

import PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
from typing import List

class DocumentIngestionAgent:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.client = chromadb.PersistentClient(path="/tmp/chroma_db")
        self.collection = self.client.get_or_create_collection(name="insurance_policies")

    def _extract_text_from_pdf(self, file_path: str) -> str:
        doc = PyMuPDF.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text

    def ingest(self, file_paths: List[str]):
        # Clear existing collection to avoid duplicates when re-uploading
        self.client.delete_collection(name="insurance_policies")
        self.collection = self.client.get_or_create_collection(name="insurance_policies")

        for file_path in file_paths:
            print(f"Ingesting {file_path}...")
            text = self._extract_text_from_pdf(file_path)
            chunks = self.text_splitter.split_text(text)
            embeddings = self.embedding_model.encode(chunks).tolist()
            # Generate unique IDs for each chunk
            ids = [f"{os.path.basename(file_path)}_{i}" for i in range(len(chunks))]
            if chunks: # Ensure there are chunks to add
                self.collection.add(
                    embeddings=embeddings,
                    documents=chunks,
                    ids=ids
                )
        print("Ingestion complete.")