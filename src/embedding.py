from sentence_transformers import SentenceTransformer
from langchain_text_splitters  import RecursiveCharacterTextSplitter
import numpy as np
import chromadb
import uuid

class EmbeddingManager:
    """
    Creates embeddings for given documents using sentence_transformer with a specified embedding model.
    """
    def __init__(self, embedding_model_name="all-miniLM-L6-v2"):
        self.embedding_model_name = embedding_model_name
        self.chunk_size = 1000
        self.model = None
        self._load_model()

    def chunk_documents(self, documents, chunk_size=1000, chunk_overlap=200):

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )

        chunks = splitter.split_documents(documents)
        print(f"Total document loaders: {len(documents)} and Number of chunks after splitting: {len(chunks)}")

        return chunks

    def _load_model(self):
        """Loads the SentenceTransformer model"""
        try:
            print(f"Loading embedding model: {self.embedding_model_name}")
            self.model = SentenceTransformer(self.embedding_model_name)
            print("Model loaded successfully. Embedding dimensions:", self.model.get_sentence_embedding_dimension())
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def generate_embeddings(self, texts):

        print(f"Generating embeddings for {len(texts)} documents...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        print(f"Embeddings generated successfully with shape {embeddings.shape}")
        
        return embeddings

    def generate_embeddings_with_chunks(self, chunks):
        texts = [chunk.page_content for chunk in chunks]
        return self.generate_embeddings(texts)

