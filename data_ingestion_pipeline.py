from src.document_loader import process_all_pdfs_directory_loader
from src.embedding import EmbeddingManager
from src.vector_store_chromadb import VectorStoreManager

if __name__ == "__main__":
    # Load and process all PDFs from a specified directory
    docs = process_all_pdfs_directory_loader("data/pdf_files")
    print(f"Processed {len(docs)} PDF documents. Documents are: {docs}")

    # Initialize Embedding Manager and generate embeddings
    embedding_manager = EmbeddingManager()
    chunks = embedding_manager.chunk_documents(docs)
    chunked_embeddings = embedding_manager.generate_embeddings_with_chunks(chunks)
    print(f"Generated embeddings for {len(chunks)} chunks.")
    print('Embeddings:', chunked_embeddings)

    # Insert embeddings along with documents into ChromaDB vector store
    vector_store_manager = VectorStoreManager()
    vector_store_manager.add_documents_to_collection(chunks, chunked_embeddings)

