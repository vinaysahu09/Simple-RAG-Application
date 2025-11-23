from src.vector_store_chromadb import VectorStoreManager


if __name__ == "__main__":
    # Initialize Vector Store Manager
    vector_store_manager = VectorStoreManager()

    # Retrieve documents and perform RAG advanced query
    query = "What are the key concerns about AI mentioned in the documents?"
    result = vector_store_manager.rag_advanced(query)
    print('Answer:', result['answer'])
    print('Sources:', result['sources'])
    print('Confidence Score:', result['confidence_score'])