from sentence_transformers import SentenceTransformer
import numpy as np
import chromadb
import uuid
import os
from src.embedding import EmbeddingManager
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

class VectorStoreManager:
    """
    Manages the creation and storage of vector embeddings using FAISS.
    """
    def __init__(self, collection_name="pdf_documents", persist_directory="data/vector_storage"):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.db_client = None
        self._initialize_storage()
        self.embedding_manager = EmbeddingManager()


    def _initialize_storage(self):
        try:
            # Create a persistent directory for vector storage
            os.makedirs(self.persist_directory, exist_ok=True)
            # Initialize ChromaDB client
            self.db_client = chromadb.PersistentClient(path=self.persist_directory)
            # Create or get the collection
            self.collection = self.db_client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "Collection of PDF document embeddings for RAG application"}
            )
            print(f"Vector storage initialized at: {self.persist_directory}")
            print(f"Existing documents in the collection name {self.collection_name} is {self.collection.count()}")

        except Exception as e:
            print(f"Error initializing storage directory: {e}")
            raise

    def add_documents_to_collection(self, documents, embeddings):
        """Adds documents and their embeddings to the vector store collection."""
        if len(documents) != len(embeddings):
            raise ValueError("The number of documents must match the number of embeddings.")
        
        print(f"Adding {len(documents)} documents to the vector store collection...")

        # Prepare data for insertion to the vector store collection
        ids = []
        metadatas = []
        documents_text = []
        embeddings_list = []

        for index, (doc, embedding) in enumerate(zip(documents, embeddings)):

            # Generate unique ID
            doc_id = f"doc_{uuid.uuid4().hex[:8]}_{index}"
            ids.append(doc_id)

            metadata = dict(doc.metadata)  # Copy existing metadata
            metadata['doc_index'] = index
            metadata['content_length'] = len(doc.page_content)
            metadatas.append(metadata)

            documents_text.append(doc.page_content)
            embeddings_list.append(embedding.tolist())

        try:
            self.collection.add(
                ids=ids,
                metadatas=metadatas,
                documents=documents_text,
                embeddings=embeddings_list
            )
            print(f"Successfully added {len(documents)} documents to the vector store collection {self.collection_name}.")
            print(f"Total documents in the collection: {self.collection.count()}")

        except Exception as e:
            print(f"Error adding documents to vector store: {e}")
            raise
    
    def retrieve(self, query, top_k=5, score_threshold=0.0):

        # Generate the embedding for the query
        query_embedding = self.embedding_manager.generate_embeddings([query])[0]
        
        # Search in Vector storage with the query embedding
        vector_results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k
        )

        # Process the vector result
        processed_results = []

        if vector_results['documents'] and vector_results['documents'][0]:
            print(f"Top {top_k} retrieved documents for the query '{query}':")
            documents = vector_results['documents'][0]
            metadatas = vector_results['metadatas'][0]
            distances = vector_results['distances'][0]
            ids = vector_results['ids'][0]

            for index, (doc_id, document, metadata, distance) in enumerate(zip(ids, documents, metadatas, distances)):
                # Convert distance to similarity score (ChromaDB uses cosine distance, lower is better)
                similarity_score = 1 - distance

                if similarity_score >= score_threshold:
                    processed_results.append({
                        'id': doc_id,
                        'content': document,
                        'metadata': metadata,
                        'similarity_score': similarity_score,
                        'distance': distance,
                        'rank': index + 1
                    })
            print(f"Retrieved {len(processed_results)} results after applying score threshold of {score_threshold}.")
        else:
            print(f"No documents retrieved for the query '{query}'.")

        return processed_results
    
    def rag_advanced(self, query, top_k=5, score_threshold=0.2, return_context=True, api_key=None):
        """
        RAG Pipeline with extra features:
        - Returns answer, sources, confidence score
        """

        groq_api_key = api_key or os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("Groq API Key not found. Please set GROQ_API_KEY in .env file or pass it as an argument.")

        llm = ChatGroq(api_key=groq_api_key, model="openai/gpt-oss-20b", temperature=0.1, max_tokens=1024)

        # Retrieve relevant documents
        retrieved_docs = self.retrieve(query, top_k=top_k, score_threshold=score_threshold)

        if not retrieved_docs:
            return {
                'answer': "No relevant documents found to answer the query.",
                'sources': [],
                'confidence_score': 0.0,
                'context': ""
            }

        # Prepare context for the LLM
        context = "\n\n".join([f"{doc['content']}" for doc in retrieved_docs]) if retrieved_docs else ""

        sources = [
            {
                'source': doc['metadata'].get('source_file', doc['metadata'].get('source', 'unknown')),
                'page': doc['metadata'].get('page', 'unknown'),
                'similarity_score': doc['similarity_score']
            }
            for doc in retrieved_docs
        ]
        confidence_scores = max([doc['similarity_score'] for doc in retrieved_docs])

        # Create the prompt
        prompt = f"""Use the following context to answer the question concisely.\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"""

        # Get response from LLM
        response = llm.invoke([prompt.format(context=context, query=query)])
        output = {
            'answer': response.content,
            'sources': sources,
            'confidence_score': confidence_scores
        }
        if return_context:
            output['context'] = context

        return output

    # query = "What are the key concerns about AI mentioned in the documents?"
    # result = rag_advanced(query, rag_retriever, llm, top_k=5, score_threshold=0.2, return_context=True)
    # print('Answer:', result['answer'])
    # print('Sources:', result['sources'])
    # print('Confidence Score:', result['confidence_score'])