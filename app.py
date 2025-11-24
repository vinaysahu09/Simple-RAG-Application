from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from src.vector_store_chromadb import VectorStoreManager
import os
# Import for file upload
from fastapi import File, UploadFile
import shutil
from langchain_community.document_loaders import PyMuPDFLoader

app = FastAPI()

# Initialize Vector Store Manager
# We initialize it once at startup to avoid reloading it for every request
try:
    vector_store_manager = VectorStoreManager()
except Exception as e:
    print(f"Error initializing VectorStoreManager: {e}")
    vector_store_manager = None

class QueryRequest(BaseModel):
    query: str

@app.post("/ask")
async def ask_question(request: QueryRequest):
    if not vector_store_manager:
        raise HTTPException(status_code=500, detail="Vector Store Manager not initialized")
    
    try:
        result = vector_store_manager.rag_advanced(request.query)
        return {
            "answer": result['answer'],
            "sources": result['sources'],
            "confidence_score": result['confidence_score']
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Mount static files for the frontend
# Ensure the static directory exists
if not os.path.exists("static"):
    os.makedirs("static")

@app.get("/files")
async def list_files():
    pdf_dir = os.path.join("data", "pdf_files")
    if not os.path.exists(pdf_dir):
        return {"files": []}
    
    files = [f for f in os.listdir(pdf_dir) if f.lower().endswith('.pdf')]
    return {"files": files}


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        # Ensure data directory exists
        pdf_dir = os.path.join("data", "pdf_files")
        if not os.path.exists(pdf_dir):
            os.makedirs(pdf_dir)
            
        # Save the file
        file_path = os.path.join(pdf_dir, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Process the file if vector store is available
        if vector_store_manager:
            # Load the document
            loader = PyMuPDFLoader(file_path)
            documents = loader.load()
            
            # Add metadata
            for doc in documents:
                doc.metadata['source'] = file.filename
                doc.metadata['file_type'] = 'pdf'
                
            # Chunk and embed
            chunks = vector_store_manager.embedding_manager.chunk_documents(documents)
            embeddings = vector_store_manager.embedding_manager.generate_embeddings_with_chunks(chunks)
            
            # Add to vector store
            vector_store_manager.add_documents_to_collection(chunks, embeddings)
            
            return {
                "message": f"File '{file.filename}' uploaded and processed successfully",
                "chunks_count": len(chunks)
            }
        else:
            return {
                "message": f"File '{file.filename}' uploaded but vector store not initialized",
                "warning": "Vector store not available"
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
