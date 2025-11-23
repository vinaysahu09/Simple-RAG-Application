from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from src.vector_store_chromadb import VectorStoreManager
import os

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

app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
