
from sentence_transformers import SentenceTransformer
import faiss
from fastapi import FastAPI, HTTPException
from langchain_core.documents import Document
from pydantic import BaseModel
import json
import uvicorn
import os
from contextlib import asynccontextmanager
from starlette.responses import JSONResponse
from typing import List

# compared RepreGuard with three other detectors

model = SentenceTransformer('all-MiniLM-L6-v2') 

@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    # app.state.models = {}

    # Load the FAISS index and chunked documents
    global faiss_index, documents

    # try load existing FAISS index and documents
    if os.path.exists("my_faiss_index.index") and os.path.exists("pdf_text_output.json"):
        faiss_index = faiss.read_index("my_faiss_index.index")
        documents = json.load(open("pdf_text_output.json", "r"))
    else:
        faiss_index = None
        documents = []
        print("FAISS index or documents file not found. Please ensure they are created before starting the server.")

    yield

    # app.state.models.clear()

app = FastAPI(lifespan=lifespan)

# Pydantic model for the search query request
class SearchQuery(BaseModel):
    query: str
    k: int = 3 # Number of top results to return

@app.get("/")
async def read_root():
    return {"message": "Welcome to the FAISS Search API"}

# API endpoint for searching
@app.post("/search/")
async def search_faiss_index(search_query: SearchQuery):
    if faiss_index is None:
        raise HTTPException(status_code=500, detail="FAISS index not loaded.")

    # Generate embedding for the query
    query_embedding = model.encode([search_query.query])

    # Perform FAISS search
    distances, indices = faiss_index.search(query_embedding, search_query.k)

    # Retrieve corresponding documents
    results = []
    for i in range(len(indices[0])):
        chunk_index = indices[0][i]
        if chunk_index != -1 and i< len(documents):  # Check if a valid index is returned
            results.append({
                "filename": documents[chunk_index]['filename'],
                "page": documents[chunk_index]['page'],
                # "distance": (distances[0][i]).item(),
                "content": documents[chunk_index]['content']
            })
        else:
            # Handle cases where FAISS index might be out of sync with documents
            print(f"Warning: chunk_index {i} out of bounds for documents list.")

    return results

# if __name__ == "__main__":
#    uvicorn.run("search:app", host="127.0.0.1", port=8000, reload=True)