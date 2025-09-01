from sentence_transformers import SentenceTransformer
import faiss
from fastapi import FastAPI, HTTPException
from langchain_core.documents import Document
from pydantic import BaseModel
import json
import uvicorn
from rank_bm25 import BM25Okapi
import os
import numpy as np
import sqlite3
from contextlib import asynccontextmanager
from starlette.responses import JSONResponse
from typing import List
import math

embedding_model = SentenceTransformer('all-MiniLM-L6-v2') 

DOC_CHUNK_FILE = "pdf_text_output.json"
DATABSE_FILE = "sqlite3_database_doc_trunks.db"
FAISS_INDEX_FILE = "faiss_index.index"

_conn = None
_cursor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    # app.state.models = {}

    # Load the FAISS index and chunked documents
    load_documents_and_faiss_index()

    # Load database
    connect_to_database()

    # Call to test the search function for hybrid search with the original query
    # hybrid_search("Normative LLMs Profiling", k=5, alpha=0.4)

    yield

    # app.state.models.clear()


def hybrid_search(query: str, k: int = 5, alpha: float = 0.6, 
                  normalize_method: str = "sigmoid", verbose: bool = True):
    """
    Perform hybrid search combining vector similarity and keyword relevance
    
    Args:
        query: Search query string
        k: Number of top results to return
        alpha: weight
        normalize_method: score normalization method ("minmax", "sigmoid", "none")
        verbose: Whether to print detailed results
    
    Returns:
        List of dictionaries containing hybrid search results
    """
    # Get vector search results (get more to have better coverage)
    vector_results = faiss_search(query, k*2)
    
    # Get keyword search results
    fts_results = sqlite_keyword_search(query, k*2, normalize_method=normalize_method)
    
    # Create dictionaries to store scores by document ID
    vector_scores = {}
    keyword_scores = {}
    
    # Process vector results
    for res in vector_results:
        # Use normalized distance as similarity score (0 to 1 range)
        vector_scores[res["chunk_idx"]] = res["normalized_distance"]
    
    # Process keyword results
    for res in fts_results:
        # Use normalized BM25 scores
        keyword_scores[res["rowid"]] = res["bm25_score_normalized"]
    
    # Get all unique document indices from both searches
    all_indices = set(vector_scores.keys()) | set(keyword_scores.keys())
    
    # Combine scores using weighted fusion
    combined_scores = {}
    
    for doc_idx in all_indices:
        vector_score = vector_scores.get(doc_idx, 0.0)
        keyword_score = keyword_scores.get(doc_idx, 0.0)
        
        # Weighted combination
        combined_score = alpha * vector_score + (1 - alpha) * keyword_score
        combined_scores[doc_idx] = {
            "combined_score": combined_score,
            "vector_score": vector_score,
            "keyword_score": keyword_score
        }
    
    # Sort by combined score (descending)
    sorted_results = sorted(combined_scores.items(), 
                           key=lambda x: x[1]["combined_score"], 
                           reverse=True)
    
    # Get top k results
    top_k_results = []
    for doc_idx, scores in sorted_results[:k]:
        # Find the document details
        doc_info = None
        
        # Try to get from vector results first
        for vec_res in vector_results:
            if vec_res["chunk_idx"] == doc_idx:
                doc_info = {
                    "doc_idx": doc_idx,
                    "filename": vec_res["filename"],
                    "page": vec_res["page"],
                    "content": vec_res["content"],
                    "combined_score": scores["combined_score"],
                    "vector_score": scores["vector_score"],
                    "keyword_score": scores["keyword_score"]
                }
                break
        
        # If not found in vector results, try keyword results
        if doc_info is None:
            for kw_res in fts_results:
                if kw_res["rowid"] == doc_idx:
                    doc_info = {
                        "doc_idx": doc_idx,
                        "filename": kw_res["filename"],
                        "page": kw_res["page"],
                        "content": kw_res["chunk"],
                        "combined_score": float(scores["combined_score"]),
                        "vector_score": float(scores["vector_score"]),
                        "keyword_score": float(scores["keyword_score"])
                    }
                    break
        
        if doc_info:
            top_k_results.append(doc_info)
    
    # Print results if verbose
    if verbose:
        print(f"\nHybrid Search Results for query: '{query}'")
        print(f"alpha={alpha}")
        print("=" * 60)
        print(f"{'Rank':<4} {'Doc ID':<6} {'Combined':<8} {'Vector':<8} {'Keyword':<8} {'Filename':<18} {'Page':<10}")
        print("-" * 60)
        
        for i, result in enumerate(top_k_results, 1):
            print(f"{i:<4} {result['doc_idx']:<6} {result['combined_score']:<8.3f} "
                  f"{result['vector_score']:<8.3f} {result['keyword_score']:<8.3f} "
                  f"{result['filename'][:14]:<18} {result['page']:<10}")

    
    return top_k_results

def load_documents_and_faiss_index():

    global faiss_index, documents

    # try load existing FAISS index and documents
    if os.path.exists(FAISS_INDEX_FILE) and os.path.exists(DOC_CHUNK_FILE):
        faiss_index = faiss.read_index(FAISS_INDEX_FILE)
        with open(DOC_CHUNK_FILE, 'r') as json_file:
            documents = json.load(json_file)
        print("FAISS index and documents files loaded")
    else:
        faiss_index = None
        documents = []
        print("FAISS index or documents file not found. Please ensure they are created before starting the server.")


def faiss_search(query: str, top_k, verbose=False):
    """
    Search the vector store for the top k most similar documents to the query.
    """
    # Generate embedding for the query
    query_embedding = embedding_model.encode([query])
    
    # Search the index
    distances, indices = faiss_index.search(query_embedding, top_k)

    # Retrieve corresponding documents
    # FAISS returns distance (lower = more similar), so we convert to similarity
    # The formula 1/(1+distance) converts distance to similarity (0 to 1 range)
    results = []
    for i in range(len(indices[0])):
        chunk_index = indices[0][i]
        if chunk_index != -1 and i< len(documents):  # Check if a valid index is returned
            results.append({
                "chunk_idx": int(chunk_index),
                "filename": documents[chunk_index]['filename'],
                "page": documents[chunk_index]['page'],
                "distance": float(distances[0][i]),
                "normalized_distance": float(1 / (1 + distances[0][i])),
                "content": documents[chunk_index]['content']
            })
        else:
            # Handle cases where FAISS index might be out of sync with documents
            print(f"Warning: chunk_index {i} out of bounds for documents list.")

    if (verbose):
        print(f"\nVector Search Results, for query: '{query}'")
        print("=" * 60)
        print(f"{'Rank':<4} {'Trunk-ID':<8} {'Nor-dist':<8} {'Filename':<18} {'page':<10}")
        print("-" * 60)
        
        for i, result in enumerate(results, 1):
            print(f"{i:<4} {result['chunk_idx']:<9}"
                    f"{result['normalized_distance']:<9.3f}"
                    f"{result['filename'][:18]:<18} {result['page']:<10}")

    return results

def connect_to_database():

    global _conn, _cursor

    # Connect to Sqlite3 database of doc chunks
    try:
        _conn = sqlite3.connect(DATABSE_FILE)
        _cursor = _conn.cursor()
        print("Successfully opened the database")
    except sqlite3.OperationalError as e:
        print("Error:", e)


def sqlite_keyword_search(query, k=5, normalize_method="sigmoid", verbose=False):
    """
    Search using SQLite FTS5 with BM25 scoring and optional normalization
    Args:
        query: Search query string
        k: Number of top results to return
        normalize_method: Normalization method - "minmax", "sigmoid", or "none"
    """
    global _conn, _cursor

    # Search sqlite database, DESC keyword causes the results to be returned from best to worst
    sql = '''SELECT documents.*, bm25(doc_fts) AS score
                FROM doc_fts
                JOIN documents ON doc_fts.rowid = documents.doc_id
                WHERE doc_fts.chunk MATCH ?
                ORDER BY score DESC
                LIMIT ?'''     
    _cursor.execute(sql, (query, k))

    output = []
    # Fetch and display the results
    results = _cursor.fetchall()
    
    # Extract raw BM25 scores
    # The bm25 score is negative, with smaller (less negative) values indicating higher relevance.
    raw_scores = [float(row[4]) for row in results]
    
    # Apply normalization if requested
    if normalize_method != "none" and raw_scores:
        if normalize_method == "minmax":
            normalized_scores = normalize_bm25_minmax(raw_scores)
        elif normalize_method == "sigmoid":
            normalized_scores = normalize_bm25_sigmoid(raw_scores)
        else:
            normalized_scores = raw_scores
    else:
        normalized_scores = raw_scores
    
    # Create output with both raw and normalized scores
    for i, row in enumerate(results):
        output.append({
            "rowid": row[0] - 1, 
            "filename": row[1], 
            "page": row[2], 
            "bm25_score_raw": raw_scores[i],
            "bm25_score_normalized": normalized_scores[i] if normalize_method != "none" else None,
            "chunk": row[3]
        })

    if (verbose):
        print(f"\nKeyword Search Results, for query: '{query}'")
        print("=" * 60)
        print(f"{'Rank':<4} {'Trunk-ID':<8} {'Norm-score':<9} {'Filename':<18} {'page':<10}")
        print("-" * 60)
        
        for i, result in enumerate(output, 1):
            print(f"{i:<4} {result['rowid']:<9}"
                    f"{result['bm25_score_normalized']:<11.3f}"
                    f"{result['filename'][:18]:<18} {result['page']:<10}")

    return output
    

def normalize_bm25_minmax(scores):
    """Min-max normalization to scale scores to [0,1] range"""
    if not scores:
        return scores
    
    min_score = min(scores)
    max_score = max(scores)
    
    if max_score == min_score:
        return [1.0] * len(scores)  # All scores are equal
    
    normalized = [(score - min_score) / (max_score - min_score) for score in scores]
    return normalized

def normalize_bm25_sigmoid(scores, temperature=1.0):
    """Sigmoid normalization to map scores to (0,1)"""
    if not scores:
        return scores
    
    mean_score = sum(scores) / len(scores)
    normalized = [1 / (1 + math.exp(-(score - mean_score) / temperature)) for score in scores]
    return normalized


app = FastAPI(lifespan=lifespan)

# Pydantic model for the search query request
class SearchQuery(BaseModel):
    query: str
    k: int = 3 # Number of top results to return

# Pydantic model for hybrid search request
class HybridSearchQuery(BaseModel):
    query: str
    k: int = 5
    alpha: float = 0.4
    normalize_method: str = "sigmoid"

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Hybrid Search API"}

# API endpoint for searching
@app.post("/faiss_search/")
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

# API endpoint for hybrid search
@app.post("/hybrid-search/")
async def hybrid_search_api(search_query: HybridSearchQuery):
    if faiss_index is None:
        raise HTTPException(status_code=500, detail="FAISS index not loaded.")
    
    try:
        results = hybrid_search(
            query=search_query.query,
            k=search_query.k,
            alpha=search_query.alpha,
            normalize_method=search_query.normalize_method,
            verbose=False  # Don't print to console for API calls
        )
        
        return {
            "query": search_query.query,
            "alpha": search_query.alpha,
            "normalize_method": search_query.normalize_method,
            "results": results 
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")

if __name__ == "__main__":
   uvicorn.run("hybrid_search:app", host="127.0.0.1", port=8000, reload=True)