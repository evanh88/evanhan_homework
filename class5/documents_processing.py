from langchain_community.document_loaders import PyMuPDFLoader
import fitz  # PyMuPDF
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
import faiss
import sqlite3
import numpy as np
import os
import json
from typing import List

def extract_pdf_text(folder):
    docs = []
    current_directory = os.getcwd()
    for file in os.listdir(folder):
        if file.endswith(".pdf"):
            path = os.path.join(folder, file)
            doc = fitz.open(path)
            # full_text = ""
            page_number = 1
            for page in doc:
                page_text = page.get_text()
                # Store text along with filename and page number
                docs.append({"filename": file, "page": page_number, "text": page_text})
                page_number += 1
    return docs

def chunk_document(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def create_vector_store(documents: List[Document], embedding_model: SentenceTransformer):
    """
    Create a vector store from the list of documents using the specified embedding model.
    """
    # Extract text content from Document objects
    texts = [doc.page_content for doc in documents]
    
    # Generate embeddings for the texts
    embeddings = embedding_model.encode(texts, convert_to_numpy=True)
    
    # Create a FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)  # using a simple L2 index
    index.add(embeddings)  # add all chunk vectors
    
    return index    

def search_results(query: str, embedding_model: SentenceTransformer, index: faiss.IndexFlatL2, documents: List[Document], top_k=3):
    """
    Search the vector store for the top k most similar documents to the query.
    """
    # Generate embedding for the query
    query_embedding = embedding_model.encode([query])
    
    # Search the index
    distances, indices = index.search(query_embedding, top_k)
    
    # Retrieve corresponding documents
    results = []
    for i in range(len(indices[0])):
        chunk_index = indices[0][i]
        if chunk_index != -1 and i< len(documents):  # Check if a valid index is returned
            results.append({
                "filename": documents[chunk_index]['filename'],
                "page": documents[chunk_index]['page'],
                "distance": distances[0][i],
                "\ncontent": documents[chunk_index]['content']
            })
        else:
            # Handle cases where FAISS index might be out of sync with documents
            print(f"Warning: chunk_index {i} out of bounds for documents list.")

    return results

#########
# Main script to extract text from PDFs, chunk it, and create a FAISS index
#########

def main():
    documents = []

    conn = sqlite3.connect("sqlite3_database_doc_trunks.db")
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS documents
             (doc_id INTEGER PRIMARY KEY, filename TEXT, page INTEGER, chunk TEXT)''')
    cursor.execute('''CREATE VIRTUAL TABLE IF NOT EXISTS doc_fts USING fts5
             (chunk, content='documents', content_rowid='doc_id')''')
    conn.commit()

    # Load PDF files from the "./pdf" folder under the current workspace, and extract the text
    pdf_texts = extract_pdf_text(folder="./pdfs")

    for doc in pdf_texts:
        # cursor.execute("INSERT INTO documents(filename) VALUES (?)", (doc["filename"],))
        chunks = chunk_document(doc["text"])
        for chunk in chunks:
            documents.append({"filename": doc["filename"], "page": doc["page"], "content": chunk})
            cursor.execute("INSERT INTO documents(filename, page, chunk) VALUES (?, ?, ?)", (doc["filename"], doc["page"], chunk))

    # Save the document chunks
    with open("pdf_text_output.json", "w") as f:
        json.dump(documents, f, indent=4)
    print("Pdf chunk file saved!")

    # Populate the FTS virtual table
    cursor.execute("INSERT INTO doc_fts (rowid, chunk) SELECT doc_id, chunk FROM documents")

    # Save the database
    conn.commit()
    print("Sqlite3 database file saved!")

    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    # model_name="sentence-transformers/all-MiniLM-L6-v2",
    # model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}

    # Creating vector store
    texts=[doc["content"] for doc in documents]
    embeddings = embedding_model.encode(texts, convert_to_numpy=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)  # using a simple L2 index
    index.add(embeddings)  # add all chunk vectors
    # Save the FAISS index to disk
    faiss.write_index(index, "my_faiss_index.index")

    # Check if the embeddings are added
    print("Number of vectors in the FAISS index:", index.ntotal)
    print("FAISS index created, populated, and saved!")


# test main
if __name__ == "__main__":
    main()