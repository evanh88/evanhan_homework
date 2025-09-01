from operator import index
from langchain_community.document_loaders import PyMuPDFLoader
import fitz  # PyMuPDF
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import sqlite3
import numpy as np
import os
import json
import re
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
                page_text = clean_page_text(page.get_text("text"))
                # Store text along with filename and page number
                docs.append({"filename": file, "page": page_number, "text": page_text})
                page_number += 1
            doc.close()

    return docs

def chunk_document(text, chunk_size=512, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def clean_page_text(text):
    if not text:
        return ""
    text = text.replace("\r", " ").replace("\t", " ")
    return text.strip()
    
def create_vector_store(documents: List[Document], embedding_model: SentenceTransformer):
    """
    Create a vector store from the list of documents using the specified embedding model.
    """
    # Extract text content from Document objects
    texts=[doc["content"] for doc in documents]
    
    # Generate embeddings for the texts
    embeddings = embedding_model.encode(texts, convert_to_numpy=True)
    
     # Create a FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)  # using a simple L2 index
    index.add(embeddings)  # add all chunk vectors   
	
	# Save the FAISS index to disk
    faiss.write_index(index, FAISS_INDEX_FILE)

    # Check if the embeddings are added
    print("Number of vectors in the FAISS index:", index.ntotal)
    print("FAISS index created, populated, and saved!")
	
    return index   


#############################################################################
# Main script to extract text from PDFs, chunk it, and create a FAISS index #
#############################################################################

DOC_CHUNK_FILE = "pdf_text_output.json"
DATABSE_FILE = "sqlite3_database_doc_trunks.db"
FAISS_INDEX_FILE = "faiss_index.index"

def build():
    documents = []

    if os.path.exists(DATABSE_FILE):
        os.remove(DATABSE_FILE)

    conn = sqlite3.connect(DATABSE_FILE)
    cur = conn.cursor()

    cur.execute('''CREATE TABLE IF NOT EXISTS documents
             (doc_id INTEGER PRIMARY KEY, filename TEXT, page INTEGER, chunk TEXT)''')
    cur.execute('''CREATE VIRTUAL TABLE IF NOT EXISTS doc_fts USING fts5
             (chunk, content='documents', content_rowid='doc_id')''')
    conn.commit()

    # Use pathLib to get path of PDF files
    root_folder = Path(__file__).parents[0]
    pdf_file_folder = root_folder / "pdfs"

    # Load PDF files from the "../pdf" folder under the current workspace, and extract the text
    pdf_texts = extract_pdf_text(folder=pdf_file_folder)

    trunk_id = 0
    for doc in pdf_texts:
        # cursor.execute("INSERT INTO documents(filename) VALUES (?)", (doc["filename"],))
        chunks = chunk_document(doc["text"])
        # chunks = text_splitter.split_text(doc["text"])
        for chunk in chunks:
            documents.append({"trunk_id": trunk_id, "filename": doc["filename"], "page": doc["page"], "content": chunk})
            cur.execute("INSERT INTO documents(filename, page, chunk) VALUES (?, ?, ?)", (doc["filename"], doc["page"], chunk))
            trunk_id += 1

    # Save the document chunks
    with open(DOC_CHUNK_FILE, "w") as f:
        json.dump(documents, f, indent=4)
        print("Pdf chunk file saved!")

    # Populate the FTS virtual table
    cur.execute("INSERT INTO doc_fts (rowid, chunk) SELECT doc_id, chunk FROM documents")

    # Save the database
    conn.commit()
    conn.close()

    print("Sqlite3 database file saved!")

    # Create vector store
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    create_vector_store(documents, embedding_model)



# test main
#if __name__ == "__main__":
#    main()