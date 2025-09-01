from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
import numpy as np
import requests
import arxiv, pandas as pd
import json
import os

def download_pdf(arxiv_id):
    url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    response = requests.get(url)
    with open(f"./pdfs/{arxiv_id}.pdf", "wb") as f:
        f.write(response.content)

def get_latest_arxiv(query, max_results):
    client = arxiv.Client(page_size=100, delay_seconds=3)
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )
    papers = list(client.results(search))
    print(f"Found {len(papers)} papers matching the query '{query}'.")
    
    dataset = []
    for paper in papers:
        arxiv_id = paper.get_short_id()

        print(f"dio: {arxiv_id}, Title: {paper.title}")
        dataset.append({
            "id": arxiv_id,
            "title": paper.title,
            "authors": [author.name for author in paper.authors],
            "summary": paper.summary,
            "link": paper.entry_id
        })

        download_pdf(arxiv_id) 
    
    with open("arxiv_papers.json", "w") as f:
        json.dump(dataset, f, indent=4)

current_directory = os.getcwd()
print(current_directory)