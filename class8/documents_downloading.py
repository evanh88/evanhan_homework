import arxiv
import requests
import os

os.makedirs("pdfs", exist_ok=True)

def download_pdf(arxiv_id):
    url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    response = requests.get(url)
    with open(f"./pdfs/{arxiv_id}.pdf", "wb") as f:
        f.write(response.content)

def get_latest_arxiv(query, max_results):
    client = arxiv.Client(page_size=100, delay_seconds=3)
    search = arxiv.Search(
        query=query,
        max_results=max_results + 20,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )
    papers = list(client.results(search))
    print(f"Found {len(papers)} papers matching the query '{query}'.")
    
    dataset = []
    for paper in papers:
        arxiv_id = paper.get_short_id()

        download_pdf(arxiv_id) 

    print(f"Downloaded {len(papers)} papers")


get_latest_arxiv("advances in quantum computing", 10)