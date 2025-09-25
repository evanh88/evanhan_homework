import arxiv


def generate_summary_of_summaries(llm, papers, query):
    """Generate a summary of all paper abstracts using the LLM"""
    try:
        # Combine all abstracts
        all_abstracts = "\n\n".join([f"Paper {i+1}: {paper['title']}\nAbstract: {paper['abstract']}" 
                                   for i, paper in enumerate(papers)])
        
        # Create a prompt for summarization
        summary_prompt = f"""You are a helpful assistant. Based on the following abstracts from academic papers related to "{query}",
provide a summary of the abstracts focusing on the following aspects:

1. The main themes and topics across all papers
2. Common methodologies or approaches used
3. Key findings or conclusions
4. Any emerging trends or patterns
5. How these papers relate to each other

Papers and Abstracts are as follows:
"{all_abstracts}"

Please provide a concise but comprehensive summary:
"""

        # Generate summary using the LLM
        summary_outputs = llm(summary_prompt, max_new_tokens=500)[0]["generated_text"]

        # Clean up the response (remove the prompt if it's included in the output)
        if "Please provide a concise but comprehensive summary:" in summary_outputs:
            summary_outputs = summary_outputs.split("Please provide a concise but comprehensive summary:")[-1].strip()
        
        return summary_outputs
        
    except Exception as e:
        return f"Error generating summary: {str(e)}"


def execute_arxiv_query(llm, query, max_results=5):
    """Execute arXiv search and return summaries"""
    try:
        # Create arXiv client
        client = arxiv.Client(page_size=100, delay_seconds=3)
        
        # Search for papers
        search = arxiv.Search(
            query=query,
            max_results=int(max_results),
            sort_by=arxiv.SortCriterion.Relevance
        )

        papers = list(client.results(search))

        results = []
        for result in papers:
            paper_info = {
                "title": result.title,
                "authors": [author.name for author in result.authors],
                "abstract": result.summary,
                "published": result.published.strftime("%Y-%m-%d"),
                "url": result.entry_id
            }
            results.append(paper_info)
        
        # Format the response
        if not results:
            return f"No papers found for query: '{query}'"
        
        response = f"Found {len(results)} papers for query: '{query}'\n\n"
        
        # Add summary of summaries
        summary_of_summaries = generate_summary_of_summaries(llm, results, query)
        response += f"## Summary of Research Papers:\n\n{summary_of_summaries}\n\n"
        response += "## Individual Papers:\n\n"
        
        for i, paper in enumerate(results, 1):
            response += f"{i}. **{paper['title']}**\n"
            response += f"   Authors: {', '.join(paper['authors'][:3])}{'...' if len(paper['authors']) > 3 else ''}\n"
            response += f"   Published: {paper['published']}\n"
            response += f"   Abstract: {paper['abstract'][:100]}{'...' if len(paper['abstract']) > 200 else ''}\n"
            response += f"   URL: {paper['url']}\n\n"
        
        return response
        
    except Exception as e:
        return f"Error searching arXiv: {str(e)}"
