from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from huggingface_hub import login
from langchain_huggingface import HuggingFacePipeline
from langchain_community.llms import HuggingFacePipeline
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents.refine import RefineDocumentsChain
# from langchain.chains.llm import LLMChain  # this is needed for HuggingFacePipeline
from langchain.llms import HuggingFacePipeline
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers.string import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from PyPDF2 import PdfReader # for getting number of pages in PDF
import json
import os
import torch
from dotenv import load_dotenv

load_dotenv()
login(token=os.getenv("LLM_KEY"))

# Initialize InferenceClient with your Hugging Face token
# client = InferenceClient(model)

# llm = pipeline("text-generation", model="avans06/Meta-Llama-3.2-8B-Instruct", device=0 if torch.cuda.is_available() else -1)

# llm = pipeline("text-generation", model="meta-llama/Llama-3.2-3B-Instruct", device=0 if torch.cuda.is_available() else -1)

# device=0 if torch.cuda.is_available() else -1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    model_id = "meta-llama/Llama-3.1-8B-Instruct"
else:
    model_id = "meta-llama/Llama-3.2-3B-Instruct"

# model_id = "Qwen/Qwen2.5-7B-Instruct"
# model_id = "meta-llama/Llama-3.2-3B-Instruct"

print("device: ", device)
print(f"model_id: {model_id}\n")

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",      # Uses GPU if available
    dtype="auto"
)

model.to(device)

# Build summarization pipeline
summarizer = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256,
    dtype=torch.bfloat16,
    temperature=0.1,
    repetition_penalty=1.1
)

llm = HuggingFacePipeline(pipeline=summarizer)
# llm = pipeline("text-generation", model=model, device=device)

if tokenizer.pad_token is None:
#    pip.tokenizer.pad_token_id = AutoModelForCausalLM.from_pretrained(model_id).config.eos_token_id
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
# llm.tokenizer.pad_token_id = model.config.eos_token_id

# === Step 4: Manual batching of chunk summaries ===
import re
import json
from pathlib import Path


def load_pdf_chunks_with_debug(pdf_path, chunk_size=1000, chunk_overlap=200):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

#            from split_doc_pages import split_documents_by_tokens_with_debug  # if in another file
#            docs = split_documents_by_tokens_with_debug(pages, tokenizer, chunk_size=2048, chunk_overlap=200)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_documents(pages)

    # === Step 3: Hugging Face text-generation pipeline ===
    # hf_pipeline = pipeline(
    #     "text-generation",
    #     model=model_id,
    #     tokenizer=tokenizer,
    #     device=0,
    #     max_new_tokens=256,
    # )

    for i, chunk in enumerate(chunks):
        tokens = tokenizer(chunk.page_content, return_tensors="pt")["input_ids"].shape[1]
        preview = chunk.page_content[:100].replace("\n", " ")
        # Overlap with previous chunk
        if i > 0:
            prev_text = chunks[i-1].page_content
            overlap_text = prev_text[-chunk_overlap:].replace("\n", " ")
        else:
            overlap_text = ""
        print(f"  Chunk {i+1}: {tokens} tokens, preview: '{preview}...'")
        if i > 0:
            print(f"    Overlap with previous chunk ({chunk_overlap} chars): '{overlap_text[:50]}...'")
    return chunks

def extract_user_assistant_pairs(text):
    """
    Extracts all <user>...<assistant> pairs in order from a single generated LLM response.
    Returns a list of tuples: (user_text, assistant_text)
    """
    pattern = re.compile(
        r"<\|start_header_id\|>user<\|end_header_id\|>\s*\n(.*?)"
        r"<\|start_header_id\|>assistant<\|end_header_id\|>\s*\n(.*?)(?=<\|start_header_id\|>|$)",
        re.DOTALL
    )
    matches = pattern.findall(text)
    return [(u.strip(), a.lstrip("assistant").lstrip("\n").strip()) for u, a in matches if a.strip()]

def summarize_chunks(
    chunks,
    hf_pipeline,
    batch_size=4,
    prompt_template=None,
    save_path="summaries.jsonl",
    snippet_chars=200
):
    all_summaries = []

    # Default prompt
    if prompt_template is None:
        prompt_template = (
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n"
            "Summarize the following in plain text:\n\n{input}\n"
            "<|start_header_id|>assistant<|end_header_id|>\n"
        )

    prompts = [prompt_template.format(input=doc.page_content) for doc in chunks]

    print("len(prompts) = ", len(prompts))

    summary_id = 0

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        chunk_batch = chunks[i:i + batch_size]
        results = hf_pipeline(batch, batch_size=batch_size)

        # print("Full output of summarizing chunk batch: \n", results)

        for j, result in enumerate(results):
            full_output = result[0]["generated_text"]
            pairs = extract_user_assistant_pairs(full_output)

            doc = chunk_batch[j]
            doc_meta = doc.metadata
            chunk_index = i + j
            page_number = doc_meta.get("page", None)
            snippet = doc.page_content.strip()[:snippet_chars].replace("\n", " ")

            if not pairs:
                all_summaries.append({
                    "summary_id": summary_id,
                    "chunk_index": chunk_index,
                    "page_number": page_number,
                    "response_index": 0,
                    "text": "[NO ASSISTANT RESPONSE FOUND]",
                    "chunk_snippet": snippet,
                    "metadata": doc_meta
                })
                summary_id += 1
                continue

            for k, (user_input, assistant_response) in enumerate(pairs):
                all_summaries.append({
                    "summary_id": summary_id,
                    "chunk_index": chunk_index,
                    "page_number": page_number,
                    "response_index": k,
                    "text": assistant_response,
                    "matched_user_input": user_input[:snippet_chars].replace("\n", " "),
                    "chunk_snippet": snippet,
                    "metadata": doc_meta
                })

                print(f"\nchunk_index: {chunk_index}, response: {assistant_response}")

                summary_id += 1

        # For testing, break after two batches are done
    #    if i == 4:
    #        break

    with open(save_path, "w", encoding="utf-8") as f:
        for summary in all_summaries:
            f.write(json.dumps(summary, ensure_ascii=False) + "\n")

    print(f"✅ Saved {len(all_summaries)} assistant responses to: {Path(save_path).absolute()}")

    return all_summaries


def extract_summary_segment_for_initial_chain(text):
    # Regex including the starting sentence
    pattern = re.compile(
        r"You are an expert summarizer. Write a short summary of the following content.\s*CONTENT:\s*(.*?)\s*SUMMARY:\s*(.*)",
        re.DOTALL
    )

    match = pattern.search(text)
    if match:
        content = match.group(1).strip()
        summary = match.group(2).strip()
        result = {
            # "Content": content,
            "Summary": summary
        }
        print(f"Initial chain summary:\n{result}")
        return summary

    else:
        print("No match found")


def extract_summary_segment_for_refine_chain(text):
    # Regex including the starting sentence
    pattern = re.compile(
        r"You are an expert summarizer.*\s*Existing summary:\s*(.*?)\s*New information:\s*(.*?)Final summary:\s*(.*)",
        re.DOTALL
    )

    match = pattern.search(text)
    if match:
        existing_summary = match.group(1).strip()
        new_information = match.group(2).strip()
        final_summary = match.group(3).strip()
        result = {
            # "Content": content,
            "Summary": final_summary
        }
        print(f"Refine chain summary:\n{result}")
        return final_summary

    else:
        print("No match found")

def extract_summary_segment_for_reduce_map(text):
    # Regex including the starting sentence
    pattern = re.compile(
        r"You are an expert summarizer.*\s*Summaries:\s*(.*?)\s*Final summary.*words\):\s*(.*)",
        re.DOTALL
    )

    match = pattern.search(text)
    if match:
        summaries = match.group(1).strip()
        final_summary = match.group(2).strip()
        result = {
            # "Content": content,
            "Summary": final_summary
        }
        print(f"Reduce map summary:\n{result}")
        return final_summary

    else:
        print("No match found")


def get_final_summary(documents, hf_pipeline_raw, refine_prompt_template, temperature=0.1):
    # Wrap HuggingFace pipeline
    llm = HuggingFacePipeline(pipeline=hf_pipeline_raw)

    # Set temperature if possible (optional)
    if hasattr(hf_pipeline_raw.model.config, "temperature"):
        hf_pipeline_raw.model.config.temperature = temperature

    # Initial prompt (first chunk)
    initial_prompt = PromptTemplate.from_template(
        """You are an expert summarizer. Write a short summary of the following content.

CONTENT:
{text}

SUMMARY:
"""
    )

    # Refine prompt (next chunks)
    refine_prompt = PromptTemplate.from_template(refine_prompt_template)

    # Assemble Runnable chains
    initial_chain = initial_prompt | llm | StrOutputParser()
    refine_chain = refine_prompt | llm | StrOutputParser()

    def refinement_runner(docs: list[Document]) -> str:
        summary_init = initial_chain.invoke({"text": docs[0].page_content})
        summary = extract_summary_segment_for_initial_chain(summary_init)
        for doc in docs[1:]:
            summary_ref = refine_chain.invoke({
                "existing_summary": summary,
                "text": doc.page_content
            })
            summary = extract_summary_segment_for_refine_chain(summary_ref)
        return summary

    refine_sequence = RunnableLambda(refinement_runner)
    return refine_sequence.invoke(documents)


def get_final_summary_map_reduce(
    summary_docs: list[Document],
    hf_pipeline_raw,
    reduce_prompt_template: str,
    temperature=0.3,
    word_limit=300
) -> str:
    """
    Final summarization using map_reduce logic:
    - Assumes all chunk summaries are already generated (as summary_docs)
    - Uses HuggingFace pipeline for reduce stage
    """

    # Use the same HF pipeline for reduce stage (already batched)
    llm = HuggingFacePipeline(pipeline=hf_pipeline_raw)

    # Combine summaries into one long string
    combined_text = "\n\n".join([doc.page_content for doc in summary_docs])

    # Optional: limit token count (you can truncate if needed)
    # token_count = len(hf_pipeline_raw.tokenizer(combined_text)["input_ids"])
    # print(f"🧮 Token count in reduce input: {token_count}")

    # Apply reduce prompt template
    reduce_prompt = PromptTemplate.from_template(reduce_prompt_template)

    reduce_chain = reduce_prompt | llm | StrOutputParser()

    # Run the final summarization
    return reduce_chain.invoke({"text": combined_text, "word_limit": 300})


# Two reduce prompts for generating alternative summaries
reduce_prompts = [
    """You are an expert summarizer. Combine the following partial summaries
into a concise summary of less than 400 words. Focus on the most important points. Avoid repetition.

Existing summary:
{existing_summary}

New information:
{text}

Final summary:
""",
    """You are an expert summarizer. Combine the following partial summaries
into a coherent summary of less than 400 words, emphasizing insights, conclusions, and actionable points.
Avoid repeating any details from chunks.

Existing summary:
{existing_summary}

New information:
{text}

Final summary:
"""
]

reduce_prompt_template = """
You are an expert summarizer.

Below are summaries of different parts of a document. Your job is to combine them into one **concise, high-quality summary**, no more than **{word_limit} words**.

Summaries:
{text}

Final summary (max {word_limit} words):
"""
def process_pdf(folder, chunk_size=2000, chunk_overlap=150):
    docs = []
    summaries = []
    summary = []
    total_papers_examined = 0

    final_summaries = []
    for file in os.listdir(folder):
        if file.endswith(".pdf"):
            pdf_path = f"{pdfs_folder}/{file}"

            print(f"Loaded : {file}")

            # Get the total number of pages, skip large file
            reader = PdfReader(pdf_path)
            num_pages = len(reader.pages)
            if (num_pages > 28):
                print(f"skipping this long PDF file, ({num_pages} pages)")
                continue

            total_papers_examined += 1

            # === Step 1: Load PDF and split into chunks with token count & overlap info ===
            chunks = load_pdf_chunks_with_debug(pdf_path, chunk_size, chunk_overlap)

            # Step 2: summarize each chunk
            # chunk_summaries = summarize_chunks(docs, summarizer, batch_size=4)
            prompt_template = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\nSummarize the following text:\n\n{input}\n<|start_header_id|>assistant<|end_header_id|>\n"
            chunk_summaries = summarize_chunks(chunks, summarizer, batch_size=32, prompt_template=prompt_template)
            
        #    with open("summaries.jsonl", "r", encoding="utf-8") as f:
        #        chunk_summaries = list(map(json.loads, f))

            print(f"\nTotal chunks: {len(chunks)}")
            print(f"Total chunk summaries: {len(chunk_summaries)}")

            # === Step 5: Combine chunk summaries into Documents ===
            summary_docs = [
                Document(
                    page_content=summary["text"],
                    metadata={
                        "summary_id": summary["summary_id"],
                        "chunk_index": summary["chunk_index"],
                        "page_number": summary.get("page_number"),
                        "response_index": summary.get("response_index"),
                        "token_count": summary.get("metadata", {}).get("token_count")
                    }
                )
                for summary in chunk_summaries
            ]

            # Step 3: generate two alternative final summaries
            # final_summaries = []

            # for i, prompt_template in enumerate(reduce_prompts):
            #    # vary temperature slightly for diversity
            #    summary_r = get_final_summary(summary_docs, summarizer, prompt_template, temperature=0.1 + 0.5*i)
            #    print(f"Generated summary version {i}")

            for i in range(2):
                summary_r = get_final_summary_map_reduce(
                    summary_docs=summary_docs,                 # from summarize_chunks_fast()
                    hf_pipeline_raw=summarizer,                # text-generation pipeline
                    reduce_prompt_template=reduce_prompt_template,
                    temperature=0.1 + 0.5*i,
                    word_limit=300
                )

                summary = extract_summary_segment_for_reduce_map(summary_r)

                if i == 0:
                    # At the first round, remember the summary obtained
                    summary1 = summary
                if i == 1:
                    # After two summaries are generated, save them to the list which is to be edited later for accept/rejct action
                    final_summaries.append({
                       # "file_#": total_papers_examined,
                        "file_name": file,
                        "chosen": summary1, # first round
                        "reject": summary   # second round
                    })
                    print("Summaries saved for file: ", file)

            # Test one paper
            if total_papers_examined == 10:
                print(f"Processed {total_papers_examined} PDF files")

                break                    

    with open("reward_data_summaries.jsonl", "w") as f:
        for item in final_summaries:
            f.write(json.dumps(item) + "\n")

"""
            # === Step 6: Final summarization using LangChain (stuff/map_reduce) ===
            llm = HuggingFacePipeline(pipeline=summarizer)

            # Use "stuff" now since input is small (just summaries)
            final_chain = load_summarize_chain(llm, chain_type="stuff")
            summary = final_chain.run(summary_docs)

            print("Final summpary: ", summary)

            summary2 = ""

            while True:
                # inp = input("Which response would you like to accept (1/2)?")
                inp = 1

                if inp == 1:
                    summaries.append({
                        "paper#": total_papers_examined,
                        "chosen": final_summaries
,
                        "chosen": summary2
                    })

                    break
                elif inp == 2:
                    summaries.append({
                        "paper#": total_papers_examined,
                        "chosen": summary2,
                        "rejected": summary
                    })

                    break
                else:
                    continue


"""



def get_abstracts_and_generate_summary_pairs(max_results=10):

    # Get arXiv papers and save the abstracts to file
    
    ### get_arxiv_papers("quantum computing", max_results)

    with open(ABSTRACTS_FILE, "r") as f:
        dataset = json.load(f)
    
    summary_pairs = []
    total_papers_examined = 0
    for paper in dataset:
        print("=" * 50)
        
        print(f"Generating summary pairs for paper# {total_papers_examined}: {paper['id']}, {paper['title']}")
         # Generate Q&A pairs for the abstract
         # list_qas = generate_qa(paper['summary'])
        
        response1 = llama_abstract_summary(paper['summary'], max_tokens=100, temperature=0.7, round=1)
        response2 = llama_abstract_summary(paper['summary'], max_tokens=100, temperature=0.7, round=2)
        if response1 == "Error: failed to generate response" or response2 == "Error: failed to generate response":
            print("Error generating summaries, go to next paper")
            continue

        print(f"Response1:\n{response1}")
        print(f"Response2:\n{response1}")

        while True:
            inp = input("Which response would you like to accept (1/2)?")

            if inp == 1:
                summary_pairs.append({
                    "paper#": {total_papers_examined},
                    "chosen": "{response1}",
                    "rejected": "{response2}"
                })

                break
            elif inp == 2:
                summary_pairs.append({
                    "paper#": {total_papers_examined},
                    "chosen": response2,
                    "rejected": response1
                })

                break
            else:
                continue

        with open("reward_data.jsonl", "w") as f:
            for item in summary_pairs:
                f.write(json.dumps(item) + "\n")

        total_papers_examined += 1
        if total_papers_examined == 10:
            break

def get_abstracts_and_generate_summary_once(max_results=10):

    # Get arXiv papers and save the abstracts to file
    
    ### get_arxiv_papers("quantum computing", max_results)

    # with open(ABSTRACTS_FILE, "r") as f:
    #    dataset = json.load(f)
    
    summaries = []
    total_papers_examined = 0
    for paper in dataset:
        print("=" * 50)
        
        total_papers_examined += 1
        
        print(f"Generating summaries for paper# {total_papers_examined}: {paper['id']}, {paper['title']}")
         # Generate Q&A pairs for the abstract
         # list_qas = generate_qa(paper['summary'])
        
        response1 = llama_abstract_summary(paper['summary'], max_tokens=512, temperature=0, round=1)
        if response1 == "Error: failed to generate response":
            continue

        print(f"Response1:\n{response1}")

        response2 = ""

        while True:
            # inp = input("Which response would you like to accept (1/2)?")
            inp = 1

            if inp == 1:
                summaries.append({
                    "paper#": total_papers_examined,
                    "chosen": response1,
                    "rejected": response2
                })

                break
            elif inp == 2:
                summaries.append({
                    "paper#": total_papers_examined,
                    "chosen": response2,
                    "rejected": response1
                })

                break
            else:
                continue

        with open("reward_data_summaries.jsonl", "w") as f:
            for item in summaries:
                f.write(json.dumps(item) + "\n")

        if total_papers_examined == 10:
            break


# Example usage
# get_abstracts_and_generate_summary_pairs(10)
# get_abstracts_and_generate_summary_once(10)

pdfs_folder = './pdfs'

process_pdf(pdfs_folder)

# get_arxiv_papers("machine learning", max_results=100)

# read_abstracts_and_generate_qas(max_results=100)

# generate_synthetic_qa_prompts()