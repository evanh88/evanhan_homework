from transformers import pipeline
import streamlit as st
import re   
import torch

from typing import Dict
from function_tools import * # file: function_tools.py


SYSTEM_PROMPT = {
    "role": "system",
    "text": "You are a helpful assistant. \
When the user asks for mathematical operations like addition or multiplication, you help the user by calling tools when available. \
Available tool for math questions: calculate(expression). \
When user asks about academic papers or research topics, use the query_arxiv tool to fetch relevant papers and provide a summary of findings. \
Available tools for academic papers: query_arxiv(query, max_results). \
To call a tool for math calculation, you must respond in this format: {\"function\": \"function_name\", \"expression\": \"the math expression\"}. \
To call a tool for academic paper search, you must respond in this format: {\"function\": \"function_name\", \"query\": \"the query string\"}. \
If the user input is not a request for a function call, respond normally."}


# Meta's LLaMA 3 instruct special tokens
BOT = "<|begin_of_text|>"
EOT = "<|end_of_text|>"
SOF = "<|start_header_id|>"
EOF = "<|end_header_id|>"
EOM = "<|eom_id|>"  # End of message, for multi-turn conversations
EOTN = "<|eot_id|>\n"  # End of turn

# login(token=os.getenv("LLM_KEY"))
llm = pipeline("text-generation", model="meta-llama/Llama-3.2-3B-Instruct", device=0 if torch.cuda.is_available() else -1)

def query_llm(chat_history):            
    # Construct the full prompt
    prompt = build_prompt(SYSTEM_PROMPT, chat_history)

    full_output = llm(prompt, max_new_tokens=500)[0]['generated_text']

    # Find the last <|start_header_id|>user<|end_header_id|> (latest user query)
    # user_blocks = list(re.finditer(r">\nuser\n", full_output))
    user_blocks = list(re.finditer(r"\|start_header_id\|>user<\|end_header_id\|>", full_output))
    if not user_blocks:
        print("No user input found. Returning everything.")
        search_start = 0
    else:
        # Get last user query position
        search_start = user_blocks[-1].end()

    # Slice from after the last user message
    remaining = full_output[search_start:]

    # Pattern to match either '\nassistant\n' or '\nassistant\n\n'
    # Find all assistant segments in the remaining text
    matches = list(re.finditer(
        # r"\|eot_id\|>\nassistant\n+(.*?)(?=(\|eot_id\|>|$))",
        # r"<\|start_header_id\|>assistant<\|end_header_id\|>",
        r"<\|start_header_id\|>assistant<\|end_header_id\|>\n+(.*?)(?=(<\|eot_id\|>|$))",
        remaining,
        flags=re.DOTALL
    ))

    function_result = ""
    function_call = None
    
    assistant_texts = [m.group(0).strip().replace('<|start_header_id|>assistant<|end_header_id|>', '').strip() for m in matches]
    if len(assistant_texts) == 1:
        # Check if the response contains a function call
        function_result = execute_function_call(llm, assistant_texts[0])
        
    # Concatenate all assistant segments
    if function_result:
        all_assistant_response = f"\nFunction Calling Result: {function_result}"
    else:
        all_assistant_response = "\n".join(assistant_texts).strip()

    return all_assistant_response

