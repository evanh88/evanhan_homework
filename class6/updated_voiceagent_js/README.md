## Week 6 Assignment --
# Voice Agent with Function Calling

This project implements a voice chatbot based on the Week 3 assignment on voice agent. Function calling feature has been added, allowing the LLM model to automatically execute tools based on voice commands.

## Modules

- Speech-to-Text: Uses Web Speech API on client browser for voice transcription (browser support: Chrome/Edge, etc.).
- Text-to-Speech: Uses Google TTS on the backend.
- Function Calling (new): LLM can automatically call functions for:
  - Mathematical calculations
  - ArXiv paper searches
- Multi-turn prompting: Maintains conversation history. Currently last turn of chatting is included in the model prompt.
- Frondend UI - rewritten in Javascript, for testing of the chatbot. The frontend is able to handle MD format text returned by the model.

## Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Voice Agent**:
   ```bash
   uvicorn function_calling_app:app --port 5001

   ```

4. **Access the Web Interface**:
   Open a browser to go to 'http://127.0.0.5001', the voice chatbot will be automatically loaded.

## Usage

The voice agent can handle three types of queries:

### 1. Mathematical Calculations (add, multiply, etc.)
- **Example**: "What is the result of adding 2 to 5?"
- **Response**: {"function": "add", "expression": "add 2 to 5"}

### 2. ArXiv Paper Searches  
- **Example**: "Find papers about neural networks"
- **Response**: {"function": "query_arxiv", "query": "neural networks"}
- The agent will search arXiv, return a summary on the abstracts of the relevant papers, and attach a list of the papers retrieved.

### 3. General Conversation

## Future enhancements

- Extend Tools: Add new tools (e.g. a weather lookup or translation). Define their function signatures and integrate them into the agent.\n",
- Tool Registry: Create a dictionary or registry of function names to callables to simplify routing logic when there are multiple tools.
- Other LLMs: Experiment with other models that support function calling (e.g. GPT-4 with its function calling API). Compare how their output format and reliability differ from Llama 3,
- Error Handling: Make sure the agent handles invalid inputs gracefully.