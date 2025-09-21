## Week 6 Assignment --
# Voice Agent with Function Calling

This project implements a voice chatbot based on the Week 3 assignment on voice agent. Function calling feature has been added, allowing the LLM model to automatically execute tools based on voice commands.

## Modules

- Speech-to-Text: Uses Web Speech API on client browser for voice transcription (browser support: Chrome/Edge, etc.)
- Text-to-Speech: Uses Google TTS on the backend.
- Function Calling: LLM can automatically call functions for:
  - Mathematical calculations including "add" and "multiple"
  - ArXiv paper searches
- Multi-turn prompting: Maintains conversation history & includes last turn of chatting in the model prompt
- Frondend UI - for testing of the chatbot

## Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Voice Agent**:
   ```bash
   uvicorn function_calling_app:app --reload --port 5001

   ```

4. **Access the Web Interface**:
   Open a browser to go to 'http://127.0.0.1:5001', the page for the voice chatbot will be automatically loaded.

## Usage

The voice agent can handle three types of queries:

### 1. Mathematical Calculations (add, multiply)
- **Example**: "Add 2 to 5?"
- **Response**: {"function": "add"}

### 2. ArXiv Paper Searches  
- **Example**: "Find papers about neural networks"
- The agent will search arXiv, return a summary on the abstracts of the relevant papers, and detailed abstracts of the papers.

### 3. General Conversation

## Future enhancements

- Extend Tools: Use Sympy to process mathematical expressions. Add new tools (e.g. a weather lookup or translation). Define their function signatures and integrate them into your agent.\n",
- Tool Registry: Create a dictionary or registry of function names to callables to simplify routing logic when there are multiple tools.
- Other LLMs: Experiment with other models that support function calling (e.g. GPT-4 with its function calling API). Compare how their output format and reliability differ from Llama 3,
- Error Handling: Make sure your agent handles invalid inputs gracefully (e.g. a malformed math expression should not crash the agent).