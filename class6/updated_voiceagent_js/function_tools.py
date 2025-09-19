
import re
import json
from arxiv_query import execute_arxiv_query  # file

# Function definitions for the LLM
FUNCTIONS = {
    "add": {
        "name": "add",
        "description": "Add two numbers together",
        "parameters": {
            "type": "object",
            "properties": {
                "a": {"type": "number", "description": "First number to add"},
                "b": {"type": "number", "description": "Second number to add"}
            },
            "required": ["a", "b"]
        }
    },
    "multiply": {
        "name": "multiply", 
        "description": "Multiply two numbers together",
        "parameters": {
            "type": "object",
            "properties": {
                "a": {"type": "number", "description": "First number to multiply"},
                "b": {"type": "number", "description": "Second number to multiply"}
            },
            "required": ["a", "b"]
        }
    },
    "query_arxiv": {
        "name": "query_arxiv",
        "description": "Search for academic papers on arXiv and get both individual paper summaries and a comprehensive summary of all research papers found",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query for academic papers"},
                "max_results": {"type": "number", "description": "Maximum number of results to return (default: 5)"}
            },
            "required": ["query"]
        }
    }
}


def execute_function(llm, function_name, arguments):
    """Execute the specified function with the given arguments"""
    try:
        if function_name == "add":
            result = arguments["a"] + arguments["b"]
            return f"The result of {arguments['a']} + {arguments['b']} is {result}"
        elif function_name == "multiply":
            result = arguments["a"] * arguments["b"]
            return f"The result of {arguments['a']} × {arguments['b']} is {result}"
        elif function_name == "query_arxiv":
            query = arguments["query"]
            max_results = arguments.get("max_results", 5)
            return execute_arxiv_query(llm, query, max_results)
        else:
            return f"Unknown function: {function_name}"
    except Exception as e:
        return f"Error executing function {function_name}: {str(e)}"

def parse_function_call(response_text):
    """Parse function call from LLM response"""
    # Look for function call pattern
    pattern = r'<function_call>(.*?)</function_call>'
    match = re.search(pattern, response_text, re.DOTALL)
    if not match:
        pattern = r'\{.*"function":.*\}'
        match = re.search(pattern, response_text, re.DOTALL)    

        if match:
            try:
                function_data = json.loads(match.group(0))
                if ((function_data["function"] == "add") |
                    (function_data["function"] == "multiply") |
                    (function_data["function"] == "query_arxiv")):
                    return function_data
                else:
                    return f"Unknown function: {function_data['function']}"
            except json.JSONDecodeError:
                return None
    return None
