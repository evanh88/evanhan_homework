
import re
import json
from arxiv_query import execute_arxiv_query  # file

def calculate(expr: str):
    """Evaluate a mathematical expression and return the result as a string"""
    
    try:
        from sympy import sympify
        result = sympify(expr)
        return str(result)
    except Exception as e:
        return f"Error: {e}"

def execute_function_call(llm, response_text):
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
                if (function_data["function"] == "calculate"):
                    print(f"Calling function 'calculate' with expression: {function_data['expression']}")
                    return calculate(function_data["expression"])
                elif (function_data["function"] == "query_arxiv"):
                    print(f"Calling function 'execute_arxiv_query' with query: {function_data['query']}")
                    return execute_arxiv_query(llm, function_data["query"])
                else:
                    return f"Unknown function: {function_data['function']}"
            except json.JSONDecodeError:
                    return f"JSON decoding error {function_name}: {str(e)}"
        return None
    return None
