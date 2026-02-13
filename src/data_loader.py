import pandas as pd
import re

def load_data(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        if 'problem' not in df.columns or 'id' not in df.columns:
            raise ValueError("Dataset missing required columns.")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame()

def clean_latex(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return " ".join(text.split())

def format_prompt(problem_text: str, template_type: str = "qwen_code", error_msg: str = None) -> str:
    """
    Wraps problem text in prompt templates.
    """
    problem_text = clean_latex(problem_text)
    
    if template_type == "qwen_code":
        return (
            "User: You are an expert mathematician. Solve the following problem by writing a Python script. "
            "Print the final integer answer at the end. Use specific libraries like `sympy` if needed.\n\n"
            f"Problem: {problem_text}\n\n"
            "Assistant: ```python\n"
        )
    elif template_type == "fix_code":
        return (
            "User: The previous code failed with this error:\n"
            f"{error_msg}\n\n"
            "Please rewrite the Python code to fix the error and solve the problem.\n\n"
            "Assistant: ```python\n"
        )
    elif template_type == "cot":
        return (
            "User: You are an expert mathematician. Solve this math problem step-by-step. "
            "Show your work clearly. At the very end, put the final integer answer inside \\boxed{}.\n\n"
            f"Problem: {problem_text}\n\n"
            "Assistant:"
        )
    return problem_text
