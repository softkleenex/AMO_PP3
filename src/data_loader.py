import pandas as pd
import re

def load_competition_data(path: str) -> pd.DataFrame:
    """Loads CSV dataset and ensures required columns exist."""
    try:
        df = pd.read_csv(path)
        if 'problem' not in df.columns or 'id' not in df.columns:
            raise ValueError("Dataset missing 'problem' or 'id' columns.")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame()

def format_prompt(problem_text: str, template_type: str = "qwen_code", error_msg: str = None) -> str:
    """
    Wraps problem text in math-optimized prompt templates.
    Designed for Qwen2.5-Math-7B behavior.
    """
    # Simple normalization
    problem_text = " ".join(problem_text.split())
    
    if template_type == "qwen_code":
        return (
            "User: You are an expert mathematician. Solve the following problem by writing a Python script. "
            "Print the final integer answer at the end.\n\n"
            f"Problem: {problem_text}\n\n"
            "Assistant: ```python\n"
        )
    elif template_type == "fix_code":
        return (
            "User: The previous Python code failed with this error:\n"
            f"{error_msg}\n\n"
            "Please rewrite the Python code to fix the error and solve the math problem correctly.\n\n"
            "Assistant: ```python\n"
        )
    elif template_type == "cot":
        return (
            "User: Solve this math problem step-by-step. Put the final integer answer inside \\boxed{}.\n\n"
            f"Problem: {problem_text}\n\n"
            "Assistant:"
        )
    return problem_text
