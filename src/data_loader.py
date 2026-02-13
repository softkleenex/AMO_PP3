import pandas as pd
import re

def load_data(path: str) -> pd.DataFrame:
    """
    Loads the competition data from a CSV file.
    
    Args:
        path (str): Path to the CSV file.
        
    Returns:
        pd.DataFrame: DataFrame with 'id', 'problem', and optional 'answer'.
    """
    try:
        df = pd.read_csv(path)
        # Ensure columns exist
        if 'problem' not in df.columns or 'id' not in df.columns:
            raise ValueError("Dataset missing required 'id' or 'problem' columns.")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame()

def clean_latex(text: str) -> str:
    """
    Basic cleaning of LaTeX problem text.
    """
    if not isinstance(text, str):
        return ""
    
    # Normalize whitespace
    text = " ".join(text.split())
    
    # Optional: Expand common abbreviations or fix common latex typos if discovered
    # For now, just return cleaned whitespace
    return text

def format_prompt(problem_text: str, template_type: str = "qwen_code") -> str:
    """
    Wraps the problem text in a specific prompt template.
    
    Args:
        problem_text (str): The raw problem.
        template_type (str): Type of prompt ('qwen_code', 'cot', 'base').
        
    Returns:
        str: Formatted prompt.
    """
    problem_text = clean_latex(problem_text)
    
    if template_type == "qwen_code":
        return (
            "User: You are an expert mathematician. Solve the following problem by writing a Python script. "
            "Print the final integer answer at the end.

"
            f"Problem: {problem_text}

"
            "Assistant: ```python
"
        )
    elif template_type == "cot":
        return (
            "User: Solve this math problem step-by-step. Put the final answer in \boxed{}.

"
            f"Problem: {problem_text}

"
            "Assistant:"
        )
    else:
        return problem_text

if __name__ == "__main__":
    # Test
    df = load_data("../data/reference.csv")
    print(f"Loaded {len(df)} rows.")
    if not df.empty:
        sample = df.iloc[0]['problem']
        print("Original:", sample[:50] + "...")
        print("Formatted:", format_prompt(sample))
