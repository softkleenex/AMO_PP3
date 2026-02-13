from utils import PythonREPL, extract_answer
from data_loader import format_prompt
import re

# Placeholder for the actual LLM generation function.
# In production/notebook, this is monkey-patched.
def mock_llm_generate_code(prompt: str) -> str:
    # Heuristic responses for testing logic flow
    if "fix the error" in prompt:
        return "print(336) # Fixed code"
    if "10^{5}" in prompt:
        return "print(1/0) # Intentionally cause error to test retry"
    if "1-1" in prompt:
        return "print(0)"
    if "step-by-step" in prompt: # CoT fallback
        return "The answer is \\boxed{42}."
    
    return "print(0)"

def solve(problem_text: str, model_generate_func=None) -> int:
    """
    Robust Solver Pipeline:
    1. Try Python Code Generation
    2. If Error, Try Self-Correction
    3. If still Error or 0, Try Chain-of-Thought
    """
    generate = model_generate_func if model_generate_func else mock_llm_generate_code
    repl = PythonREPL(timeout=7)
    
    # --- ATTEMPT 1: Code Gen ---
    prompt = format_prompt(problem_text, "qwen_code")
    response = generate(prompt)
    
    # Extract code
    code = response
    match = re.search(r'```python\n(.*?)\n```', response, re.DOTALL)
    if match:
        code = match.group(1)
        
    output = repl.execute(code)
    
    # --- ATTEMPT 2: Self-Correction (if needed) ---
    if "Error" in output:
        # print(f"DEBUG: Error encountered: {output}. Retrying...")
        prompt = format_prompt(problem_text, "fix_code", error_msg=output)
        response = generate(prompt)
        
        match = re.search(r'```python\n(.*?)\n```', response, re.DOTALL)
        if match:
            code = match.group(1)
        output = repl.execute(code)
    
    # Extract answer from code output
    ans = extract_answer(output)
    
    # --- ATTEMPT 3: Chain-of-Thought Fallback ---
    # If answer is 0 (default/failure) or we had persistent errors
    if ans == 0 or "Error" in output:
        # print("DEBUG: Fallback to CoT...")
        prompt = format_prompt(problem_text, "cot")
        response = generate(prompt)
        ans = extract_answer(response)
        
    # Final Modulo Constraint (AIMO often requires answer modulo 1000 or similar, but check rules)
    # Rules say: "non-negative integers between 0 and 99999"
    return int(ans) % 100000
