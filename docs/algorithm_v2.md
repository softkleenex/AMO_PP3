# Algorithm V2: Robust Tool-Integrated Reasoning

## Pipeline Overview

The `solver.py` implements a multi-stage reasoning pipeline designed to maximize accuracy by leveraging Python's computational accuracy while maintaining a semantic fallback.

### Stage 1: Code Generation (Primary)
- **Model**: Qwen2.5-Math-1.5B-Instruct
- **Prompt**: `format_prompt(..., "qwen_code")`
- **Action**: Generates a Python script.
- **Execution**: Run in `PythonREPL` with 7s timeout.
- **Success**: If `stdout` contains a number, parse and return.

### Stage 2: Self-Correction (Retry)
- **Trigger**: If Stage 1 results in `Error:` (SyntaxError, Timeout, ZeroDivision, etc.).
- **Prompt**: `format_prompt(..., "fix_code", error_msg=...)`. Feeds the specific error back to the model.
- **Action**: Generates corrected code.
- **Execution**: Run again.

### Stage 3: Chain-of-Thought (Fallback)
- **Trigger**: If Stage 2 fails (Error) OR if the extracted answer is `0` (indicating failure to print a result).
- **Prompt**: `format_prompt(..., "cot")`. "Solve step-by-step... put answer in \boxed{}".
- **Action**: Parse text output looking for `\boxed{}` or "The answer is X".

## Modules
- `src/solver.py`: Main logic.
- `src/utils.py`: `PythonREPL` and `extract_answer` (robust regex parsing).
- `src/data_loader.py`: Prompt templates.
