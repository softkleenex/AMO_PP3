# AIMO Prize 3 Submission Protocol

## 🚨 Critical: The "Two Modes" Trap

Kaggle Code Competitions have two distinct runtime environments. Failing to handle **both** will result in "Submission File Not Found" errors.

| Mode | Trigger | Environment | API Status | Requirement |
| :--- | :--- | :--- | :--- | :--- |
| **Validation** | "Save Version" (Commit) | Interactive/Batch Session | **MISSING** or Local | **MUST** create `submission.parquet` (dummy is fine) |
| **Inference** | "Submit" Button | Private Re-run | **PRESENT** | **MUST** use `InferenceServer` to generate real predictions |

**Common Failure:** The code checks for the API, doesn't find it (during Validation), skips the prediction loop, and exits *without* creating `submission.parquet`. Kaggle then rejects the notebook because the output file is missing.

---

## ✅ The Robust Code Pattern

Use this exact structure in the final cell of your submission notebook. It handles both modes and finds the API module wherever it hides.

```python
import sys
import os
import glob
import pandas as pd

# ==========================================
# AIMO 3 API: Robust Setup
# ==========================================

# 1. Locate the API Module
# The API file 'aimo_3_inference_server.py' moves around depending on the environment.
api_files = glob.glob("/kaggle/input/**/aimo_3_inference_server.py", recursive=True)
if not api_files:
    # Fallback for local testing
    api_files = glob.glob("data/**/aimo_3_inference_server.py", recursive=True)

aimo_server_mod = None

if api_files:
    api_path = os.path.dirname(api_files[0])
    # Case A: It's inside a package (e.g., kaggle_evaluation)
    if os.path.basename(api_path) == 'kaggle_evaluation':
        parent_dir = os.path.dirname(api_path)
        if parent_dir not in sys.path: sys.path.append(parent_dir)
        try:
            from kaggle_evaluation import aimo_3_inference_server as aimo_server_mod
        except ImportError: pass
    
    # Case B: It's a standalone module
    if not aimo_server_mod:
        if api_path not in sys.path: sys.path.append(api_path)
        try:
            import aimo_3_inference_server as aimo_server_mod
        except ImportError: pass

# 2. Define Callback
def predict(*args, **kwargs):
    """
    Callback triggered by the Gateway.
    Args: usually a tuple, where args[0] is the problem (or a DataFrame containing it).
    """
    try:
        input_data = args[0] if args else None
        problem_text = "What is 0+0?" # Default
        
        # Extraction Logic for various data types (Polars/Pandas/Str)
        if hasattr(input_data, 'columns') and 'problem' in input_data.columns:
            problem_text = str(input_data['problem'][0])
        elif isinstance(input_data, str):
            problem_text = input_data
            
        # --- YOUR SOLVER CALL HERE ---
        # answer = solver.solve(problem_text)
        return 0 # Return integer answer
    except Exception as e:
        print(f"Predict Error: {e}")
        return 0

# 3. Execution Logic
if aimo_server_mod:
    print("Initializing Inference Server...")
    server = aimo_server_mod.AIMO3InferenceServer(predict)
    
    if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
        # [Mode: Inference]
        print("Starting Server (Blocking)...")
        server.serve()
    else:
        # [Mode: Validation / Local]
        print("Running Local Gateway...")
        try:
            server.run_local_gateway()
            print("✅ Local Gateway run finished.")
        except Exception as e:
            print(f"❌ Local Gateway failed: {e}")
            # Fallback is handled below
else:
    print("⚠️ API not found.")

# 4. FINAL FAIL-SAFE (CRITICAL)
# Ensure submission.parquet exists no matter what happened above.
if not os.path.exists('submission.parquet'):
    print("⚠️ Generating dummy submission.parquet for validation pass.")
    pd.DataFrame({'id': ['test_id'], 'answer': [0]}).to_parquet('submission.parquet', index=False)
```

## Troubleshooting

### "Submission File Not Found"
- **Cause:** Your code skipped the generation step because it couldn't find the API, or the API crashed.
- **Fix:** Ensure the **FAIL-SAFE** block (Step 4 above) is present at the very end of your notebook.

### "Must pass at least one endpoint listener"
- **Cause:** You tried to use `env.iter_test()` (the old API style).
- **Fix:** You must instantiate `AIMO3InferenceServer(predict)`.

### "Exception calling application"
- **Cause:** Your `predict` function raised an unhandled exception.
- **Fix:** Wrap the entire body of `predict` in a `try-except` block and return a default value (e.g., `0`) on error.
