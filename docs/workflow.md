# Development Workflow

This document outlines the pipeline for working on the AI Mathematical Olympiad using Gemini CLI, GitHub, and Kaggle.

## 1. Setup & Tools
- **Gemini CLI:** Assistant for coding, refactoring, and documentation.
- **GitHub:** Version control and remote storage.
- **Kaggle API:** For dataset management and submission.

## 2. Development Cycle
1.  **Local Dev:** Write core logic in `src/`. Use standard Python modules.
2.  **Experiment:** Use `notebooks/` to import `src` modules and test on sample data.
3.  **Commit:** `git commit -am "Update logic"` -> `git push`.

## 3. Kaggle Deployment Strategy
Since internet access is disabled during inference, external libraries must be uploaded as Datasets.

**Option A: Zip Source Code**
1.  Zip the `src/` folder: `zip -r src.zip src/`
2.  Upload/Update Kaggle Dataset:
    ```bash
    kaggle datasets version -p path/to/dataset -m "Update source"
    ```
3.  In the Kaggle Inference Notebook, add the dataset and:
    ```python
    import sys
    sys.path.append('/kaggle/input/my-source-code-dataset/src')
    import my_module
    ```

**Option B: Single Script (Simpler)**
- Use a build script to concatenate `src/` files into a single submission notebook if the logic is simple enough.

**Option C: OOP Baseline Utility Script (Recommended)**
- We have created a robust, self-contained baseline in `src/kaggle_baseline.py`.
- **Steps:**
    1. Upload `src/kaggle_baseline.py` as a Kaggle Dataset (e.g., named `aimo-pp3-modules`).
    2. In your Kaggle Notebook, add this dataset.
    3. Import the baseline:
       ```python
       import sys
       sys.path.append('/kaggle/input/aimo-pp3-modules')
       from kaggle_baseline import CompetitionConfig, AIMSolver
       ```
    4. The code automatically detects if it's running Locally (using `./data`) or on Kaggle (using `/kaggle/input`).

## 4. Submission Protocol (AIMO 3 Specific)

### API Pattern: Inference Server
The AIMO 3 competition uses the **Inference Server** pattern, unlike previous iterations that used `iter_test`.

**Correct Implementation:**
```python
# Import the API module (often nested deep in kaggle_evaluation)
import aimo_3_inference_server

# Define a callback function
def predict(*args, **kwargs):
    problem_text = str(args[0]) # Extract problem
    answer = solve(problem_text)
    return answer

# Initialize Server
server = aimo_3_inference_server.AIMO3InferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    # Blocking call for actual inference
    server.serve()
else:
    # Local validation / Save Version step
    # This generates submission.parquet automatically using local data
    server.run_local_gateway()
```

### Critical Submission Requirements
1.  **`submission.parquet` MUST exist:** Even if the API fails to load or during the "Save Version" step (validation mode), the notebook **must** produce a `submission.parquet` file.
2.  **Fail-safe Logic:** Always wrap the API interaction in a `try-except` block. If the API is missing or crashes, **create a dummy `submission.parquet`** manually to ensure the "Output File" check passes.
    ```python
    try:
        if server: server.run_local_gateway()
    except Exception:
        # Fallback
        pd.DataFrame({'id': ['test'], 'answer': [0]}).to_parquet('submission.parquet')
    ```

## 5. Recording Results
- Keep a `submissions/log.md` to track experiment ID, CV score, and LB score.
