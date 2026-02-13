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

## 4. Submission
- Use the official Kaggle API notebook template.
- Ensure the `aimo` evaluation API is handled correctly.
- Submit via UI or CLI: `kaggle competitions submit -c ai-mathematical-olympiad-progress-prize-3 -f submission.csv -m "Message"` (Note: Code competitions usually require Notebook submission, so we submit the notebook).

## 5. Recording Results
- Keep a `submissions/log.md` to track experiment ID, CV score, and LB score.
