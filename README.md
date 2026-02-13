# AI Mathematical Olympiad - Progress Prize 3

## Competition Overview
**Goal:** Solve international-level mathematics problems using AI.
**Problem Type:** 110 problems (AIME/IMO level), text-only (LaTeX), answers are non-negative integers (0-99999).
**Evaluation:** Average accuracy on a private test set (rerun twice for robustness).
**Submission:** Python Code Competition via API.

## Project Structure
- `data/`: Competition datasets (ignored by git).
- `notebooks/`: Jupyter notebooks for experiments.
- `src/`: Source code for solution pipelines.
- `submissions/`: Generated submission files and logs.
- `docs/`: Project documentation and workflows.

## Workflow
See `docs/workflow.md` for details on the development pipeline using Gemini CLI, Kaggle, and GitHub.
See `docs/algorithm_v2.md` for the detailed logic of the solver (TIR + CoT Fallback).
