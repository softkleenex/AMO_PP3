# AI Mathematical Olympiad (AIMO) Prize 3 - AI Solver

![Kaggle](https://img.shields.io/badge/Kaggle-AIMO%203-blue?logo=kaggle)
![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![LLM](https://img.shields.io/badge/Model-Qwen2.5--Math--7B-green)

This repository contains the source code, experimentation notebooks, and submission pipeline for the **AI Mathematical Olympiad (AIMO) Progress Prize 3**. The goal is to solve international-level mathematics problems (AIME/IMO) using AI models.

---

## 📺 Project Video
- **YouTube:** [AIMO 3 Solver Strategy & Demo](https://youtu.be/r7_SRmbvdk8)
- **Local:** `assets/solution_demo.mp4`

---

## 🏆 Competition Overview
- **Name:** [AI Mathematical Olympiad - Progress Prize 3](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3)
- **Goal:** Build an AI system capable of solving international math olympiad problems.
- **Problem Set:** 110 problems, LaTeX format, non-negative integer answers (0-99999).
- **Evaluation:** Average accuracy on a hidden private test set.
- **Competition Period:** Jan 2026 – April 15, 2026.

---

## 🗓️ Development Period
- **Period:** Feb 13, 2026 – Feb 21, 2026
- **Status:** Initial Baseline (OOP) + vLLM Integration + API Protocol Fixes completed.

---

## 📁 Directory Structure
```text
.
├── assets/             # Demo videos and project assets
├── data/               # Local test data and Kaggle evaluation modules (Ignored)
├── docs/               # Technical documentation and algorithm notes
│   ├── algorithm_v2.md     # Solver logic (TIR + CoT Fallback)
│   ├── submission_protocol.md # Kaggle API implementation details
│   └── workflow.md         # Development & Deployment guide
├── notebooks/          # Experimentation and Kaggle submission notebooks
├── scripts/            # Utility, diagnostic, and deployment scripts
├── src/                # Core Python modules (Solver, Executor, Utils)
│   ├── kaggle_baseline.py  # Robust OOP Baseline for Kaggle
│   └── solver.py           # Core math solving logic
└── submissions/        # Submission logs and local evaluation results
```

---

## 🚀 Key Features & Implementation
- **Tool-Integrated Reasoning (TIR):** Uses Python code execution to solve complex mathematical steps.
- **CoT Fallback:** Automatically falls back to Chain-of-Thought reasoning if code execution fails.
- **Majority Voting (Self-Consistency):** Runs multiple independent attempts and selects the most frequent answer.
- **Robust Kaggle API Integration:** Custom monkey-patches for the `InferenceServer` pattern to handle data passing and column naming bugs.

---

## 🛠️ Usage
1. **Local Validation:** Use `src/kaggle_baseline.py` to run the solver against local datasets.
2. **Kaggle Deployment:**
   - Update the utility script dataset: `make deploy` (via `scripts/deploy.sh`).
   - Push the notebook: `kaggle kernels push -p notebooks/`.

---

## 📜 License
This project is for educational and portfolio purposes. Data and API components are subject to Kaggle Competition Rules.
