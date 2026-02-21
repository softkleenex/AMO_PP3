# AI Mathematical Olympiad (AIMO) Prize 3 - AI Solver

![Kaggle](https://img.shields.io/badge/Kaggle-AIMO%203-blue?logo=kaggle)
![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![LLM](https://img.shields.io/badge/Model-Qwen2.5--Math--7B-green)

This repository contains the source code, experimentation notebooks, and submission pipeline for the **AI Mathematical Olympiad (AIMO) Progress Prize 3**. The goal is to build an autonomous AI system capable of solving international-level mathematics problems (AIME/IMO level).

---

## 🏆 Competition Overview
- **Name:** [AI Mathematical Olympiad - Progress Prize 3](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3)
- **Goal:** Solve 110 international math olympiad problems.
- **Answer Format:** Non-negative integers (0-99999).
- **Evaluation:** Accuracy on a hidden private test set.
- **Competition Period:** Jan 2026 – April 15, 2026.

---

## 🚀 Key Features & Implementation
- **Tool-Integrated Reasoning (TIR):** Bridges the gap between LLM reasoning and mathematical precision by generating and executing Python code.
- **Self-Correction Logic:** Automatically captures code execution errors and prompts the LLM to fix its own code.
- **CoT Fallback:** Provides a robust fallback mechanism using Chain-of-Thought reasoning if symbolic/code methods fail.
- **Thread-Safe Code Execution:** Custom executor designed to handle timeouts and signal handling in multi-threaded environments (Kaggle Inference Server).
- **Kaggle API Integration:** Includes advanced monkey-patches for the `InferenceServer` pattern to ensure reliable data passing.

---

## 📁 Directory Structure
```text
.
├── docs/               # Technical documentation
│   ├── algorithm_v2.md     # Solver logic details (TIR + CoT)
│   ├── submission_protocol.md # Kaggle API implementation & Fail-safes
│   └── workflow.md         # Development lifecycle guide
├── src/                # Core Python modules
│   ├── solver.py           # AIMSolver class (Main pipeline)
│   ├── utils.py            # Code execution & answer extraction
│   ├── data_loader.py      # Prompt formatting & data handling
│   └── kaggle_baseline.py  # Self-contained OOP utility for Kaggle
├── notebooks/          # Kaggle submission template
├── scripts/            # Deployment & evaluation utilities
├── data/               # Local test datasets (Ignored)
└── README.md           # Project overview
```

---

## 🛠️ Usage
1. **Local Validation:** 
   ```bash
   python3 scripts/evaluate_v2.py
   ```
2. **Kaggle Deployment:**
   - Update source code: `make deploy msg="Your commit message"`
   - Push notebook: `kaggle kernels push -p notebooks/`

---

## 🗓️ Development Period
- **Phase 1:** Feb 13 – Feb 21, 2026 (Initial Baseline & API Infrastructure)
- **Status:** Architecture verified, API issues resolved, and successful submission completed.

---

## 📜 License
This project is for educational and portfolio purposes. Data and competition-specific API components are subject to Kaggle Rules.
