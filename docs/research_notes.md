# Research Notes: AIMO Progress Prize 3 Winning Approaches

## 1. Key Strategies
Based on previous AIMO Progress Prize 2 winning solutions, the following approaches are dominant:

### A. Models
- **DeepSeek-Math / DeepSeek-R1-Distill-Qwen-14B**: Widely used due to strong mathematical reasoning capabilities.
- **Quantization (AWQ)**: Essential for fitting large models into Kaggle's limited GPU memory (2x T4 or P100).
- **Fine-tuning**:
  - **SFT (Supervised Fine-Tuning)**: Using datasets like OpenR1 Math, NuminaMath.
  - **DPO (Direct Preference Optimization)**: Used to encourage concise reasoning and correct answers.

### B. Inference Techniques
- **Tool-Integrated Reasoning (TIR)**:
  - The model generates Python code to solve the problem (e.g., using `sympy`, `numpy`, or brute force loops).
  - Code is executed in a sandbox, and the output is fed back or used as the answer.
  - *Why?* LLMs are bad at arithmetic but good at logic; Python is perfect for arithmetic.
- **Majority Voting (Self-Consistency)**:
  - Generate $N$ solutions (e.g., $N=5$ to $N=20$).
  - Pick the most frequent answer.
  - *Optimization*: Stop early if the first $k$ answers agree.
- **Iterative Refinement / Redrafting**:
  - Generate a solution, then ask the model to "review your work and correct errors".

## 2. Recommended Pipeline for PP3
1.  **Preprocessing**: None (problems are text/LaTeX).
2.  **Solver Core**:
    - **Prompt**: "You are an expert mathematician. Solve the problem by writing a Python script..."
    - **Execution**: Run the generated code.
    - **Fallback**: If code fails, try pure text reasoning (Chain-of-Thought).
3.  **Ensemble**:
    - Run the solver $k$ times.
    - Select majority vote.

## 3. Libraries to Explore
- `sympy`: For symbolic math (calculus, algebra).
- `networkx`: For graph theory problems.
- `itertools`: For combinatorics.

## 4. Action Items
- [ ] Implement a `PythonREPL` class in `src/utils.py` to execute model-generated code safely.
- [ ] Update `src/solver.py` to use a "Prompt -> Code -> Execute" loop.
- [ ] Find a small, high-quality Math LLM compatible with Kaggle (e.g., `deepseek-math-7b-rl`, `Qwen2.5-Math-7B`).
