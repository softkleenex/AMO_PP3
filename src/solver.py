import re
from .utils import CodeExecutor, extract_answer
from .data_loader import format_prompt

class AIMSolver:
    """
    Main Solver Logic for AI Mathematical Olympiad.
    Implements Tool-Integrated Reasoning (TIR) with Self-Correction and CoT Fallback.
    """
    def __init__(self, llm_generate_func, timeout=7):
        self.generate = llm_generate_func
        self.executor = CodeExecutor(timeout=timeout)

    def extract_python_code(self, text: str) -> str:
        """Parses markdown-wrapped Python code blocks."""
        match = re.search(r'```python\n(.*?)\n```', text, re.DOTALL)
        return match.group(1) if match else text

    def solve_single_attempt(self, problem_text: str) -> int:
        """
        Single solving attempt: Code Gen -> Retry -> CoT Fallback.
        """
        # 1. Initial Python Code Generation
        prompt = format_prompt(problem_text, "qwen_code")
        response = self.generate(prompt)
        code = self.extract_python_code(response)
        output = self.executor.execute(code)
        
        # 2. Self-Correction Retry if Error
        if "Error" in output:
            prompt = format_prompt(problem_text, "fix_code", error_msg=output)
            response = self.generate(prompt)
            code = self.extract_python_code(response)
            output = self.executor.execute(code)

        # 3. Final Answer Extraction
        try:
            # If code printed a number, use it as the primary source
            ans = int(float(output.strip()))
            return ans % 100000
        except:
            pass

        # 4. Fallback to step-by-step Chain-of-Thought
        prompt = format_prompt(problem_text, "cot")
        response = self.generate(prompt)
        return extract_answer(response)

    def solve(self, problem_text: str, n_repetitions: int = 1) -> int:
        """
        Solves with optional majority voting.
        """
        from collections import Counter
        
        answers = []
        for _ in range(n_repetitions):
            ans = self.solve_single_attempt(problem_text)
            if ans >= 0:
                answers.append(ans)
        
        if not answers:
            return 0
            
        # Majority Vote (Self-Consistency)
        counts = Counter(answers)
        most_common, count = counts.most_common(1)[0]
        return most_common
