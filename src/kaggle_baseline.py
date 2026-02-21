import os
import sys
import re
import io
import contextlib
import signal
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from collections import Counter

# ==========================================
# 1. Configuration & Environment Handling
# ==========================================
class CompetitionConfig:
    """
    Central configuration that adapts to Local vs Kaggle environment.
    """
    def __init__(self):
        self.is_kaggle = os.path.exists("/kaggle/input")
        
        if self.is_kaggle:
            self.base_dir = "/kaggle/input"
            self.work_dir = "/kaggle/working"
            # Path to the specific model directory in Kaggle
            # User must add 'Qwen/Qwen2.5-Math-7B-Instruct' as a model/dataset
            self.model_path = "/kaggle/input/qwen2.5-math-7b-instruct" 
        else:
            self.base_dir = "./data"
            self.work_dir = "./submissions"
            self.model_path = "./models/qwen2.5-math" # Local path

        # Common Settings
        self.timeout_seconds = 7
        self.max_retries = 2
        self.n_repetitions = 5 # Majority Voting Count
        self.debug = True

    def get_dataset_path(self, filename: str) -> str:
        """Finds a file in common locations to avoid path errors."""
        possible_paths = [
            os.path.join(self.base_dir, filename),
            os.path.join(self.base_dir, "ai-mathematical-olympiad-prize", filename),
            filename # relative path fallback
        ]
        for path in possible_paths:
            if os.path.exists(path):
                return path
        if self.debug:
            print(f"Warning: File {filename} not found in {possible_paths}")
        return filename

# ==========================================
# 2. Abstract Model Interface
# ==========================================
class LLMInterface(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        pass

class MockLLM(LLMInterface):
    """Used for testing pipeline without loading heavy models."""
    def generate(self, prompt: str) -> str:
        if "step-by-step" in prompt:
            return "The answer is \\boxed{42}."
        # Randomize slightly for voting test
        import random
        return f"print({random.choice([42, 42, 42, 0])})"

# Placeholder for real vLLM or HF implementation
class LocalVLLM(LLMInterface):
    def __init__(self, model_path: str):
        # try:
        #     from vllm import LLM, SamplingParams
        #     self.model = LLM(model=model_path, trust_remote_code=True)
        #     self.params = SamplingParams(temperature=0.7, max_tokens=1024)
        # except ImportError:
        #     print("vLLM not installed. Using Mock.")
        pass

    def generate(self, prompt: str) -> str:
        # output = self.model.generate([prompt], self.params)
        # return output[0].outputs[0].text
        return "print(0) # Placeholder for vLLM"

# ==========================================
# 3. Code Execution Environment
# ==========================================
class CodeExecutor:
    def __init__(self, timeout: int = 5):
        self.timeout = timeout

    def execute(self, code: str) -> str:
        output_buffer = io.StringIO()
        
        def timeout_handler(signum, frame):
            raise TimeoutError("Execution timed out")

        # Signal handling only works in main thread usually
        # In AIMO 3, predict() is called in a worker thread, so this will crash if not handled.
        use_timeout = False
        if hasattr(signal, 'SIGALRM'):
            try:
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(self.timeout)
                use_timeout = True
            except ValueError:
                # Likely running in a background thread (gRPC worker)
                # Skip timeout protection to avoid crash
                pass
        
        try:
            with contextlib.redirect_stdout(output_buffer):
                # Unsafe exec - in Kaggle this is isolated, locally be careful
                exec_globals = {}
                exec(code, exec_globals)
        except TimeoutError:
            return "Error: Execution timed out."
        except Exception as e:
            return f"Error: {str(e)}"
        finally:
            if use_timeout and hasattr(signal, 'SIGALRM'):
                signal.alarm(0)
        
        return output_buffer.getvalue().strip()

# ==========================================
# 4. Main Solver Logic
# ==========================================
class AIMSolver:
    def __init__(self, config: CompetitionConfig, llm: LLMInterface):
        self.config = config
        self.llm = llm
        self.executor = CodeExecutor(timeout=config.timeout_seconds)

    def extract_code(self, text: str) -> str:
        match = re.search(r'```python\n(.*?)\n```', text, re.DOTALL)
        return match.group(1) if match else text

    def extract_answer(self, text: str) -> int:
        # 1. Boxed
        match = re.search(r'\\boxed\{(\d+)\}', text)
        if match: return int(match.group(1))
        # 2. Last number fallback
        numbers = re.findall(r'\d+', text)
        if numbers:
            try:
                val = int(numbers[-1])
                return val % 1000 
            except: pass
        return -1 # Indicator for no answer found

    def format_prompt(self, problem: str, mode: str = "code", error: str = None) -> str:
        if mode == "code":
            examples = (
                "Problem: Find the sum of the first 10 integers.\n"
                "Code:\n"
                "```python\n"
                "print(sum(range(1, 11)))\n"
                "```\n\n"
                "Problem: What is the remainder when 123 is divided by 10?\n"
                "Code:\n"
                "```python\n"
                "print(123 % 10)\n"
                "```\n\n"
            )
            return (
                "Write a Python script to solve this math problem. "
                "Print the answer at the end.\n\n"
                f"{examples}"
                f"Problem: {problem}\n\nCode:"
            )
        elif mode == "fix":
            return (
                f"Previous code error: {error}\n"
                "Fix the code and solve:\n\nCode:"
            )
        return f"Solve step-by-step. Put answer in \\boxed{{}}.\nProblem: {problem}"

    def _solve_single(self, problem_text: str) -> int:
        # Strategy: Code Gen -> Retry -> CoT
        
        # 1. Initial Code Attempt
        prompt = self.format_prompt(problem_text, "code")
        response = self.llm.generate(prompt)
        code = self.extract_code(response)
        output = self.executor.execute(code)
        
        # 2. Retry if Error
        if "Error" in output:
            prompt = self.format_prompt(problem_text, "fix", error=output)
            response = self.llm.generate(prompt)
            code = self.extract_code(response)
            output = self.executor.execute(code)

        # 3. Answer Extraction
        try:
            # If code printed a number, use it
            ans = int(float(output.strip()))
            return ans % 1000 # Modulo check
        except:
            pass

        # 4. Fallback to CoT
        prompt = self.format_prompt(problem_text, "cot")
        response = self.llm.generate(prompt)
        return self.extract_answer(response)

    def solve(self, problem_text: str) -> int:
        """
        Executes majority voting (self-consistency).
        """
        answers = []
        for _ in range(self.config.n_repetitions):
            ans = self._solve_single(problem_text)
            if ans >= 0: # valid answer
                answers.append(ans)
        
        if not answers:
            return 0
            
        # Majority Vote
        counts = Counter(answers)
        most_common, count = counts.most_common(1)[0]
        return most_common
