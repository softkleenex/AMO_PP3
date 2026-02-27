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
    def __init__(self):
        self.is_kaggle = os.path.exists("/kaggle/input")
        
        if self.is_kaggle:
            self.base_dir = "/kaggle/input"
            # Kaggle에 추가할 Qwen 모델 경로 (예시)
            self.model_path = "/kaggle/input/qwen2.5-math-7b-instruct/transformers/default/1" 
        else:
            self.base_dir = "./data"
            self.model_path = "Qwen/Qwen2.5-Math-1.5B-Instruct" # 로컬 테스트용 가벼운 모델

        # 하이퍼파라미터 (Top-tier 커널 참고)
        self.timeout_seconds = 10
        self.n_repetitions = 16  # 한 문제당 16번 다르게 풀기 시도
        self.temperature = 0.7   # 다양한 풀이 경로를 위한 높은 온도
        self.max_tokens = 2048
        self.gpu_memory_utilization = 0.95

# ==========================================
# 2. Code Execution Environment (Stateful)
# ==========================================
class CodeExecutor:
    """Thread-safe, optionally stateful Python executor."""
    def __init__(self, timeout: int = 5):
        self.timeout = timeout
        self.globals_dict = {} # 상태 유지를 위한 전역 변수 사전

    def execute(self, code: str, reset_state: bool = True) -> str:
        if reset_state:
            self.globals_dict = {}
            
        output_buffer = io.StringIO()
        
        def timeout_handler(signum, frame):
            raise TimeoutError("Execution timed out")

        use_timeout = False
        if hasattr(signal, 'SIGALRM'):
            try:
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(self.timeout)
                use_timeout = True
            except ValueError:
                pass # 백그라운드 스레드 무시
        
        try:
            with contextlib.redirect_stdout(output_buffer):
                # 기본적인 수학 라이브러리 자동 임포트
                exec("import math\nimport sympy\nimport numpy as np\n", self.globals_dict)
                exec(code, self.globals_dict)
        except TimeoutError:
            return "Error: Execution timed out."
        except Exception as e:
            return f"Error: {type(e).__name__}: {str(e)}"
        finally:
            if use_timeout and hasattr(signal, 'SIGALRM'):
                signal.alarm(0)
        
        return output_buffer.getvalue().strip()

# ==========================================
# 3. Model Interface (vLLM Batched)
# ==========================================
class LLMInterface(ABC):
    @abstractmethod
    def generate_batch(self, prompts: List[str]) -> List[str]:
        pass

class VLLMEngine(LLMInterface):
    """Real vLLM integration for high-throughput batch generation."""
    def __init__(self, config: CompetitionConfig):
        try:
            from vllm import LLM, SamplingParams
            print(f"Loading vLLM model from {config.model_path}...")
            # VRAM을 꽉 채워 쓰도록 설정
            self.model = LLM(
                model=config.model_path, 
                trust_remote_code=True,
                tensor_parallel_size=1,
                gpu_memory_utilization=config.gpu_memory_utilization,
                max_model_len=4096, # 컨텍스트 길이 최적화
                enforce_eager=True # Kaggle 환경 호환성
            )
            self.sampling_params = SamplingParams(
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                top_p=0.9,
                stop=["```\n", "User:", "<|im_end|>"]
            )
            self.is_mock = False
            print("vLLM loaded successfully.")
        except ImportError:
            print("⚠️ vLLM not installed. Falling back to MockLLM.")
            self.is_mock = True
            
    def generate_batch(self, prompts: List[str]) -> List[str]:
        if self.is_mock:
            import random
            return [f"The answer is \\boxed{{{random.randint(1, 100)}}}" for _ in prompts]
            
        outputs = self.model.generate(prompts, self.sampling_params, use_tqdm=False)
        return [output.outputs[0].text for output in outputs]

# ==========================================
# 4. Main Solver Logic
# ==========================================
class AIMSolver:
    def __init__(self, config: CompetitionConfig, llm: LLMInterface):
        self.config = config
        self.llm = llm
        
    def format_prompt(self, problem: str) -> str:
        """Qwen Math Instruct Template"""
        system = "You are an expert mathematician. Solve the problem step-by-step. If you write Python code, enclose it in ```python\n...\n```. Always put your final answer inside \\boxed{}."
        return f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{problem}<|im_end|>\n<|im_start|>assistant\n"

    def extract_answer(self, text: str) -> int:
        match = re.search(r'\\boxed\{([0-9,]+)\}', text)
        if match: 
            try:
                return int(match.group(1).replace(',', '')) % 100000
            except: pass
        
        # Fallback
        match = re.search(r'final answer is\s*([0-9,]+)', text, re.IGNORECASE)
        if match:
            try:
                return int(match.group(1).replace(',', '')) % 100000
            except: pass
        return -1

    def solve(self, problem_text: str) -> int:
        """
        Batched generation for Majority Voting.
        1. 16개의 프롬프트를 한 번에 생성
        2. 병렬로 응답 수집
        3. 가장 많이 나온 답 선택
        """
        # Create N identical prompts for sampling diverse paths (due to temp=0.7)
        prompts = [self.format_prompt(problem_text)] * self.config.n_repetitions
        
        # Batch Generate (vLLM handles this extremely efficiently)
        responses = self.llm.generate_batch(prompts)
        
        valid_answers = []
        for resp in responses:
            ans = self.extract_answer(resp)
            if ans >= 0:
                valid_answers.append(ans)
                
        if not valid_answers:
            print("No valid answers found. Returning 0.")
            return 0
            
        # Majority Vote
        counts = Counter(valid_answers)
        most_common, count = counts.most_common(1)[0]
        print(f"Votes: {dict(counts)} -> Selected: {most_common}")
        return most_common
