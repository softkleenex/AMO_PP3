import os
import sys
import re
import math
import time
import queue
import threading
import contextlib
from typing import Optional, List, Dict, Any
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import pandas as pd
import polars as pl
from jupyter_client import KernelManager

# ==========================================
# 1. Configuration for P100 Compatibility
# ==========================================
class CFG:
    is_kaggle = os.path.exists("/kaggle/input")
    
    # Model Paths
    model_path = '/kaggle/input/qwen2-5-math-7b-instruct' if is_kaggle else 'Qwen/Qwen2.5-Math-1.5B-Instruct'
    
    # Device Management
    # P100 has 16GB VRAM. 7B model in FP16 takes ~14GB. 
    # We must be extremely careful with memory.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 # P100 does NOT support bfloat16
    
    # Generation Params
    temperature = 0.7
    top_p = 0.9
    max_new_tokens = 1024
    
    # Concurrency
    # Since we use Transformers (not vLLM), we can't easily do parallel inference on one GPU.
    # We will process attempts sequentially but keep the stateful sandbox logic.
    n_repetitions = 8  # Reduced from 16 to save time (Transformers is slower)
    max_turns = 5      # Max TIR turns
    
    system_prompt = (
        "You are an expert mathematician. Solve the problem by thinking step-by-step. "
        "You can write Python code to help you calculate or verify. "
        "Enclose your code in ```python\n...\n``` blocks. "
        "Always print the results of your calculations. "
        "Once you are absolutely sure of the final integer answer, place it inside \\boxed{}."
    )

# ==========================================
# 2. Stateful Jupyter Sandbox (Same as before)
# ==========================================
class PythonSandbox:
    _port_lock = threading.Lock()
    _next_port = 50000

    @classmethod
    def _get_ports(cls) -> list:
        with cls._port_lock:
            ports = list(range(cls._next_port, cls._next_port + 5))
            cls._next_port += 5
            return ports

    def __init__(self, timeout: float = 10):
        self.timeout = timeout
        ports = self._get_ports()
        env = os.environ.copy()
        self.km = KernelManager()
        self.km.shell_port, self.km.iopub_port, self.km.stdin_port, self.km.hb_port, self.km.control_port = ports
        self.km.start_kernel(env=env, extra_arguments=['--Application.log_level=CRITICAL'])
        self.client = self.km.blocking_client()
        self.client.start_channels()
        self.client.wait_for_ready(timeout=10)
        self.execute("import math, sympy\nfrom sympy import *\n")

    def execute(self, code: str) -> str:
        msg_id = self.client.execute(code, store_history=True, allow_stdin=False)
        stdout_parts, stderr_parts = [], []
        start_time = time.time()
        while True:
            if time.time() - start_time > self.timeout:
                self.km.interrupt_kernel()
                return "[Error] Timeout"
            try:
                msg = self.client.get_iopub_msg(timeout=0.5)
            except queue.Empty: continue
            if msg.get('parent_header', {}).get('msg_id') != msg_id: continue
            msg_type = msg.get('msg_type')
            if msg_type == 'stream':
                stdout_parts.append(msg['content']['text'])
            elif msg_type == 'error':
                stderr_parts.append('\n'.join(msg['content']['traceback']))
            elif msg_type == 'status' and msg['content']['execution_state'] == 'idle':
                break
        res = ''.join(stdout_parts + stderr_parts).strip()
        return re.sub(r'\x1b\[[0-9;]*m', '', res) if res else "[Success]"

    def close(self):
        try: self.client.stop_channels(); self.km.shutdown_kernel(now=True)
        except: pass

# ==========================================
# 3. Transformers Engine
# ==========================================
class HFEngine:
    def __init__(self, cfg: CFG):
        print(f"Loading Model: {cfg.model_path} on {cfg.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model_path,
            device_map="auto", # Automatically balances across GPU/CPU
            torch_dtype=cfg.dtype,
            trust_remote_code=True
        )
        self.cfg = cfg

    def generate(self, messages: List[Dict]) -> str:
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.cfg.max_new_tokens,
                do_sample=True,
                temperature=self.cfg.temperature,
                top_p=self.cfg.top_p,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, outputs)]
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

# ==========================================
# 4. Solver Logic
# ==========================================
class AIMSolver:
    def __init__(self, cfg: CFG):
        self.cfg = cfg
        self.engine = HFEngine(cfg)
        self.sandbox = PythonSandbox()

    def solve(self, problem: str) -> int:
        print(f"\n--- Solving: {problem[:50]}... ---")
        answers = []
        for i in range(self.cfg.n_repetitions):
            ans = self._solve_single(problem)
            if ans is not None:
                answers.append(ans)
                # Early stop if we have a strong consensus
                counts = Counter(answers)
                if counts.most_common(1)[0][1] >= 3: break
        
        if not answers: return 0
        final = Counter(answers).most_common(1)[0][0]
        print(f"Result: {final} (Votes: {dict(Counter(answers))})")
        return final

    def _solve_single(self, problem: str) -> Optional[int]:
        messages = [
            {"role": "system", "content": self.cfg.system_prompt},
            {"role": "user", "content": problem}
        ]
        for _ in range(self.cfg.max_turns):
            response = self.engine.generate(messages)
            messages.append({"role": "assistant", "content": response})
            
            # Answer extraction
            match = re.search(r'\\boxed\{([0-9,]+)\}', response)
            if match: return int(match.group(1).replace(',', '')) % 100000
            
            # Code execution
            code_match = re.search(r'```python\n(.*?)\n```', response, re.DOTALL)
            if code_match:
                output = self.sandbox.execute(code_match.group(1))
                messages.append({"role": "user", "content": f"```output\n{output}\n```"})
            else:
                break
        return None
