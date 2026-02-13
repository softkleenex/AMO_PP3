import sys
import io
import contextlib
import signal
import re

class PythonREPL:
    """
    A simple class to execute Python code and capture its output.
    Useful for Tool-Integrated Reasoning (TIR) where the LLM generates code.
    """
    def __init__(self, timeout=5):
        self.timeout = timeout

    def execute(self, code: str) -> str:
        """
        Executes the provided Python code and returns the stdout.
        """
        output_buffer = io.StringIO()
        
        def timeout_handler(signum, frame):
            raise TimeoutError("Execution timed out")

        if hasattr(signal, 'SIGALRM'):
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(self.timeout)
        
        try:
            with contextlib.redirect_stdout(output_buffer):
                exec_globals = {}
                exec(code, exec_globals)
        except TimeoutError:
            return "Error: Execution timed out."
        except Exception as e:
            return f"Error: {str(e)}"
        finally:
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)
        
        return output_buffer.getvalue().strip()

def extract_answer(text: str) -> int:
    """
    Extracts the final integer answer from text.
    Prioritizes \boxed{...}, then last number.
    """
    if not isinstance(text, str):
        return 0
        
    # 1. Check for \boxed{number}
    boxed_match = re.search(r'\\boxed\{(\d+)\}', text)
    if boxed_match:
        return int(boxed_match.group(1))
    
    # 2. Check for "The answer is: number"
    answer_match = re.search(r'answer is\W*(\d+)', text, re.IGNORECASE)
    if answer_match:
        return int(answer_match.group(1))
        
    # 3. Fallback: Last number found (risky but common fallback)
    numbers = re.findall(r'-?\d+', text)
    if numbers:
        try:
            val = int(numbers[-1])
            if 0 <= val <= 999999: # Reasonable range check
                return val
        except:
            pass
            
    return 0
