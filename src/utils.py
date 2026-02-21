import sys
import io
import contextlib
import signal
import re

class CodeExecutor:
    """
    A robust class to execute Python code and capture its output.
    Includes timeout handling and is thread-safe for background worker usage.
    """
    def __init__(self, timeout: int = 7):
        self.timeout = timeout

    def execute(self, code: str) -> str:
        """
        Executes the provided Python code and returns the stdout.
        Gracefully handles timeouts even when called from non-main threads.
        """
        output_buffer = io.StringIO()
        
        def timeout_handler(signum, frame):
            raise TimeoutError("Execution timed out")

        # Signal handling (timeouts) only works in the main thread.
        # Background worker threads (e.g. Kaggle Inference Server) will skip this to avoid crash.
        use_timeout = False
        if hasattr(signal, 'SIGALRM'):
            try:
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(self.timeout)
                use_timeout = True
            except ValueError:
                # Likely running in a background thread
                pass
        
        try:
            with contextlib.redirect_stdout(output_buffer):
                # Execute in an isolated global scope
                exec_globals = {}
                exec(code, exec_globals)
        except TimeoutError:
            return "Error: Execution timed out."
        except Exception as e:
            return f"Error: {str(e)}"
        finally:
            if use_timeout and hasattr(signal, 'SIGALRM'):
                signal.alarm(0) # Disable alarm
        
        return output_buffer.getvalue().strip()

def extract_answer(text: str) -> int:
    """
    Robust answer extraction from LLM response text.
    Order of priority:
    1. \boxed{number}
    2. Last found sequence of digits
    """
    if not isinstance(text, str):
        return 0
        
    # 1. Look for LaTeX boxed content
    boxed_match = re.search(r'\\boxed\{(\d+)\}', text)
    if boxed_match:
        return int(boxed_match.group(1))
    
    # 2. Fallback: Find all digit sequences and take the last one
    # This is a common heuristic for math problems.
    numbers = re.findall(r'\d+', text)
    if numbers:
        try:
            val = int(numbers[-1])
            return val % 100000 # AIMO range constraint
        except:
            pass
            
    return 0
