import sys
import io
import contextlib
import signal

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
        
        Args:
            code (str): The Python code to execute.
            
        Returns:
            str: The captured stdout, or error message.
        """
        # Create a string buffer to capture stdout
        output_buffer = io.StringIO()
        
        # Handler for timeout
        def timeout_handler(signum, frame):
            raise TimeoutError("Execution timed out")

        # Register signal for timeout (only works on Unix-like systems)
        if hasattr(signal, 'SIGALRM'):
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(self.timeout)
        
        try:
            # Redirect stdout to our buffer
            with contextlib.redirect_stdout(output_buffer):
                # Create a restricted globals dictionary if needed, 
                # but for Kaggle competitions full access is usually fine/required.
                exec_globals = {}
                exec(code, exec_globals)
        except TimeoutError:
            return "Error: Execution timed out."
        except Exception as e:
            return f"Error: {str(e)}"
        finally:
            # Disable alarm
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)
        
        return output_buffer.getvalue().strip()

if __name__ == "__main__":
    # Simple test
    repl = PythonREPL()
    code = "print(1 + 1)"
    result = repl.execute(code)
    print(f"Code: {code}
Result: {result}")
