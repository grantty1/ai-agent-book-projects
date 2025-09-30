"""Generic execution tools: code interpreter and virtual terminal."""

import subprocess
import sys
import io
import traceback
from typing import Dict, Any
from contextlib import redirect_stdout, redirect_stderr
from llm_helper import LLMHelper
from config import Config


class ExecutionTools:
    """Generic execution tools with safety and result analysis."""
    
    def __init__(self, llm_helper: LLMHelper):
        """Initialize execution tools with LLM helper."""
        self.llm_helper = llm_helper
    
    async def code_interpreter(
        self,
        code: str,
        language: str = "python"
    ) -> Dict[str, Any]:
        """
        Execute code in a sandboxed environment.
        
        Args:
            code: Code to execute
            language: Programming language (currently only python supported)
            
        Returns:
            Result dictionary with output and analysis
        """
        if language != "python":
            return {
                "success": False,
                "error": f"Language {language} not supported yet"
            }
        
        # Verify syntax first
        if Config.AUTO_VERIFY_CODE:
            is_valid, error_msg = self.llm_helper.verify_code_syntax(code, language)
            if not is_valid:
                return {
                    "success": False,
                    "error": f"Syntax error: {error_msg}",
                    "verification": "failed"
                }
        
        # Check for dangerous operations
        if Config.REQUIRE_APPROVAL_FOR_DANGEROUS_OPS:
            dangerous_patterns = [
                'os.system', 'subprocess', 'eval', 'exec',
                'open(', '__import__', 'compile'
            ]
            
            if any(pattern in code for pattern in dangerous_patterns):
                approved, reason = self.llm_helper.request_approval(
                    "code_execution",
                    {
                        "code": code,
                        "detected_patterns": [p for p in dangerous_patterns if p in code]
                    }
                )
                
                if not approved:
                    return {
                        "success": False,
                        "error": f"Execution not approved: {reason}"
                    }
        
        # Execute code
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        
        try:
            # Create a restricted namespace
            namespace = {
                '__builtins__': __builtins__,
                'print': print
            }
            
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                exec(code, namespace)
            
            stdout_value = stdout_capture.getvalue()
            stderr_value = stderr_capture.getvalue()
            
            # Summarize stdout if it is too long (>10000 characters)
            if Config.AUTO_SUMMARIZE_COMPLEX_OUTPUT and len(stdout_value) > 10000:
                stdout_value = self.llm_helper.summarize_output(
                    "code_interpreter",
                    stdout_value
                )
            # Summarize stderr if it is too long (>10000 characters)
            if Config.AUTO_SUMMARIZE_COMPLEX_OUTPUT and len(stderr_value) > 10000:
                stderr_value = self.llm_helper.summarize_output(
                    "code_interpreter",
                    stderr_value
                )
            
            return {
                "success": True,
                "stdout": stdout_value,
                "stderr": stderr_value,
                "verification": "passed" if Config.AUTO_VERIFY_CODE else "skipped"
            }
            
        except Exception as e:
            error_output = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            return {
                "success": False,
                "error": error_output,
            }
    
    async def virtual_terminal(
        self,
        command: str,
        timeout: int = 30
    ) -> Dict[str, Any]:
        """
        Execute shell command in a virtual terminal.
        
        Args:
            command: Shell command to execute
            timeout: Timeout in seconds
            
        Returns:
            Result dictionary with output and analysis
        """
        # Check for dangerous commands
        if Config.REQUIRE_APPROVAL_FOR_DANGEROUS_OPS:
            dangerous_commands = [
                'rm -rf', 'dd', 'mkfs', 'format',
                '> /dev/', 'chmod -R', 'chown -R'
            ]
            
            if any(dangerous in command for dangerous in dangerous_commands):
                approved, reason = self.llm_helper.request_approval(
                    "terminal_command",
                    {
                        "command": command,
                        "detected_patterns": [p for p in dangerous_commands if p in command]
                    }
                )
                
                if not approved:
                    return {
                        "success": False,
                        "error": f"Command execution not approved: {reason}"
                    }
        
        # Execute command
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=Config.WORKSPACE_DIR
            )
            
            stdout = result.stdout
            stderr = result.stderr
            
            # Summarize long output (>10000 characters)
            if Config.AUTO_SUMMARIZE_COMPLEX_OUTPUT:
                if len(stdout) > 10000:
                    stdout = self.llm_helper.summarize_output(
                        "virtual_terminal",
                        stdout
                    )
                if len(stderr) > 10000:
                    stderr = self.llm_helper.summarize_output(
                        "virtual_terminal",
                        stderr
                    )
            
            response = {
                "success": result.returncode == 0,
                "returncode": result.returncode,
                "stdout": stdout,
                "stderr": stderr
            }
            return response
            
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": f"Command timed out after {timeout} seconds"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Command execution failed: {str(e)}"
            }
