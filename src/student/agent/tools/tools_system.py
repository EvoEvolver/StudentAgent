import subprocess
import traceback

from pydantic_ai import RunContext

from .tools import RaspaTool
from ..utils import *



class SystemAgent(RaspaTool):
    def __init__(self, path=None):
        name = "SystemAgent"
        description = """
        Use this tool to execute system tasks using natural language instructions.
        This tool can read files, write files, and execute system commands based on your query.
        Provide a natural language instruction describing what you want to accomplish.
        You MUST make the query very specific and carry all the necessary information to perform the task.
        """
        super().__init__(name, description, path)

    def run(self, query: str, timeout: int = 300):
        """
        Execute a natural language query using Claude CLI.

        Args:
            query: Natural language instruction for file operations or system commands
            timeout: Maximum time in seconds to wait for completion (default: 300)

        Returns:
            The output from Claude CLI
        """
        try:
            # Get the working directory path
            work_dir = self.get_path(full=True)

            print(f"[SystemAgent] Executing query in: {work_dir}")
            print(f"[SystemAgent] Query: {query[:100]}..." if len(query) > 100 else f"[SystemAgent] Query: {query}")

            # Execute claude command with the query
            # IMPORTANT: Close stdin to prevent child process from waiting for input
            process = subprocess.Popen(
                ['claude', '--dangerously-skip-permissions', '-p', query],
                cwd=work_dir,
                text=True,
                stdin=subprocess.DEVNULL,  # Close stdin to prevent hanging
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,  # Unbuffered
                universal_newlines=True
            )

            print(f"[SystemAgent] Process started (PID: {process.pid}), waiting up to {timeout}s...")

            try:
                stdout, stderr = process.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                print(f"[SystemAgent] Process timed out after {timeout}s, terminating...")
                process.kill()
                stdout, stderr = process.communicate()
                return self.get_output(
                    e=f"Command timed out after {timeout} seconds.\nPartial stdout: {stdout[:500]}\nPartial stderr: {stderr[:500]}"
                )

            print(f"[SystemAgent] Process completed with return code: {process.returncode}")

            if process.returncode != 0:
                error_msg = f"Command failed with return code {process.returncode}"
                if stderr:
                    error_msg += f"\nError: {stderr}"
                if stdout:
                    error_msg += f"\nStdout: {stdout}"
                return self.get_output(e=error_msg)

            # Return the output
            return self.get_output(content=stdout if stdout else "(No output produced)")

        except FileNotFoundError:
            return self.get_output(
                e="Claude CLI not found. Please ensure 'claude' command is installed and available in PATH.")
        except Exception as e:
            return self.get_output(e=f"Error executing system agent: {str(e)}\n{traceback.format_exc()}")


class ReportToHuman(RaspaTool):
    """Tool to generate markdown reports with datetime-based filenames."""

    def __init__(self, path=None):
        name = "report_to_human"
        description = """
        Use this tool to report to human your result in markdown when you finished or failed your task.
        """
        super().__init__(name, description, path)

    def run(self, report_content: str):
        """
        Generate a markdown report with a datetime-based filename.

        Args:
            report_content: The content of the markdown report to write

        Returns:
            Success message with the filename, or error message
        """
        try:
            from datetime import datetime

            # Generate filename with current datetime
            timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
            filename = f"report_{timestamp}.md"

            # Get the full path for saving the report
            full_path = self.get_path(full=True)
            os.makedirs(full_path, exist_ok=True)

            # Full file path
            file_path = os.path.join(full_path, filename)

            # Write the report content to the file
            with open(file_path, "w") as f:
                f.write(report_content)

            result = f"Successfully generated markdown report: {filename}\nLocation: {file_path}"
            return self.get_output(content=result)

        except Exception as e:
            return self.get_output(e=f"Error generating markdown report: {str(e)}")

def report_to_human(ctx: RunContext, report_content: str):
    """
    Use this tool to report to human your result in markdown when you finished or failed your task.
    """
    path = ctx.deps["cwd"]
    return ReportToHuman(path=path).run(report_content)

class AskHuman(RaspaTool):
    """Tool to ask questions to a human user via console input."""

    def __init__(self, path=None):
        name = "ask_human"
        description = """
        Use this tool when you need to ask the human user a question during execution.
        This is useful when you need clarification, additional information, or decisions from the user.
        Provide a clear question, and the tool will prompt the user for input via the console.
        """
        super().__init__(name, description, path)

    def run(self, question: str):
        """
        Ask a question to the human user and get their response.

        Args:
            question: The question to ask the user

        Returns:
            The user's response from console input
        """
        try:
            # Print the question to console
            print(f"\n[AGENT QUESTION] {question}")
            print("[Waiting for your input...]")

            # Get input from user
            user_response = input("Your answer: ").strip()

            if not user_response:
                return self.get_output(e="No response provided by user.")

            result = f"Question: {question}\nUser's answer: {user_response}"
            return self.get_output(content=result)

        except EOFError:
            return self.get_output(e="Input stream closed. Cannot read from console.")
        except Exception as e:
            return self.get_output(e=f"Error getting user input: {str(e)}")


def ask_human(ctx: RunContext, question: str):
    """
    Use this tool when you need to ask the human user a question during execution.
    This is useful when you need clarification, additional information, or decisions from the user.
    Provide a clear question, and the tool will prompt the user for input via the console.
    """
    return AskHuman().run(question)

class ImageQuestionTool(RaspaTool):
    """Tool to ask questions about images using OpenAI's vision API."""

    def __init__(self, path=None):
        name = "ask_image_question"
        description = """
        Ask a question about an image using AI vision capabilities.
        Provide a query (question) and the path to an image file.
        Supported formats: JPG, JPEG, PNG, GIF, WebP.
        The tool will analyze the image and return an answer to your question.
        """
        super().__init__(name, description, path)
        self._init_vision_client()

    def _init_vision_client(self):
        """Initialize OpenAI client for vision API."""
        try:
            from openai import OpenAI
            self.client = OpenAI()
            self.vision_model = "gpt-4o"
        except ImportError:
            self.client = None
            print("Warning: OpenAI package not found. Image question tool will not work.")

    def run(self, query: str, image_path: str):
        """
        Ask a question about an image.

        Args:
            query: The question to ask about the image
            image_path: Path to the image file (relative to working directory or absolute)

        Returns:
            The answer from the vision model
        """
        if self.client is None:
            return self.get_output(e="OpenAI client not initialized. Please install openai package.")

        try:
            import base64

            # Handle both absolute and relative paths
            if not os.path.isabs(image_path):
                # Try relative to full path first
                full_image_path = os.path.join(self.get_path(full=True), image_path)
                if not os.path.exists(full_image_path):
                    # Try relative to base path
                    full_image_path = os.path.join(self.get_path(full=False), image_path)
                    if not os.path.exists(full_image_path):
                        # Try as-is
                        full_image_path = image_path
            else:
                full_image_path = image_path

            # Validate image exists
            if not os.path.exists(full_image_path):
                return self.get_output(e=f"Image file not found: {full_image_path}")

            # Read and encode image
            with open(full_image_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode('utf-8')

            # Determine image format from file extension
            ext = os.path.splitext(full_image_path)[1].lower()
            mime_types = {
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.png': 'image/png',
                '.gif': 'image/gif',
                '.webp': 'image/webp'
            }

            if ext not in mime_types:
                return self.get_output(e=f"Unsupported image format: {ext}. Supported: {list(mime_types.keys())}")

            mime_type = mime_types[ext]

            # Call OpenAI vision API
            response = self.client.chat.completions.create(
                model=self.vision_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": query
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{image_data}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=1000
            )

            answer = response.choices[0].message.content.strip()
            result = f"Question: {query}\nImage: {image_path}\n\nAnswer: {answer}"
            return self.get_output(content=result)

        except Exception as e:
            return self.get_output(e=f"Error analyzing image: {str(e)}")