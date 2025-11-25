import re
from typing import Any, Dict, List

from .agent_student import StudentAgent


class TodoListAgent(StudentAgent):
    name = "TodoListAgent"
    current_todo_list: str
    ask_human: bool

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.current_todo_list = ""
        self.ask_human = self.tools.get("ask_human") is not None

    def print_progress(self, message_type: str, details: str):
        """Print progress messages during execution."""
        print(f"[{message_type}]: {details}")

    def run(self, prompt: str, max_iter: int = 25) -> str:
        if self.verbose:
            print(f"[{self.name}] received instruction: {prompt}")

        # Add initial progress message
        self.print_progress(
            "execution_start", f"[{self.name}] starting execution of: {prompt[:100]}..."
        )

        try:
            # Step 1: Decompose the instruction into a todo list
            todo_list_markdown = self.decompose_task(prompt)

            if not todo_list_markdown:
                return "Could not decompose the instruction into actionable tasks. Please provide a more specific instruction."

            # Extract tasks for display
            tasks = self._extract_tasks_from_markdown(todo_list_markdown)

            if self.verbose:
                print(f"[DECOMPOSE] Decomposed into {len(tasks)} tasks:")
                print(todo_list_markdown)

            # Step 2: Execute the todo list using tools
            self.print_progress("execution_phase", "Starting execution of todo list")
            execution_results = self.execute_todo_list_with_tools(
                todo_list_markdown, max_iter
            )

            # Step 3: Prepare summary
            self.print_progress("summary_generation", "Generating execution summary")
            summary = self._generate_summary(prompt, execution_results)

            # Step 4: Store execution in memory for future reference
            # self._store_execution_in_memory(prompt, execution_results)

            self.print_progress(
                "execution_final_complete", f"{self.name} execution complete."
            )

            return summary

        except Exception as e:
            self.print_progress("execution_error", f"Execution failed: {str(e)}")
            if self.verbose:
                print(f"[ERROR] {e}")
            return e

    def _create_memory_from_feedback(
        self, context: str, query: str, user_answer: str
    ) -> str:
        """
        Create and store a memory based on user feedback using LLM to generate content.

        Args:
            context: The context/task being worked on
            query: The specific query
            user_answer: The user's feedback/answer

        Returns:
            The generated memory content as a string, or empty string if failed
        """
        try:
            # Check if user wants to skip
            if user_answer.lower() in ["skip", "no", "n", ""]:
                if self.verbose:
                    print("[INFO] User skipped memory creation")
                return ""

            # Use LLM to generate structured memory content based on user feedback and context
            memory_generation_prompt = f"""Generate a structured memory entry based on the following information.
The memory should be informative and useful for future similar tasks.

Task Context: {context}
Query: {query}
User Feedback: {user_answer}

Create a comprehensive memory entry that includes:
1. The task or scenario this applies to
2. Key lessons or guidance
3. Best practices or tips
4. Any warnings or common pitfalls to avoid

Format it as a clear, concise paragraph that would be useful for an AI agent to reference in the future.
"""

            memory_content = self.single_run(
                memory_generation_prompt,
                expensive=self.expensive,
                info="create_memory_from_feedback",
            )

            if memory_content:
                # Store the generated memory
                title = self.memory.learn(memory_content.strip())
                if self.verbose:
                    print(f"[MEMORY] Stored new memory with title: {title}")
                self.print_progress("memory_created", f"New memory created: {title}")
                return memory_content.strip()

        except Exception as e:
            if self.verbose:
                print(f"[WARNING] Failed to create memory from user input: {str(e)}")

        return ""

    def decompose_task(self, instruction: str) -> str:
        """Generate a markdown todo list directly from the instruction."""
        if self.verbose:
            print(f"[{self.name}] Generating todo list directly from instruction...")

        # Add progress message
        self.print_progress(
            "decomposition_start",
            f"Starting task decomposition for: {instruction[:100]}...",
        )

        # Query memory for relevant past experiences
        relevant_memories = ""
        try:
            if self.verbose:
                print(f"[{self.name}] Querying memory for relevant past experiences...")

            memories = self.memory.retrieve(instruction, top_k=3)

            if memories:
                if self.verbose:
                    print(f"[{self.name}] Found {len(memories)} relevant memories")

                relevant_memories = "\n\nRelevant Past Experiences:\n"
                for i, mem in enumerate(memories, 1):
                    relevant_memories += (
                        f"\n{i}. {mem.title}\n   {mem.content[:300]}...\n"
                    )

                self.print_progress(
                    "memory_retrieved", f"Retrieved {len(memories)} relevant memories"
                )
            else:
                if self.verbose:
                    print(f"[{self.name}] No relevant memories found.")

                # Ask human for input
                ask_human_tool = self.tools.get("ask_human")
                if ask_human_tool and self.ask_human:
                    question = f"""No relevant memories found for: "{instruction}"

Do you have any guidance, best practices, or lessons learned for this type of task?
If yes, please share. If no, you can skip by typing 'skip' or 'no'."""

                    result = ask_human_tool.run(question)

                    # Check if the result contains an error
                    if result and "error" not in result.lower():
                        # Parse the user's answer from the tool output
                        user_answer = ""
                        if "User's answer:" in result:
                            user_answer = result.split("User's answer:")[-1].strip()
                        else:
                            user_answer = result.strip()

                        # Create memory from the answer
                        new_memory = self._create_memory_from_feedback(
                            context=f"Task decomposition for: {instruction}",
                            query=instruction,
                            user_answer=user_answer,
                        )

                        if new_memory:
                            # Add the newly created memory to the context
                            relevant_memories = f"\n\nRelevant Past Experiences (newly added):\n\n1. {new_memory}\n"
                    elif self.verbose:
                        print("[WARNING] Could not get user input for memory")
        except Exception as e:
            if self.verbose:
                print(f"[WARNING] Could not query memory: {str(e)}")
            # Continue without memories if retrieval fails

        # Direct prompt to generate todo list
        todo_generation_prompt = f"""
        Analyze the following instruction and break it down into a actionable markdown todo list.
        You must ensure that the todo list you generate can be solved by the tools

        Instruction: {instruction}

        Available Tools:
        {self.get_tool_schema(json=False)}
        {relevant_memories}

        Rules:
        - Use [ ] for incomplete tasks in markdown format
        - Each task should specify which tool to use from the available tools
        - The steps should be actionable with the available tools
        - Keep tasks focused and specific
        - The todo list can be as simple as one or two items
        - If relevant past experiences are provided, learn from them to create a better todo list

        Return only the markdown todo list, nothing else:
        - [ ] Task 1
        - [ ] Task 2
        """

        try:
            response = self.single_run(
                todo_generation_prompt, expensive=self.expensive, info="decompose_task"
            )
            self.current_todo_list = response.strip()

            # Add progress message to display the generated todo list in chat
            self.print_progress(
                "todo_list_generated",
                f"Generated todo list:\n\n{self.current_todo_list}",
            )

            self.print_progress(
                "decomposition_complete",
                f"Task decomposed into todo list with {len(self._extract_tasks_from_markdown(self.current_todo_list))} tasks",
            )
            return self.current_todo_list
        except Exception as e:
            error_msg = f"Could not generate todo list: {str(e)}"
            if self.verbose:
                print(f"[ERROR] {error_msg}")
            self.print_progress("decomposition_error", error_msg)
            return ""

    def update_todo_list_after_task(self, completed_task: str, task_result: str) -> str:
        """Update the todo list by marking completed task and potentially adding new tasks."""
        if not self.current_todo_list:
            return self.current_todo_list

        update_prompt = f"""
        Current todo list:
        {self.current_todo_list}

        A task was just completed:
        Task: {completed_task}
        Result: {task_result[:500]}...

        Please update the todo list by:
        1. Mark the completed task with [x] instead of [ ]
        2. If the result suggests new tasks are needed, add them as new [ ] items
        3. Return the complete updated todo list in markdown format

        Only return the updated todo list, nothing else.
        """

        try:
            response = self.single_run(
                update_prompt, expensive=self.expensive, info="update_todo_list"
            )
            updated_list = response.strip()
            self.current_todo_list = updated_list
            return updated_list
        except Exception as e:
            if self.verbose:
                print(f"[WARNING] Could not update todo list: {str(e)}")
            return self.current_todo_list

    def _suggest_next_action(
        self, todo_list_markdown: str, current_results: Dict[str, Any]
    ) -> str:
        """Use LLM to suggest the next action based on the todo list and current progress."""

        # Build context from completed tasks
        completed_context = ""
        if current_results["completed_tasks"]:
            completed_context = "\n\nCompleted tasks so far:\n"
            for task in current_results["completed_tasks"]:
                completed_context += (
                    f"- {task['task']}\n  Result: {task['result'][:1000]}...\n"
                )

        # Find the next unfinished task and query memory for it
        next_incomplete_task = self._get_next_incomplete_task(todo_list_markdown)
        relevant_memories = ""

        if next_incomplete_task:
            try:
                if self.verbose:
                    print(
                        f"[{self.name}] Searching memory for next task: {next_incomplete_task}..."
                    )

                # Query memory based on the specific next task
                memories = self.memory.retrieve(next_incomplete_task, top_k=2)

                if memories:
                    if self.verbose:
                        print(
                            f"[{self.name}] Found {len(memories)} relevant memories for next action"
                        )

                    relevant_memories = "\n\nRelevant Past Experiences for This Task:\n"
                    for i, mem in enumerate(memories, 1):
                        relevant_memories += (
                            f"\n{i}. {mem.title}\n   {mem.content[:1000]}...\n"
                        )

                    self.print_progress(
                        "memory_for_task",
                        f"Retrieved {len(memories)} memories for task: {next_incomplete_task[:80]}...",
                    )
                else:
                    if self.verbose:
                        print(
                            f"[{self.name}] No relevant memories found for next task."
                        )

                    # Ask human for input
                    ask_human_tool = self.tools.get("ask_human")
                    if ask_human_tool and self.ask_human:
                        question = f"""No relevant memories found for: "{next_incomplete_task[:200]}"

Context: Executing task: {next_incomplete_task}. Previous tasks: {completed_context[:300]}

Do you have any guidance, best practices, or lessons learned for this type of task?
If yes, please share. If no, you can skip by typing 'skip' or 'no'."""

                        result = ask_human_tool.run(question)

                        # Check if the result contains an error
                        if result and "error" not in result.lower():
                            # Parse the user's answer from the tool output
                            user_answer = ""
                            if "User's answer:" in result:
                                user_answer = result.split("User's answer:")[-1].strip()
                            else:
                                user_answer = result.strip()

                            # Create memory from the answer
                            new_memory = self._create_memory_from_feedback(
                                context=f"Executing task: {next_incomplete_task}. Previous tasks: {completed_context[:500]}",
                                query=next_incomplete_task,
                                user_answer=user_answer,
                            )

                            if new_memory:
                                # Add the newly created memory to the context
                                relevant_memories = f"\n\nRelevant Past Experiences for This Task (newly added):\n\n1. {new_memory}\n"
                        elif self.verbose:
                            print("[WARNING] Could not get user input for memory")
            except Exception as e:
                if self.verbose:
                    print(f"[WARNING] Could not query memory for next task: {str(e)}")
                # Continue without memories if retrieval fails

        next_action_prompt = f"""
        Current Todo List:
        {todo_list_markdown}

        Available Tools:
        {self.get_tool_schema(json=False)}
        {completed_context}
        {relevant_memories}

        Based on the current todo list and progress, what should be the next action to take?

        Notice:
        - Return a action description that contains all the information to run the tool
        - Look at the todo list and identify the next incomplete task (marked with [ ])
        - Provide specific details needed to execute the task
        """

        try:
            response = self.single_run(
                next_action_prompt, expensive=self.expensive, info="suggest_next_action"
            )
            next_action = response.strip()
            return next_action
        except Exception as e:
            if self.verbose:
                print(f"[WARNING] Could not suggest next action: {str(e)}")
            return "done"

    def _get_next_incomplete_task(self, todo_list_markdown: str) -> str:
        """Get the next incomplete task from the todo list."""
        lines = todo_list_markdown.split("\n")

        for line in lines:
            line = line.strip()
            # Match incomplete task: - [ ]
            match = re.match(r"^-\s*\[\s*\]\s*(.+)$", line)
            if match:
                return match.group(1).strip()

        # No incomplete tasks found
        return ""

    def _extract_tool_results_from_chat(self) -> str:
        """Extract tool results from chat messages when no response text is provided."""
        tool_results = []

        # Iterate through chat messages in reverse to get recent tool results
        for message in reversed(self.chat.messages):
            if message.get("role") == "tool":
                tool_name = message.get("name", "unknown")
                content = message.get("content", {})
                if isinstance(content, dict) and content.get("type") == "tool_result":
                    result = content.get("result", "")
                    success = content.get("success", True)
                    status = "✓" if success else "✗"
                    tool_results.append(f"{status} {tool_name}: {result}")

        if tool_results:
            # Return the most recent tool results (reverse back to chronological order)
            return "\n".join(reversed(tool_results))

        return "Task executed (no output captured)"

    def _extract_tasks_from_markdown(self, markdown_todo: str) -> List[str]:
        """Extract task descriptions from markdown todo list."""
        tasks = []
        lines = markdown_todo.split("\n")

        for line in lines:
            line = line.strip()
            # Match markdown checkbox format: - [ ] or - [x]
            match = re.match(r"^-\s*\[\s*[x ]?\s*\]\s*(.+)$", line)
            if match:
                tasks.append(match.group(1).strip())

        return tasks

    def _all_tasks_completed(self, todo_list_markdown: str) -> bool:
        """Check if all tasks in the todo list are completed (marked with [x])."""
        if not todo_list_markdown.strip():
            return True  # Empty todo list means nothing to do

        lines = todo_list_markdown.split("\n")
        has_tasks = False

        for line in lines:
            line = line.strip()
            # Check for markdown checkbox format
            if re.match(r"^-\s*\[", line):
                has_tasks = True
                # If we find any incomplete task ([ ]), return False
                if re.match(r"^-\s*\[\s*\]\s*", line):
                    return False

        # If we found tasks and none were incomplete, all are completed
        # If we found no tasks at all, consider it completed
        return has_tasks

    def execute_todo_list_with_tools(
        self, todo_list_markdown: str, max_iter: int
    ) -> Dict[str, Any]:
        """Execute the markdown todo list using tools through the Agent framework."""
        results = {
            "completed_tasks": [],
            "failed_tasks": [],
            "total_tasks": 0,
        }

        max_iterations = min(max_iter, 20)  # Prevent infinite loops
        iteration = 0

        # Build context from previous completions
        context_summary = ""

        while iteration < max_iterations:
            # Check if all tasks in the todo list are completed (marked with [x])
            if self._all_tasks_completed(self.current_todo_list):
                if self.verbose:
                    print(
                        f"[{self.name}] All tasks completed - todo list shows all [x]"
                    )
                self.print_progress(
                    "execution_complete",
                    "All tasks completed - todo list shows all [x]",
                )
                break

            iteration += 1

            # Use LLM to suggest next action based on current todo list
            next_action = self._suggest_next_action(self.current_todo_list, results)

            if self.verbose:
                print(f"\n[{self.name}] Iteration {iteration}: {next_action}")

            self.print_progress(
                "task_start",
                f"Iteration {iteration} - Starting task {iteration}: {next_action}",
            )

            try:
                # Build action prompt with context from previous tasks
                if context_summary:
                    action_prompt = f"""Execute this specific task: {next_action}

Context from previous completed tasks:
{context_summary}

Please execute the task taking into account the previous work done."""
                else:
                    action_prompt = f"Execute this specific task: {next_action}"

                # Get tool mask for current state
                remove_tools = self.get_tool_mask()
                self.reset_chat()

                result = super().run(
                    action_prompt, remove_tools=remove_tools, max_iter=3
                )

                # If result is empty, try to extract tool results from chat messages
                if not result or result.strip() == "":
                    result = self._extract_tool_results_from_chat()

                # Print the actual result for visibility
                if self.verbose:
                    print(f"[RESULT] Task result: {result[:500]}...")

                self.print_progress("task_result", f"Result: {result[:300]}...")

                # Update context summary with this completion
                context_summary += f"\nTask: {next_action}\nResult: {result[:500]}\n"

                # Update to-do list after task completion
                self.update_todo_list_after_task(next_action, result)

                self.print_progress("todo_list", f"{self.current_todo_list}")

                results["completed_tasks"].append(
                    {"task": next_action, "result": result, "status": "completed"}
                )

            except Exception as e:
                error_result = {
                    "task": next_action,
                    "error": str(e),
                    "status": "failed",
                }
                results["failed_tasks"].append(error_result)
                self.print_progress(
                    "task_error", f"Task failed: {next_action}. Error: {str(e)}"
                )

                if self.verbose:
                    print(f"[FAILED] Task failed: {str(e)}")

        results["total_tasks"] = len(results["completed_tasks"]) + len(
            results["failed_tasks"]
        )
        return results

    def _generate_summary(
        self, original_instruction: str, execution_results: Dict[str, Any]
    ) -> str:
        """Generate a comprehensive summary of the execution."""
        summary = f"""
## Execution Summary

**Original Instruction:** {original_instruction}

### Final Todo List Status:
{self.current_todo_list}

### Completed Tasks ({len(execution_results['completed_tasks'])}):
"""

        for i, task_result in enumerate(execution_results["completed_tasks"], 1):
            summary += f"\n**Task {i}:** {task_result['task']}\n"
            summary += f"**Result:** {task_result['result'][:1000] if task_result['result'] else 'No result captured'}\n"
            if len(task_result["result"]) > 1000:
                summary += "... (result truncated)\n"

        if execution_results["failed_tasks"]:
            summary += (
                f"\n### Failed Tasks ({len(execution_results['failed_tasks'])}):\n"
            )
            for i, task_result in enumerate(execution_results["failed_tasks"], 1):
                summary += f"\n**Task {i}:** {task_result['task']}\n"
                summary += f"**Error:** {task_result['error']}\n"

        summary += f"\n### Total: {execution_results['total_tasks']} tasks processed\n"
        return summary
