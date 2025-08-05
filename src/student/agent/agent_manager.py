from .agent_student import StudentAgent
from .agent_raspa import RaspaAgent
from .agent import Agent
from .tools.tools import Tool
from typing import Dict, List, Any
import json
import re
import os
from mllm import Chat

class CallRaspaAgentTool(Tool):
    def __init__(self, raspa_agent):
        super().__init__("call_raspa_agent", "Execute a simulation task using the RASPA agent")
        self.raspa_agent = raspa_agent
    
    def run(self, task: str) -> str:
        """Execute a simulation task using the RASPA agent."""
        try:
            self.raspa_agent.set_auto(True)
            result = self.raspa_agent.run(task, max_iter=25)
            return result
        except Exception as e:
            return f"Error executing RASPA task: {str(e)}"
        finally:
            self.raspa_agent.set_auto(False)

class RespondHumanTool(Tool):
    def __init__(self):
        super().__init__("respond_human", "Provide a response or update to the human user")
    
    def run(self, message: str) -> str:
        """Provide a response or update to the human user."""
        return f"Response to human: {message}"

class ManagerAgent(Agent):
    def __init__(self, output_path="output", csd_path=None, version="v1", provider="anthropic", verbose=False, active_learning=True):
        self.student_agent = StudentAgent(
            version=version, 
            provider=provider, 
            verbose=verbose, 
            active_learning=active_learning
        )
        
        self.raspa_agent = RaspaAgent(
            path=output_path,
            csd_path=csd_path,
            version=version,
            provider=provider,
            verbose=verbose,
            active_learning=active_learning
        )
        
        # Create tools
        manager_tools = [
            CallRaspaAgentTool(self.raspa_agent),
            RespondHumanTool()
        ]
        
        tools = {
            tool.name: tool
            for tool in manager_tools
        }
        
        super().__init__(tools=tools, version=version, provider=provider, verbose=verbose)
        
        self.output_path = output_path
        self.current_todo_list = ""

    def print_progress(self, message_type: str, details: str):
        print(f"[{message_type}]: {details}")

    def get_tool_list_prompt(self):
        # Generate tool descriptions dynamically
        tool_descriptions = []
        tool_names = []
        for tool_name, tool in self.tools.items():
            tool_descriptions.append(f"- {tool_name}: {tool.description}")
            tool_names.append(tool_name)

        tools_text = "\n".join(tool_descriptions)
        return tools_text

    def decompose_task(self, instruction: str) -> str:
        """Generate a markdown todo list directly from the instruction."""
        if self.verbose:
            print("[MANAGER] Generating todo list directly from instruction...")
        
        # Add progress message
        self.print_progress("decomposition_start", f"Starting task decomposition for: {instruction[:100]}...")

        
        # Direct prompt to generate todo list
        todo_generation_prompt = f"""
        Analyze the following instruction and break it down into a actionable markdown todo list.
        You must ensure that the todo list you generate can be solved by the tools
        
        Instruction: {instruction}
        
        Available Tools:
        {self.get_tool_list_prompt()}
        
        Rules:
        - Use [ ] for incomplete tasks in markdown format
        - Each task should specify which tool to use from the available tools
        - The steps should be actionable with the available tools
        - Keep tasks focused and specific
        - The todo list can be as simple as one or two items
        
        Return only the markdown todo list, nothing else:
        - [ ] Task 1
        - [ ] Task 2 
        """
        
        try:
            response = Chat(todo_generation_prompt).complete(cache=False, expensive=True)
            self.current_todo_list = response.strip()
            
            # Add progress message to display the generated todo list in chat
            self.print_progress("todo_list_generated", f"Generated todo list:\n\n{self.current_todo_list}")
            
            self.print_progress("decomposition_complete", f"Task decomposed into todo list with {len(self._extract_tasks_from_markdown(self.current_todo_list))} tasks")
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
            response = Chat(update_prompt).complete(cache=False, expensive=True)
            updated_list = response.strip()
            self.current_todo_list = updated_list
            return updated_list
        except Exception as e:
            if self.verbose:
                print(f"[WARNING] Could not update todo list: {str(e)}")
            return self.current_todo_list

    
    def _suggest_next_action(self, todo_list_markdown: str, current_results: Dict[str, Any]) -> str:
        """Use LLM to suggest the next action based on the todo list and current progress."""
        
        next_action_prompt = f"""
        Current Todo List:
        {todo_list_markdown}
        
        Available Tools:
        {self.get_tool_list_prompt()}
        
        Based on the current todo list and progress, what should be the next action to take?
        
        Notice:
        - Return a action description that contains all the information to run the tool
        """
        
        try:
            response = Chat(next_action_prompt).complete(cache=False, expensive=True)
            next_action = response.strip()
            return next_action
        except Exception as e:
            if self.verbose:
                print(f"[WARNING] Could not suggest next action: {str(e)}")
            return "done"
    
    def _extract_tasks_from_markdown(self, markdown_todo: str) -> List[str]:
        """Extract task descriptions from markdown todo list."""
        tasks = []
        lines = markdown_todo.split('\n')
        
        for line in lines:
            line = line.strip()
            # Match markdown checkbox format: - [ ] or - [x]
            match = re.match(r'^-\s*\[\s*[x ]?\s*\]\s*(.+)$', line)
            if match:
                tasks.append(match.group(1).strip())
        
        return tasks
    
    def _all_tasks_completed(self, todo_list_markdown: str) -> bool:
        """Check if all tasks in the todo list are completed (marked with [x])."""
        if not todo_list_markdown.strip():
            return True  # Empty todo list means nothing to do
        
        lines = todo_list_markdown.split('\n')
        has_tasks = False
        
        for line in lines:
            line = line.strip()
            # Check for markdown checkbox format
            if re.match(r'^-\s*\[', line):
                has_tasks = True
                # If we find any incomplete task ([ ]), return False
                if re.match(r'^-\s*\[\s*\]\s*', line):
                    return False
        
        # If we found tasks and none were incomplete, all are completed
        # If we found no tasks at all, consider it completed
        return has_tasks
    
    def execute_todo_list_with_tools(self, todo_list_markdown: str, max_iter: int, schema: str, remove_tools: List[str]) -> Dict[str, Any]:
        """Execute the markdown todo list using tools through the Agent framework."""
        results = {
            'completed_tasks': [],
            'failed_tasks': [],
            'total_tasks': 0,
        }
        
        max_iterations = min(max_iter, 20)  # Prevent infinite loops
        iteration = 0
        
        while iteration < max_iterations:
            # Check if all tasks in the todo list are completed (marked with [x])
            if self._all_tasks_completed(self.current_todo_list):
                if self.verbose:
                    print("[MANAGER] All tasks completed - todo list shows all [x]")
                self.print_progress("execution_complete", "All tasks completed - todo list shows all [x]")
                break

            iteration += 1
            
            # Use LLM to suggest next action based on current todo list
            next_action = self._suggest_next_action(self.current_todo_list, results)
            
            if self.verbose:
                print(f"\n[MANAGER] Iteration {iteration}: {next_action}")
            
            self.print_progress("task_start", f"Starting task {iteration}: {next_action}")
            
            try:
                # Execute the next action using tools through the parent Agent's framework
                # Create a temporary prompt for this specific action
                action_prompt = f"Execute this specific task: {next_action}"
                
                # Use parent Agent's run method to execute this single action
                result = super().run(action_prompt, max_iter=5, schema=schema, remove_tools=remove_tools)
                
                # Update todo list after task completion
                self.update_todo_list_after_task(next_action, result)
                
                results['completed_tasks'].append({
                    'task': next_action,
                    'result': result,
                    'status': 'completed'
                })
                
                if self.verbose:
                    print(f"[SUCCESS] Task completed successfully")
                    
            except Exception as e:
                error_result = {
                    'task': next_action,
                    'error': str(e),
                    'status': 'failed'
                }
                results['failed_tasks'].append(error_result)
                self.print_progress("task_error", f"Task failed: {next_action}. Error: {str(e)}")
                
                if self.verbose:
                    print(f"[FAILED] Task failed: {str(e)}")
        
        results['total_tasks'] = len(results['completed_tasks']) + len(results['failed_tasks'])
        return results
    
    def run(self, prompt: str, max_iter: int = 10, schema: str = None, remove_tools: List[str] = None) -> str:
        """Main execution method that coordinates between agents using tools."""
        if remove_tools is None:
            remove_tools = []
            
        if self.verbose:
            print(f"[MANAGER] Manager Agent received instruction: {prompt}")
        
        # Add initial progress message
        self.print_progress("execution_start", f"Manager Agent starting execution of: {prompt[:100]}...")
        
        try:
            # Step 1: Decompose the instruction
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
            execution_results = self.execute_todo_list_with_tools(todo_list_markdown, max_iter, schema, remove_tools)
            
            # Step 3: Prepare summary
            self.print_progress("summary_generation", "Generating execution summary")
            summary = self._generate_summary(prompt, todo_list_markdown, execution_results)
            
            self.print_progress("execution_final_complete", "Manager Agent execution complete.")
            
            return summary
            
        except Exception as e:
            error_msg = f"Manager Agent encountered an error: {str(e)}"
            self.print_progress("execution_error", f"Manager Agent execution failed: {str(e)}")
            if self.verbose:
                print(f"[ERROR] {error_msg}")
            return error_msg
    
    def _generate_summary(self, original_instruction: str, todo_list_markdown: str, results: Dict[str, Any]) -> str:
        """Generate a comprehensive summary of the execution."""
        summary = f"""
## Execution Summary

**Original Instruction:** {original_instruction}

### Current Todo List:
{self.current_todo_list}

### Completed Tasks:
"""
        
        for task_result in results['completed_tasks']:
            summary += f"\n **{task_result['task']}**\n"
            if isinstance(task_result['result'], str) and len(task_result['result']) > 200:
                summary += f"   Result: {task_result['result'][:200]}...\n"
            else:
                summary += f"   Result: {task_result['result']}\n"
        
        if results['failed_tasks']:
            summary += "\n### Failed Tasks:\n"
            for task_result in results['failed_tasks']:
                summary += f"\n[FAILED] **{task_result['task']}**\n"
                summary += f"   Error: {task_result['error']}\n"
        
        return summary
    
    def get_student_agent(self) -> StudentAgent:
        """Get the student agent instance."""
        return self.student_agent
    
    def get_raspa_agent(self) -> RaspaAgent:
        """Get the raspa agent instance."""
        return self.raspa_agent
    
    def get_memory_agent(self):
        """Get the memory agent instance from the student agent."""
        return self.student_agent.get_memory_agent()