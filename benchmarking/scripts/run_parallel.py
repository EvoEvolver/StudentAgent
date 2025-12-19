"""
Run multiple RASPA benchmark tasks in parallel.

Each task runs in its own folder with output captured to files.
"""

import asyncio
import json
import os
import pickle

from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from run_benchmark import run_task_i

from student.agent.logger import Logger

# Type alias for task result
TaskResult = Tuple[int, Any, Any, Optional[str]]


async def run_single_task_with_logging(
    task_id: int,
    base_path: str,
    agent_params: Dict[str, Any],
    single: bool = True,
    give_hint: bool = True,
    variant_index: int = 0,
) -> TaskResult:
    """
    Run a single task with output captured to a file.

    Args:
        task_id: The task index
        base_path: Base output directory (e.g., output/run_20251220_123456/)
        agent_params: Base agent parameters (path will be overridden)
        single: Whether to use single-framework tasks
        give_hint: Whether to provide hints
        variant_index: Task variant index

    Returns:
        Tuple of (task_id, result, agent, error_message or None)
    """
    # Create task-specific paths
    task_folder = os.path.join(base_path, f"task_{task_id}")
    simulation_folder = os.path.join(task_folder, "simulation")
    stdout_file = os.path.join(task_folder, "std.out")
    agent_pickle_file = os.path.join(task_folder, "agent.pkl")
    log_file = os.path.join(task_folder, "agent_log.json")

    # Create directories
    os.makedirs(simulation_folder, exist_ok=True)

    # Create task-specific agent params with dedicated logger
    task_agent_params = agent_params.copy()
    task_agent_params["path"] = simulation_folder
    task_agent_params["logger"] = Logger(file=log_file, format="json", auto_load=False)
    # Disable verbose output since we're capturing to file
    task_agent_params["verbose"] = False

    result = None
    agent = None
    error_msg = None

    # Redirect stdout and stderr to file during task execution
    with open(stdout_file, "w") as f:
        f.write(f"=== Task {task_id} STARTED at {datetime.now().isoformat()} ===\n")
        f.write(f"Simulation folder: {simulation_folder}\n")
        f.write("=" * 60 + "\n\n")
        f.flush()

        with redirect_stdout(f), redirect_stderr(f):
            try:
                result, agent = await run_task_i(
                    i=task_id,
                    agent_params=task_agent_params,
                    single=single,
                    give_hint=give_hint,
                    variant_index=variant_index,
                )
                f.write(f"\n{'=' * 60}\n")
                f.write(
                    f"=== Task {task_id} COMPLETED at {datetime.now().isoformat()} ===\n"
                )
                f.write(f"Result: {result}\n")

            except Exception as e:
                error_msg = str(e)
                f.write(f"\n{'=' * 60}\n")
                f.write(
                    f"=== Task {task_id} FAILED at {datetime.now().isoformat()} ===\n"
                )
                f.write(f"Error: {error_msg}\n")

            # Save the agent state
            if agent is not None:
                try:
                    save_agent_state(agent, agent_pickle_file)
                except Exception as e:
                    f.write(f"Warning: Failed to pickle agent: {e}\n")

    return task_id, result, agent, error_msg


def save_agent_state(agent, filepath: str) -> None:
    """
    Save agent state to a pickle file.

    Note: Some objects (like loggers with file handles) may not be picklable,
    so we save a state dict instead of the full object.
    """
    state = {
        "path": agent.path,
        "model_name": agent.model_name,
        "verbose": agent.verbose,
        "csd_path": agent.csd_path,
        "ask_human": agent.ask_human,
        "retrieve_memory": agent.retrieve_memory,
        "todo_list": getattr(agent, "todo_list", ""),
        "memory_path": agent.memory_path,
        "memory": (
            agent.memory.to_dict()
            if hasattr(agent, "memory") and agent.memory
            else None
        ),
    }
    with open(filepath, "wb") as f:
        pickle.dump(state, f)


def load_agent_state(filepath: str) -> Dict[str, Any]:
    """Load agent state from a pickle file."""
    with open(filepath, "rb") as f:
        return pickle.load(f)


async def run_tasks_parallel(
    task_ids: List[int],
    base_path: str,
    agent_params: Dict[str, Any],
    single: bool = True,
    give_hint: bool = True,
    variant_index: int = 0,
    max_concurrent: Optional[int] = None,
) -> List[TaskResult]:
    """
    Run multiple tasks in parallel.

    Args:
        task_ids: List of task indices to run
        base_path: Base output directory
        agent_params: Base agent parameters
        single: Whether to use single-framework tasks
        give_hint: Whether to provide hints
        variant_index: Task variant index
        max_concurrent: Maximum concurrent tasks (None = no limit)

    Returns:
        List of (task_id, result, agent, error_message) tuples
    """

    async def run_task(task_id: int) -> TaskResult:
        return await run_single_task_with_logging(
            task_id=task_id,
            base_path=base_path,
            agent_params=agent_params,
            single=single,
            give_hint=give_hint,
            variant_index=variant_index,
        )

    # Apply concurrency limit if specified
    if max_concurrent is not None:
        semaphore = asyncio.Semaphore(max_concurrent)

        async def limited_task(task_id: int) -> TaskResult:
            async with semaphore:
                return await run_task(task_id)

        coroutines = [limited_task(tid) for tid in task_ids]
    else:
        coroutines = [run_task(tid) for tid in task_ids]

    results = await asyncio.gather(*coroutines, return_exceptions=True)

    # Convert exceptions to error tuples
    return [
        (task_ids[i], None, None, str(res)) if isinstance(res, Exception) else res
        for i, res in enumerate(results)
    ]


async def main():
    """Main entry point for parallel task execution."""
    # Configuration
    run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    base_path = os.path.join(os.path.dirname(__file__), "..", "output", run_name)
    task_ids = list(range(4))

    agent_params = {
        "model_name": "openai:gpt-5-mini",
        "retrieve_memory": True,
        "memory_path": os.path.join(
            os.path.dirname(__file__), "..", "memory", "initial_knowledge.json"
        ),
        "verbose": True,
    }

    print(f"Starting parallel run: {run_name}")
    print(f"Output directory: {base_path}")
    print(f"Running {len(task_ids)} tasks: {task_ids}")
    print("=" * 60)

    os.makedirs(base_path, exist_ok=True)

    # Save run configuration
    _save_json(
        os.path.join(base_path, "run_config.json"),
        {
            "run_name": run_name,
            "task_ids": task_ids,
            "agent_params": {k: v for k, v in agent_params.items() if k != "logger"},
            "started_at": datetime.now().isoformat(),
        },
    )

    # Run tasks in parallel
    results = await run_tasks_parallel(
        task_ids=task_ids,
        base_path=base_path,
        agent_params=agent_params,
        single=True,
        give_hint=True,
        variant_index=0,
        max_concurrent=8,
    )

    # Print summary
    _print_summary(results, base_path)

    # Save summary
    successful = sum(1 for _, _, _, err in results if err is None)
    failed = len(results) - successful
    _save_json(
        os.path.join(base_path, "run_summary.json"),
        {
            "run_name": run_name,
            "completed_at": datetime.now().isoformat(),
            "successful": successful,
            "failed": failed,
            "results": [
                {"task_id": tid, "success": err is None, "error": err}
                for tid, _, _, err in results
            ],
        },
    )


def _save_json(filepath: str, data: dict) -> None:
    """Save data to a JSON file."""
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)


def _print_summary(results: List[TaskResult], base_path: str) -> None:
    """Print results summary to stdout."""
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    successful = 0
    failed = 0
    for task_id, _, _, error in results:
        if error:
            print(f"Task {task_id}: FAILED - {error}")
            failed += 1
        else:
            print(f"Task {task_id}: SUCCESS")
            successful += 1

    print(f"\nTotal: {successful} successful, {failed} failed")
    print(f"Output saved to: {base_path}")


if __name__ == "__main__":
    asyncio.run(main())
