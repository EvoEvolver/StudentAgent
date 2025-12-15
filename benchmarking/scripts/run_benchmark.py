"""
Run the RASPA benchmark for an agent
"""

import json
import os

from student.agent.agent_raspa import RaspaAgent

TEST = False
ADDITIONAL_INSTRUCTIONS = "This is a test of your capabilitys to run molecular simulations using RASPA. Strictly follow the steps to solve the task (no verifications, just report the result or error)."


def get_hint(task_name: str, single: bool = True) -> str:
    hints = json.load(
        open(os.path.join(os.path.dirname(__file__), "..", "input", "hints.json"), "r")
    )
    ts = task_name.split("___")

    # Base hint
    hint = hints.get(ts[0], "")

    # Append for large molecules:
    if "l" in ts[-1]:
        hint = hint + hints["l"]

        if ts[-1] == "sl":
            hint = hint + hints["sl"]

    # Append subtask hints for multi-step tasks
    if single is False:
        hint = hints["multi"] + "\n\n" + hint
        if "l" in ts[-1]:
            hint = hint + "\n\n" + hints["rosenbluth"]
        if "iso" in ts[0]:
            hint = hint + "\n\n" + hints["multi_iso"] + ":\n\n" + hints["hvf"]
    print("HINT:", hint)
    return hint


async def run_task(
    agent: RaspaAgent, task: str, give_hint: bool = False, single: bool = True
):
    """
    Run the given task using the provided agent.

    Parameters:
    agent (StudentAgent): The agent to run the task.
    task (str): The task to be executed.
    give_hint (bool): Whether to provide a hint to the agent.

    Returns:
    The result of the task execution.
    """
    task_name = task[0]
    task_instructions = task[1]

    hint = None
    if give_hint:
        hint = get_hint(task_name, single=single)
        if hint is None:
            raise RuntimeWarning(f"No hint found for task {task_name}")
        else:
            task_instructions = (
                f"{task_instructions} \n\nIMPORTANT: HINT to solve the task:\n\n{hint}"
            )

    if TEST is True:
        return {"task": task_name, "hint": hint}
    task_instructions = ADDITIONAL_INSTRUCTIONS + task_instructions
    return await agent.run(task_instructions)


async def run_task_i(
    i: int,
    agent_params: dict,
    give_hint: bool = False,
    single: bool = True,
    variant_index: int = 0,
):
    """
    Run a specific task by index using the provided agent parameters.

    Parameters:
    i (int): The index of the task to run from the RASPA tasks list.
    agent_params (dict): Configuration parameters for the agent.
    give_hint (bool): Whether to provide a hint to the agent.
    single (bool): Whether to use single-framework tasks or multi-framework tasks.

    Returns:
    The result of the task execution.
    """
    # Load the RASPA benchmark tasks
    raspa_tasks_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "input",
        f"tasks_{'single' if single else 'multi'}.json",
    )
    with open(raspa_tasks_path, "r") as f:
        raspa_tasks_data = json.load(f)

    raspa_tasks = [(k, v[variant_index]) for k, v in raspa_tasks_data.items()]

    if i < 0 or i >= len(raspa_tasks):
        raise ValueError(
            f"Task index {i} is out of range. Available tasks: {len(raspa_tasks)}"
        )

    task = raspa_tasks[i]
    task_name = task[0]

    # Adjust path for this specific task
    base_path = agent_params.get("path", "output/raspa_benchmark/")
    task_path = os.path.join(base_path, f"task_{i}_{task_name}/")

    # Create a copy of params to avoid side effects
    current_agent_params = agent_params.copy()
    current_agent_params["path"] = task_path

    agent = RaspaAgent(**current_agent_params)
    result = await run_task(agent, task, give_hint=give_hint, single=single)

    # TODO: save agent state

    return result, agent


if __name__ == "__main__":

    print(run_task_i(1, {}, give_hint=True, single=True))
