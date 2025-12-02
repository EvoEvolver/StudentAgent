"""
Run the RASPA benchmark for an agent
"""

import json
import os

import tqdm

from student.agent.agent_raspa import RaspaAgent
from student.agent.agent_student import StudentAgent

TEST = False
ADDITIONAL_INSTRUCTIONS = "This is a test of your capabilitys to run molecular simulations using RASPA. Strictly follow the steps to solve the task (no verifications, just report the result or error)."
RASPA_TASKS = [
    "total___l",
    "hvf___x",
    "rosenbluth___l",
    "henry___s",
    "henry___sl",
    "henry___l",
    "ads_dil___s",
    "ads_iso___s",
]


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

    return hint


def run_task(
    agent: StudentAgent, task: str, give_hint: bool = False, single: bool = True
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
    return agent.run(task_instructions)


def run_multiple_tasks(
    agent: StudentAgent,
    tasks: list,
    give_hint: bool = False,
    single: bool = True,
):
    """
    Run multiple tasks using the provided agent.

    Parameters:
    agent (StudentAgent): The agent to run the tasks.
    tasks (list): A list of tasks to be executed.

    Returns:
    A list of results from the task executions.
    """
    results = []
    for task in tqdm.tqdm(tasks):
        result = run_task(agent, task, give_hint=give_hint, single=single)
        results.append(result)
    return results


def test_agent_n_tasks(
    n: int, agent: RaspaAgent = None, single: bool = True, give_hint=False, task_index=0
):
    """
    Test the agent on n tasks from the RASPA benchmark.

    Parameters:
    n (int): The number of tasks to test.
    agent (RaspaAgent): The agent to be tested. If None, a new RaspaAgent will be created.
    single (bool): Whether to use single-framework tasks or multi-framework tasks.
    give_hint (bool): Whether to provide hints to the agent.
    task_index (int): The index of the task formulation variant to use (0, 1 or 2).

    Returns:
    A list of results from the task executions.
    """
    if agent is None:
        agent = RaspaAgent(path="../output/testing/")
        agent.set_auto(True)

    # Load the RASPA benchmark tasks
    raspa_tasks_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "input",
        f"tasks_{'single' if single else 'multi'}.json",
    )
    with open(raspa_tasks_path, "r") as f:
        raspa_tasks = json.load(f)
    raspa_tasks = [
        (k, v[task_index]) for k, v in raspa_tasks.items() if k in RASPA_TASKS
    ]

    # Run the tasks using the provided agent
    results = run_multiple_tasks(
        agent,
        raspa_tasks[:n],
        give_hint=give_hint,
        single=single,
    )

    return results


if __name__ == "__main__":

    print(test_agent_n_tasks(1, give_hint=True, single=True, task_index=0))
