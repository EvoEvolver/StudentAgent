"""
Run the RASPA benchmark for an agent
"""

import json
import os

import tqdm

from student.agent.agent_raspa import RaspaAgent
from student.agent.agent_student import StudentAgent

TEST = False


def get_hint(task_name: str, single: bool = True) -> str:
    hints = json.load(
        open(os.path.join(os.path.dirname(__file__), "..", "input", "hints.json"), "r")
    )
    ts = task_name.split("___")
    hint = ""

    if ts[0] in hints:
        hint = hints.get(ts[0], "")
    if len(ts) > 1:
        if "l" in ts[-1]:
            hint = hint + hints.get("l", "")
        if ts[-1] == "sl":
            hint = hint + hints.get("sl", "")
    if single is False:
        hint = hint + "\nIMPORTANT: " + hints.get("multi", "")
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
    return agent.run(task_instructions)


def run_multiple_tasks(
    agent: StudentAgent, tasks: list, give_hint: bool = False, single: bool = True
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
    n: int, agent: RaspaAgent = None, single: bool = True, give_hint=False
):
    """
    Test the agent on n tasks from the RASPA benchmark.

    Parameters:
    n (int): The number of tasks to test.
    agent (RaspaAgent): The agent to be tested. If None, a new RaspaAgent will be created.

    Returns:
    A list of results from the task executions.
    """
    if agent is None:
        agent = RaspaAgent(path="../output/testing/")

    # Load the RASPA benchmark tasks
    raspa_tasks_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "input",
        f"tasks_{'single' if single else 'multi'}.json",
    )
    with open(raspa_tasks_path, "r") as f:
        raspa_tasks = json.load(f)
    raspa_tasks = [(k, v) for k, v in raspa_tasks.items()]

    # Run the tasks using the provided agent
    results = run_multiple_tasks(
        agent, raspa_tasks[:n], give_hint=give_hint, single=single
    )

    return results


if __name__ == "__main__":

    print(test_agent_n_tasks(1, give_hint=True, single=True))
