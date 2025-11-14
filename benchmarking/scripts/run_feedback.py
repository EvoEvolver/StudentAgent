"""
Functions to give feedback to an agent.
"""

from student.agent.agent_student import StudentAgent


def give_feedback(agent: StudentAgent, feedback: str) -> str:
    """
    Give feedback to the agent.

    Parameters:
    agent (StudentAgent): The agent to give feedback to.
    feedback (str): The feedback message.
    """
    return agent.run(feedback)
