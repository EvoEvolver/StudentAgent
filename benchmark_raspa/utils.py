import os
from student.agent.agent_student import StudentAgent

def save(agent : StudentAgent, filename : str):
    filename = "checkpoints/"+filename
    os.makedirs(filename, exist_ok=True)
    agent.save_conversation(f"{filename}/conversation.txt")
    agent.get_memory_agent().save_memory(f"{filename}/memory.txt")
    agent.get_memory_agent().save_conversation(f"{filename}/conversation_memory.txt")

def load(agent : StudentAgent, filename : str):
    filename = "checkpoints/"+filename
    agent.load_conversation(f"{filename}/conversation.txt")
    agent.get_memory_agent().load_memory(f"{filename}/memory.txt")
    agent.get_memory_agent().load_conversation(f"{filename}/conversation_memory.txt")


def read_file(filename):
    try:
        with open(filename, 'r') as file:
            content = file.read()
        return content
    except:
        print(f"Error reading file: {filename}")
        return None

def example_simulation(path):
    ex = {
        "goal" : read_file(os.path.join(path, "goal.txt")),
        "input" : read_file(os.path.join(path, "simulation.input")),
        "output" : read_file(os.path.join(path, "output.txt")),
        "pre" : read_file(os.path.join(path, "prerequisite.txt")),
        "annotation" : read_file(os.path.join(path, "annotation.txt")),
    }
    return ex

