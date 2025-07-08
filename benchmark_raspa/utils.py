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



from latex_parsing import *

def parse_tex(filename):
    with open(filename) as f:
            latex_text = f.read()
    return construct_tree(split_latex_sections(latex_text, depth=0))