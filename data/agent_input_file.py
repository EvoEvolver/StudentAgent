import os
from textwrap import dedent
from typing import List

from mllm import Chat
from mllm.utils import p_map

from latex_parsing import split_latex_sections
from student.agent.agent_memory import Memory, MemoryNode

def decompose_instruction(instruction: str):
    prompt = f"""
    <goal>
    You are an assistant who are helping running a RASPA simulation.
    You task is to help clarify the instruction.
    A instruction can be decomposed into 4 parts:
    - Simulation: the goal of the simulation
    - System: in what setup the simulation of molecule is performed
    - Molecule: the names of the molecule that are being simulated
    - Other parameters
    You are required to decompose the users' instruction into these 4 parts.
    </goal> 
    <instruction>
    {instruction}
    </instruction>
    <output>
    You are required to output a JSON object with the following keys. If the information is not available, the value should be an empty string.
    - "simulation" (str): the description of the simulation.
    - "system" (str): the system of the simulation.
    - "molecule" (str): the molecules of the simulation separated with "." (e.j. "butane . carbon dioxide . nitrogen").
    - "other" (str): other parameters of the simulation.
    </output>
    """
    chat = Chat(dedent=True)
    chat += prompt
    res = chat.complete(parse="dict", cache=True, expensive=True)
    return res


def add_feedback_to_instructions(instruction: str, feedback: list[str]):
    if len(feedback) == 0:
        return instruction

    assert type(feedback) == list

    new_instruction =f"""
    The original instructions are
    <old>
    {instruction}
    </old>
    The user provided additional feedback after reviewing the decomposed instructions from the old instructions:
    <feedback>
    {" <next/> ".join(feedback)}
    </feedback>
    """
    return new_instruction


def extract_input_example(content)->List[str]:
    """
    Extract all the contents surrounded by the \begin{verbatim} and \end{verbatim} and do not include the verbatim tags
    :param content:
    :return:
    """
    import re
    pattern = re.compile(r'\\begin\{verbatim\}(.*?)\\end\{verbatim\}', re.DOTALL)
    match = pattern.search(content)
    if match is None:
        return []
    # return all the matches
    return [dedent(m).strip() for m in pattern.findall(content)]


def decompose_input_file(content: str):
    # Split the content by the first Box or Framework
    first_box = content.find("\nBox ")
    first_framework = content.find("\nFramework ")
    if first_box == -1 and first_framework == -1:
        raise ValueError("No Box or Framework found in the content")
    if first_box == -1:
        first_box = first_framework
    if first_framework == -1:
        first_framework = first_box
    first = min(first_box, first_framework)
    return content[:first], content[first:]


def init_input_file_memory():
    this_path = os.path.dirname(os.path.abspath(__file__))
    with open(this_path + "/data/examples.tex") as f:
        latex_text = f.read()
    sections = split_latex_sections(latex_text, depth=1)

    for section in sections:
        section.title = section.title[section.title.find(":") + 1:].strip()

    simulation_inputs = {}

    for i in range(len(sections)):
        for c in extract_input_example(sections[i].content):
            if c.startswith("SimulationType"):
                running_setup, system_setup = decompose_input_file(c)
                simulation_inputs[sections[i].title] = f"# {sections[i].title}"+"\n"+running_setup+"\n"+system_setup

    memory = Memory()

    for key, res in p_map(decompose_instruction, simulation_inputs.keys()):
        
        content = simulation_inputs[key]
        
        keys = key
        keys.append(res["simulation"])
        
        memory_node = MemoryNode(content = content, keys = keys)
        memory.add(memory_node)

    memory.save(this_path + "/input_file_memory.json")
    return memory


input_file_memory = None

def get_input_file_memory():
    
    global input_file_memory
    if input_file_memory is not None:
        return input_file_memory
    

    this_path = os.path.dirname(os.path.abspath(__file__))
    memory_path = this_path + "/input_file_memory.json"
    if not os.path.exists(memory_path):
        input_file_memory = init_input_file_memory()
    else:
        with open(memory_path) as f:
            input_file_memory.load(memory_path)
    return input_file_memory