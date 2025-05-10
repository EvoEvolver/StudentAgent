import os
import shutil
import streamlit as st
import subprocess

from student_draft.agent_input_file import get_input_file_memory, decompose_instruction

from student_draft.input_gen.simulation import input_generation, generate_simulation_input
from student_draft.input_gen.generate_mol_definition import generate_molecule_def

from student_draft.io_interface import echo, echo_code
from student_draft.memory import Memory
from input_gen.memory_molecule import init_molecule_name_memory
from input_gen.memory_framework import init_framework_memory, generate_framework_file


def set_path(path="test/"):
    global TEMP_PATH 
    TEMP_PATH = path
    return path

def get_path():
    return TEMP_PATH

def setup_folder(path):
    existing_folders = [
        d for d in os.listdir(path)
        if os.path.isdir(os.path.join(path, d)) and d.isdigit()
    ]
    if existing_folders:
        next_num = max(int(folder) for folder in existing_folders) + 1
    else:
        next_num = 1
    new_path = os.path.join(path, str(next_num))
    os.makedirs(new_path, exist_ok=True)
    shutil.copy(os.path.join(path,"run.sh"), new_path)

    return set_path(new_path)
    

    
def assistant_decompose_instructions(user_input):
    decomposed_instruction = decompose_instruction(user_input)
    if decomposed_instruction["simulation"] == "":
        echo("Please specify the goal of the simulation.")
        return None
    if decomposed_instruction["molecule"] == "":
        echo("Please specify the molecule involved in the simulation.")
        return None
    if decomposed_instruction["system"] == "":
        echo("Please specify the system where the simulation is performed (e.g., a MOF).")
        return None

    echo(f"""Instruction:
        Decomposed instruction:
        Simulation: {decomposed_instruction['simulation']}
        Molecule: {decomposed_instruction['molecule']}
        System: {decomposed_instruction['system']}
    """)

    return decomposed_instruction


def assistant_search_memory(memory: Memory, user_input, query):
    st.write("Searching memory for relevant information...")
    top_excited_nodes = memory.self_consistent_search(user_input, [user_input, query], top_k=3)
    if top_excited_nodes:
        echo("Memory found:")
        for node in top_excited_nodes:
            echo_code(node.content)
    else:
        echo("No memory found.")

    return top_excited_nodes


def assistant_find_molecule(names):
    molecule_names = [name.replace(" ", "_") for name in names]

    mol_memory = init_molecule_name_memory()       # this should be done placed somewhere so that the initialization is not repeated
    res = mol_memory.search(molecule_names, top_k=len(molecule_names))

    if len(res) ==  0:
        echo("No corresponding molecule found in the Trappe database.")
        return

    # node = res[0]
    # name = node.content
    #path = f"{TEMP_PATH}{name}.def"
    names_i = [i.content for i in res]
    names = [name.replace(" ", "_") for name in names_i]

    ids = [i.data['molecule_ID'] for i in res]
    try:
        generate_molecule_def(molecule_ids=ids, names=names, output_dir=TEMP_PATH)
        echo(f"Generated molecule.def file for '{', '.join(names)}' from Trappe Data")
    except Exception as e:
        echo(f"There was some error with the molecule file generation: {e}")
        return None 
    
    return names


def assistant_find_framework(framework):
    
    coremof_memory = init_framework_memory()
    
    res = coremof_memory.search([framework], top_k=1)

    if len(res) ==  0:
        echo("No corresponding molecule found in the coremof database.")
        return None
    elif res[0].data['name'] == "box":
        return "box"
    
    try:
        path = os.path.join(TEMP_PATH, "framework.cif")
        mof_name = res[0].data['name']
        dataset = res[0].data["datasets"][-1]

        generate_framework_file(mof_name, dataset, output_file=path)
        echo(f"Generated framework file for '{mof_name}' from coremof data")
    except Exception as e:
        echo(f"There was some error with the MOF file generation: {e}")
        return None 
    
    return framework




def assistant_simulation_file(input, top_excited_nodes):
    '''
    Simulation Input File Generation
    '''
    input_file = input_generation(input, top_excited_nodes)
    echo_code(input_file)
    
    generate_simulation_input(input_file, output_dir=TEMP_PATH)
    echo("Input file generated.")

    return





if __name__ == '__main__':
    memory = get_input_file_memory()
    user_input = "Run monte carlo for CO2 in box"

    instructions = assistant_decompose_instructions(user_input)
    assistant_search_memory(memory, user_input, query=instructions['simulation'])