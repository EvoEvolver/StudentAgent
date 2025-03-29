import streamlit as st
import subprocess

from student.agent_input_file import get_input_file_memory, decompose_instruction

from student.input_gen.simulation import input_generation, generate_simulation_input
from student.input_gen.generate_mol_definition import generate_molecule_def

from student.io_interface import echo, echo_code
from student.memory import Memory
from input_gen.memory_molecule import init_molecule_name_memory
from input_gen.memory_framework import init_framework_memory, generate_framework_file

TEMP_PATH = "test/"


    
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

    echo(f"Instruction: {user_input}")
    echo("Decomposed instruction:")
    echo(f"Simulation: {decomposed_instruction['simulation']}")
    echo(f"Molecule: {decomposed_instruction['molecule']}")
    echo(f"System: {decomposed_instruction['system']}")

    return decomposed_instruction


def assistant_search_memory(memory: Memory, user_input, query):
    echo("Searching memory for relevant information...")
    top_excited_nodes = memory.self_consistent_search(user_input, [user_input, query], top_k=3)
    echo("Memory found:")
    for node in top_excited_nodes:
        echo_code(node.content)
    if not top_excited_nodes:
        echo("No memory found.")

    return top_excited_nodes


def assistant_find_molecule(molecule):
    mol_memory = init_molecule_name_memory()       # this should be done placed somewhere so that the initialization is not repeated
    res = mol_memory.search([molecule], top_k=1)

    if len(res) ==  0:
        echo("No corresponding molecule found in the Trappe database.")
        return

    node = res[0]
    name = node.content
    #path = f"{TEMP_PATH}{name}.def"
    path = f"{TEMP_PATH}molecule.def"
    try:
        generate_molecule_def(molecule_id=node.data['molecule_ID'], name=name, family=node.data['family'], output_file=path)
        echo(f"Generated molecule.def file for '{name}' from Trappe Data")
    except Exception as e:
        echo(f"There was some error with the molecule file generation: {e}")
        return None 
    
    return name


def assistant_find_framework(framework):
    
    coremof_memory = init_framework_memory()
    res = coremof_memory.search([framework], top_k=1)

    if len(res) ==  0:
        echo("No corresponding molecule found in the coremof database.")
        return
    
    try:
        path = f"{TEMP_PATH}framework.cif"
        mof_name = res[0].data['name']
        dataset = res[0].data["datasets"][-1]

        generate_framework_file(mof_name, dataset, output_file=path)
        echo(f"Generated molecule.def file for '{mof_name}' from coremof data")
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