import os
import shutil

import litellm
import mllm.config
import subprocess
import streamlit as st
from dotenv import load_dotenv
load_dotenv()

from student.agent_input_file import get_input_file_memory, add_feedback_to_instructions
from student.io_interface import set_streamlit_output_interface, echo_with_type, update_chat
from student.mode_assistant import *
from student.mode_library import run_library_mode
from student.mode_student import run_student_mode

mllm.config.default_models.expensive = "openai/gpt-4o"



this_path = os.path.dirname(os.path.realpath(__file__))
memory = get_input_file_memory()
memory_path = os.path.join(this_path, "input_file_memory.json")


if "mode" not in st.session_state:
    st.session_state.mode = "assistant"

if "message_list" not in st.session_state:
    st.session_state.message_list = []

if "input_file" not in st.session_state:
    st.session_state.input_file = None

if "stage" not in st.session_state:
    st.session_state.stage = 0

if "user_input" not in st.session_state:
    st.session_state.user_input = None

if "feedback" not in st.session_state:
    st.session_state.feedback = []

if "instruction" not in st.session_state:
    st.session_state.instruction = None

if "input_file_feedback" not in st.session_state:
    st.session_state.input_file_feedback = None

if "temp_path" not in st.session_state:
    st.session_state.temp_path = set_path("test/")

set_streamlit_output_interface(st)


with st.sidebar:
    st.title('Student AI')
    #st.write(f"Memory path: `{memory_path}`")
    st.radio(
        "Mode",
        key="mode",
        options=["assistant", "student", "library"],
    )
    save_memory_button = st.button("Save memory")
    if save_memory_button:
        memory.save(memory_path)
        st.write("Memory saved to", memory_path)
    
    reset_button = st.button("Reset", on_click=st.session_state.clear)
        
    #openai_api_key = st.text_input("OpenAI API key")



def add_feedback(success):
    if success:
        st.session_state.stage += 1
    else:
        st.session_state.stage = "feedback"

def reset(process=None):
    if process is not None:
        process.terminate()
    add_feedback(False)


if st.session_state.mode == "assistant":
    
    _internal_notes = '''
    1. Get Instructions 
        a) deconstruct
        b) load memory 
        c) ask for overall feedback on deconstruction:
            a) if good: continue
            b) if bad: generate new deconstruction depending on the written feedback
    2. Propose files
        a) Does the molecule id correspond to the desired molecule name?
        b) Does the input file correspond to the desired simulation?
        c) ask for feedback
    3. Simulation Input File
        a) Generation
        b) Feedback
    4. Execute
    5. Ask for feedback
    '''
    
    # echo(st.session_state["stage"])

    input_label = "Your instructions:" if st.session_state.stage == 0 else "Feedback:"
    user_input = st.chat_input(input_label, key="chat_input")
    
    if st.session_state.stage in [0, "feedback"]:
        if user_input:
            message = {"role": "user", "content": user_input}
            #st.session_state.message_list.append(message)
            #update_chat([message])
            echo(user_input, role="user")

            if st.session_state.stage == 0:
                st.session_state.user_input = user_input
            else: 
                st.session_state.feedback.append(user_input)
            st.session_state.stage = 1


    if st.session_state.stage == 1:
        user_input = st.session_state["user_input"]
        feedback = st.session_state["feedback"]

        instructions_prompt = add_feedback_to_instructions(user_input, feedback)
        # a)
        instructions = assistant_decompose_instructions(instructions_prompt)
        if instructions is None:
            add_feedback(False)
        else:
            st.session_state["instructions"] = instructions
            
            # b)
            top_excited_nodes = assistant_search_memory(memory, user_input, query=instructions['simulation'])
            st.session_state["top_excited_nodes"] = top_excited_nodes

            # c)
            st.button("Good", key="input_good", on_click=lambda: add_feedback(True))
            st.button("Bad", key="input_bad", on_click=lambda: add_feedback(False))

    elif st.session_state.stage == 2:
        st.session_state.temp_path = setup_folder(st.session_state.temp_path)
        # a)
        molecule_name = assistant_find_molecule(st.session_state["instructions"]["molecule"])
        st.session_state["instructions"]["other"] += " Remove the MoleculeDefinition row and make sure that the MoleculeName is molecule."
        st.session_state["instructions"]["other"] += "Use 1000 cycles and 100 initializations."
        
        framework = assistant_find_framework(st.session_state["instructions"]["system"])
        st.session_state["instructions"]["other"] += " If a MOF framework is specified (and not 'box'), use the FrameworkName 'mof'. "

        if molecule_name is None or framework is None:
            add_feedback(False)
        else:
            # c)
            st.button("Good", key="select_good", on_click=lambda: add_feedback(True))
            st.button("Bad", key="select_bad", on_click=lambda: add_feedback(False))

    

    elif st.session_state.stage == 3:
        input = st.session_state["user_input"] + st.session_state["instructions"]["other"]
        assistant_simulation_file(input=input, top_excited_nodes=st.session_state["top_excited_nodes"])
        st.button("Good", key="simulation_good", on_click=lambda: add_feedback(True))
        st.button("Bad", key="simulation_bad", on_click=lambda: add_feedback(False))


    elif st.session_state.stage == 4:
        execute = st.button("Execute")
        feedback = st.button("Add feedback")

        process = None

        if execute:
            echo("Executing... (not finished)")

            process = subprocess.Popen(['bash', 'run.sh'], cwd=TEMP_PATH, text=True)
            stdout, stderr = process.communicate()
            echo(stdout)

            if process.returncode != 0:
                echo(f"Script failed: {stderr}")

        if process is not None:
            st.button("Stop Execution", key="stop", on_click=lambda: reset(process))




elif st.session_state.mode == "student":
    user_input = st.chat_input("You teaching instruction")
    if user_input:
        run_student_mode(user_input, memory)
elif st.session_state.mode == "library":
    user_input = st.chat_input("You stimuli for search")
    if user_input:
        run_library_mode(user_input, memory)

