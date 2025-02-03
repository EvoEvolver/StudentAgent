import os, json
import streamlit as st

from student.io_interface import set_streamlit_output_interface
from student.mode_library import run_library_mode
from student.iraspa import get_raspa_memory
from student.mode_student import run_student_mode

this_path = os.path.dirname(os.path.realpath(__file__))
memory = get_raspa_memory()
memory_path = this_path + "/raspa_memory.json"

if "mode" not in st.session_state:
    st.session_state.mode = "assistant"

with st.sidebar:
    st.title('Student AI')
    st.write(f"Memory path: `{memory_path}`")
    st.radio(
        "Mode",
        key="mode",
        options=["assistant", "student", "library"],
    )
    save_memory_button = st.button("Save memory")
    if save_memory_button:
        memory.save(memory_path)
        st.write("Memory saved to", memory_path)


if st.session_state.mode == "assistant":
    user_input = st.chat_input("You instruction for execution")
elif st.session_state.mode == "student":
    user_input = st.chat_input("You teaching instruction")
elif st.session_state.mode == "library":
    user_input = st.chat_input("You stimuli for search")

if user_input:
    set_streamlit_output_interface(st)
    if st.session_state.mode == "assistant":
        ...
    elif st.session_state.mode == "student":
        run_student_mode(user_input, memory)
    elif st.session_state.mode == "library":
        run_library_mode(user_input, memory)