import os
import shutil
import litellm
import mllm.config

import streamlit as st
from dotenv import load_dotenv
load_dotenv()

from app_utils import *

path = "output/st/"
memory_path = "memory/test.txt"


if "chat" not in st.session_state:
    st.title("Student Agent")
    
    mode = st.radio("Select Mode:", ["Student", "RASPA"], key="mode")

    if "load_mem" not in st.session_state:
        load_mem = st.checkbox(
            "Load memory",
            value=st.session_state.get("load_mem", True),
            key="load_mem"
        )
    
    load_agent(st, mode, path)
    setup_path(path) # only relevant for RASPA agent: creates a new subdir

    if st.button("Start Chat"):
        st.session_state.chat = True
        st.session_state.agent_mode = mode
        
        if st.session_state.get("load_mem", False):
            load_memory(st, memory_path)

        st.rerun()



if st.session_state.get("chat", False):
    mode = st.session_state.agent_mode

    with st.sidebar:
        st.header("Settings")

        # Button: Save memory
        if st.button("💾 Save Memory", key="save_memory"):
            save_memory(st, memory_path)
            st.write("memory saved")
        
        # Checkbox: show reasoning?
        show_reasoning = st.checkbox(
            "Show reasoning",
            value=st.session_state.get("show_reasoning", True),
            key="show_reasoning"
        )

        # Checkbox: manual or automatic raspa usage?
        if mode == "RASPA":
            auto = st.checkbox(
                "RASPA auto run",
                value=st.session_state.get("auto_raspa", False),
                key="auto_raspa"
            )
            if auto:

                # Button: Manually run RASPA
                if st.button("Run RASPA", key="run_raspa"):
                    run_raspa(st)


        
        # rename subfolder or add custom note + conversation history  (TODO: add load/save messages to agent)
        # some kind of progress bar for running raspa run             (TODO advanced: parse output, ...)
        
        # Button to delete the conversational history for the agent   (TODO: add st message history OR rather make a mask for the agent's messages in the running)
        # Button to reset the agent + chat                            (TODO: reset some session_states, reset the agent + rerun())


    load_agent(st, mode, path)

    load_history(st)

    run_agent(st)

    display_chat(st, show_reasoning)

    


# TODO: file manager (view only)