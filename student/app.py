import os
import shutil
import litellm
import mllm.config

import streamlit as st
from dotenv import load_dotenv
load_dotenv()

from app_utils import *

path = "output/st/test_henrik/"
memory_path = "memory/test.txt"


if "sidebar_state" not in st.session_state:
    st.session_state.sb_state = "expanded"

st.set_page_config(
    page_title="StudentAgent",
    layout="wide",
    initial_sidebar_state=st.session_state.sb_state,
)


if "chat" not in st.session_state:
    st.title("Student Agent")
    
    provider = st.radio("Select LLM Provider:", ["Anthropic","OpenAI"], key="provider")
    mode = st.radio("Select Mode:", ["Student", "RASPA"], key="mode")

    load_mem = st.checkbox(
        "Load memory",
        value=st.session_state.get("load_mem", True),
        key="load_mem"
    )

    load_agent(st, mode, path)
    
    if st.button("Start Chat"):
        st.session_state.chat = True
        st.session_state.agent_mode = mode
        
        setup_path(path) # only relevant for RASPA agent: creates a new subdir
        if st.session_state.get("load_mem", False):
            load_memory(st, memory_path)

        st.rerun()


if st.session_state.get("chat", False):
    mode = st.session_state.agent_mode
    

    with st.sidebar:
        
        st.header("Settings")
        empty_line(st, 2)

        # Checkbox: show reasoning?
        show_reasoning = st.checkbox(
            "Show reasoning",
            value=st.session_state.get("show_reasoning", True),
            key="show_reasoning"
        )
        show_mem = st.checkbox("Show MemoryAgent conversation")
        st.divider()

        # Checkbox: manual or automatic raspa usage?
        if mode == "RASPA":
            auto = st.checkbox(
                "RASPA auto run",
                value=st.session_state.get("auto_raspa", False),
                key="auto_raspa"
            )
            set_auto(st, auto)

            if not auto:
                # Button: Manually run RASPA
                if st.button("Run RASPA", key="run_raspa_auto"):
                    run_raspa(st)
                    # some kind of progress bar for running raspa run             (TODO advanced: parse output, ...)
            else:
                empty_line(st, 1)

        empty_line(st, 3)
        

        # Button: Save memory
        if st.button("💾 Save Memory", key="save_memory"):
            save_memory(st, memory_path)
            st.success("Memory saved")
        
        empty_line(st, 3)


        # Button: Save conversation
        note = st.text_input("Conversation description", key="note")
        if st.button("💾 Save Conversation", key="save_conversation"):
            if note:
                save_conversation(st, note, path)
                st.success("Conversation saved")
            else:
                st.warning("Please enter a note before saving.")
        else:
            empty_line(st, 2)

        empty_line(st, 3)


        # Button to delete the conversational history for the agent
        if st.button("🔄 Reset Agent", key="reset_messages"):
            reset_messages(st)

        # Button: reset the agent + chat                           
        if st.button("Reset All", key="reset"):
            st.session_state.clear()
            st.rerun()    
    


    ##### Conversation #####
    if show_mem:
        st.header("MemoryAgent Conversation")
        display_chat(st, show_reasoning=show_reasoning, memory=True)
        
    else:
        st.header("🗨️ StudentAgent")
        load_agent(st, mode, path)
        load_history(st)                    # TODO: limit conversation history of the agent (how does mllm do it?
        run_agent(st)
        display_chat(st, show_reasoning)  


    

# TODO: file manager (view only)
# checkbox: show_file_manager