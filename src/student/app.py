import os
import shutil
import litellm
import mllm.config

import streamlit as st
from dotenv import load_dotenv
load_dotenv()

from student.app_utils import *
from student.session_manager import create_session, load_session, save_session, SessionState


if "sidebar_state" not in st.session_state:
    st.session_state.sb_state = "expanded"

if "session_dir" not in st.session_state:
    st.session_state.session_dir = None

st.set_page_config(
    page_title="StudentAgent",
    layout="wide",
    initial_sidebar_state=st.session_state.sb_state,
)

if "chat" not in st.session_state:
    st.title("Student Agent")
    
    load_id = st.text_input("Session ID:", key="load_id")
    session_dir = st.text_input("Session Directory (leave empty for default):", key="session_directory")
    if st.button("Load Session"):
        session = load_session(load_id, session_dir=session_dir)
        if session is None:
            st.write("Invalid ID")
        else:
            if session_dir is not None:
                st.session_state.session_dir = session_dir
            setup_agent(st, session)
            st.rerun()

    empty_line(st, 3)
    provider = st.radio("Select LLM Provider:", ["Anthropic","OpenAI"], key="provider_selection")
    mode = st.radio("Select Mode:", ["Student", "RASPA", "Boring"], key="mode")
    if mode == "RASPA":
        load_raspa_agent = st.checkbox("Load RASPA Agent memory")
    if st.button("New Session"):
        new_id = create_session(agent_type=mode, provider=provider)
        session = load_session(new_id, session_dir=st.session_state.session_dir)
        
        setup_agent(st, session)
        if load_raspa_agent is True:
            load_raspa_memory(st)
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
        # Checkbox: show conversation of memory agent instead
        show_mem = st.checkbox("Show MemoryAgent conversation")

        # Checkbox: render memory instead of chat
        show_memory = st.checkbox("Show Memory")

        # Change active learning of agent
        if mode in ["RASPA", "Student"]:
            active_learning = st.checkbox(
                "Enable learning",
                value=st.session_state.get("active_learning", False),
                key="active_learning"
            )
            if active_learning:
                update_active_learning(st, active_learning)
        st.divider()

        # Checkbox: manual or automatic raspa usage?
        show_files = None
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
            else:
                empty_line(st, 1)
            
            # Show file System
            show_files = st.checkbox("Show file manager", key="file_manager")
            st.divider()

        empty_line(st, 3)
        
        
        # Button: Save Session
        session_id = st.session_state.session_id
        st.info(f"Session Id: {session_id}")
        if st.button("💾 Save Session", key="save_conversation"):
            state = st.session_state.session
            save_session(session_id, state, session_dir=st.session_state.session_dir)
            save(st)
            st.success(f"Session saved!")
        else:
            empty_line(st, 2)
        
        empty_line(st, 2)
        st.divider()
        empty_line(st,2)


        # Button to delete the conversational history for the agent
        if st.button("🔄 Reset Agent", key="reset_messages"):
            reset_messages(st)

        # Button: reset the agent + chat                           
        if st.button("Reset All", key="reset"):
            st.session_state.clear()
            st.rerun()    
    
    if show_files:
        path = get_path(st, full=False)
        initial = get_path(st, full=True)
        file_manager = StreamlitFileManager(root_path=path, initial_path=initial)
        file_manager.render()
    

    ##### Conversation #####
    elif show_mem:
        st.header("MemoryAgent Conversation")
        display_chat(st, show_reasoning=show_reasoning, memory=True)
    
    elif show_memory:
        st.header("Memory")
        display_memory(st)
    else:
        st.header("🗨️ StudentAgent")
        load_history(st)
        run_agent(st)
        display_chat(st, show_reasoning)  