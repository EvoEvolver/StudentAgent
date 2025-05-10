import os
import shutil
import litellm
import mllm.config

import streamlit as st
from dotenv import load_dotenv
load_dotenv()

from app_utils import *

TEMP_PATH = "test/"


if "set_mode" not in st.session_state:
    st.title("Student Agent")
    st.session_state.mode = st.radio("Select Mode:", ["Student", "RASPA"])
    if st.button("Start Chat"):
        st.session_state.set_mode = True


if st.session_state.get("set_mode", False):

    mode = st.session_state.mode

    
    show_reasoning = st.checkbox(
        "Show reasoning",
        key="show_reasoning"
    )

    load_agent(st, mode)
    load_history(st)

    run_agent(st)

    display_chat(st, show_reasoning)


