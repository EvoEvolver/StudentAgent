from agent.agent_raspa import RaspaAgent
from agent.agent_student import StudentAgent

import streamlit as st
from streamlit.components.v1 import html


def load_agent(st, mode):
    if (
        "agent" not in st.session_state
        or st.session_state.agent_mode != mode
    ):
        st.session_state.agent_mode = mode
        if mode == "RASPA":
            st.session_state.agent = RaspaAgent()
        else:
            st.session_state.agent = StudentAgent()
        

def load_history(st):
    if "history" not in st.session_state:
        st.session_state.history = []


def run_agent(st):
    user_input = st.chat_input("Type your message…")
    if user_input:
        st.session_state.history.append(("user", user_input))
        with st.spinner("Thinking…"):
            reply = st.session_state.agent.run(prompt=user_input)
        st.session_state.history.append(("assistant", reply))


def render_content(st, message):
    return st.session_state.agent.render_content(message)


def display_chat(st, show_reasoning=False):
    if show_reasoning is False:
        for role, msg in st.session_state.history:
            st.chat_message(role).write(msg)
            
    else:
        messages = st.session_state.agent.chat.messages
        for message in messages:
            role = message['role']
            content = render_content(st, message)
            st.chat_message(role).markdown(content, unsafe_allow_html=True)
