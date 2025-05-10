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
    return st.session_state.agent.render_content(message, no_background=True)


def display_chat(st, show_reasoning=False):
    if show_reasoning is False:
        for role, msg in st.session_state.history:
            st.chat_message(role).write(msg)
            
    else:
        messages = st.session_state.agent.chat.messages
        for message in messages:
            role = message['role']
            content = render_content(st, message)
            #add_message(st, role, content, html=True)
            with st.chat_message(role):
                st.html(content)




def add_message(st, role, content, html=True):
 
    background_color="light_amber"
    background_color_set = {
        'light_orange': '#FFF7EB',
        'light_blue': '#F0F8FF',
        'light_green': '#F0FFF0',
        'light_red': '#FFF0F5',
        'light_yellow': '#FFFFE0',
        'light_purple': '#F8F8FF',
        'light_pink': '#FFF0F5',
        'light_cyan': '#E0FFFF',
        'light_lime': '#F0FFF0',
        'light_teal': '#E0FFFF',
        'light_mint': '#F0FFF0',
        'light_lavender': '#F8F8FF',
        'light_peach': '#FFEFD5',
        'light_rose': '#FFF0F5',
        'light_amber': '#FFFFE0',
        'light_emerald': '#F0FFF0',
        'light_platinum': '#F1EEE9',
    }

    if background_color in background_color_set:
        background_color = background_color_set[background_color]
    if not html:
        content = html.escape(content)
        content = content.replace('\n', '<br/>')

    output_html = f'''
    <p style="background-color: {background_color}; padding: 20px; border-radius: 8px; color: #333;">
        <strong>{role}</strong> 
        <br/>
        {content}
    </p>
    '''
    with st.chat_message(role):
        st.html(output_html)
