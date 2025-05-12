import os
import re
import shutil
from agent.agent_raspa import RaspaAgent
from agent.agent_student import StudentAgent

import streamlit as st
from streamlit.components.v1 import html


############ Agent utils ############

def load_agent(st, mode, path):
    if (
        "agent" not in st.session_state
        or st.session_state.agent_mode != mode
    ):
        st.session_state.agent_mode = mode
        if mode == "RASPA":
            st.session_state.agent = RaspaAgent(path=path)
        else:
            st.session_state.agent = StudentAgent()

def load_memory(st, memory_path):
    agent = get_agent(st)
    return agent.load_memory(memory_path)

def save_memory(st, memory_path):
    agent = get_agent(st)
    return agent.save_memory(memory_path)

def load_history(st):
    if "history" not in st.session_state:
        st.session_state.history = []

def set_auto(st, auto):
    agent = get_agent(st)
    agent.set_auto(auto)

def save_conversation(st, note, path):
    agent = get_agent(st)
    file = next_note(path)
    if type(agent) == RaspaAgent:
        path = agent.get_path(full=True)
    
    path = os.path.join(path, "conversations")
    os.makedirs(path, exist_ok=True)

    file = os.path.join(path, file)
    agent.save_conversation(filename=file, note=note)
    return


def next_note(path: str) -> str:
    """
    Scan the given directory for files named note_<i>.txt,
    find the highest i, and return the next filename in sequence.
    """
    pattern = re.compile(r"^note_(\d+)\.txt$")
    max_index = -1

    for fname in os.listdir(path):
        match = pattern.match(fname)
        if match:
            idx = int(match.group(1))
            if idx > max_index:
                max_index = idx

    next_index = max_index + 1
    return f"note_{next_index}.txt"


def run_agent(st):
    user_input = st.chat_input("Type your message…")
    if user_input:
        st.session_state.history.append(("user", user_input))
        with st.spinner("Thinking…"):
            agent = get_agent(st)
            reply = agent.run(prompt=user_input)
            #st.session_state.history.append(new_messages)
        st.session_state.history.append(("assistant", reply))


def get_agent(st) -> StudentAgent:
    return st.session_state.agent


def setup_path(path):
    # st.session_state.path = path
    agent = get_agent(st)
    if type(agent) == RaspaAgent:
        new, path = next_folder(path)
        agent.set_path_add(new)
    return path


def reset_messages(st):
    agent = get_agent(st)
    agent.reset_chat()
    return

############ RASPA utils ############

def run_raspa(st):
    with st.spinner("Running..."):
        agent = get_agent(st)
        if type(agent, RaspaAgent):
            agent.tools['raspa'].run()
        else:
            raise Exception("Error running RASPA manually.")
    return True


def next_folder(path):
    os.makedirs(path, exist_ok=True)
    existing_folders = [
        d for d in os.listdir(path)
        if os.path.isdir(os.path.join(path, d)) and d.isdigit()
    ]
    if existing_folders:
        next_num = max(int(folder) for folder in existing_folders) + 1
    else:
        next_num = 1
    new = str(next_num)
    new_path = os.path.join(path, new)
    os.makedirs(new_path, exist_ok=True)

    return new, new_path



############ Streamlit stuff ############


def empty_line(st, n):
    for i in range(n):
        st.write("")


def render_content(st, message):
    return st.session_state.agent.render_content(message, no_background=True)


def display_chat(st, show_reasoning=False):
    if show_reasoning is False:
        for role, msg in st.session_state.history:
            st.chat_message(role).write(msg)
            
    else:
        agent = get_agent(st)
        messages = agent.get_conversation()
        for message in messages:
            if message == "reset":
                st.info("🔄 Conversation has been reset.")
            else:
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
