import html

import streamlit as st

output_component = None
message_list: None | list = None

def set_streamlit_output_interface(message_component):
    global message_list
    message_list = st.session_state.get("message_list", [])
    global output_component
    output_component = message_component

    for message in message_list:
        if message["role"] == "user":
            st.chat_message("user").write(message["content"])
        else:
            st.chat_message("assistant").write(message["content"])


def echo_with_type(message_type, message):
    global output_component
    #print("echo_with_type", message_type, message)
    #message_list.append((message_type, message))
    if message_type == "echo":
        st.write(message)
    elif message_type.startswith("code"):
        code_type = message_type.split("_")[1]
        st.code(message, language=code_type)
    elif message_type == "html":
        st.html(message)
    
    st.session_state["message_list"].append({"role": "assistant", "content": message})


def echo(args):
    if output_component is None:
        print(args)
        return
    echo_with_type("echo", args)


def echo_code(code, language="python"):
    if output_component is None:
        print(code)
        return
    echo_with_type(f"code_{language}", code)


def echo_box(content: str, title: str, background_color: str="light_blue", html_content: bool=False):
    """
    Display a chat message formatted with the given parameters.

    Args:
    agent_name (str): Name of the agent speaking.
    background_color (str): Background color for the chat bubble.
    content (str): Content of the message, which may include plain text or HTML.

    Returns:
    str: HTML formatted string representing the chat message.
    """
    if output_component is None:
        print(title+":")
        print(content)
        return

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
    if not html_content:
        content = html.escape(content)
        content = content.replace('\n', '<br/>')

    output_html = f'''
    <p style="background-color: {background_color}; padding: 20px; border-radius: 8px; color: #333;">
        <strong>{title}</strong> 
        <br/>
        {content}
    </p>
    '''
    echo_with_type("html", output_html)