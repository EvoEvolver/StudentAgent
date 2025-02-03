import html

output_component = None

def set_streamlit_output_interface(message_component):
    global output_component
    output_component = message_component

def echo(*args):
    if output_component is None:
        print(*args)
        return
    output_component.write(" ".join(map(str, args)))

def echo_json(data):
    if output_component is None:
        print(data)
        return
    output_component.json(data)

def echo_html(content):
    if output_component is None:
        print(content)
        return
    output_component.html(content)
def echo_code(code, language="python"):
    if output_component is None:
        print(code)
        return
    output_component.code(code, language=language)


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
    output_component.html(output_html)