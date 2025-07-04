import os
import re
import shutil
from student.agent.agent_raspa import RaspaAgent
from student.agent.agent_student import StudentAgent
from student.agent.agent_memory import MemoryAgent

import streamlit as st
from streamlit.components.v1 import html


############ Agent utils ############

def load_agent(st, mode, path):
    if (
        "agent" not in st.session_state
        or st.session_state.agent_mode != mode
    ):
        st.session_state.agent_mode = mode
        provider = st.session_state.provider
        version = "v3"
        
        if mode == "RASPA":
            r = RaspaAgent(path=path, version=version, provider=provider)
            r.tools["framework loader"].coremof = False
            print("Not using CoreMOF database!")
            st.session_state.agent = r
        else:
            st.session_state.agent = StudentAgent(version=version, provider=provider)

def load_memory(st, memory_path):
    agent = get_memory_agent(st)
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
        path = agent.get_full_path()
    
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

def get_memory_agent(st) -> MemoryAgent:
    agent = get_agent(st)
    return agent.get_memory_agent()


def setup_path(path):
    agent = get_agent(st)
    if type(agent) == RaspaAgent:
        new, path = next_folder(path)
        agent.set_path_add(new)
    return new


def reset_messages(st):
    agent = get_agent(st)
    agent.reset_chat()
    return

############ RASPA utils ############

def run_raspa(st):
    with st.spinner("Running..."):
        agent = get_agent(st)
        if type(agent) == RaspaAgent:
            agent.tools["execute raspa"].run()
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
        max_num = max(int(folder) for folder in existing_folders)
        
        if not os.listdir(os.path.join(path, str(max_num))): # empty
            next_num = max_num

        next_num = max_num +1

    else:
        next_num = 1
    new = str(next_num)
    new_path = os.path.join(path, new)
    os.makedirs(new_path, exist_ok=True)

    return new, new_path

    

############ Streamlit stuff ############

def toggle_sidebar():
    st.session_state.sb_state = (
        "collapsed" if st.session_state.sb_state == "expanded" else "expanded"
    )
    st.rerun()

def empty_line(st, n):
    for i in range(n):
        st.write("")


def render_content(st, message):
    return st.session_state.agent.render_content(message, no_background=True)


def display_chat(st, show_reasoning=False, memory=False):
    if show_reasoning is False:
        for role, msg in st.session_state.history:
            st.chat_message(role).write(msg)
            
    else:
        agent = get_agent(st) if memory is False else get_memory_agent(st)
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



### File Manager


import streamlit as st
import os
import shutil
from pathlib import Path
import math
from typing import Optional, List, Dict, Any

class StreamlitFileManager:
    def __init__(
        self,
        root_path: str = "files",
        key_prefix: str = "",
        items_per_page_options: List[int] = [10, 25, 50, 100],
        initial_path: Optional[str] = None

    ):
        """
        Initialize the File Manager component.
        
        Args:
            root_path (str): Root directory for the file manager
            key_prefix (str): Prefix for session state keys to allow multiple instances
            items_per_page_options (List[int]): Options for items per page in pagination
        """
        self.root_path = os.path.abspath(os.path.normpath(root_path))
        if initial_path:
            initial_path_abs = os.path.abspath(
                os.path.normpath(initial_path)
                if os.path.isabs(initial_path)
                else os.path.join(self.root_path, initial_path)
            )
        else:
            initial_path_abs = self.root_path

        # Only use initial_path if it is inside root and exists
        try:
            if os.path.commonpath([self.root_path, initial_path_abs]) != self.root_path:
                self.initial_path = self.root_path
            elif not os.path.exists(initial_path_abs):
                self.initial_path = self.root_path
            else:
                self.initial_path = initial_path_abs
        except ValueError:
            self.initial_path = self.root_path

        self.key_prefix = key_prefix
        self.items_per_page_options = items_per_page_options
        self._init_session_state()
        if not os.path.exists(self.root_path):
            os.makedirs(self.root_path)
        
    def _get_state_key(self, key: str) -> str:
        """Generate a unique session state key."""
        return f"{self.key_prefix}{key}"

    def _init_session_state(self):
        """Initialize session state variables."""
        state_vars = {
            'current_path': self.initial_path,
            'previous_path': None,
            'show_new_folder_input': False,
            'show_upload': False,
            'current_page': 1,
            'items_per_page': 10,
            'upload_success': [],
            'upload_progress': 0,
            'preview_path': None,
            'show_preview': False,
        }

        for key, default_value in state_vars.items():
            state_key = self._get_state_key(key)
            if state_key not in st.session_state:
                st.session_state[state_key] = default_value
    
    
        
    
    def _get_files_and_folders(self) -> List[Dict[str, Any]]:
        """Get list of files and folders in current directory."""
        items = []
        try:
            for item in os.listdir(st.session_state[self._get_state_key('current_path')]):
                full_path = os.path.join(
                    st.session_state[self._get_state_key('current_path')], 
                    item
                )
                is_directory = os.path.isdir(full_path)
                items.append({
                    'name': item,
                    'path': full_path,
                    'is_directory': is_directory,
                    'size': os.path.getsize(full_path) if not is_directory else 0,
                    'modified': os.path.getmtime(full_path)
                })
        except Exception as e:
            print(f"Error accessing path: {e}")
        return items

    def _handle_file_upload(self, uploaded_files) -> bool:
        """Handle file upload process with progress tracking."""
        if not uploaded_files:
            return False
        
        total_files = len(uploaded_files)
        st.session_state[self._get_state_key('upload_success')] = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, uploaded_file in enumerate(uploaded_files, 1):
            try:
                file_path = os.path.join(
                    st.session_state[self._get_state_key('current_path')],
                    uploaded_file.name
                )
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.session_state[self._get_state_key('upload_success')].append(
                    {"name": uploaded_file.name, "status": "success"}
                )
            except Exception as e:
                st.session_state[self._get_state_key('upload_success')].append(
                    {"name": uploaded_file.name, "status": "error", "message": str(e)}
                )
            
            progress = int(idx * 100 / total_files)
            progress_bar.progress(progress)
            status_text.text(f"Uploading file {idx} of {total_files}: {uploaded_file.name}")
        
        success_count = sum(1 for item in st.session_state[self._get_state_key('upload_success')] 
                          if item["status"] == "success")
        st.success(f"Successfully uploaded {success_count} of {total_files} files")
        
        errors = [item for item in st.session_state[self._get_state_key('upload_success')] 
                 if item["status"] == "error"]
        if errors:
            st.error("Failed to upload the following files:")
            for error in errors:
                st.write(f"- {error['name']}: {error['message']}")
        
        return success_count == total_files

    def _create_new_folder(self, folder_name: str) -> bool:
        """Create a new folder in the current directory."""
        try:
            new_folder_path = os.path.join(
                st.session_state[self._get_state_key('current_path')],
                folder_name
            )
            if os.path.exists(new_folder_path):
                st.error("A folder with this name already exists!")
                return False
            os.makedirs(new_folder_path)
            return True
        except Exception as e:
            st.error(f"Error creating folder: {e}")
            return False

    def _delete_item(self, path: str) -> bool:
        """Delete a file or folder."""
        try:
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
            return True
        except Exception as e:
            st.error(f"Error deleting item: {e}")
            return False

    def _format_size(self, size: int) -> str:
        """Format file size in human-readable format."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} TB"

    def _render_pagination(self, total_items: int):
        """Render pagination controls."""
        total_pages = math.ceil(total_items / st.session_state[self._get_state_key('items_per_page')])
        
        if total_pages > 1:
            col1, col2, col3, col4, col5 = st.columns([1, 2, 1, 2, 1])
            current_page = st.session_state[self._get_state_key('current_page')]
            
            with col1:
                if st.button("⏮️", disabled=current_page == 1, key=f"{self.key_prefix}first"):
                    st.session_state[self._get_state_key('current_page')] = 1
                    st.rerun()
            
            with col2:
                if st.button("◀️", disabled=current_page == 1, key=f"{self.key_prefix}prev"):
                    st.session_state[self._get_state_key('current_page')] -= 1
                    st.rerun()
            
            with col3:
                st.write(f"Page {current_page} of {total_pages}")
            
            with col4:
                if st.button("▶️", disabled=current_page == total_pages, key=f"{self.key_prefix}next"):
                    st.session_state[self._get_state_key('current_page')] += 1
                    st.rerun()
            
            with col5:
                if st.button("⏭️", disabled=current_page == total_pages, key=f"{self.key_prefix}last"):
                    st.session_state[self._get_state_key('current_page')] = total_pages
                    st.rerun()

    def render(self):
        """Render the file manager component."""
            # Custom styling example
        st.html("""
            <style>
                /* Style the main container */
                .st-key-file_manager_container {
                    padding: unset;
                    gap: unset;
                }
                .st-key-file_manager_container .stButton button{   
                    padding: unset;
                    border: 0px;
                }
                .st-key-file_manager_container .stButton button:active{   
                    padding: unset;
                    border: 0px;
                    background-color:unset;
                    color:unset;
                }
                .st-key-file_manager_container hr{   
                    margin-top: 15px;
                }

    
            </style>
        """)
        with st.container(border=True, key=f"{self.key_prefix}file_manager_container"):
            # Top navigation bar
            col3, col4, col5 = st.columns([1, 1, 1])
            

            
            with col3:
                with st.popover('📁 New Folder',):
                    with st.container(border=False):
                        folder_name = st.text_input(
                            "Enter folder name:",
                            key=f"{self.key_prefix}new_folder_name"
                        )
                        col1, col2 = st.columns([1, 1])
                        with col1:
                            if st.button("Create", key=f"{self.key_prefix}create_folder"):
                                if folder_name and folder_name.strip():
                                    if self._create_new_folder(folder_name):
                                        st.success(f"Created folder: {folder_name}")
                                        st.session_state[self._get_state_key('show_new_folder_input')] = False
                                        st.rerun()
                                else:
                                    st.error("Please enter a folder name")
            
            with col4:
                if st.button('📤 Upload', key=f"{self.key_prefix}upload"):
                    st.session_state[self._get_state_key('show_upload')] = True
            
            with col5:
                items_per_page = st.selectbox(
                    "Items per page",
                    options=self.items_per_page_options,
                    key=f"{self.key_prefix}items_per_page_selector",
                    label_visibility="collapsed"
                )
                if items_per_page != st.session_state[self._get_state_key('items_per_page')]:
                    st.session_state[self._get_state_key('items_per_page')] = items_per_page
                    st.session_state[self._get_state_key('current_page')] = 1
                    st.rerun()
             # Upload Component
            if st.session_state[self._get_state_key('show_upload')]:
                with st.container(border=True):
                    uploaded_files = st.file_uploader(
                        "Choose files",
                        accept_multiple_files=True,
                        key=f"{self.key_prefix}file_uploader"
                    )
                    
                    col1, col2 = st.columns([1, 5])
                    with col1:
                        if st.button("Close", key=f"{self.key_prefix}close_upload"):
                            st.session_state[self._get_state_key('show_upload')] = False
                            st.session_state[self._get_state_key('upload_success')] = []
                            st.rerun()
                    
                    if uploaded_files:
                        if self._handle_file_upload(uploaded_files):
                            st.session_state[self._get_state_key('show_upload')] = False
                            st.rerun()
            st.divider()
            col1, col2 = st.columns([1,1])
            # New Folder Input
            with col1:
                st.text(f"Current: {st.session_state[self._get_state_key('current_path')]}")
            with col2:
                if st.session_state[self._get_state_key('current_path')] != self.root_path:
                    if st.button('⬆️ Up', key=f"{self.key_prefix}up"):
                        st.session_state[self._get_state_key('current_path')] = str(
                            Path(st.session_state[self._get_state_key('current_path')]).parent
                        )
                        st.session_state[self._get_state_key('show_new_folder_input')] = False
                        st.session_state[self._get_state_key('current_page')] = 1
                        st.rerun()
                       
            st.divider()
            
            # File/Folder List
            items = self._get_files_and_folders()
            items.sort(key=lambda x: (not x['is_directory'], x['name'].lower()))
            
            start_idx = (st.session_state[self._get_state_key('current_page')] - 1) * \
                       st.session_state[self._get_state_key('items_per_page')]
            end_idx = start_idx + st.session_state[self._get_state_key('items_per_page')]
            paginated_items = items[start_idx:end_idx]
            
            for item in paginated_items:
                col1, col2, col3, col4 = st.columns([4, 1, 1, 1])
                is_active = (st.session_state[self._get_state_key('previous_path')] == item['path'])
                
                with col1:
                    if item['is_directory']:
                        if st.button(
                            f"📁 {item['name']}", 
                            key=f"{self.key_prefix}dir_{item['path']}", 
                        ):
                            st.session_state[self._get_state_key('previous_path')] = \
                                st.session_state[self._get_state_key('current_path')]
                            st.session_state[self._get_state_key('current_path')] = item['path']
                            st.session_state[self._get_state_key('show_new_folder_input')] = False
                            st.session_state[self._get_state_key('current_page')] = 1
                            st.rerun()
                    else:
                        st.text(f"📄 {item['name']}")
                        #if st.button("👁️ Preview", key=f"{self.key_prefix}preview_{item['path']}"):
                        #    st.session_state[self._get_state_key('preview_path')] = item['path']
                        #    st.session_state[self._get_state_key('show_preview')] = True
                        #    st.rerun()
                
                with col2:
                    if not item['is_directory']:
                        st.text(self._format_size(item['size']))
                
                with col3:
                    if st.button('🗑️', key=f"{self.key_prefix}del_{item['path']}", help="Delete item"):
                        if self._delete_item(item['path']):
                            st.success(f"Deleted {item['name']}")
                            st.rerun()
                with col4:
                    if not item['is_directory']:
                        if st.button("👁️ Preview", key=f"{self.key_prefix}preview_{item['path']}"):
                            st.session_state[self._get_state_key('preview_path')] = item['path']
                            st.session_state[self._get_state_key('show_preview')] = True
                            st.rerun()
            
            st.divider()

            if st.session_state.get(self._get_state_key('show_preview'), False):
                preview_path = st.session_state.get(self._get_state_key('preview_path'), None)
                if preview_path and os.path.isfile(preview_path):
                    st.subheader(f"Preview: {os.path.basename(preview_path)}")
                    # Only preview for small text files for simplicity
                    try:
                        with open(preview_path, "r", encoding="utf-8") as f:
                            file_content = f.read(20000)  # read only first 5000 chars
                        st.code(file_content)
                    except Exception as e:
                        st.warning(f"Cannot preview this file: {e}")
                    if st.button("Close Preview", key=f"{self.key_prefix}close_preview"):
                        st.session_state[self._get_state_key('show_preview')] = False
                        st.session_state[self._get_state_key('preview_path')] = None
                        st.rerun()


            self._render_pagination(len(items))

    @property
    def current_path(self) -> str:
        """Get current directory path."""
        return st.session_state[self._get_state_key('current_path')]

    @property
    def selected_items(self) -> List[str]:
        """Get list of selected items (for future implementation)."""
        return []  # Placeholder for future feature
    

# def main():
#    st.title("File Manager Demo")
    # Basic usage with default settings
#    file_manager = StreamlitFileManager(root_path="/home/")
#    file_manager.render()
    # You can access the current path
#    st.write(f"Current path: {file_manager.current_path}")
#main()