from __future__ import annotations

import html
import os
import re
from fibers.tree import Node, Attr
from pydantic import BaseModel



class Section(BaseModel):
    title: str
    content: str
    children: list[Section]

class Item(BaseModel):
    title: str
    content: str
    children: list[Item]


class TypeAttr(Attr):
    def __init__(self, node, type: str):
        super().__init__(node)
        self.type: str = type

    def render(self, rendered):
        rendered.tabs["Type"] = self.type

class DescriptionAttr(Attr):
    def __init__(self, node, description: str):
        super().__init__(node)
        self.description: str = description

    def render(self, rendered):
        rendered.tabs["Description"] = html.escape(self.description)


class DefaultValueAttr(Attr):
    def __init__(self, node, default_value: str):
        super().__init__(node)
        self.default_value: str = default_value

    def render(self, rendered):
        rendered.tabs["Default Value"] = self.default_value

def split_latex_sections(latex_text: str, depth: int = 0) -> list[Section]:
    identifier = depth * "sub" + "section"
    pattern = rf'\\{identifier}\*?\{{(.*?)\}}(.*?)(?=(\\{identifier}\*?\{{|$))'
    matches = re.findall(pattern, latex_text, re.DOTALL)

    sections = []
    for title, body, _ in matches:
        child_sections = split_latex_sections(body, depth + 1)
        # if child_sections:
        #     first_child_start = body.find(rf"\{(depth + 1) * 'sub'}section")
        #     content = body[:first_child_start].strip()
        # else:
        #     content = body.strip()
        content = body.strip()
        sections.append(Section(
            title=title.strip(),
            content=content,
            children=child_sections
        ))

    return sections

def split_latex_items(latex_text: str) -> list[Item]:
    itemize_pattern = r'\\begin\{itemize\}(.*?)\\end\{itemize\}'
    itemize_matches = re.findall(itemize_pattern, latex_text, re.DOTALL)

    items = []
    # the double for loop concatenates items from different itemize blocks
    for block in itemize_matches:
        item_pattern = r'\\item\{(.*?)\}(.*?)(?=(\\item|$))'
        item_matches = re.findall(item_pattern, block, re.DOTALL)

        for title, body, _ in item_matches:
            child_items = split_latex_items(body)
            content = body.strip()
            items.append(Item(
                title=title.strip(),
                content=content,
                children=child_items
            ))

    return items


def construct_tree(sections: list[Section]) -> list[Node]:
    nodes = []
    for section in sections:
        node = Node(title=clean_verb(section.title))
        node.content = clean_verb(section.content).replace(r"\\", "\n")
        if section.children:
            for child_node in construct_tree(section.children):
                node.add_child(child_node)
        else:
            for child_node in construct_item_tree(split_latex_items(section.content)):
                node.add_child(child_node)

        nodes.append(node)
    return nodes

def construct_item_tree(items: list[Item]) -> list[Node]:
    nodes = []
    for item in items:
        item.title = clean_verb(item.title)
        item.content = clean_verb(item.content).replace(r"\\", "\n")
        if detect_square_brackets(item.title):
            node = Node(title=item.title.split("[", 1)[0])
            node.content = item.content
            type = "["+item.title.split("[", 1)[1]
            type = type.replace("$", "")
            TypeAttr(node, type=type)
            if "Default:" in item.content:
                description = item.content.split("Default:", 1)[0]
                default_value = item.content.split("Default:", 1)[1]
                DescriptionAttr(node, description=description)
                DefaultValueAttr(node, default_value=default_value)
                node.content = description
        else:
            node = Node(title=item.title)
            node.content = item.content
        if item.children:
            for child_node in construct_item_tree(item.children):
                node.add_child(child_node)
        nodes.append(node)
    return nodes


def clean_verb(text: str) -> str:
    pattern = r'\\verb(.)(.*?)\1'

    cleaned_text = re.sub(pattern, r'\2', text)
    return cleaned_text


def detect_square_brackets(input_text: str) -> bool:
    pattern = r'\[(.*?)\]'
    return bool(re.search(pattern, input_text))


"""
1.    Provide simulation files for running an energy minimization for CO2 in NU-1000
2.    Find me a structure that has Fm-3m topology and then run methane adsorption in it
3.    please tell me the performance of reinsertion and partial reinsertion moves for benzene adsorption in MFI type zeolites
4.    Run a tertiary mixture adsorption of hexane and its isomers in MFI zeolite at 298K and pressures 0 - 100 Pa
5.    Create simulation files for finding a minimum energy location of methane in MFI zeolite?
6.    Please provide simulation files for running a adsorption simulation of methane in IFMOF-1
7.    Provide simulation input files to find the Henry law coefficient of methane in NU-1000
8.    Provide Input files for running GEMC for methane in NU-1000
9.    simulate dynamic behaviour of hexane in MFI zeolite
10.   Run simulation to find the critical temperature of heptane
"""
# if you need to simulate dynamic behaviour, run molecular dynamics


def init_raspa_memory():
    from student.memory import MemoryNode, Memory
    this_path = os.path.dirname(os.path.abspath(__file__))
    with open(this_path+"/raw_knowledge/input_files.tex") as f:
        latex_text = f.read()

    root = Node(title="RASPA")
    nodes = construct_tree(split_latex_sections(latex_text, depth=0))
    for n in nodes:
        root.add_child(n)
    for node in root.get_nodes_in_subtree():
        if node.has_child():
            node.content = node.title
        node.content = node.content.strip()
    memory_nodes = []
    # flatten the tree
    for node in root.get_nodes_in_subtree():
        if node.content == "":
            continue
        memory_node = MemoryNode()
        memory_node.content = node.title + ":" + node.content
        memory_node.src.append(node.content)
        if node.content != node.title:
            memory_node.src.append(node.title)
        if node.parent() is not None:
            memory_node.src.append(node.parent().content)
            node._parent = root
        memory_nodes.append(memory_node)


    memory = Memory()
    memory.memory = memory_nodes

    memory.save(this_path + "/raspa_memory.json")
    return memory


raspa_memory = None

def get_raspa_memory():
    from student.memory import Memory
    global raspa_memory
    this_path = os.path.dirname(os.path.abspath(__file__))
    if raspa_memory is None:
        raspa_memory = Memory()
    else:
        return raspa_memory
    memory_path = this_path + "/raspa_memory.json"
    if not os.path.exists(memory_path):
        raspa_memory = init_raspa_memory()
    else:
        with open(memory_path) as f:
            raspa_memory.load(memory_path)
    return raspa_memory

if __name__ == '__main__':
    init_raspa_memory()