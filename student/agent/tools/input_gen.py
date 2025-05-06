from student.input_gen.trappe_loader import get_molecule_list
from student.memory import Memory, MemoryNode

from typing import List
from collections import defaultdict

import CoRE_MOF


def get_coremof_list():
    structures = defaultdict(list)
    for dataset in CoRE_MOF.load.__datasets:
        for name in CoRE_MOF.list_structures(dataset):
            structures[name].append(dataset)
    return dict(structures)



def init_framework_memory():
    structures = get_coremof_list()
    memory = Memory()
    for name, dataset in structures.items():

        new_node = MemoryNode()
        # name = molecule["name"].replace("<em>", "").replace("</em>", "")
        new_node.content = name
        new_node.abstract = name
        new_node.data = {
            "name": name,
            "datasets" : dataset
        }
        new_node.src.append(name)
        memory.memory.append(new_node)
    
    box_node = MemoryNode()
    box_node.content = "box"
    box_node.abstract = "box"
    box_node.data = {
        "name": "box",
        "dataset" : ""
    }
    box_node.src.append("box")
    memory.memory.append(box_node)

    return memory


def init_molecule_name_memory(families : List[str] =["UA", "small"]):
    '''
    Parameters 
      families:     a list of strings of valid trappe families. Available are {'EH', 'UA', 'pol', 'small'}. Only UA and small have been tested.
    '''
    molecular_list = get_molecule_list()
    memory = Memory()
    for molecule in molecular_list:

        if molecule['family'] not in families:
            continue

        new_node = MemoryNode()
        name = molecule["name"].replace("<em>", "").replace("</em>", "")
        new_node.content = name
        new_node.abstract = name
        new_node.data = {
            "molecule_ID": molecule["molecule_ID"],
            "family": molecule["family"]
        }
        new_node.src.append(name)
        memory.memory.append(new_node)
    return memory

