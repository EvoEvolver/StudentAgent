from student.input_gen.trappe_loader import get_molecule_list
from student.memory import Memory, MemoryNode
from typing import List

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


if __name__ == '__main__':
    memory = init_molecule_name_memory()
    res = memory.search(["Methane"], top_k=1)
    for node in res:
        print(node.content)