'''
Functions to generate frameworks for the RASPA simulations.
'''
from student.memory import Memory, MemoryNode
import CoRE_MOF
# mof = CoRE_MOF.get_structure("2019-ASR", "ZUZZEB_clean")


def generate_framework_file(name, dataset, output_file="mof.cif"):     
    mof = CoRE_MOF.get_structure(dataset, name)
    mof.to_file(output_file)
    return mof


from collections import defaultdict

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
        #name = molecule["name"].replace("<em>", "").replace("</em>", "")
        new_node.content = name
        new_node.abstract = name
        new_node.data = {
            "name": name,
            "dataset" : dataset
        }
        new_node.src.append(name)
        memory.memory.append(new_node)
    return memory




if __name__ == '__main__':
    memory = init_framework_memory()
    res = memory.search(["ZUZZEB"], top_k=1)
    for node in res:
        print(node.content)