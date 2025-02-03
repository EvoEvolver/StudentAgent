from student.io_interface import echo, echo_box
from student.memory import generate_non_stop_words, Memory
from student.iraspa import get_raspa_memory


def run_library_mode(user_input, memory: Memory):
    echo("Extracting keywords from the input...")
    src_list = []
    src_list += generate_non_stop_words(user_input)
    src_list.append(user_input)
    src_list = list(set(src_list))
    echo("Input keywords:"+str(src_list))
    echo("Searching the memory...")
    top_excited_nodes = memory.search(src_list)
    echo("Search results:")
    for node in top_excited_nodes:
        echo_box(node.content+"\n'''\n"+f'{repr(node.src)}'+"\n'''", title="Memory")


if __name__ == '__main__':
    memory = get_raspa_memory()
    run_library_mode("Run a simulation to find the critical temperature of heptane", memory)