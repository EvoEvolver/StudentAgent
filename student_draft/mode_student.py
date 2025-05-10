from mllm import Chat
from student_draft.io_interface import echo, echo_box
from student_draft.latex_parsing import get_raspa_memory
from student_draft.memory import Memory, MemoryNode


def add_knowledge_by_instruction(memory: Memory, instruction: str):
    prompt = f"""
    <task>
    You are required to analyze an instruction about learning and break the instruction into two parts: source and target.
    The target of the instruction is what you need to recall in the situation.
    The source of the instruction is the situation when you need to recall the target.
    </task>
    <example>
    Instruction: "Do not do something when XXX."
    Source: "XXX"
    </example>
    <example>
    Instruction: "XXX means Something"
    Source: "XXX"
    </example>
    <example>
    Instruction: "If you need to do XXX, then do B"
    Source: "XXX"
    <instruction>
    {instruction}
    </instruction>
    <output>
    You are required to output a JSON object with the following
    "analysis" (str): the analysis of the instruction.
    "source" (str): the source of the instruction.
    """
    chat = Chat(dedent=True)
    chat += prompt
    res = chat.complete(parse="dict", cache=True, expensive=True)

    new_memory_node = MemoryNode()
    new_memory_node.content = instruction
    new_memory_node.src.append(res["source"])
    new_memory_node.abstract = instruction
    new_memory_node.terminal = False


    print(len(memory.memory))

    memory.memory.append(new_memory_node)

    print(len(memory.memory))

    echo_box(f"{res['analysis']}", "Analysis")
    echo_box(f"{res['source']}", "Stimuli of memory")
    echo_box(f"{instruction}", "Content of memory")


def run_student_mode(user_input, memory: Memory):
    add_knowledge_by_instruction(memory,
                                 user_input)

if __name__ == '__main__':
    memory = get_raspa_memory()
    instruction = "remember that GEMC means Gibbs ensemble Monte Carlo"
    run_student_mode(instruction, memory)

    from student.mode_library import run_library_mode
    run_library_mode("Run GEMC",
                     memory)