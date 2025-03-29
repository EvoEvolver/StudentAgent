from mllm import Chat


def input_generation(instruction, retrieved_memory):

    retrieved_memory_in_prompt = ""
    for node in retrieved_memory:
        retrieved_memory_in_prompt += "<memory_item>" + node.content + "</memory_item>\n"
    
    prompt = f"""
    You are required to generate an input for the RASPA software.
    You are required to generate a input file for the following instruction:
    <instruction>
    {instruction}
    </instruction>
    You have retrieved the following memory from your knowledge base for reference:
    <memory>
    {retrieved_memory_in_prompt}
    </memory>
    <output>
    You are required to output a JSON object with the following
    - "input" (str): the input file for the RASPA software.
    </output>
    """
    # print(prompt)
    
    chat = Chat(dedent=True)
    chat += prompt
    res = chat.complete(parse="dict", cache=True, expensive=True)
    return res["input"]


def generate_simulation_input(input_string, output_dir):
    with open(f"{output_dir}simulation.input", "w") as f:
        f.write(input_string)
    