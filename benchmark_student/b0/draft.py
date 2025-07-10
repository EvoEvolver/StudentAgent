from student.agent.agent_student import StudentAgent
from typing import List, Tuple
from mllm import Chat

def eval_answer(question, answer, correct_answer):
    prompt = f"""
    You are an evaluator for a knowledge test. 
    Your task is to decide if the <test_answer> is fully correct, given the <question> and the <correct_answer>. 
    Use only the information provided. Do not make assumptions beyond the given answers.

    Inputs:
    <question>
    {question}
    </question>

    <test_answer>
    {answer}
    </test_answer>

    <correct_answer>
    {correct_answer}
    </correct_answer>

    Instructions:
    - Output **CORRECT** if <test_answer> answers the question as good as <correct_answer>.
    - Output **INCORRECT** if <test_answer> is incorrect or does not sufficiently match <correct_answer>.

    Example outputs:
    CORRECT (nothing else)
    INCORRECT + explaination
    
    """

    chat = Chat(dedent=True)
    chat += prompt
    res = chat.complete(cache=False, expensive=True)

    return res

def check_possible(question, correct_answer, knowledge):
    prompt = f"""
    You are an evaluator for a knowledge test. 
    Your task is to decide it is possible to give the <correct_answer> to a <question>, given a <text_knowledge>.
    Use only the information provided. Do not make assumptions beyond the given answers.

    Inputs:
    <question>{question}</question>
    <correct_answer>{correct_answer}</correct_answer>
    <text_knowledge>{knowledge}</text_knowledge>

    Instructions:
    - Output True if it is possible to correctly answer the <question> based on the <text_knowledge>.
    - Output False else

    Example outputs (IMPORTANT: nothing else except True or False):
    True
    False
    """

    chat = Chat(dedent=True)
    chat += prompt
    res = chat.complete(cache=False, expensive=True)

    return res




import pandas as pd
import os

# load data
data = pd.read_parquet("hf://datasets/tomg-group-umd/fictionalqa/fict_qa/train-00000-of-00001.parquet")
fict = pd.read_parquet("hf://datasets/tomg-group-umd/fictionalqa/fictions/train-00000-of-00001.parquet")
styles = [
    '_style_blog_num_000',
    '_style_blog_num_001',
    '_style_corporate_num_000',
    '_style_corporate_num_001',
    '_style_corporate_num_002',
    '_style_encyclopedia_num_000',
    '_style_encyclopedia_num_001',
    '_style_news_num_000',
    '_style_news_num_001',
    '_style_news_num_002',
    '_style_news_num_003',
    '_style_news_num_004',
    '_style_social_num_000',
    '_style_social_num_001',
    '_style_social_num_002'
]

# specific data
run_id = run_id # from cli
event_id = data.event_id.unique()[n_event] # from cli
style = styles[n_style] # from cli
fiction_id = event_id + style

knowledge = fict.loc[fict["fiction_id"] == fiction_id, "fiction"]
if knowledge.empty:
    print(identifier, ": knowledge empty")
    raise e
    
knowledge = knowledge.iloc[0]
qa_test = [(row["question"], row["natural_answer"]) for i, row in data[data["fiction_id"] == fiction_id].iterrows()]


# checkpoint setup
identifier = os.path.join(style, event_id, run_id)
path = "checkpoints/"+identifier
os.makedirs(path, exist_ok=True)
def checkpoint():
    agent.save(path)

# learn
try:
    agent.reset_system_prompt(prompt, append=True)
    agent.run(knowledge)
    checkpoint()

except Exception as e:
    print(identifier, ": learning")
    raise e
    
    
# test
predictions = []
possibility = []
try:
    for question, correct_answer in d["test"]:
        agent.reset_system_prompt(test_prompt, append=True)
        answer = agent.run(question)
        predictions.append(answer)
        
        possible = check_possible(question, correct_answer, knowledge)
        possibility.append(possible)
        checkpoint()

except Exception as e:
    print(identifier, ": test")
    raise e


# baseline








# store results

