import os
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from typing import Tuple, List, Dict

from student.agent.agent_student import StudentAgent
from mllm import Chat

RESULTS_DIR = "results_wikidyk_jsonl"
os.makedirs(RESULTS_DIR, exist_ok=True)
MAX_WORKERS = 4

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    filename='logs/wikidyk_eval.log',
    filemode='a'
)

def eval_answer(question, answer, correct_answer) -> Tuple[str, str]:
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
    - Output CORRECT if <test_answer> answers the question as good as <correct_answer>.
    - Output INCORRECT if <test_answer> is incorrect or does not sufficiently match <correct_answer>.

    Example outputs (IMPORTANT: never start anyway different):
    CORRECT (+ explaination)
    INCORRECT (+ explaination)
    
    """

    chat = Chat(dedent=True)
    chat += prompt
    res = chat.complete(cache=False, expensive=True).strip()
    if res.startswith("CORRECT"):
        return "CORRECT", ""
    elif res.startswith("INCORRECT"):
        return "INCORRECT", res
    else:
        return "ERROR", res  # handle unexpected output

def run_eval_experiment(fact_id, fact, eval_type, qa, run_id):
    identifier = f"{run_id}__{fact_id}__{eval_type}"
    out_file = os.path.join(RESULTS_DIR, f"{identifier}.jsonl")
    logging.info(f"Running: {identifier}")

    question = qa["prompt"]
    correct_answer = qa["answer"][0] if isinstance(qa["answer"], list) else qa["answer"]

    system_prompt = """You are the world's most studious student of fact about the world. 
    Your task is to answer questions about these facts as concisely and accurately as possible based on all the information that you know about the facts of the world. 

    IMPORTANT: You should answer every single question.
    IMPORTANT: If you do not know the answer to a question, make a best effort guess or write "UNKNOWN_ANSWER". Do not apologize for your lack of knowledge about the question.
    IMPORTANT: Learn all facts from the input. Try not to miss any details!
    """


    # checkpoint setup
    identifier = os.path.join(run_id, fact_id, eval_type)
    path = "checkpoints/"+identifier
    os.makedirs(path, exist_ok=True)
    path = os.path.join(path, "agent")

    agent = StudentAgent(provider="anthropic")
    agent.reset_system_prompt(system_prompt, append=True)
    agent.run(fact)
    agent.save(path)

    try:
        prediction = agent.run(question)
        agent.save(path)
    except Exception as e:
        prediction = ""
        logging.error(f"{identifier} failed on question: {question} | {e}")

    # Evaluate the answer
    try:
        verdict, explanation = eval_answer(question, prediction, correct_answer)
    except Exception as e:
        verdict, explanation = "ERROR", str(e)

    result = {
        "identifier": identifier,
        "fact_id": fact_id,
        "eval_type": eval_type,
        "run_id": run_id,
        "question": question,
        "correct_answer": correct_answer,
        "fact": fact,
        "prediction": prediction,
        "verdict": verdict,
        "explanation": explanation,
    }
    with open(out_file, "w") as fout:
        fout.write(json.dumps(result, ensure_ascii=False) + "\n")

    logging.info(f"Completed: {identifier}")
    return f"{identifier}: complete"

def main():
    wikidyk = pd.read_parquet("hf://datasets/YWZBrandon/wikidyk/data/test-00000-of-00001.parquet")
    data = wikidyk[["fact", "eval"]].drop_duplicates()

    run_id = "test1_wikidyk"

    jobs = []
    for i, row in data[:3].iterrows():
        
        fact_id = str(i)
        fact = row["fact"]
        evals = json.loads(row["eval"]) if isinstance(row["eval"], str) else row["eval"]
        for eval_type, qa in evals.items():
            jobs.append((fact_id, fact, eval_type, qa, run_id))

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(run_eval_experiment, *args) for args in jobs]
        for f in as_completed(futures):
            try:
                result = f.result()
                logging.info(result)
            except Exception as e:
                logging.error(f"Experiment failed: {e}")

if __name__ == "__main__":
    main()
