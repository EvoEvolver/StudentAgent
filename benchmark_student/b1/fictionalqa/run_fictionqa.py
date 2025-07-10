import os
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple
import pandas as pd

from student.agent.agent_student import StudentAgent
from mllm import Chat

# ========================
# Configuration
# ========================

RESULTS_DIR = "results/results_fictionqa"
os.makedirs(RESULTS_DIR, exist_ok=True)
MAX_WORKERS = 4

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    filename='logs/fictionqa_run_eval.log',
    filemode='a'
)

# Your list of styles
styles = [
    '_style_blog_num_000',
    #'_style_blog_num_001',
    '_style_corporate_num_000',
    #'_style_corporate_num_001',
    #'_style_corporate_num_002',
    #'_style_encyclopedia_num_000',
    #'_style_encyclopedia_num_001',
    #'_style_news_num_000',
    #'_style_news_num_001',
    #'_style_news_num_002',
    #'_style_news_num_003',
    #'_style_news_num_004',
    '_style_social_num_000',
    #'_style_social_num_001',
    #'_style_social_num_002'
]

# ========================
# Evaluation Functions
# ========================

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
    CORRECT
    INCORRECT (+ very short explaination)
    
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

def check_possible(question, correct_answer, knowledge) -> bool:
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
    res = chat.complete(cache=False, expensive=True).strip()
    return res == "True"

# ========================
# Main Per-Run Logic
# ========================

def run_experiment(event_id: str, style: str, run_id: str,
                  data: pd.DataFrame, fict: pd.DataFrame,
                  prompt: str, test_prompt: str) -> str:
    identifier = f"{run_id}__{event_id}__{style}"
    out_file = os.path.join(RESULTS_DIR, f"{identifier}.jsonl")
    logging.info(f"Running: {identifier}")

    fiction_id = event_id + style
    knowledge_row = fict.loc[fict["fiction_id"] == fiction_id, "fiction"]
    if knowledge_row.empty:
        logging.warning(f"{identifier}: knowledge empty")
        return f"{identifier}: knowledge empty"

    knowledge = knowledge_row.iloc[0]
    qa_data = data[data["fiction_id"] == fiction_id]
    if qa_data.empty:
        logging.warning(f"{identifier}: no questions for fiction")
        return f"{identifier}: no questions for fiction"

    # checkpoint setup
    identifier = os.path.join(run_id, style, event_id)
    path = "checkpoints/single/"+identifier
    os.makedirs(path, exist_ok=True)
    path = os.path.join(path, "agent")


    # Init agent
    agent = StudentAgent(provider="anthropic")
    agent.reset_system_prompt(prompt, append=True)
    agent.run(knowledge)  # Learning step
    agent.save(path)

    # Run predictions and evaluations
    with open(out_file, "w") as fout:
        for _, row in qa_data.iterrows():
            question = row["question"]
            correct_answer = row["natural_answer"]

            # Generate answer
            agent.reset_system_prompt(test_prompt, append=True)
            try:
                prediction = agent.run(question)
                agent.save(path)

            except Exception as e:
                prediction = ""
                logging.error(f"{identifier} failed on question: {question} | {e}")

            # Is answer possible with knowledge?
            try:
                possible = check_possible(question, correct_answer, knowledge)
            except Exception as e:
                possible = None
                logging.error(f"{identifier} possibility check failed: {e}")

            # Evaluation
            try:
                verdict, explanation = eval_answer(question, prediction, correct_answer)
            except Exception as e:
                verdict, explanation = "ERROR", str(e)

            result = {
                "identifier": identifier,
                "event_id": event_id,
                "style": style,
                "run_id": run_id,
                "fiction_id": fiction_id,
                "question": question,
                "correct_answer": correct_answer,
                "prediction": prediction,
                "possible": possible,
                "verdict": verdict,
                "explanation": explanation
            }
            fout.write(json.dumps(result, ensure_ascii=False) + "\n")
            fout.flush()  # Ensure it's written even if interrupted

    logging.info(f"Completed: {identifier}")
    return f"{identifier}: complete"

# ========================
# Loading and Aggregate Evaluation
# ========================

def load_all_results(results_dir: str) -> pd.DataFrame:
    records = []
    for fname in os.listdir(results_dir):
        if fname.endswith(".jsonl"):
            with open(os.path.join(results_dir, fname), "r") as fin:
                for line in fin:
                    records.append(json.loads(line))
    return pd.DataFrame(records)

def aggregate_metrics(df: pd.DataFrame):
    total = len(df)
    correct = df['verdict'].eq('CORRECT').sum()
    possible = df['possible'].eq(True).sum()
    print(f"Total: {total}")
    print(f"Correct: {correct} ({correct/total:.2%})")
    print(f"Possible: {possible} ({possible/total:.2%})")
    # More detailed metrics as desired

# ========================
# Main Entrypoint
# ========================

def main():
    # Load data
    data = pd.read_parquet("hf://datasets/tomg-group-umd/fictionalqa/fict_qa/train-00000-of-00001.parquet")
    fict = pd.read_parquet("hf://datasets/tomg-group-umd/fictionalqa/fictions/train-00000-of-00001.parquet")

    test_prompt = "Consider all the information that you know. Answer the following question as short and precise as possible:"

    prompts = [
        (
        "fictionqa_run1_1",
        """<specialization>
        You are the world's most studious student of all factual, historical, or fictional events in the world. 
        Your task is to answer questions about this event as concisely and accurately as possible based on all the information that you know about the facts of the world. 

        IMPORTANT: If you do not know the answer to a question, make a best effort guess. Do not apologize for your lack of knowledge about the question.
        IMPORTANT: Learn all facts from the input. Try not to miss any details!
        <specialization>
        """
        ),
        (
        "fictionqa_run1_2", 
        """<specialization>
        You are the world's most studious student of all factual, historical, or fictional events in the world. 
        Your task is to memorize and learn all the details.
        You should use this knowledge to answer questions as concisely and accurately as possible based on all the information that you learned. 

        IMPORTANT: If you do not know the answer to a question, make a best effort guess. Do not apologize for your lack of knowledge about the question.
        IMPORTANT: Learn all facts from the input. Try not to miss any details!
        <specialization>
        """
        )
    ]

    # Prepare all (event, style) runs
    runs = [
        (event_id, style, run_id, data, fict, prompt, test_prompt)
        for event_id in data.event_id.unique()[:20]
        for style in styles
        for run_id, prompt in prompts
    ]

    # Run in parallel
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(run_experiment, *args) for args in runs]
        for f in as_completed(futures):
            try:
                result = f.result()
                logging.info(result)
            except Exception as e:
                logging.error(f"Experiment failed: {e}")

    # Load and aggregate results
    df = load_all_results(RESULTS_DIR)
    aggregate_metrics(df)

if __name__ == "__main__":
    main()
